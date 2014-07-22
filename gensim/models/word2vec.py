#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Deep learning via word2vec's "skip-gram and CBOW models", using either
hierarchical softmax or negative sampling [1]_ [2]_.

The training algorithms were originally ported from the C package https://code.google.com/p/word2vec/
and extended with additional functionality.

For a blog tutorial on gensim word2vec, with an interactive web app trained on GoogleNews, visit http://radimrehurek.com/2014/02/word2vec-tutorial/

**Install Cython with `pip install cython` to use optimized word2vec training** (70x speedup [3]_).

Initialize a model with e.g.::

>>> model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

Persist a model to disk with::

>>> model.save(fname)
>>> model = Word2Vec.load(fname)  # you can continue training with the loaded model!

The model can also be instantiated from an existing file on disk in the word2vec C format::

  >>> model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
  >>> model = Word2Vec.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format

You can perform various syntactic/semantic NLP word tasks with the model. Some of them
are already built-in::

  >>> model.most_similar(positive=['woman', 'king'], negative=['man'])
  [('queen', 0.50882536), ...]

  >>> model.doesnt_match("breakfast cereal dinner lunch".split())
  'cereal'

  >>> model.similarity('woman', 'man')
  0.73723527

  >>> model['computer']  # raw numpy vector of a word
  array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)

and so on.

If you're finished training a model (=no more updates, only querying), you can do

  >>> model.init_sims(replace=True)

to trim unneeded model memory = use (much) less RAM.

.. [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
.. [2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.
.. [3] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/
"""

import logging
import sys
import os
import heapq
import time
from copy import deepcopy
import threading
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from numpy import exp, dot, zeros, outer, random, dtype, get_include, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum

logger = logging.getLogger("gensim.models.word2vec")


from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange


try:
    from gensim_addons.models.word2vec_inner import train_sentence_sg, train_sentence_cbow, FAST_VERSION
except ImportError:
    try:
        # try to compile and use the faster cython version
        import pyximport
        models_dir = os.path.dirname(__file__) or os.getcwd()
        pyximport.install(setup_args={"include_dirs": [models_dir, get_include()]})
        from word2vec_inner import train_sentence_sg, train_sentence_cbow, FAST_VERSION
    except:
        # failed... fall back to plain numpy (20-80x slower training than the above)
        FAST_VERSION = -1

        def train_sentence_sg(model, sentence, alpha, work=None):
            """
            Update skip-gram model by training on a single sentence.

            The sentence is a list of Vocab objects (or None, where the corresponding
            word is not in the vocabulary. Called internally from `Word2Vec.train()`.

            This is the non-optimized, Python version. If you have cython installed, gensim
            will use the optimized version from word2vec_inner instead.

            """
            if model.negative:
                # precompute negative labels
                labels = zeros(model.negative + 1)
                labels[0] = 1.0

            for pos, word in enumerate(sentence):
                if word is None:
                    continue  # OOV word in the input sentence => skip
                reduced_window = random.randint(model.window)  # `b` in the original word2vec code

                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)
                for pos2, word2 in enumerate(sentence[start : pos + model.window + 1 - reduced_window], start):
                    # don't train on OOV words and on the `word` itself
                    if word2 and not (pos2 == pos):
                        l1 = model.syn0[word2.index]
                        neu1e = zeros(l1.shape)

                        if model.hs:
                            # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
                            l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
                            fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  #  propagate hidden -> output
                            ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
                            model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
                            neu1e += dot(ga, l2a) # save error

                        if model.negative:
                            # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
                            word_indices = [word.index]
                            while len(word_indices) < model.negative + 1:
                                w = model.table[random.randint(model.table.shape[0])]
                                if w != word.index:
                                    word_indices.append(w)
                            l2b = model.syn1neg[word_indices] # 2d matrix, k+1 x layer1_size
                            fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output
                            gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
                            model.syn1neg[word_indices] += outer(gb, l1) # learn hidden -> output
                            neu1e += dot(gb, l2b) # save error

                        model.syn0[word2.index] += neu1e  # learn input -> hidden

            return len([word for word in sentence if word is not None])

        def train_sentence_cbow(model, sentence, alpha, work=None, neu1=None):
            """
            Update CBOW model by training on a single sentence.

            The sentence is a list of Vocab objects (or None, where the corresponding
            word is not in the vocabulary. Called internally from `Word2Vec.train()`.

            This is the non-optimized, Python version. If you have cython installed, gensim
            will use the optimized version from word2vec_inner instead.

            """
            if model.negative:
                # precompute negative labels
                labels = zeros(model.negative + 1)
                labels[0] = 1.

            for pos, word in enumerate(sentence):
                if word is None:
                    continue  # OOV word in the input sentence => skip
                reduced_window = random.randint(model.window) # `b` in the original word2vec code
                start = max(0, pos - model.window + reduced_window)
                window_pos = enumerate(sentence[start : pos + model.window + 1 - reduced_window], start)
                word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
                l1 = np_sum(model.syn0[word2_indices], axis=0) # 1 x layer1_size
                if word2_indices and model.cbow_mean:
                    l1 /= len(word2_indices)
                neu1e = zeros(l1.shape)

                if model.hs:
                    l2a = model.syn1[word.point] # 2d matrix, codelen x layer1_size
                    fa = 1. / (1. + exp(-dot(l1, l2a.T))) # propagate hidden -> output
                    ga = (1. - word.code - fa) * alpha # vector of error gradients multiplied by the learning rate
                    model.syn1[word.point] += outer(ga, l1) # learn hidden -> output
                    neu1e += dot(ga, l2a) # save error

                if model.negative:
                    # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
                    word_indices = [word.index]
                    while len(word_indices) < model.negative + 1:
                        w = model.table[random.randint(model.table.shape[0])]
                        if w != word.index:
                            word_indices.append(w)
                    l2b = model.syn1neg[word_indices] # 2d matrix, k+1 x layer1_size
                    fb = 1. / (1. + exp(-dot(l1, l2b.T))) # propagate hidden -> output
                    gb = (labels - fb) * alpha # vector of error gradients multiplied by the learning rate
                    model.syn1neg[word_indices] += outer(gb, l1) # learn hidden -> output
                    neu1e += dot(gb, l2b) # save error

                model.syn0[word2_indices] += neu1e # learn input -> hidden, here for all words in the window separately

            return len([word for word in sentence if word is not None])


class Vocab(object):
    """A single vocabulary item, used internally for constructing binary trees (incl. both word leaves and inner nodes)."""
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"


class Word2Vec(utils.SaveLoad):
    """
    Class for training, using and evaluating neural networks described in https://code.google.com/p/word2vec/

    The model can be stored/loaded via its `save()` and `load()` methods, or stored/loaded in a format
    compatible with the original word2vec implementation via `save_word2vec_format()` and `load_word2vec_format()`.

    """
    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
        sample=0, seed=1, workers=1, min_alpha=0.0001, sg=1, hs=1, negative=0, cbow_mean=0):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        this module for such examples.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `sg` defines the training algorithm. By default (`sg=1`), skip-gram is used. Otherwise, `cbow` is employed.
        `size` is the dimensionality of the feature vectors.
        `window` is the maximum distance between the current and predicted word within a sentence.
        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).
        `seed` = for the random number generator.
        `min_count` = ignore all words with total frequency lower than this.
        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
                default is 0 (off), useful value is 1e-5.
        `workers` = use this many worker threads to train the model (=faster training with multicore machines)
        `hs` = if 1 (default), hierarchical sampling will be used for model training (else set to 0)
        `negative` = if > 0, negative sampling will be used, the int for negative
                specifies how many "noise words" should be drawn (usually between 5-20)
        `cbow_mean` = if 0 (default), use the sum of the context word vectors. If 1, use the mean.
                Only applies when cbow is used.
        """
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.sg = int(sg)
        self.table = None # for negative sampling --> this needs a lot of RAM! consider setting back to None before saving
        self.layer1_size = int(size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.window = int(window)
        self.seed = seed
        self.min_count = min_count
        self.sample = sample
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        if sentences is not None:
            self.build_vocab(sentences)
            self.train(sentences)

    def make_table(self, table_size=100000000, power=0.75):
        """
        Create a table using stored vocabulary word counts for drawing random words in the negative
        sampling training routines.

        Called internally from `build_vocab()`.

        """
        logger.info("constructing a table with noise distribution from %i words" % len(self.vocab))
        # table (= list of words) of noise distribution for negative sampling
        vocab_size = len(self.index2word)
        self.table = zeros(table_size, dtype=uint32)

        if not vocab_size:
            logger.warning("empty vocabulary in word2vec, is this intended?")
            return

        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.vocab[word].count**power for word in self.vocab]))
        # go through the whole table and fill it up with the word indexes proportional to a word's count**power
        widx = 0
        # normalize count^0.75 by Z
        d1 = self.vocab[self.index2word[widx]].count**power / train_words_pow
        for tidx in xrange(table_size):
            self.table[tidx] = widx
            if 1.0 * tidx / table_size > d1:
                widx += 1
                d1 += self.vocab[self.index2word[widx]].count**power / train_words_pow
            if widx >= vocab_size:
                widx = vocab_size - 1

    def create_binary_tree(self):
        """
        Create a binary Huffman tree using stored vocabulary word counts. Frequent words
        will have shorter binary codes. Called internally from `build_vocab()`.

        """
        logger.info("constructing a huffman tree from %i words" % len(self.vocab))

        # build the huffman tree
        heap = list(itervalues(self.vocab))
        heapq.heapify(heap)
        for i in xrange(len(self.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(self.vocab), left=min1, right=min2))

        # recurse over the tree, assigning a binary code to each vocabulary word
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(self.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(self.vocab)], dtype=uint32)
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))

            logger.info("built huffman tree with maximum node depth %i" % max_depth)

    def precalc_sampling(self):
        """Precalculate each vocabulary item's threshold for sampling"""
        if self.sample:
            logger.info("frequent-word downsampling, threshold %g; progress tallies will be approximate" % (self.sample))
            total_words = sum(v.count for v in itervalues(self.vocab))
            threshold_count = float(self.sample) * total_words
        for v in itervalues(self.vocab):
            prob = (sqrt(v.count / threshold_count) + 1) * (threshold_count / v.count) if self.sample else 1.0
            v.sample_probability = min(prob, 1.0)

    def build_vocab(self, sentences):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        logger.info("collecting all words and their counts")
        sentence_no, vocab = -1, {}
        total_words = 0
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % 10000 == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
                    (sentence_no, total_words, len(vocab)))
            for word in sentence:
                total_words += 1
                if word in vocab:
                    vocab[word].count += 1
                else:
                    vocab[word] = Vocab(count=1)
        logger.info("collected %i word types from a corpus of %i words and %i sentences" %
            (len(vocab), total_words, sentence_no + 1))

        # assign a unique index to each word
        self.vocab, self.index2word = {}, []
        for word, v in iteritems(vocab):
            if v.count >= self.min_count:
                v.index = len(self.vocab)
                self.index2word.append(word)
                self.vocab[word] = v
        logger.info("total %i word types after removing those with count<%s" % (len(self.vocab), self.min_count))

        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_table()
        # precalculate downsampling thresholds
        self.precalc_sampling()
        self.reset_weights()


    def train(self, sentences, total_words=None, word_count=0, chunksize=100):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("Cython compilation failed, training will be slow. Do you have Cython installed? `pip install cython`")
        logger.info("training model with %i workers on %i vocabulary and %i features, "
            "using 'skipgram'=%s 'hierarchical softmax'=%s 'subsample'=%s and 'negative sampling'=%s" %
            (self.workers, len(self.vocab), self.layer1_size, self.sg, self.hs, self.sample, self.negative))

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), [1.0]
        word_count = [word_count]
        total_words = total_words or int(sum(v.count * v.sample_probability for v in itervalues(self.vocab)))
        jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of sentences from the jobs queue."""
            work = zeros(self.layer1_size, dtype=REAL)  # each thread must have its own work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)

            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                # update the learning rate before every job
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words))
                # how many words did we train on? out-of-vocabulary (unknown) words do not count
                if self.sg:
                    job_words = sum(train_sentence_sg(self, sentence, alpha, work) for sentence in job)
                else:
                    job_words = sum(train_sentence_cbow(self, sentence, alpha, work, neu1) for sentence in job)
                with lock:
                    word_count[0] += job_words
                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        logger.info("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %
                            (100.0 * word_count[0] / total_words, alpha, word_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        def prepare_sentences():
            for sentence in sentences:
                # avoid calling random_sample() where prob >= 1, to speed things up a little:
                sampled = [self.vocab[word] for word in sentence
                    if word in self.vocab and (self.vocab[word].sample_probability >= 1.0 or self.vocab[word].sample_probability >= random.random_sample())]
                yield sampled

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(utils.grouper(prepare_sentences(), chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in xrange(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i words took %.1fs, %.0f words/s" %
            (word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))

        return word_count[0]


    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        random.seed(self.seed)
        self.syn0 = empty((len(self.vocab), self.layer1_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(len(self.vocab)):
            self.syn0[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size
        if self.hs:
            self.syn1 = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        self.syn0norm = None


    def save_word2vec_format(self, fname, fvocab=None, binary=False):
        """
        Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        """
        if fvocab is not None:
            logger.info("Storing vocabulary in %s" % (fvocab))
            with utils.smart_open(fvocab, 'wb') as vout:
                for word, vocab in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):
                    vout.write(utils.to_utf8("%s %s\n" % (word, vocab.count)))
        logger.info("storing %sx%s projection weights into %s" % (len(self.vocab), self.layer1_size, fname))
        assert (len(self.vocab), self.layer1_size) == self.syn0.shape
        with utils.smart_open(fname, 'wb') as fout:
            fout.write(utils.to_utf8("%s %s\n" % self.syn0.shape))
            # store in sorted order: most frequent words at the top
            for word, vocab in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):
                row = self.syn0[vocab.index]
                if binary:
                    fout.write(utils.to_utf8(word) + b" " + row.tostring())
                else:
                    fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))


    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, norm_only=True):
        """
        Load the input-hidden weight matrix from the original C word2vec-tool format.

        Note that the information stored in the file is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.

        `binary` is a boolean indicating whether the data is in binary word2vec format.
        `norm_only` is a boolean indicating whether to only store normalised word2vec vectors in memory.
        Word counts are read from `fvocab` filename, if set (this is the file generated
        by `-save-vocab` flag of the original C tool).
        """
        counts = None
        if fvocab is not None:
            logger.info("loading word counts from %s" % (fvocab))
            counts = {}
            with utils.smart_open(fvocab) as fin:
                for line in fin:
                    word, count = utils.to_unicode(line).strip().split()
                    counts[word] = int(count)

        logger.info("loading projection weights from %s" % (fname))
        with utils.smart_open(fname) as fin:
            header = utils.to_unicode(fin.readline())
            vocab_size, layer1_size = map(int, header.split())  # throws for invalid file format
            result = Word2Vec(size=layer1_size)
            result.syn0 = zeros((vocab_size, layer1_size), dtype=REAL)
            if binary:
                binary_len = dtype(REAL).itemsize * layer1_size
                for line_no in xrange(vocab_size):
                    # mixed text and binary: read text first, then binary
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)
                            word.append(ch)
                    word = utils.to_unicode(b''.join(word))
                    if counts is None:
                        result.vocab[word] = Vocab(index=line_no, count=vocab_size - line_no)
                    elif word in counts:
                        result.vocab[word] = Vocab(index=line_no, count=counts[word])
                    else:
                        logger.warning("vocabulary file is incomplete")
                        result.vocab[word] = Vocab(index=line_no, count=None)
                    result.index2word.append(word)
                    result.syn0[line_no] = fromstring(fin.read(binary_len), dtype=REAL)
            else:
                for line_no, line in enumerate(fin):
                    parts = utils.to_unicode(line).split()
                    if len(parts) != layer1_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                    word, weights = parts[0], map(REAL, parts[1:])
                    if counts is None:
                        result.vocab[word] = Vocab(index=line_no, count=vocab_size - line_no)
                    elif word in counts:
                        result.vocab[word] = Vocab(index=line_no, count=counts[word])
                    else:
                        logger.warning("vocabulary file is incomplete")
                        result.vocab[word] = Vocab(index=line_no, count=None)
                    result.index2word.append(word)
                    result.syn0[line_no] = weights
        logger.info("loaded %s matrix from %s" % (result.syn0.shape, fname))
        result.init_sims(norm_only)
        return result


    def most_similar(self, positive=[], negative=[], topn=10):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words, and corresponds to the `word-analogy` and
        `distance` scripts in the original word2vec implementation.

        Example::

          >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]

        """
        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, string_types + (ndarray,))
                                else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, string_types + (ndarray,))
                                 else word for word in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            elif word in self.vocab:
                mean.append(weight * self.syn0norm[self.vocab[word].index])
                all_words.add(self.vocab[word].index)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        dists = dot(self.syn0norm, mean)
        if not topn:
            return dists
        best = argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]


    def doesnt_match(self, words):
        """
        Which word from the given list doesn't go with the others?

        Example::

          >>> trained_model.doesnt_match("breakfast cereal dinner lunch".split())
          'cereal'

        """
        self.init_sims()

        words = [word for word in words if word in self.vocab]  # filter out OOV words
        logger.debug("using words %s" % words)
        if not words:
            raise ValueError("cannot select a word from an empty list")
        vectors = vstack(self.syn0norm[self.vocab[word].index] for word in words).astype(REAL)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = dot(vectors, mean)
        return sorted(zip(dists, words))[0][1]


    def __getitem__(self, word):
        """
        Return a word's representations in vector space, as a 1D numpy array.

        Example::

          >>> trained_model['woman']
          array([ -1.40128313e-02, ...]

        """
        return self.syn0[self.vocab[word].index]


    def __contains__(self, word):
        return word in self.vocab


    def similarity(self, w1, w2):
        """
        Compute cosine similarity between two words.

        Example::

          >>> trained_model.similarity('woman', 'man')
          0.73723527

          >>> trained_model.similarity('woman', 'woman')
          1.0

        """
        return dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))


    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        if getattr(self, 'syn0norm', None) is None or replace:
            logger.info("precomputing L2-norms of word weight vectors")
            if replace:
                for i in xrange(self.syn0.shape[0]):
                    self.syn0[i, :] /= sqrt((self.syn0[i, :] ** 2).sum(-1))
                self.syn0norm = self.syn0
                if hasattr(self, 'syn1'):
                    del self.syn1
            else:
                self.syn0norm = (self.syn0 / sqrt((self.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)


    def accuracy(self, questions, restrict_vocab=30000):
        """
        Compute accuracy of the model. `questions` is a filename where lines are
        4-tuples of words, split into sections by ": SECTION NAME" lines.
        See https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt for an example.

        The accuracy is reported (=printed to log and returned as a list) for each
        section separately, plus there's one aggregate summary at the end.

        Use `restrict_vocab` to ignore all questions containing a word whose frequency
        is not in the top-N most frequent words (default top 30,000).

        This method corresponds to the `compute-accuracy` script of the original C word2vec.

        """
        ok_vocab = dict(sorted(iteritems(self.vocab),
                               key=lambda item: -item[1].count)[:restrict_vocab])
        ok_index = set(v.index for v in itervalues(ok_vocab))

        def log_accuracy(section):
            correct, incorrect = section['correct'], section['incorrect']
            if correct + incorrect > 0:
                logger.info("%s: %.1f%% (%i/%i)" %
                    (section['section'], 100.0 * correct / (correct + incorrect),
                    correct, correct + incorrect))

        sections, section = [], None
        for line_no, line in enumerate(utils.smart_open(questions)):
            # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
            line = utils.to_unicode(line)
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': 0, 'incorrect': 0}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
                try:
                    a, b, c, expected = [word.lower() for word in line.split()]  # TODO assumes vocabulary preprocessing uses lowercase, too...
                except:
                    logger.info("skipping invalid line #%i in %s" % (line_no, questions))
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    logger.debug("skipping line #%i with OOV words: %s" % (line_no, line))
                    continue

                ignore = set(self.vocab[v].index for v in [a, b, c])  # indexes of words to ignore
                predicted = None
                # find the most likely prediction, ignoring OOV words and input words
                for index in argsort(self.most_similar(positive=[b, c], negative=[a], topn=False))[::-1]:
                    if index in ok_index and index not in ignore:
                        predicted = self.index2word[index]
                        if predicted != expected:
                            logger.debug("%s: expected %s, predicted %s" % (line.strip(), expected, predicted))
                        break
                section['correct' if predicted == expected else 'incorrect'] += 1
        if section:
            # store the last section, too
            sections.append(section)
            log_accuracy(section)

        total = {'section': 'total', 'correct': sum(s['correct'] for s in sections), 'incorrect': sum(s['incorrect'] for s in sections)}
        log_accuracy(total)
        sections.append(total)
        return sections


    def __str__(self):
        return "Word2Vec(vocab=%s, size=%s, alpha=%s)" % (len(self.index2word), self.layer1_size, self.alpha)


    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm']) # don't bother storing the cached normalized vectors
        super(Word2Vec, self).save(*args, **kwargs)


class BrownCorpus(object):
    """Iterate over sentences from the Brown corpus (part of NLTK data)."""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for line in utils.smart_open(fname):
                line = utils.to_unicode(line)
                # each file line is a single sentence in the Brown corpus
                # each token is WORD/POS_TAG
                token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                if not words:  # don't bother sending out empty sentences
                    continue
                yield words


class Text8Corpus(object):
    """Iterate over sentences from the "text8" corpus, unzipped from http://mattmahoney.net/dc/text8.zip ."""
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest, max_sentence_length = [], b'', 1000
        with utils.smart_open(self.fname) as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    sentence.extend(rest.split()) # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(b' ')  # the last token may have been split in two... keep it for the next iteration
                words, rest = (utils.to_unicode(text[:last_token]).split(), text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= max_sentence_length:
                    yield sentence[:max_sentence_length]
                    sentence = sentence[max_sentence_length:]


class LineSentence(object):
    """Simple format: one sentence = one line; words already preprocessed and separated by whitespace."""
    def __init__(self, source):
        """
        `source` can be either a string or a file object.

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = source

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in self.source:
                yield utils.to_unicode(line).split()
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for line in fin:
                    yield utils.to_unicode(line).split()



# Example: ./word2vec.py ~/workspace/word2vec/text8 ~/workspace/word2vec/questions-words.txt ./text8
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))
    logging.info("using optimization %s" % FAST_VERSION)

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    infile = sys.argv[1]
    from gensim.models.word2vec import Word2Vec  # avoid referencing __main__ in pickle

    seterr(all='raise')  # don't ignore numpy errors

    # model = Word2Vec(LineSentence(infile), size=200, min_count=5, workers=4)
    model = Word2Vec(Text8Corpus(infile), size=200, min_count=5, workers=1)

    if len(sys.argv) > 3:
        outfile = sys.argv[3]
        model.save(outfile + '.model')
        model.save_word2vec_format(outfile + '.model.bin', binary=True)
        model.save_word2vec_format(outfile + '.model.txt', binary=False)

    if len(sys.argv) > 2:
        questions_file = sys.argv[2]
        model.accuracy(sys.argv[2])

    logging.info("finished running %s" % program)
