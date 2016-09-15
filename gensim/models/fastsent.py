#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
FastSent model for sentence representation.

The FastSent algorithm was originally inspired by the C package https://code.google.com/p/fastsent/.

For a blog tutorial on gensim fastsent, which is closely related, visit http://radimrehurek.com/2014/02/fastsent-tutorial/

**Make sure you have a C compiler before installing gensim, to use optimized (compiled) fastsent training**
(70x speedup compared to plain NumPy implementation [3]_).

Initialize a model with e.g.::

>>> model = FastSent(sentences, size=100, window=5, min_count=5, workers=4)

Persist a model to disk with::

>>> model.save(fname)
>>> model = FastSent.load(fname)  # you can continue training with the loaded model!

The model can also be instantiated from an existing file on disk in the fastsent C format::

  >>> model = FastSent.load_fastsent_format('/tmp/vectors.txt', binary=False)  # C text format
  >>> model = FastSent.load_fastsent_format('/tmp/vectors.bin', binary=True)  # C binary format
"""

from __future__ import division  # py3 "true division"
import logging
import sys
import os
from timeit import default_timer
import threading
try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.models.word2vec import Word2Vec, Vocab, BrownCorpus, LineSentence, Text8Corpus
from six.moves import xrange
from types import GeneratorType

logger = logging.getLogger("gensim.models.fastsent")
MAX_WORDS_IN_BATCH = 10000

try:
    from gensim.models.fastsent_inner import train_sentence_fastsent, FAST_VERSION
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1

    def train_sentence_fastsent(model, sentences, alpha, work=None, neu1=None):
        """
        Update parameters based on three consecutive sentences from the training data
        model: the model object
        sentences: an ordered list of three sentences as lists of words
        alpha: the learning rate
        """

        current_sent = sentences[1]
        if model.autoencode:
            context_sents = sentences[0] + sentences[1]+ sentences[2]
        else:
            context_sents = sentences[0] + sentences[2]
        word_vocabs = [model.vocab[w] for w in current_sent if w in model.vocab and
                       model.vocab[w].sample_int > model.random.rand() * 2**32]
        context_vocabs = [model.vocab[w] for w in context_sents if w in model.vocab and
                          model.vocab[w].sample_int > model.random.rand() * 2**32]
        word2_indices = [word.index for word in word_vocabs]
        l1 = np_sum(model.syn0[word2_indices], axis=0)  # 1 x vector_size
        if word2_indices and model.fastsent_mean:
            l1 /= len(word2_indices)
        for word in context_vocabs:
            train_fastsent_pair(model, word, word2_indices, l1, alpha)
        return len(context_vocabs)


def train_fastsent_pair(model, word, input_word_indices, l1, alpha, learn_vectors=True, learn_hidden=True):
    """
    Update parameters based on a middle (source) sentence and a single word
    in an adjacent (context) sentence
    """
    neu1e = zeros(l1.shape)

    l2a = model.syn1[word.point]  # 2d matrix, codelen x layer1_size
    fa = 1. / (1. + exp(-dot(l1, l2a.T)))  # propagate hidden -> output
    ga = (1. - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
    if learn_hidden:
        model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
    neu1e += dot(ga, l2a)  # save error

    if learn_vectors:
        # learn input -> hidden, here for all words in the window separately
        if not model.fastsent_mean and input_word_indices:
            neu1e /= len(input_word_indices)
        for i in input_word_indices:
            model.syn0[i] += neu1e * model.syn0_lockf[i]

    return neu1e


class FastSent(Word2Vec):
    """
    Class for training, using and evaluating neural networks described in http://arxiv.org/abs/1602.03483

    The model can be stored/loaded via its `save()` and `load()` methods, or stored/loaded in a format
    compatible with the original fastsent implementation via `save_fastsent_format()` and `load_fastsent_format()`.

    """
    def __init__(
            self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=0, seed=1, workers=1, min_alpha=0.0001,
            fastsent_mean=0, hashfxn=hash, iter=1, null_word=0, autoencode=0, noverlap=0):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        this module for such examples.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the current and predicted word within a sentence.

        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).

        `seed` = for the random number generator. Initial vectors for each
        word are seeded with a hash of the concatenation of word + str(seed).

        `min_count` = ignore all words with total frequency lower than this.

        `max_vocab_size` = limit RAM during vocabulary building; if there are more unique
        words than this, then prune the infrequent ones. Every 10 million word types
        need about 1GB of RAM. Set to `None` for no limit (default).

        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
            default is 0 (off), useful value is 1e-5.

        `workers` = use this many worker threads to train the model (=faster training with multicore machines).

        `fastsent_mean` = if 0 (default), use the sum of the context word vectors. If 1, use the mean.

        `hashfxn` = hash function to use to randomly initialize weights, for increased
        training reproducibility. Default is Python's rudimentary built in hash function.

        `iter` = number of iterations (epochs) over the corpus.

        `autoencode` = predict the present sentence as well as neighbouring sentences?

        'noverlap' = do not predict target words if found in source sentence

        """
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.cum_table = None  # for negative sampling
        self.vector_size = int(size)
        self.layer1_size = int(size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.window = int(window)
        self.max_vocab_size = max_vocab_size
        self.seed = seed
        self.random = random.RandomState(seed)
        self.min_count = min_count
        self.sample = sample
        self.workers = int(workers)
        self.min_alpha = float(min_alpha)
        self.fastsent_mean = int(fastsent_mean)
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.train_count = 0
        self.total_train_time = 0
        self.autoencode = autoencode
        self.noverlap = noverlap
        self.hs, self.negative, self.sorted_vocab = True, False, False   # for inheriting some methods from word2vec
        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")
            self.build_vocab(sentences)
            self.train(sentences)

    def _do_train_job(self, job, alpha, inits):
        work, neu1 = inits
        tally = 0
        raw_tally = 0
        # loop over sentences in this job
        for i in range(0, len(job)-2):
            if self.noverlap:
                ss = set(job[i+1])
                t1 = [w for w in job[i] if not w in ss]
                t2 = [w for w in job[i+2] if not w in ss]
                sentences = [t1,job[i+1],t2]
            else:
                sentences = [job[i], job[i+1], job[i+2]]
            tally += train_sentence_fastsent(self, sentences, alpha, work, neu1)
            if self.autoencode:
                raw_tally += len(sentences[0]) + len(sentences[1]) + len(sentences[2])
            else:
                raw_tally += len(sentences[0]) + len(sentences[2])
        return (tally, raw_tally)

    def train(self, sentences, total_words=None, word_count=0, chunksize=100, 
              total_examples=None, queue_factor=2, report_delay=1.0):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For FastSent, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)

        To support linear learning-rate decay from (initial) alpha to min_alpha, either total_examples
        (count of sentences) or total_words (count of raw words in sentences) should be provided, unless the
        sentences are the same as those that were used to initially build the vocabulary.

        """
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension not loaded for FastSent, training will be slow. "
                          "Install a C compiler and reinstall gensim for fast training.")
            self.neg_labels = []

        logger.info(
            "training model with %i workers on %i vocabulary and %i features, "
            "using sample=%s",
            self.workers, len(self.vocab), self.layer1_size,
            self.sample)

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")
        if not hasattr(self, 'syn0'):
            raise RuntimeError("you must first finalize vocabulary before training the model")

        if total_words is None and total_examples is None:
            if self.corpus_count:
                total_examples = self.corpus_count
                logger.info("expecting %i examples, matching count from corpus used for vocabulary survey", total_examples)
            else:
                raise ValueError("you must provide either total_words or total_examples, to enable alpha and progress calculations")

        if self.iter > 1:
            sentences = utils.RepeatCorpusNTimes(sentences, self.iter)
            total_words = total_words and total_words * self.iter
            total_examples = total_examples and total_examples * self.iter

        def worker_init():
            work = matutils.zeros_aligned(self.layer1_size, dtype=REAL)  # per-thread private work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            return (work, neu1)

        def worker_one_job(job, inits):
            items, alpha = job
            if items is None:  # signal to finish
                return False
            # train & return tally
            tally, raw_tally = self._do_train_job(items, alpha, inits)
            progress_queue.put((len(items), tally, raw_tally))  # report progress
            return True

        # loop of a given worker: fetches the data from the queue and then
        # launches the worker_one_job function
        def worker_loop():
            """Train the model, lifting lists of sentences from the jobs queue."""
            init = worker_init()
            while True:
                job = job_queue.get()
                if not worker_one_job(job, init):
                    break

        start, next_report = default_timer(), 1.0

        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        if self.workers > 0:
            job_queue = Queue(maxsize=queue_factor * self.workers)
        else:
            job_queue = FakeJobQueue(worker_init, worker_one_job)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()
        pushed_words = 0
        pushed_examples = 0
        example_count = 0
        trained_word_count = 0
        raw_word_count = word_count
        push_done = False
        done_jobs = 0
        next_alpha = self.alpha
        jobs_source = enumerate(utils.grouper(sentences, chunksize))
        # fill jobs queue with (sentence, alpha) job tuples
        while True:
            try:
                job_no, items = next(jobs_source)
                logger.debug("putting job #%i in the queue at alpha %.05f", job_no, next_alpha)
                job_queue.put((items, next_alpha))
                # update the learning rate before every next job
                if self.min_alpha < next_alpha:
                    if total_examples:
                        # examples-based decay
                        pushed_examples += len(items)
                        next_alpha = self.alpha - (self.alpha - self.min_alpha) * (pushed_examples / total_examples)
                    else:
                        # words-based decay
                        pushed_words += self._raw_word_count(items)
                        next_alpha = self.alpha - (self.alpha - self.min_alpha) * (pushed_words / total_words)
                    next_alpha = max(next_alpha, self.min_alpha)
            except StopIteration:
                logger.info("reached end of input; waiting to finish %i outstanding jobs",
                    job_no - done_jobs + 1)
                for _ in xrange(self.workers):
                    job_queue.put((None, 0))  # give the workers heads up that they can finish -- no more work!
                push_done = True
            try:
                while done_jobs < (job_no+1) or not push_done:
                    examples, trained_words, raw_words = progress_queue.get(push_done)  # only block after all jobs pushed
                    example_count += examples
                    trained_word_count += trained_words  # only words in vocab & sampled
                    raw_word_count += raw_words
                    done_jobs += 1
                    elapsed = default_timer() - start
                    if elapsed >= next_report:
                        if total_examples:
                            # examples-based progress %
                            logger.info(
                                "FASTSENT MODEL PROGRESS: at %.2f%% examples, %.0f words/s",
                                100.0 * example_count / total_examples, trained_word_count / elapsed)
                        else:
                            # words-based progress %
                            logger.info(
                                "FASTSENT MODEL PROGRESS: at %.2f%% words, %.0f words/s",
                                100.0 * raw_word_count / total_words, trained_word_count / elapsed)
                        next_report = elapsed + report_delay  # don't flood log, wait report_delay seconds
                else:
                    # loop ended by job count; really done
                    break
            except Empty:
                pass  # already out of loop; continue to next push

        # all done; report the final stats
        elapsed = default_timer() - start
        logger.info(
            "training on %i raw words took %.1fs, %.0f trained words/s",
            raw_word_count, elapsed, trained_word_count / elapsed if elapsed else 0.0)

        if total_examples and total_examples != example_count:
            logger.warn("supplied example count (%i) did not equal expected count (%i)", example_count, total_examples)
        if total_words and total_words != raw_word_count:
            logger.warn("supplied raw word count (%i) did not equal expected count (%i)", raw_word_count, total_words)

        self.train_count += 1  # number of times train() has been called
        self.total_train_time += elapsed
        self.clear_sims()
        return trained_word_count

    def save_fastsent_format(self, fname, fvocab=None, binary=False):
        """
        Please refer to Word2Vec.save_word2vec_format
        """
        super(FastSent,self).save_word2vec_format(fname=fname, fvocab=fvocab, binary=binary)

    @classmethod
    def load_fastsent_format(cls, fname, fvocab=None, binary=False, norm_only=True, encoding='utf8', unicode_errors='strict',
                            limit=None, datatype=REAL):
        """
        Please refer to Word2Vec.load_word2vec_format
        """
        result=super(FastSent,cls).load_word2vec_format(fname, fvocab=fvocab, binary=binary, encoding=encoding, unicode_errors=unicode_errors, limit=limit, datatype=datatype) 
        result.init_sims(norm_only)
        return result

    def __getitem__(self, sentence):

        """
        Accepts a sentence (string) as input
        where 'words' are separated by white space.
        Returns a vector for that setence (or word).

        Example::

          >>> trained_model['office']
          array([ -1.40128313e-02, ...])

          >>> trained_model['I  go to the office']]
          array([ -1.328313e-02, ...])

        """
        if self.fastsent_mean:
            return array([self.syn0[self.vocab[word].index] for word in sentence.split()]).mean(axis=0)
        else:
            return array([self.syn0[self.vocab[word].index] for word in sentence.split()]).sum(axis=0)

    def sentence_similarity(self, s1, s2, delimiter=' '):
        """
        Compute cosine similarity between two sentences (as strings).

        Example::

          >>> trained_model.sentence_similarity('the red cat sat on the blue mat', 'the yellow dog sat on the brown carpet')
          0.61540466561049689

        """
        words1 = s1.split(delimiter)
        words2 = s2.split(delimiter)
        v1 = [self[word] for word in words1]
        v2 = [self[word] for word in words2]
        if self.fastsent_mean:
            return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))
        else:
            return dot(matutils.unitvec(array(v1).sum(axis=0)), matutils.unitvec(array(v2).sum(axis=0)))



class FakeJobQueue(object):
    """Pretends to be a Queue; does equivalent of work_loop in calling thread."""
    def __init__(self, init_fn, job_fn):
        self.inits = init_fn()
        self.job_fn = job_fn

    def put(self, job):
        self.job_fn(job, self.inits)


if __name__ == "__main__":

    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    logging.info("running %s", " ".join(sys.argv))
    logging.info("using optimization %s", FAST_VERSION)

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    infile = sys.argv[1]
    from gensim.models.fastsent import Word2Vec  # avoid referencing __main__ in pickle
    

    seterr(all='raise')  # don't ignore numpy errors

    # model = Word2Vec(LineSentence(infile), size=200, min_count=5, workers=4)
    model = FastSent(Text8Corpus(infile), size=200, min_count=5, workers=1)

    if len(sys.argv) > 3:
        outfile = sys.argv[3]
        model.save(outfile + '.model')
        model.save_fastsent_format(outfile + '.model.bin', binary=True)
        model.save_fastsent_format(outfile + '.model.txt', binary=False)

    if len(sys.argv) > 2:
        questions_file = sys.argv[2]
        model.accuracy(sys.argv[2])

    logging.info("finished running %s", program)