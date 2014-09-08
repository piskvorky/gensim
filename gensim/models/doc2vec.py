#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Deep learning via the distributed memory and distributed bag of words models from
[1]_, using either hierarchical softmax or negative sampling [2]_ [3]_.

**Install Cython with `pip install cython` to use optimized doc2vec training** (70x speedup [4]_).

Initialize a model with e.g.::

>>> model = Doc2Vec(sentences, size=100, window=8, min_count=5, workers=4)

Persist a model to disk with::

>>> model.save(fname)
>>> model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

The model can also be instantiated from an existing file on disk in the doc2vec C format::

  >>> model = Doc2Vec.load_doc2vec_format('/tmp/vectors.txt', binary=False)  # C text format
  >>> model = Doc2Vec.load_doc2vec_format('/tmp/vectors.bin', binary=True)  # C binary format

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

.. [1] Quoc Le and Tomas Mikolov. Distributed Representations of Sentences and Documents. http://arxiv.org/pdf/1405.4053v2.pdf
.. [2] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
.. [3] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.
.. [4] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/

"""

import logging
import sys
import os

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from numpy import zeros, random, dtype, get_include, float32 as REAL, \
    seterr, fromstring, empty, sum as np_sum

logger = logging.getLogger("gensim.models.doc2vec")

from gensim import utils  # utility fnc for pickling, common scipy operations etc
from six.moves import xrange
from word2vec import Word2Vec, Vocab, train_cbow_pair, train_sg_pair


try:
    from gensim_addons.models.doc2vec_inner import train_sentence_dbow, train_sentence_dm, FAST_VERSION
except ImportError:
    try:
        # try to compile and use the faster cython version
        import pyximport
        models_dir = os.path.dirname(__file__) or os.getcwd()
        pyximport.install(setup_args={"include_dirs": [models_dir, get_include()]})
        from doc2vec_inner import train_sentence_dbow, train_sentence_dm, FAST_VERSION
    except:
        # failed... fall back to plain numpy (20-80x slower training than the above)
        FAST_VERSION = -1

        def train_sentence_dbow(model, sentence, lbls, alpha, work=None):
            """
            Update distributed bag of words model by training on a single sentence.

            The sentence is a list of Vocab objects (or None, where the corresponding
            word is not in the vocabulary. Called internally from `Doc2Vec.train()`.

            This is the non-optimized, Python version. If you have cython installed, gensim
            will use the optimized version from doc2vec_inner instead.

            """
            neg_labels = []
            if model.negative:
                # precompute negative labels
                neg_labels = zeros(model.negative + 1)
                neg_labels[0] = 1.0

            for word in lbls:
                if word is None:
                    continue  # OOV word in the input sentence => skip

                # now go over all words from the sentence, predicting each one in turn
                for word2 in sentence:
                    # don't train on OOV words
                    if word2:
                        train_sg_pair(model, word, word2, alpha, neg_labels)

            return len([word for word in sentence if word is not None])

        def train_sentence_dm(model, sentence, lbls, alpha, work=None, neu1=None):
            """
            Update distributed memory model by training on a single sentence.

            The sentence is a list of Vocab objects (or None, where the corresponding
            word is not in the vocabulary. Called internally from `Doc2Vec.train()`.

            This is the non-optimized, Python version. If you have cython installed, gensim
            will use the optimized version from doc2vec_inner instead.

            """
            lbl_indices = [lbl.index for lbl in lbls if lbl is not None]
            lbl_sum = np_sum(model.syn0[lbl_indices], axis=0)
            lbl_len = len(lbl_indices)
            neg_labels = []
            if model.negative:
                # precompute negative labels
                neg_labels = zeros(model.negative + 1)
                neg_labels[0] = 1.

            for pos, word in enumerate(sentence):
                if word is None:
                    continue  # OOV word in the input sentence => skip
                reduced_window = random.randint(model.window)  # `b` in the original doc2vec code
                start = max(0, pos - model.window + reduced_window)
                window_pos = enumerate(sentence[start : pos + model.window + 1 - reduced_window], start)
                word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
                l1 = np_sum(model.syn0[word2_indices], axis=0) + lbl_sum  # 1 x layer1_size
                if word2_indices and model.cbow_mean:
                    l1 /= (len(word2_indices) + lbl_len)
                neu1e = train_cbow_pair(model, word, word2_indices, l1, alpha, neg_labels)
                model.syn0[lbl_indices] += neu1e

            return len([word for word in sentence if word is not None])


class LabeledText(object):
    """A single labeled text item. Replaces list of words for each sentence from Word2Vec."""
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __str__(self):
        return 'LabeledText(' + str(self.text) + ', ' + str(self.labels) + ')'


class Doc2Vec(Word2Vec):
    """
    Class for training, using and evaluating neural networks described in https://code.google.com/p/doc2vec/

    The model can be stored/loaded via its `save()` and `load()` methods, or stored/loaded in a format
    compatible with the original doc2vec implementation via `save_doc2vec_format()` and `load_doc2vec_format()`.

    """
    def __init__(self, sentences=None, size=300, alpha=0.025, window=8, min_count=5,
                 sample=0, seed=1, workers=1, min_alpha=0.0001, dm=1, hs=1, negative=0,
                 dm_mean=0, initial=None):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of LabeledText objects that will be used for training.

        The `sentences` iterable can be simply a list of LabeledText elements, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `dm` defines the training algorithm. By default (`dm=1`), distributed memory is used.
                Otherwise, `dbow` is employed.
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
        `dm_mean` = if 0 (default), use the sum of the context word vectors. If 1, use the mean.
                Only applies when dm is used.
        """
        Word2Vec.__init__(self, size=size, alpha=alpha, window=window, min_count=min_count,
                          sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,
                          sg=(1+dm) % 2, hs=hs, negative=negative, cbow_mean=dm_mean)
        self.initial = initial
        if sentences is not None:
            self.build_vocab(sentences)
            self.train(sentences)

    @staticmethod
    def _vocab_from(sentences):
        sentence_no, vocab = -1, {}
        total_words = 0
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % 10000 == 0:
                logger.info("PROGRESS: at item #%i, processed %i words and %i word types" %
                            (sentence_no, total_words, len(vocab)))
            sentence_length = len(sentence.text)
            for label in sentence.labels:
                total_words += 1
                if label in vocab:
                    vocab[label].count += sentence_length
                else:
                    vocab[label] = Vocab(count=sentence_length)
            for word in sentence.text:
                total_words += 1
                if word in vocab:
                    vocab[word].count += 1
                else:
                    vocab[word] = Vocab(count=1)
        logger.info("collected %i word types from a corpus of %i words and %i items" %
                    (len(vocab), total_words, sentence_no + 1))
        return vocab

    def _prepare_sentences(self, sentences):
        for sentence in sentences:
            # avoid calling random_sample() where prob >= 1, to speed things up a little:
            sampled = [self.vocab[word] for word in sentence.text
                       if word in self.vocab and (self.vocab[word].sample_probability >= 1.0 or
                                                  self.vocab[word].sample_probability >= random.random_sample())]
            yield (sampled, [self.vocab[word] for word in sentence.labels if word in self.vocab])

    def _get_job_words(self, alpha, work, job, neu1):
        if self.sg:
            return sum(train_sentence_dbow(self, sentence, lbls, alpha, work) for sentence, lbls in job)
        else:
            return sum(train_sentence_dm(self, sentence, lbls, alpha, work, neu1) for sentence, lbls in job)

    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        random.seed(self.seed)
        self.syn0 = empty((len(self.vocab), self.layer1_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(len(self.vocab)):
            self.syn0[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size
        if self.initial is not None:
            _, _, stream = self.load_word2vec_stream(self.initial, None)
            i = 0
            for _, word, _, weights in stream():
                if word.lower() in self.vocab:
                    i += 1
                    self.syn0[self.vocab[word.lower()].index] = weights
            logger.info("initialized %d weights from %s" % (i, self.initial))
        if self.hs:
            self.syn1 = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        self.syn0norm = None

    @classmethod
    def load_word2vec_stream(cls, fname, counts, binary=True):
        logger.info("loading projection weights from %s" % (fname))
        with utils.smart_open(fname) as fin:
            header = utils.to_unicode(fin.readline())
            vocab_size, layer1_size = map(int, header.split())  # throws for invalid file format

        def stream():
            with utils.smart_open(fname) as fin:
                header = utils.to_unicode(fin.readline())
                vocab_size, layer1_size = map(int, header.split())  # throws for invalid file format
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
                            vocab = Vocab(index=line_no, count=vocab_size - line_no)
                        elif word in counts:
                            vocab = Vocab(index=line_no, count=counts[word])
                        else:
                            logger.warning("vocabulary file is incomplete")
                            vocab = Vocab(index=line_no, count=None)
                        yield(line_no, word, vocab, fromstring(fin.read(binary_len), dtype=REAL))
                else:
                    for line_no, line in enumerate(fin):
                        parts = utils.to_unicode(line).split()
                        if len(parts) != layer1_size + 1:
                            raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                        word, weights = parts[0], map(REAL, parts[1:])
                        if counts is None:
                            vocab = Vocab(index=line_no, count=vocab_size - line_no)
                        elif word in counts:
                            vocab = Vocab(index=line_no, count=counts[word])
                        else:
                            logger.warning("vocabulary file is incomplete")
                            vocab = Vocab(index=line_no, count=None)
                        yield(line_no, word, vocab, weights)
        return vocab_size, layer1_size, stream

    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, norm_only=True):
        """
        Load the input-hidden weight matrix from the original C word2vec-tool format.

        Note that the information stored in the file is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.

        `binary` is a boolean indicating whether the data is in binary doc2vec format.
        `norm_only` is a boolean indicating whether to only store normalised doc2vec vectors in memory.
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
    
        vocab_size, layer1_size, stream = cls.load_word2vec_stream(fname, counts, binary=binary)
        result = Doc2Vec(size=layer1_size)
        result.syn0 = zeros((vocab_size, layer1_size), dtype=REAL)
        for line_no, word, vocab, weights in stream():
            result.vocab[word] = vocab
            result.index2word.append(word)
            result.syn0[line_no] = weights
        logger.info("loaded %s matrix from %s" % (result.syn0.shape, fname))
        result.init_sims(norm_only)
        return result

    def load_doc2vec_format(self, fname, fvocab=None, binary=False):
        self.load_word2vec_format(fname, fvocab, binary)

    def save_doc2vec_format(self, fname, fvocab=None, binary=False):
        self.save_word2vec_format(fname, fvocab, binary)

    def __str__(self):
        return "Doc2Vec(vocab=%s, size=%s, alpha=%s)" % (len(self.index2word), self.layer1_size, self.alpha)

    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm'])  # don't bother storing the cached normalized vectors
        super(Doc2Vec, self).save(*args, **kwargs)


class BrownCorpus(object):
    """Iterate over sentences from the Brown corpus (part of NLTK data)."""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for item_no, line in enumerate(utils.smart_open(fname)):
                line = utils.to_unicode(line)
                # each file line is a single sentence in the Brown corpus
                # each token is WORD/POS_TAG
                token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                if not words:  # don't bother sending out empty sentences
                    continue
                yield LabeledText(words, [fname+'_SENT_'+str(item_no)])


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
            for item_no, line in enumerate(self.source):
                yield LabeledText(utils.to_unicode(line).split(), ['SENT_'+str(item_no)])
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledText(utils.to_unicode(line).split(), ['SENT_'+str(item_no)])


# Example: ./doc2vec.py ~/workspace/doc2vec/textfile ~/workspace/doc2vec/questions-words.txt ./textfile
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
    from gensim.models.doc2vec import Doc2Vec  # avoid referencing __main__ in pickle

    seterr(all='raise')  # don't ignore numpy errors

    model = Doc2Vec(LineSentence(infile), size=200, min_count=5, workers=4)

    if len(sys.argv) > 3:
        outfile = sys.argv[3]
        model.save(outfile + '.model')
        model.save_doc2vec_format(outfile + '.model.bin', binary=True)
        model.save_doc2vec_format(outfile + '.model.txt', binary=False)

    if len(sys.argv) > 2:
        questions_file = sys.argv[2]
        model.accuracy(sys.argv[2])

    logging.info("finished running %s" % program)

