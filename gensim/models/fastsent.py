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
            fastsent_mean=0, hashfxn=hash, iter=1, null_word=0, autoencode=0, 
            noverlap=0, batch_words=MAX_WORDS_IN_BATCH):
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

        `batch_words` = target size (in words) for batches of examples passed to worker threads (and
        thus cython routines). Default is 10000. (Larger batches will be passed if individual
        texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
        """
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.cum_table = None  # for negative sampling
        self.vector_size = int(size)
        self.layer1_size = int(size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.min_alpha_yet_reached = float(alpha) # To warn user if alpha increases
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
        self.batch_words = batch_words
        self.hs, self.negative, self.sorted_vocab, self.sg = True, False, False, 0   # for inheriting some methods from word2vec
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