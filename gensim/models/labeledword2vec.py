#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Deep learning via word2vec's "skip-gram and CBOW models", using either
hierarchical softmax or negative sampling [1]_ [2]_.

The training algorithms were originally ported from the C package https://code.google.com/p/word2vec/
and extended with additional functionality.

For a blog tutorial on gensim word2vec, with an interactive web app trained on GoogleNews, visit http://radimrehurek.com/2014/02/word2vec-tutorial/

**Make sure you have a C compiler before installing gensim, to use optimized (compiled) word2vec training**
(70x speedup compared to plain NumPy implementation [3]_).

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

Note that there is a :mod:`gensim.models.phrases` module which lets you automatically
detect phrases longer than one word. Using phrases, you can learn a word2vec model
where "words" are actually multiword expressions, such as `new_york_times` or `financial_crisis`:

>>> bigram_transformer = gensim.models.Phrases(sentences)
>>> model = Word2Vec(bigram_transformer[sentences], size=100, ...)

.. [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
.. [2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.
.. [3] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/
"""

from __future__ import division  # py3 "true division"

import logging
import sys

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import train_cbow_pair, Vocab

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from numpy import exp, dot, zeros, outer, float32 as REAL, \
    uint32, empty, sum as np_sum, prod, ones, vstack, apply_along_axis, copy

from scipy.special import expit

from gensim import matutils  # utility fnc for pickling, common scipy operations etc

logger = logging.getLogger(__name__)

try:
    from .word2vec_inner import train_batch_labeled_cbow, score_document_labeled_cbow as sdlc
    from .word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH

    def score_document_labeled_cbow(model, document, labels=None, work=None, neu1=None):
        if work is None:
            work = ones(len(model.lvocab) if labels is None else len(labels), dtype=REAL)
        if neu1 is None:
            neu1 = matutils.zeros_aligned(model.layer1_size, dtype=REAL)
        labels = labels or model.lvocab.keys()
        scores = sdlc(model, document, labels, work, neu1)
        return zip(labels, scores)

except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000


    def train_cbow_pair_softmax(model, target, input_word_indices, l1, alpha, learn_vectors=True, learn_hidden=True):
        neu1e = zeros(l1.shape)

        target_vect = zeros(model.syn1neg.shape[0])
        target_vect[target.index] = 1.
        l2 = copy(model.syn1neg)
        fa = expit(dot(l1, l2.T))  # propagate hidden -> output
        ga = (target_vect - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1neg += outer(ga, l1)  # learn hidden -> output
        neu1e += dot(ga, l2)  # save error

        if learn_vectors:
            # learn input -> hidden, here for all words in the window separately
            if not model.cbow_mean and input_word_indices:
                neu1e /= len(input_word_indices)
            for i in input_word_indices:
                model.wv.syn0[i] += neu1e * model.syn0_lockf[i]

        return neu1e


    def train_batch_labeled_cbow(model, sentences, alpha, work=None, neu1=None):
        result = 0
        for sentence in sentences:
            document, target = sentence
            word_vocabs = [model.wv.vocab[w] for w in document if w in model.wv.vocab and
                           model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32]
            target_vocabs = [model.lvocab[t] for t in target if t in model.lvocab]
            for target in target_vocabs:
                word2_indices = [w.index for w in word_vocabs]
                l1 = np_sum(model.wv.syn0[word2_indices], axis=0)  # 1 x vector_size
                if word2_indices and model.cbow_mean:
                    l1 /= len(word2_indices)
                if model.softmax:
                    train_cbow_pair_softmax(model, target, word2_indices, l1, alpha)
                else:
                    train_cbow_pair(model, target, word2_indices, l1, alpha)
            result += len(word_vocabs)
        return result


    def score_document_labeled_cbow(model, document, labels=None, work=None, neu1=None):

        word_vocabs = [model.wv.vocab[w] for w in document if w in model.wv.vocab]

        if labels is not None:
            targets = [model.lvocab[label] for label in labels]
        else:
            targets = model.lvocab.values()
            labels = model.lvocab.keys()

        word2_indices = [word2.index for word2 in word_vocabs]
        l1 = np_sum(model.wv.syn0[word2_indices], axis=0)  # 1 x layer1_size
        if word2_indices and model.cbow_mean:
            l1 /= len(word2_indices)
        return zip(labels, score_cbow_labeled_pair(model, targets, l1))


    def score_cbow_labeled_pair(model, targets, l1):
        if model.hs:
            prob = []
            # FIXME this cycle should be executed internally in numpy
            for target in targets:
                l2a = model.syn1[target.point]
                sgn = (-1.0) ** target.code  # ch function, 0-> 1, 1 -> -1
                prob.append(prod(expit(sgn * dot(l1, l2a.T))))
        # Softmax
        else:
            def exp_dot(x):
                return exp(dot(l1, x.T))

            prob_num = exp_dot(model.syn1neg[[t.index for t in targets]])
            prob_den = np_sum(apply_along_axis(exp_dot, 1, model.syn1neg))
            prob = prob_num / prob_den
        return prob


class LabeledWord2Vec(Word2Vec):
    def __init__(self, loss='softmax', **kwargs):
        """
        Exactly as the parent class `Word2Vec <https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec>`_.
        Some parameter values are overwritten (e.g. sg=0 because we never use skip-gram here), look at the code for details.
        Argument names must be explicit!

        `loss` = one value in {ns, hs, softmax}. If "ns" is selected negative sampling will be used
        as loss function, together with the parameter `negative`. With "hs" hierarchical softmax will be used,
        while with "softmax" (default) the sandard softmax function (the other two are "approximations").
         The `hs` argument does not exist anymore.

        It basically builds two vocabularies, one for the sample words and one for the labels,
        so that the input layer is only made of words, while the output layer is only made of labels.
        **Parent class methods that are not overridden here are not tested and not safe to use**.
        """
        self.lvocab = {}  # Vocabulary of labels only
        self.index2label = []
        kwargs['sg'] = 0
        kwargs['window'] = sys.maxsize
        kwargs['sentences'] = None
        self.softmax = self.init_loss(kwargs, loss)
        super(LabeledWord2Vec, self).__init__(**kwargs)

    def init_loss(self, kwargs, loss):
        if loss == 'hs':
            kwargs['hs'] = 1
            kwargs['negative'] = 0
        elif loss == 'ns':
            kwargs['hs'] = 0
            assert kwargs['negative'] > 0
        elif loss == 'softmax':
            kwargs['hs'] = 0
            kwargs['negative'] = 0
            return True
        else:
            raise ValueError('loss argument must be set with "ns", "hs" or "softmax"')
        return False

    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return sum(len(sentence[0]) for sentence in job)

    def _do_train_job(self, sentences, alpha, inits):
        """
        Train a single batch of sentences (documents and labels in LabeledWord2Vec).
        Return 2-tuple `(effective word count after ignoring unknown words and sentence length trimming,
        total word count)`.
        """
        work, neu1 = inits
        tally = 0
        if self.sg:
            raise NotImplementedError('Supervised learning in fastText is based only on the CBOW model')
        else:
            tally += train_batch_labeled_cbow(self, sentences, alpha, work, neu1)
        return tally, self._raw_word_count(sentences)

    # TODO use TaggedDocument from Gensim?
    # FIXME pass just an iterator over (doc, label) like for train
    def build_vocab(self, sentences, labels, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False):
        """
        Build vocabularies from a sequence of sentences/documents (can be a once-only generator stream) and the set of labels.
        Each sentence must be a list of unicode strings. `labels` is an iterable over the label names.

        """
        # Build words and labels vocabularies in two different objects
        self.scan_vocab(sentences, progress_per=progress_per, trim_rule=trim_rule)
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, update=update)
        self.finalize_vocab(update=update)  # build tables & arrays
        self.build_lvocab(labels, progress_per=progress_per, update=update)

    def build_lvocab(self, labels, progress_per=10000, update=False):
        """Only build data structures for labels. `labels` is an iterable over the label names."""

        class FakeSelf(LabeledWord2Vec):
            def __init__(self, max_vocab_size, min_count, sample, estimate_memory):
                self.max_vocab_size = max_vocab_size
                self.corpus_count = 0
                self.raw_vocab = None
                self.wv = KeyedVectors()
                self.min_count = min_count
                self.sample = sample
                self.estimate_memory = estimate_memory

        # FIXME set the right estimate memory for labels
        labels_vocab = FakeSelf(sys.maxsize, 0, 0, self.estimate_memory)
        self.__class__.scan_vocab(labels_vocab, [labels], progress_per=progress_per, trim_rule=None)
        self.__class__.scale_vocab(labels_vocab, min_count=None, sample=None, keep_raw_vocab=False, trim_rule=None,
                                   update=update)
        self.lvocab = labels_vocab.wv.vocab
        self.index2label = labels_vocab.wv.index2word
        # If we want to sample more negative labels that their count
        if self.negative and self.negative >= len(self.index2label) > 0:
            self.negative = len(self.index2label) - 1
        self.finalize_lvocab(update=update)

    def finalize_vocab(self, update=False):
        """Build tables and model weights based on final word vocabulary settings."""
        if not self.wv.index2word:
            self.scale_vocab()
        if self.sorted_vocab and not update:
            self.sort_vocab()
        if self.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input – never predicted – so count, huffman-point, etc doesn't matter
            word, v = '\0', Vocab(count=1, sample_int=0)
            v.index = len(self.wv.vocab)
            self.wv.index2word.append(word)
            self.wv.vocab[word] = v
        # set initial input/projection and hidden weights
        if not update:
            self.reset_weights(outputs=False)
        else:
            self.update_weights(outputs=False)

    def finalize_lvocab(self, update=False):
        """Build tables and model weights based on final label vocabulary settings."""
        if self.hs:
            class FakeSelf(LabeledWord2Vec):
                def __init__(self, vocab):
                    self.wv = KeyedVectors()
                    self.wv.vocab = vocab

            # add info about each word's Huffman encoding
            self.__class__.create_binary_tree(FakeSelf(self.lvocab))
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()
        if not update:
            self.reset_weights(inputs=False)
        else:
            self.update_weights(inputs=False)

    def update_weights(self, inputs=True, outputs=True):
        """
        Copy all the existing weights, and reset the weights for the newly
        added vocabulary.
        """
        logger.info("updating layer weights")

        if inputs:
            gained_vocab = len(self.wv.vocab) - len(self.wv.syn0)
            newsyn0 = empty((gained_vocab, self.vector_size), dtype=REAL)

            # randomize the remaining words
            for i in range(len(self.wv.syn0), len(self.wv.vocab)):
                # construct deterministic seed from word AND seed argument
                newsyn0[i - len(self.wv.syn0)] = self.seeded_vector(
                    (self.wv.index2word[i] if isinstance(self.wv.index2word[i], basestring) else
                        str(self.wv.index2word[i])) + str(self.seed))
            self.wv.syn0 = vstack([self.wv.syn0, newsyn0])
            self.wv.syn0norm = None

            # do not suppress learning for already learned words
            self.syn0_lockf = ones(len(self.wv.vocab), dtype=REAL)  # zeros suppress learning

        if outputs:
            gained_vocab = len(self.lvocab) - len(self.syn1 if self.hs else self.syn1neg)
            if self.hs:
                self.syn1 = vstack([self.syn1, zeros((gained_vocab, self.layer1_size), dtype=REAL)])
            if self.negative or self.softmax:
                self.syn1neg = vstack([self.syn1neg, zeros((gained_vocab, self.layer1_size), dtype=REAL)])

    def reset_weights(self, inputs=True, outputs=True):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        if inputs:
            self.wv.syn0 = empty((len(self.wv.vocab), self.vector_size), dtype=REAL)
            # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
            for i in range(len(self.wv.vocab)):
                # construct deterministic seed from word AND seed argument
                self.wv.syn0[i] = self.seeded_vector(
                    (self.wv.index2word[i] if isinstance(self.wv.index2word[i], basestring) else
                        str(self.wv.index2word[i])) + str(self.seed))
            self.wv.syn0norm = None
            self.syn0_lockf = ones(len(self.wv.vocab), dtype=REAL)  # zeros suppress learning
        if outputs:
            # Output layer is only made of labels
            if self.hs:
                self.syn1 = zeros((len(self.lvocab), self.layer1_size), dtype=REAL)
            # Use syn1neg also for softmax outputs
            if self.negative or self.softmax:
                self.syn1neg = zeros((len(self.lvocab), self.layer1_size), dtype=REAL)

    def reset_from(self, other_model):
        """
        Borrow shareable pre-built structures (like vocab) from the other_model. Useful
        if testing multiple models in parallel on the same corpus.
        """
        self.lvocab = getattr(other_model, 'lvocab', {})
        self.index2label = getattr(other_model, 'index2label', [])
        super(LabeledWord2Vec, self).reset_from(other_model)

    def make_cum_table(self, power=0.75, domain=2 ** 31 - 1):
        """
        Create a cumulative-distribution table using stored vocabulary label counts for
        drawing random labels in the negative-sampling training routines.

        To draw a label index, choose a random integer up to the maximum value in the
        table (cum_table[-1]), then finding that integer's sorted insertion point
        (as if by bisect_left or ndarray.searchsorted()). That insertion point is the
        drawn index, coming up in proportion equal to the increment at that slot.

        Called internally from 'build_vocab()'.
        """
        vocab_size = len(self.index2label)
        self.cum_table = zeros(vocab_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.lvocab[word].count ** power for word in self.lvocab]))
        cumulative = 0.0
        for word_index in range(vocab_size):
            cumulative += self.lvocab[self.index2label[word_index]].count ** power
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def train(self, sentences, total_words=None, word_count=0,
              total_examples=None, queue_factor=2, report_delay=1.0):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For Word2Vec, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)

        To support linear learning-rate decay from (initial) alpha to min_alpha, either total_examples
        (count of sentences) or total_words (count of raw words in sentences) should be provided, unless the
        sentences are the same as those that were used to initially build the vocabulary.

        """
        if (self.model_trimmed_post_training):
            raise RuntimeError("Parameters for training were discarded using model_trimmed_post_training method")
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension not loaded for Word2Vec, training will be slow. "
                          "Install a C compiler and reinstall gensim for fast training.")
            self.neg_labels = []
            if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.
        return super(LabeledWord2Vec, self).train(sentences, total_words, word_count,
                                                  total_examples, queue_factor, report_delay)

    @classmethod
    def load_from(cls, other_model):
        """
        Import data and parameter values from other model
        :param other_model: A ``LabeledWord2Vec`` object, or a ``Word2Vec`` or ``KeyedVectors`` object of Gensim
        """
        softmax = getattr(other_model, 'softmax', False)
        if softmax:
            loss = 'softmax'
        elif not other_model.hs and other_model.negative:
            loss = 'ns'
        else:
            loss = 'hs'
        new_model = LabeledWord2Vec(
            loss=loss,
            negative=other_model.negative if loss == 'ns' else 0,
            size=other_model.vector_size,
            seed=other_model.seed
        )
        new_model.reset_from(other_model)
        for attr in vars(other_model):
            if hasattr(new_model, attr):
                if not isinstance(other_model, LabeledWord2Vec) and (attr == 'syn1' or attr == 'syn1neg'):
                    continue
                value = getattr(other_model, attr, getattr(new_model, attr))
                if isinstance(value, KeyedVectors):
                    new_model.wv.syn0 = value.syn0
                    new_model.wv.syn0norm = value.syn0norm
                else:
                    setattr(new_model, attr, value)
        return new_model

    def __str__(self):
        return "%s(vocab=%s, labels=%s, size=%s, alpha=%s)" % (
            self.__class__.__name__, len(self.wv.index2word), len(self.index2label), self.vector_size, self.alpha)

    def score(self, **kwargs):
        raise NotImplementedError('This method has no reason to exist in this class (for now)')

    def save_word2vec_format(self, **kwargs):
        raise NotImplementedError('This is not a word2vec model')

    @classmethod
    def load_word2vec_format(cls, **kwargs):
        raise NotImplementedError('This is not a word2vec model')

    def intersect_word2vec_format(self, **kwargs):
        raise NotImplementedError('This is not a word2vec model')

    def accuracy(self, **kwargs):
        raise NotImplementedError('This method has no reason to exist in this class (for now)')
