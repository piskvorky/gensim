#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Prakhar Pratyush (er.prakhar2b@gmail.com)

import logging

from gensim.models.word2vec import Word2Vec
from gensim.models.wrappers.fasttext import FastTextKeyedVectors
from gensim.models.wrappers.fasttext import FastText as Ft_Wrapper

import numpy as np
from numpy import dot, zeros, ones, vstack, outer, random, sum as np_sum, empty, float32 as REAL

from scipy.special import expit
from types import GeneratorType
from copy import deepcopy
from six import string_types

from gensim.utils import call_on_class_only

logger = logging.getLogger(__name__)


"""
TO-DO : description of FastText and the API
"""

MAX_WORDS_IN_BATCH = 10000


def train_batch_sg(model, sentences, alpha, work=None):
    """
    Update skip-gram model by training on a sequence of sentences.

    Each sentence is a list of string tokens, which are looked up in the model's
    vocab dictionary. Called internally from `Word2Vec.train()`.

    This is the non-optimized, Python version. If you have cython installed, gensim
    will use the optimized version from word2vec_inner instead.

    """
    result = 0
    for sentence in sentences:
        word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2**32]
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code

            # now go over all words from the (reduced) window, predicting each one in turn
            start = max(0, pos - model.window + reduced_window)
            for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):

                word2_subwords = ['<' + model.wv.index2word[word2.index] + '>']
                word2_subwords += Ft_Wrapper.compute_ngrams(model.wv.index2word[word2.index], model.min_n, model.max_n)
                word2_subwords = set(word2_subwords)

                subwords_indices = []
                for subword in word2_subwords:
                    subwords_indices.append(model.wv.ngrams[subword])

                # don't train on the `word` itself
                if pos2 != pos:
                    train_sg_pair(model, model.wv.index2word[word.index], subwords_indices, alpha)
                    # train_sg_pair(model, model.wv.index2word[word.index], word2.index, alpha)

        result += len(word_vocabs)
    return result


def train_batch_cbow(model, sentences, alpha, work=None, neu1=None):
    """
    Update CBOW model by training on a sequence of sentences.

    """
    result = 0
    for sentence in sentences:
        word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2**32]
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code  ## why random
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]

            word2_subwords = []

            for indices in word2_indices:
                word2_subwords += ['<' + model.wv.index2word[indices] + '>']
                word2_subwords += Ft_Wrapper.compute_ngrams(model.wv.index2word[indices], model.min_n, model.max_n)
            word2_subwords = set(word2_subwords)

            subwords_indices = []
            for subword in word2_subwords:
                subwords_indices.append(model.wv.ngrams[subword])

            l1 = np_sum(model.wv.syn0_all[subwords_indices], axis=0)  # 1 x vector_size
            if subwords_indices and model.cbow_mean:
                l1 /= len(subwords_indices)

            train_cbow_pair(model, word, subwords_indices, l1, alpha)  # train on the sliding window for target word
        result += len(word_vocabs)
    return result


def train_sg_pair(model, word, context_subwords_index, alpha, learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None):
    if context_vectors is None:
        context_vectors = model.wv.syn0_all
    if context_locks is None:
        context_locks = model.syn0_all_lockf

    if word not in model.wv.vocab:
        return
    predict_word = model.wv.vocab[word]  # target word (NN output)

    l1 = np_sum(context_vectors[context_subwords_index], axis=0)
    if context_subwords_index:
        l1 /= len(context_subwords_index)

    # lock_factor = context_locks[context_subwords_index]

    neu1e = zeros(l1.shape)

    if model.hs:
        # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
        l2a = deepcopy(model.syn1[predict_word.point])  # 2d matrix, codelen x layer1_size
        fa = expit(dot(l1, l2a.T))  # propagate hidden -> output
        ga = (1 - predict_word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1[predict_word.point] += outer(ga, l1)  # learn hidden -> output
        neu1e += dot(ga, l2a)  # save error

    if model.negative:
        # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
        word_indices = [predict_word.index]
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != predict_word.index:
                word_indices.append(w)
        l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
        fb = expit(dot(l1, l2b.T))  # propagate hidden -> output
        gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
        neu1e += dot(gb, l2b)  # save error

    if learn_vectors:
        # l1 += neu1e * lock_factor  # learn input -> hidden (mutates model.wv.syn0[word2.index], if that is l1)
        if context_subwords_index:
            neu1e /= len(context_subwords_index)
        for i in context_subwords_index:
            model.wv.syn0_all[i] += neu1e * model.syn0_all_lockf[i]

    return neu1e


def train_cbow_pair(model, word, input_subword_indices, l1, alpha, learn_vectors=True, learn_hidden=True):
    neu1e = zeros(l1.shape)

    if model.hs:
        l2a = model.syn1[word.point]  # 2d matrix, codelen x layer1_size
        fa = expit(dot(l1, l2a.T))  # propagate hidden -> output
        ga = (1. - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
        neu1e += dot(ga, l2a)  # save error

    if model.negative:
        # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
        word_indices = [word.index]  # through word index get all subwords indices (need to make the changes in code)
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != word.index:
                word_indices.append(w)
        l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
        fb = expit(dot(l1, l2b.T))  # propagate hidden -> output
        gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
        neu1e += dot(gb, l2b)  # save error

    if learn_vectors:
        # learn input -> hidden, here for all words in the window separately
        if not model.cbow_mean and input_subword_indices:
            neu1e /= len(input_subword_indices)
        for i in input_subword_indices:
            model.wv.syn0_all[i] += neu1e * model.syn0_all_lockf[i]

    return neu1e


class FastText(Word2Vec):
    def __init__(self, sentences=None, sg=0, hs=0, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, word_ngrams=1, loss='ns', sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, min_n=3, max_n=6, sorted_vocab=1, bucket=2000000,
            trim_rule=None, batch_words=MAX_WORDS_IN_BATCH):

        # TO-D0: later with supervised, take care of model - sg, hs

        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (utf-8 encoded strings) that will be used for training.

        `model` defines the training algorithm. By default, cbow is used. Accepted values are
        'cbow', 'skipgram', (later 'supervised').

        `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the current and predicted word within a sentence.

        `alpha` is the initial learning rate.

        `min_count` = ignore all words with total occurrences lower than this.

        `word_ngram` = max length of word ngram

        `loss` = defines training objective. Allowed values are `hs` (hierarchical softmax),
        `ns` (negative sampling) and `softmax`. Defaults to `ns`

        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
            default is 1e-3, useful range is (0, 1e-5).

        `negative` = the value for negative specifies how many "noise words" should be drawn
        (usually between 5-20). Default is 5. If set to 0, no negative samping is used.
        Only relevant when `loss` is set to `ns`

        `iter` = number of iterations (epochs) over the corpus. Default is 5.

        `min_n` = min length of char ngrams to be used for training word representations. Default is 3.

        `max_n` = max length of char ngrams to be used for training word representations. Set `max_n` to be
        lesser than `min_n` to avoid char ngrams being used. Default is 6.

        `sorted_vocab` = if 1 (default), sort the vocabulary by descending frequency before
        assigning word indexes.
        """

        self.load = call_on_class_only
        self.initialize_word_vectors()

        self.vector_size = size
        self.layer1_size = int(size)

        self.sg = int(sg)
        self.cum_table = None  # for negative sampling
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")

        self.alpha = float(alpha)
        self.min_alpha_yet_reached = float(alpha)  # To warn user if alpha increases
        self.window = int(window)
        self.max_vocab_size = max_vocab_size
        self.seed = seed
        self.random = random.RandomState(seed)

        self.min_count = min_count
        self.min_alpha = float(min_alpha)
        self.workers = int(workers)
        self.word_ngrams = word_ngrams
        self.loss = loss
        self.sample = sample
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.iter = iter
        self.bucket = bucket
        self.null_word = null_word

        self.min_n = min_n
        self.max_n = max_n

        self.wv.min_n = min_n
        self.wv.max_n = max_n

        if self.word_ngrams <= 1 and self.max_n == 0:
            self.bucket = 0

        self.train_count = 0
        self.total_train_time = 0
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.model_trimmed_post_training = False

        self.neg_labels = []
        if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.

        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")
            self.build_vocab(sentences, trim_rule=trim_rule)
            self.train(sentences, total_examples=self.corpus_count, epochs=self.iter,
                       start_alpha=self.alpha, end_alpha=self.min_alpha)
        else:
            if trim_rule is not None:
                logger.warning("The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part of the model. ")
                logger.warning("Model initialized without sentences. trim_rule provided, if any, will be ignored.")

    def train(self, sentences, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None,
              word_count=0,
              queue_factor=2, report_delay=1.0):
        Word2Vec.train(self, sentences, total_examples=self.corpus_count, epochs=self.iter,
                       start_alpha=self.alpha, end_alpha=self.min_alpha)
        self.word_vec_invocab()

    def initialize_word_vectors(self):
        self.wv = FastTextKeyedVectors()

    def __getitem__(self, words):
        if isinstance(words, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.word_vec(words)

        return vstack([self.word_vec(word) for word in words])

    def word_vec_invocab(self):
        """ Store word vectors for invocab words in syn0"""

        # print(model['trees'])
        # print(model.wv.syn0[model.wv.vocab['trees'].index])
        # These two gives same result
        for w, v in self.wv.vocab.items():
            word_vec = np.zeros(self.wv.syn0_all.shape[1])
            ngrams = ['<' + w + '>']
            # We also include the word w itself in the set of its n -grams -- from research paper

            ngrams = Ft_Wrapper.compute_ngrams(w, self.min_n, self.max_n)
            ngrams = set(ngrams)
            ngram_weights = self.wv.syn0_all
            for ngram in ngrams:
                word_vec += ngram_weights[self.wv.ngrams[ngram]]
            word_vec / len(ngrams)

            self.wv.syn0[v.index] = word_vec

    def word_vec(self, word, use_norm=False):
        if word in self.wv.vocab:
            if use_norm:
                return self.wv.syn0norm[self.wv.vocab[word].index]
            else:
                return self.wv.syn0[self.wv.vocab[word].index]
        else:
            logger.info("out of vocab")
            word_vec = np.zeros(self.wv.syn0_all.shape[1])
            ngrams = Ft_Wrapper.compute_ngrams(word, self.min_n, self.max_n)
            ngrams = [ng for ng in ngrams if ng in self.wv.ngrams]
            if use_norm:
                ngram_weights = self.wv.syn0_all_norm
            else:
                ngram_weights = self.wv.syn0_all
            for ngram in ngrams:
                word_vec += ngram_weights[self.wv.ngrams[ngram]]
            if word_vec.any():
                return word_vec / len(ngrams)
            else:  # No ngrams of the word are present in self.ngrams
                raise KeyError('all ngrams for word %s absent from model' % word)

    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False):
        """ In word2vec, we built unigram dictionary, here we will make n-grams dictionary """

        self.scan_vocab(sentences, progress_per=progress_per, trim_rule=trim_rule)  # initial survey
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, update=update)  # trim by min_count & precalculate downsampling
        self.finalize_vocab(update=update)  # build tables & arrays

        # super(build_vocab, self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False)

        self.init_ngrams()

    def reset_ngram_weights(self):
        for ngram in self.wv.ngrams:
            # construct deterministic seed from word AND seed argument
            self.wv.syn0_all[self.wv.ngrams[ngram]] = self.seeded_vector(ngram + str(self.seed))

    def init_ngrams(self):
        self.wv.ngrams = {}
        self.wv.syn0_all = empty((self.bucket + len(self.wv.vocab), self.vector_size), dtype=REAL)
        self.syn0_all_lockf = ones((self.bucket + len(self.wv.vocab), self.vector_size), dtype=REAL)  # zeros suppress learning

        all_ngrams = []
        for w, v in self.wv.vocab.items():
            all_ngrams += ['<' + w + '>']  # for special sequence, for ex- <where> for word 'where' -- from research paper
            all_ngrams += Ft_Wrapper.compute_ngrams(w, self.min_n, self.max_n)
        all_ngrams = set(all_ngrams)
        self.num_ngram_vectors = len(all_ngrams)
        logger.info("Total number of ngrams in the vocab is %d", self.num_ngram_vectors)

        ngram_indices = []
        for i, ngram in enumerate(all_ngrams):
            ngram_hash = Ft_Wrapper.ft_hash(ngram)
            ngram_indices.append(len(self.wv.vocab) + ngram_hash % self.bucket)
            self.wv.ngrams[ngram] = i

        self.wv.syn0_all = self.wv.syn0_all.take(ngram_indices, axis=0)
        # indices in syn0_all now re-arranged according to hashed indices

        self.reset_ngram_weights()

    def _do_train_job(self, sentences, alpha, inits):
        """
        Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.
        """
        work, neu1 = inits
        tally = 0
        if self.sg:
            tally += train_batch_sg(self, sentences, alpha, work)
        else:
            tally += train_batch_cbow(self, sentences, alpha, work, neu1)

        return tally, self._raw_word_count(sentences)
