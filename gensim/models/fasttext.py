#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Prakhar Pratyush (er.prakhar2b@gmail.com)

import logging

from gensim.models.word2vec import Word2Vec
from gensim.models.wrapper.fasttext import FastTextKeyedVectors as ft_keyedvectors
from gensim.models.wrapper.fasttext import FastText as ft_wrapper

import numpy as np
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray, vstack, logaddexp

from scipy.special import expit

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

            # subword_indices = []
            # subword_indices += get_subwords(word)

            # now go over all words from the (reduced) window, predicting each one in turn
            start = max(0, pos - model.window + reduced_window)
            for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                # don't train on the `word` itself

                subword2_indices = get_subwords(model, word2)
                if pos2 != pos:
                    train_sg_pair(model, model.wv.index2word[word.index], subword2_indices, alpha)
                    # train_sg_pair(model, model.wv.index2word[word.index], word2.index, alpha)

        result += len(word_vocabs)
    return result

def train_batch_cbow(model, sentences, alpha, work=None, neu1=None):
    """
    Update CBOW model by training on a sequence of sentences.

    """
    result = 0
    for sentence in sentences:
        # word_vocabs bow in fasttext.cc
        word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2**32]
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code  ## why random
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
            # l1 = np_sum(model.wv.syn0[word2_indices], axis=0)  # 1 x vector_size
            # if word2_indices and model.cbow_mean:
            #   l1 /= len(word2_indices)

            # for loop for ngrams for words in this window here
            word2_subwords_indices = []

            for indices in word2_indices:
                word2_subwords_indices += get_subwords(model, model.wv.syn0[indices])  # subwords for each word in window except target word

            l1 = np_sum(model.wv.syn0_all[word2_subwords_indices], axis=0)  # 1 x vector_size
            if word2_subwords_indices and model.cbow_mean:
                l1 /= len(word2_subwords_indices)

            train_cbow_pair(model, word, word2_subwords_indices, l1, alpha)  # train on the sliding window for target word
        result += len(word_vocabs)
    return result

def train_sg_pair(model, word, context_index, alpha, learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None):
    if context_vectors is None:
        context_vectors = model.wv.syn0
    if context_locks is None:
        context_locks = model.syn0_lockf

    if word not in model.wv.vocab:
        return
    predict_word = model.wv.vocab[word]  # target word (NN output)

    l1 = context_vectors[context_index]  # input word (NN input/projection layer)
    lock_factor = context_locks[context_index]

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
        l1 += neu1e * lock_factor  # learn input -> hidden (mutates model.wv.syn0[word2.index], if that is l1)
    return neu1e

def train_cbow_pair(model, word, input_word_indices, l1, alpha, learn_vectors=True, learn_hidden=True):
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
        word_indices = [word.index]
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
        if not model.cbow_mean and input_word_indices:
            neu1e /= len(input_word_indices)
        for i in input_word_indices:
            model.wv.syn0_all[i] += neu1e * model.syn0_lockf[i]  # maybe need to create syn0_all_lockf

    return neu1e

def get_subwords(self, word):
    """
    int32_t i = getId(word);
    if (i >= 0) {
        return getNgrams(i);
    }
    """

    subword_indices = []
    all_subwords = compute_subwords(word, self.wv.min_n, self.wv.max_n)
    for subword in all_subwords:
        # int32_t h = hash(ngram) % args_->bucket;
        # ngrams.push_back(nwords_ + h);
        subword_hash = ft_hash(subword)
        subword_indices.append(len(self.wv.vocab) + subword_hash % self.bucket)

    # self.wv.syn0_all = self.wv.syn0_all.take(subword_indices, axis=0)  # self.wv.syn0_all[subword_indices]
    return subword_indices


def compute_subwords(word, min_n, max_n):
        BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
        extended_word = BOW + word + EOW
        subwords = []

        for subword_length in range(min_n, min(len(extended_word), max_n) + 1):
            for i in range(0, len(extended_word) - subword_length + 1):
                subwords.append(extended_word[i:i + subword_length])  # append or += ? discuss
                # As of now, we have string subwords, we want to do hashing now
        return subwords

@staticmethod
def ft_hash(string):
    """
    Reproduces [hashing trick](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)
    used in fastText.

    """
    # Runtime warnings for integer overflow are raised, this is expected behaviour. These warnings are suppressed.
    old_settings = np.seterr(all='ignore')
    h = np.uint32(2166136261)
    for c in string:
        h = h ^ np.uint32(ord(c))
        h = h * np.uint32(16777619)
    np.seterr(**old_settings)
    return h

@staticmethod
def compute_ngrams(word, min_n, max_n):
    ngram_indices = []
    BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
    extended_word = BOW + word + EOW
    ngrams = []
    for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
        for i in range(0, len(extended_word) - ngram_length + 1):
            ngrams.append(extended_word[i:i + ngram_length])
    return ngrams




class FastText(Word2Vec):
    def __init__(self, model='cbow', hs=0, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, word_ngrams=1, loss='ns', sample=1e-3,seed=1, workers=3, min_alpha=0.0001,
            negative=5, iter=5, min_n=3, max_n=6, sorted_vocab=1, bucket=2000000,
            trim_rule=None, batch_words=MAX_WORDS_IN_BATCH):

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

        self.model = model
        self.vector_size = size
        self.layer1_size = int(size)
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
        self.iter = iter
        self.bucket = bucket

        # what is null_word in word2vec ?

        self.min_n = min_n
        self.max_n = max_n

        if self.word_ngrams <= 1 and self.max_n == 0:
            self.bucket = 0

        self.train_count = 0
        self.total_train_time = 0
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.model_trimmed_post_training = False

        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")
            self.build_vocab(sentences, trim_rule=trim_rule)
            # let word2vec code run here

            self.train(sentences, total_examples=self.corpus_count, epochs=self.iter,
                       start_alpha=self.alpha, end_alpha=self.min_alpha)
        else :
            if trim_rule is not None :
                logger.warning("The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part of the model. ")
                logger.warning("Model initialized without sentences. trim_rule provided, if any, will be ignored." )

    def initialize_word_vectors():

        self.wv = ft_keyedvectors.FastTextKeyedVectors()
        # TO-DO : wv or word_vec

    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False):

        super(build_vocab, self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False)

        # syn0_all for all the ngrams
        self.wv.syn0_all = np.zeros(shape=((self.bucket + len(self.wv.vocab)), self.vector_size))

        ft_wrapper.init_ngrams()
        # while training, use word index to get all the ngrams already computed

    
    def _do_train_job(self, sentences, alpha, inits):
        """
        Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.
        """

        work, neu1 = inits
        tally = 0
        if self.model == 'cbow':
            tally += train_batch_cbow(self, sentences, alpha, work, neu1)
        elif self.model == 'skipgram':
            tally += train_batch_sg(self, sentences, alpha, work)
        return tally, self._raw_word_count(sentences)


