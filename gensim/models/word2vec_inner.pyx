#!/usr/bin/env cython
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Optimized cython functions for training :class:`~gensim.models.word2vec.Word2Vec` model."""

import cython
import numpy as np

cimport numpy as np

from libc.math cimport exp
from libc.math cimport log
from libc.string cimport memset

import scipy.linalg.blas as fblas

REAL = np.float32

DEF MAX_SENTENCE_LEN = 10000

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0


# for when fblas.sdot returns a double
cdef REAL_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>dsdot(N, X, incX, Y, incY)

# for when fblas.sdot returns a float
cdef REAL_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>sdot(N, X, incX, Y, incY)

# for when no blas available
cdef REAL_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    # not a true full dot()-implementation: just enough for our cases
    cdef int i
    cdef REAL_t a
    a = <REAL_t>0.0
    for i from 0 <= i < N[0] by 1:
        a += X[i] * Y[i]
    return a

# for when no blas available
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil:
    cdef int i
    for i from 0 <= i < N[0] by 1:
        Y[i * (incY[0])] = (alpha[0]) * X[i * (incX[0])] + Y[i * (incY[0])]

cdef void w2v_fast_sentence_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, REAL_t *words_lockf,
    const np.uint32_t lockf_len, const int _compute_loss, REAL_t *_running_training_loss_param) nogil:
    """Train on a single effective word from the current batch, using the Skip-Gram model.

    In this model we are using a given word to predict a context word (a word that is
    close to the one we are using as training). Hierarchical softmax is used to speed-up
    training.

    Parameters
    ----------
    word_point
        Vector representation of the current word.
    word_code
        ASCII (char == uint8) representation of the current word.
    codelen
        Number of characters (length) in the current word.
    syn0
        Embeddings for the words in the vocabulary (`model.wv.vectors`)
    syn1
        Weights of the hidden layer in the model's trainable neural network.
    size
        Length of the embeddings.
    word2_index
        Index of the context word in the vocabulary.
    alpha
        Learning rate.
    work
        Private working memory for each worker.
    words_lockf
        Lock factors for each word. A value of 0 will block training.
    _compute_loss
        Whether or not the loss should be computed at this step.
    _running_training_loss_param
        Running loss, used to debug or inspect how training progresses.

    """

    cdef long long a, b
    cdef long long row1 = <long long>word2_index * <long long>size, row2, sgn
    cdef REAL_t f, g, f_dot, lprob

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = <long long>word_point[b] * <long long>size
        f_dot = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha

        if _compute_loss == 1:
            sgn = (-1)**word_code[b]  # ch function: 0-> 1, 1 -> -1
            lprob = sgn*f_dot
            if lprob <= -MAX_EXP or lprob >= MAX_EXP:
                continue
            lprob = LOG_TABLE[<int>((lprob + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - lprob

        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)

    our_saxpy(&size, &words_lockf[word2_index % lockf_len], work, &ONE, &syn0[row1], &ONE)


# to support random draws from negative-sampling cum_table
cdef inline unsigned long long bisect_left(np.uint32_t *a, unsigned long long x, unsigned long long lo, unsigned long long hi) nogil:
    cdef unsigned long long mid
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo

# this quick & dirty RNG apparently matches Java's (non-Secure)Random
# note this function side-effects next_random to set up the next number
cdef inline unsigned long long random_int32(unsigned long long *next_random) nogil:
    cdef unsigned long long this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random

cdef unsigned long long w2v_fast_sentence_sg_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, REAL_t *words_lockf,
    const np.uint32_t lockf_len, const int _compute_loss, REAL_t *_running_training_loss_param) nogil:
    """Train on a single effective word from the current batch, using the Skip-Gram model.

    In this model we are using a given word to predict a context word (a word that is
    close to the one we are using as training). Negative sampling is used to speed-up
    training.

    Parameters
    ----------
    negative
        Number of negative words to be sampled.
    cum_table
        Cumulative-distribution table using stored vocabulary word counts for
        drawing random words (with a negative label).
    cum_table_len
        Length of the `cum_table`
    syn0
        Embeddings for the words in the vocabulary (`model.wv.vectors`)
    syn1neg
        Weights of the hidden layer in the model's trainable neural network.
    size
        Length of the embeddings.
    word_index
        Index of the current training word in the vocabulary.
    word2_index
        Index of the context word in the vocabulary.
    alpha
        Learning rate.
    work
        Private working memory for each worker.
    next_random
        Seed to produce the index for the next word to be randomly sampled.
    words_lockf
        Lock factors for each word. A value of 0 will block training.
    _compute_loss
        Whether or not the loss should be computed at this step.
    _running_training_loss_param
        Running loss, used to debug or inspect how training progresses.

    Returns
    -------
    Seed to draw the training word for the next iteration of the same routine.

    """
    cdef long long a
    cdef long long row1 = <long long>word2_index * <long long>size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = <long long>target_index * <long long>size
        f_dot = our_dot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot

        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)

    our_saxpy(&size, &words_lockf[word2_index % lockf_len], work, &ONE, &syn0[row1], &ONE)

    return next_random


cdef void w2v_fast_sentence_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, REAL_t *words_lockf, const np.uint32_t lockf_len,
    const int _compute_loss, REAL_t *_running_training_loss_param) nogil:
    """Train on a single effective word from the current batch, using the CBOW method.

    Using this method we train the trainable neural network by attempting to predict a
    given word by its context (words surrounding the one we are trying to predict).
    Hierarchical softmax method is used to speed-up training.

    Parameters
    ----------
    word_point
        Vector representation of the current word.
    word_code
        ASCII (char == uint8) representation of the current word.
    codelens
        Number of characters (length) for all words in the context.
    neu1
        Private working memory for every worker.
    syn0
        Embeddings for the words in the vocabulary (`model.wv.vectors`)
    syn1
        Weights of the hidden layer in the model's trainable neural network.
    size
        Length of the embeddings.
    word2_index
        Index of the context word in the vocabulary.
    alpha
        Learning rate.
    work
        Private working memory for each worker.
    i
        Index of the word to be predicted from the context.
    j
        Index of the word at the beginning of the context window.
    k
        Index of the word at the end of the context window.
    cbow_mean
        If 0, use the sum of the context word vectors as the prediction. If 1, use the mean.
    words_lockf
        Lock factors for each word. A value of 0 will block training.
    _compute_loss
        Whether or not the loss should be computed at this step.
    _running_training_loss_param
        Running loss, used to debug or inspect how training progresses.

    """
    cdef long long a, b
    cdef long long row2, sgn
    cdef REAL_t f, g, count, inv_count = 1.0, f_dot, lprob
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[<long long>indexes[m] * <long long>size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = <long long>word_point[b] * <long long>size
        f_dot = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha

        if _compute_loss == 1:
            sgn = (-1)**word_code[b]  # ch function: 0-> 1, 1 -> -1
            lprob = sgn*f_dot
            if lprob <= -MAX_EXP or lprob >= MAX_EXP:
                continue
            lprob = LOG_TABLE[<int>((lprob + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - lprob

        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)

    for m in range(j, k):
        if m == i:
            continue
        else:
            our_saxpy(&size, &words_lockf[indexes[m] % lockf_len], work, &ONE, &syn0[<long long>indexes[m] * <long long>size], &ONE)


cdef unsigned long long w2v_fast_sentence_cbow_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random, REAL_t *words_lockf,
    const np.uint32_t lockf_len, const int _compute_loss, REAL_t *_running_training_loss_param) nogil:
    """Train on a single effective word from the current batch, using the CBOW method.

    Using this method we train the trainable neural network by attempting to predict a
    given word by its context (words surrounding the one we are trying to predict).
    Negative sampling is used to speed-up training.

    Parameters
    ----------
    negative
        Number of negative words to be sampled.
    cum_table
        Cumulative-distribution table using stored vocabulary word counts for
        drawing random words (with a negative label).
    cum_table_len
        Length of the `cum_table`
    codelens
        Number of characters (length) for all words in the context.
    neu1
        Private working memory for every worker.
    syn0
        Embeddings for the words in the vocabulary (`model.wv.vectors`)
    syn1neg
        Weights of the hidden layer in the model's trainable neural network.
    size
        Length of the embeddings.
    indexes
        Indexes of the context words in the vocabulary.
    alpha
        Learning rate.
    work
        Private working memory for each worker.
    i
        Index of the word to be predicted from the context.
    j
        Index of the word at the beginning of the context window.
    k
        Index of the word at the end of the context window.
    cbow_mean
        If 0, use the sum of the context word vectors as the prediction. If 1, use the mean.
    next_random
        Seed for the drawing the predicted word for the next iteration of the same routine.
    words_lockf
        Lock factors for each word. A value of 0 will block training.
    _compute_loss
        Whether or not the loss should be computed at this step.
    _running_training_loss_param
        Running loss, used to debug or inspect how training progresses.

    """
    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count = 1.0, label, log_e_f_dot, f_dot
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = indexes[i]

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[<long long>indexes[m] * <long long>size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = <long long>target_index * <long long>size
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot

        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)

    for m in range(j,k):
        if m == i:
            continue
        else:
            our_saxpy(&size, &words_lockf[indexes[m] % lockf_len], work, &ONE, &syn0[<long long>indexes[m] * <long long>size], &ONE)

    return next_random


cdef init_w2v_config(Word2VecConfig *c, model, alpha, compute_loss, _work, _neu1=None):
    c[0].hs = model.hs
    c[0].negative = model.negative
    c[0].sample = (model.sample != 0)
    c[0].cbow_mean = model.cbow_mean
    c[0].window = model.window
    c[0].workers = model.workers

    c[0].compute_loss = (1 if compute_loss else 0)
    c[0].running_training_loss = model.running_training_loss

    c[0].syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    c[0].words_lockf = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_lockf))
    c[0].words_lockf_len = len(model.wv.vectors_lockf)
    c[0].alpha = alpha
    c[0].size = model.wv.vector_size

    if c[0].hs:
        c[0].syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if c[0].negative:
        c[0].syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        c[0].cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
        c[0].cum_table_len = len(model.cum_table)
    if c[0].negative or c[0].sample:
        c[0].next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    c[0].work = <REAL_t *>np.PyArray_DATA(_work)

    if _neu1 is not None:
        c[0].neu1 = <REAL_t *>np.PyArray_DATA(_neu1)


def train_batch_sg(model, sentences, alpha, _work, compute_loss):
    """Update skip-gram model by training on a batch of sentences.

    Called internally from :meth:`~gensim.models.word2vec.Word2Vec.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.word2Vec.Word2Vec`
        The Word2Vec model instance to train.
    sentences : iterable of list of str
        The corpus used to train the model.
    alpha : float
        The learning rate
    _work : np.ndarray
        Private working memory for each worker.
    compute_loss : bool
        Whether or not the training loss should be computed in this batch.

    Returns
    -------
    int
        Number of words in the vocabulary actually used for training (They already existed in the vocabulary
        and were not discarded by negative sampling).

    """
    cdef Word2VecConfig c
    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end
    cdef np.uint32_t *vocab_sample_ints

    init_w2v_config(&c, model, alpha, compute_loss, _work)
    if c.sample:
        vocab_sample_ints = <np.uint32_t *>np.PyArray_DATA(model.wv.expandos['sample_int'])
    if c.hs:
        vocab_codes = model.wv.expandos['code']
        vocab_points = model.wv.expandos['point']

    # prepare C structures so we can go "full C" and release the Python GIL
    c.sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            if token not in model.wv.key_to_index:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            word_index = model.wv.key_to_index[token]
            if c.sample and vocab_sample_ints[word_index] < random_int32(&c.next_random):
                continue
            c.indexes[effective_words] = word_index
            if c.hs:
                c.codelens[effective_words] = <int>len(vocab_codes[word_index])
                c.codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(vocab_codes[word_index])
                c.points[effective_words] = <np.uint32_t *>np.PyArray_DATA(vocab_points[word_index])
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        c.sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    if model.shrink_windows:
        for i, item in enumerate(model.random.randint(0, c.window, effective_words)):
            c.reduced_windows[i] = item
    else:
        for i in range(effective_words):
            c.reduced_windows[i] = 0

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = c.sentence_idx[sent_idx]
            idx_end = c.sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - c.window + c.reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + c.window + 1 - c.reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                for j in range(j, k):
                    if j == i:
                        continue
                    if c.hs:
                        w2v_fast_sentence_sg_hs(c.points[i], c.codes[i], c.codelens[i], c.syn0, c.syn1, c.size, c.indexes[j], c.alpha, c.work, c.words_lockf, c.words_lockf_len, c.compute_loss, &c.running_training_loss)
                    if c.negative:
                        c.next_random = w2v_fast_sentence_sg_neg(c.negative, c.cum_table, c.cum_table_len, c.syn0, c.syn1neg, c.size, c.indexes[i], c.indexes[j], c.alpha, c.work, c.next_random, c.words_lockf, c.words_lockf_len, c.compute_loss, &c.running_training_loss)

    model.running_training_loss = c.running_training_loss
    return effective_words


def train_batch_cbow(model, sentences, alpha, _work, _neu1, compute_loss):
    """Update CBOW model by training on a batch of sentences.

    Called internally from :meth:`~gensim.models.word2vec.Word2Vec.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.word2vec.Word2Vec`
        The Word2Vec model instance to train.
    sentences : iterable of list of str
        The corpus used to train the model.
    alpha : float
        The learning rate.
    _work : np.ndarray
        Private working memory for each worker.
    _neu1 : np.ndarray
        Private working memory for each worker.
    compute_loss : bool
        Whether or not the training loss should be computed in this batch.

    Returns
    -------
    int
        Number of words in the vocabulary actually used for training (They already existed in the vocabulary
        and were not discarded by negative sampling).
    """
    cdef Word2VecConfig c
    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end
    cdef np.uint32_t *vocab_sample_ints

    init_w2v_config(&c, model, alpha, compute_loss, _work, _neu1)
    if c.sample:
        vocab_sample_ints = <np.uint32_t *>np.PyArray_DATA(model.wv.expandos['sample_int'])
    if c.hs:
        vocab_codes = model.wv.expandos['code']
        vocab_points = model.wv.expandos['point']

    # prepare C structures so we can go "full C" and release the Python GIL
    c.sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            if token not in model.wv.key_to_index:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            word_index = model.wv.key_to_index[token]
            if c.sample and vocab_sample_ints[word_index] < random_int32(&c.next_random):
                continue
            c.indexes[effective_words] = word_index
            if c.hs:
                c.codelens[effective_words] = <int>len(vocab_codes[word_index])
                c.codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(vocab_codes[word_index])
                c.points[effective_words] = <np.uint32_t *>np.PyArray_DATA(vocab_points[word_index])
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        c.sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    if model.shrink_windows:
        for i, item in enumerate(model.random.randint(0, c.window, effective_words)):
            c.reduced_windows[i] = item
    else:
        for i in range(effective_words):
            c.reduced_windows[i] = 0

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = c.sentence_idx[sent_idx]
            idx_end = c.sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - c.window + c.reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + c.window + 1 - c.reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                if c.hs:
                    w2v_fast_sentence_cbow_hs(c.points[i], c.codes[i], c.codelens, c.neu1, c.syn0, c.syn1, c.size, c.indexes, c.alpha, c.work, i, j, k, c.cbow_mean, c.words_lockf, c.words_lockf_len, c.compute_loss, &c.running_training_loss)
                if c.negative:
                    c.next_random = w2v_fast_sentence_cbow_neg(c.negative, c.cum_table, c.cum_table_len, c.codelens, c.neu1, c.syn0, c.syn1neg, c.size, c.indexes, c.alpha, c.work, i, j, k, c.cbow_mean, c.next_random, c.words_lockf, c.words_lockf_len, c.compute_loss, &c.running_training_loss)

    model.running_training_loss = c.running_training_loss
    return effective_words


def score_sentence_sg(model, sentence, _work):
    """Obtain likelihood score for a single sentence in a fitted skip-gram representation.

    Notes
    -----
    This scoring function is only implemented for hierarchical softmax (`model.hs == 1`).
    The model should have been trained using the skip-gram model (`model.sg` == 1`).

    Parameters
    ----------
    model : :class:`~gensim.models.word2vec.Word2Vec`
        The trained model. It **MUST** have been trained using hierarchical softmax and the skip-gram algorithm.
    sentence : list of str
        The words comprising the sentence to be scored.
    _work : np.ndarray
        Private working memory for each worker.

    Returns
    -------
    float
        The probability assigned to this sentence by the Skip-Gram model.

    """
    cdef Word2VecConfig c
    c.syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    c.size = model.wv.vector_size

    c.window = model.window

    cdef int i, j, k
    cdef long result = 0
    cdef int sentence_len

    c.syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    # convert Python structures to primitive types, so we can release the GIL
    c.work = <REAL_t *>np.PyArray_DATA(_work)

    vocab_codes = model.wv.expandos['code']
    vocab_points = model.wv.expandos['point']
    i = 0
    for token in sentence:
        word_index = model.wv.key_to_index[token] if token in model.wv.key_to_index else None
        if word_index is None:
            # For score, should this be a default negative value?
            #
            # See comment by @gojomo at https://github.com/RaRe-Technologies/gensim/pull/2698/files#r445827846 :
            #
            # These 'score' functions are a long-ago contribution from @mataddy whose
            # current function/utility is unclear.
            # I've continued to apply mechanical updates to match other changes, and the code
            # still compiles & passes the one (trivial, form-but-not-function) unit test. But it's an
            # idiosyncratic technique, and only works for the non-default hs mode. Here, in lieu of the
            # previous cryptic # should drop the comment, I've asked if for the purposes of this
            # particular kind of 'scoring' (really, loss-tallying indicating how divergent this new
            # text is from what the model learned during training), shouldn't completely missing
            # words imply something very negative, as opposed to nothing-at-all? But probably, this
            # functionality should be dropped. (And ultimately, a talented cleanup of the largely-broken
            # loss-tallying functions might provide a cleaner window into this same measure of how
            # well a text contrasts with model expectations - such as a way to report loss from a
            # single invocation of one fo the inner train methods, without changing the model.)
            continue
        c.indexes[i] = word_index
        c.codelens[i] = <int>len(vocab_codes[word_index])
        c.codes[i] = <np.uint8_t *>np.PyArray_DATA(vocab_codes[word_index])
        c.points[i] = <np.uint32_t *>np.PyArray_DATA(vocab_points[word_index])
        result += 1
        i += 1
        if i == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?
    sentence_len = i

    # release GIL & train on the sentence
    c.work[0] = 0.0

    with nogil:
        for i in range(sentence_len):
            if c.codelens[i] == 0:
                continue
            j = i - c.window
            if j < 0:
                j = 0
            k = i + c.window + 1
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or c.codelens[j] == 0:
                    continue
                score_pair_sg_hs(c.points[i], c.codes[i], c.codelens[i], c.syn0, c.syn1, c.size, c.indexes[j], c.work)

    return c.work[0]

cdef void score_pair_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, REAL_t *work) nogil:

    cdef long long b
    cdef long long row1 = <long long>word2_index * <long long>size, row2, sgn
    cdef REAL_t f

    for b in range(codelen):
        row2 = <long long>word_point[b] * <long long>size
        f = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        sgn = (-1)**word_code[b] # ch function: 0-> 1, 1 -> -1
        f *= sgn
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = LOG_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        work[0] += f

def score_sentence_cbow(model, sentence, _work, _neu1):
    """Obtain likelihood score for a single sentence in a fitted CBOW representation.

    Notes
    -----
    This scoring function is only implemented for hierarchical softmax (`model.hs == 1`).
    The model should have been trained using the skip-gram model (`model.cbow` == 1`).

    Parameters
    ----------
    model : :class:`~gensim.models.word2vec.Word2Vec`
        The trained model. It **MUST** have been trained using hierarchical softmax and the CBOW algorithm.
    sentence : list of str
        The words comprising the sentence to be scored.
    _work : np.ndarray
        Private working memory for each worker.
    _neu1 : np.ndarray
        Private working memory for each worker.

    Returns
    -------
    float
        The probability assigned to this sentence by the Skip-Gram model.

    """
    cdef Word2VecConfig c

    c.cbow_mean = model.cbow_mean
    c.syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    c.size = model.wv.vector_size
    c.window = model.window

    cdef int i, j, k
    cdef long result = 0

    c.syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    # convert Python structures to primitive types, so we can release the GIL
    c.work = <REAL_t *>np.PyArray_DATA(_work)
    c.neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    vocab_codes = model.wv.expandos['code']
    vocab_points = model.wv.expandos['point']
    i = 0
    for token in sentence:
        word_index = model.wv.key_to_index[token] if token in model.wv.key_to_index else None
        if word_index is None:
            continue  # for score, should this be a default negative value?
        c.indexes[i] = word_index
        c.codelens[i] = <int>len(vocab_codes[word_index])
        c.codes[i] = <np.uint8_t *>np.PyArray_DATA(vocab_codes[word_index])
        c.points[i] = <np.uint32_t *>np.PyArray_DATA(vocab_points[word_index])
        result += 1
        i += 1
        if i == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?
    sentence_len = i

    # release GIL & train on the sentence
    c.work[0] = 0.0
    with nogil:
        for i in range(sentence_len):
            if c.codelens[i] == 0:
                continue
            j = i - c.window
            if j < 0:
                j = 0
            k = i + c.window + 1
            if k > sentence_len:
                k = sentence_len
            score_pair_cbow_hs(c.points[i], c.codes[i], c.codelens, c.neu1, c.syn0, c.syn1, c.size, c.indexes, c.work, i, j, k, c.cbow_mean)

    return c.work[0]

cdef void score_pair_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], REAL_t *work,
    int i, int j, int k, int cbow_mean) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count, sgn
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[<long long>indexes[m] * <long long>size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)

    for b in range(codelens[i]):
        row2 = <long long>word_point[b] * <long long>size
        f = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        sgn = (-1)**word_code[b] # ch function: 0-> 1, 1 -> -1
        f *= sgn
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = LOG_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        work[0] += f


def init():
    """Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized into table EXP_TABLE.
     Also calculate log(sigmoid(x)) into LOG_TABLE.

    Returns
    -------
    {0, 1, 2}
        Enumeration to signify underlying data type returned by the BLAS dot product calculation.
        0 signifies double, 1 signifies double, and 2 signifies that custom cython loops were used
        instead of BLAS.

    """
    global our_dot
    global our_saxpy

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
        LOG_TABLE[i] = <REAL_t>log( EXP_TABLE[i] )

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if abs(d_res - expected) < 0.0001:
        our_dot = our_dot_double
        our_saxpy = saxpy
        return 0  # double
    elif abs(p_res[0] - expected) < 0.0001:
        our_dot = our_dot_float
        our_saxpy = saxpy
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        our_dot = our_dot_noblas
        our_saxpy = our_saxpy_noblas
        return 2

FAST_VERSION = init()  # initialize the module
MAX_WORDS_IN_BATCH = MAX_SENTENCE_LEN
