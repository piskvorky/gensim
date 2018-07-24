#!/usr/bin/env cython
# distutils: language = c++
# distutils: sources = fast_line_sentence.cpp
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

from gensim.utils import any2utf8
from six import iteritems

cimport numpy as np

from libc.math cimport exp
from libc.math cimport log
from libc.string cimport memset
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool as bool_t

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
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


cdef extern from "fast_line_sentence.h":
    cdef cppclass FastLineSentence:
        FastLineSentence() except +
        FastLineSentence(string&, size_t) except +
        vector[string] ReadSentence() nogil except +
        bool_t IsEof() nogil
        void Reset() nogil


cdef struct VocabItem:
    long long sample_int
    np.uint32_t index
    np.uint8_t *code
    int code_len
    np.uint32_t *point


def rebuild_cython_line_sentence(source, max_sentence_length):
    return CythonLineSentence(source, max_sentence_length=max_sentence_length)


@cython.final
cdef class CythonLineSentence:
    cdef FastLineSentence* _thisptr
    cdef public string source
    cdef public size_t max_sentence_length, max_words_in_batch, offset
    cdef vector[vector[string]] buf_data

    def __cinit__(self, source, offset, max_sentence_length=MAX_SENTENCE_LEN):
        self._thisptr = new FastLineSentence(source, offset)

    def __init__(self, source, offset=0, max_sentence_length=MAX_SENTENCE_LEN):
        self.source = source
        self.offset = offset
        self.max_sentence_length = max_sentence_length
        self.max_words_in_batch = max_sentence_length

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    cpdef bool_t is_eof(self) nogil:
        return self._thisptr.IsEof()

    cpdef vector[string] read_sentence(self) nogil except *:
        return self._thisptr.ReadSentence()

    cpdef vector[vector[string]] _read_chunked_sentence(self) nogil except *:
        cdef vector[string] sent = self.read_sentence()
        return self._chunk_sentence(sent)

    cpdef vector[vector[string]] _chunk_sentence(self, vector[string] sent) nogil:
        cdef vector[vector[string]] res
        cdef vector[string] chunk
        cdef size_t cur_idx = 0

        if sent.size() > self.max_sentence_length:
            while cur_idx < sent.size():
                chunk.clear()
                for i in range(cur_idx, min(cur_idx + self.max_sentence_length, sent.size())):
                    chunk.push_back(sent[i])

                res.push_back(chunk)
                cur_idx += chunk.size()
        else:
            res.push_back(sent)

        return res

    cpdef void reset(self) nogil:
        self._thisptr.Reset()

    def __iter__(self):
        self.reset()
        while not self.is_eof():
            chunked_sentence = self._read_chunked_sentence()
            for chunk in chunked_sentence:
                if not chunk.empty():
                    yield chunk

    def __reduce__(self):
        return rebuild_cython_line_sentence, (self.source, self.max_sentence_length)

    cpdef vector[vector[string]] next_batch(self) nogil except *:
        cdef:
            vector[vector[string]] job_batch
            vector[vector[string]] chunked_sentence
            vector[string] data
            size_t batch_size = 0
            size_t last_idx = 0
            size_t tmp = 0
            int idx

        # Try to read data from previous calls which was not returned
        if not self.buf_data.empty():
            job_batch = self.buf_data
            self.buf_data.clear()

            for sent in job_batch:
                batch_size += sent.size()

        while not self.is_eof() and batch_size <= self.max_words_in_batch:
            data = self.read_sentence()

            chunked_sentence = self._chunk_sentence(data)
            for chunk in chunked_sentence:
                job_batch.push_back(chunk)
                batch_size += chunk.size()

        if batch_size > self.max_words_in_batch:
            # Save data which doesn't fit in batch in order to return it later.
            self.buf_data.clear()

            tmp = batch_size
            idx = job_batch.size() - 1
            while idx >= 0:
                if tmp - job_batch[idx].size() <= self.max_words_in_batch:
                    last_idx = idx + 1
                    break
                else:
                    tmp -= job_batch[idx].size()

                idx -= 1

            for i in range(last_idx, job_batch.size()):
                self.buf_data.push_back(job_batch[i])
            job_batch.resize(last_idx)

        return job_batch


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

cdef void fast_sentence_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param) nogil:
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
    word_locks
        Lock factors for each word. A value of 0 will block training.
    _compute_loss
        Whether or not the loss should be computed at this step.
    _running_training_loss_param
        Running loss, used to debug or inspect how training progresses.

    """

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2, sgn
    cdef REAL_t f, g, f_dot, lprob

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f_dot = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha

        if _compute_loss == 1:
            sgn = (-1)**word_code[b]  # ch function: 0-> 1, 1 -> -1
            lprob = -1*sgn*f_dot
            if lprob <= -MAX_EXP or lprob >= MAX_EXP:
                continue
            lprob = LOG_TABLE[<int>((lprob + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - lprob

        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)

    our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn0[row1], &ONE)


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

cdef unsigned long long fast_sentence_sg_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param) nogil:
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
    word_locks
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
    cdef long long row1 = word2_index * size, row2
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

        row2 = target_index * size
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

    our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn0[row1], &ONE)

    return next_random


cdef void fast_sentence_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, REAL_t *word_locks,
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
    word_locks
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
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f_dot = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha

        if _compute_loss == 1:
            sgn = (-1)**word_code[b]  # ch function: 0-> 1, 1 -> -1
            lprob = -1*sgn*f_dot
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
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m] * size], &ONE)


cdef unsigned long long fast_sentence_cbow_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param) nogil:
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
    word_locks
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
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
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

        row2 = target_index * size
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
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m]*size], &ONE)

    return next_random


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
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)

    cdef int _compute_loss = (1 if compute_loss else 0)
    cdef REAL_t _running_training_loss = model.running_training_loss

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if sample and word.sample_int < random_int32(&next_random):
                continue
            indexes[effective_words] = word.index
            if hs:
                codelens[effective_words] = <int>len(word.code)
                codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, window, effective_words)):
        reduced_windows[i] = item

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - window + reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + window + 1 - reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                for j in range(j, k):
                    if j == i:
                        continue
                    if hs:
                        fast_sentence_sg_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], _alpha, work, word_locks, _compute_loss, &_running_training_loss)
                    if negative:
                        next_random = fast_sentence_sg_neg(negative, cum_table, cum_table_len, syn0, syn1neg, size, indexes[i], indexes[j], _alpha, work, next_random, word_locks, _compute_loss, &_running_training_loss)

    model.running_training_loss = _running_training_loss
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
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int cbow_mean = model.cbow_mean

    cdef int _compute_loss = (1 if compute_loss == True else 0)
    cdef REAL_t _running_training_loss = model.running_training_loss

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if sample and word.sample_int < random_int32(&next_random):
                continue
            indexes[effective_words] = word.index
            if hs:
                codelens[effective_words] = <int>len(word.code)
                codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, window, effective_words)):
        reduced_windows[i] = item

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - window + reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + window + 1 - reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                if hs:
                    fast_sentence_cbow_hs(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, _alpha, work, i, j, k, cbow_mean, word_locks, _compute_loss, &_running_training_loss)
                if negative:
                    next_random = fast_sentence_cbow_neg(negative, cum_table, cum_table_len, codelens, neu1, syn0, syn1neg, size, indexes, _alpha, work, i, j, k, cbow_mean, next_random, word_locks, _compute_loss, &_running_training_loss)

    model.running_training_loss = _running_training_loss
    return effective_words


cdef void prepare_c_structures_for_batch(vector[vector[string]] &sentences, int sample, int hs, int window, int *total_words,
                                         int *effective_words, int *effective_sentences, unsigned long long *next_random,
                                         unordered_map[string, VocabItem] &vocab, int *sentence_idx, np.uint32_t *indexes,
                                         int *codelens, np.uint8_t **codes, np.uint32_t **points,
                                         np.uint32_t *reduced_windows) nogil:
    cdef VocabItem word
    cdef string token
    cdef vector[string] sent

    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if sent.empty():
            continue # ignore empty sentences; leave effective_sentences unchanged
        total_words[0] += sent.size()

        for token in sent:
            # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if vocab.find(token) == vocab.end():
                continue

            word = vocab[token]
            if sample and word.sample_int < random_int32(next_random):
                continue
            indexes[effective_words[0]] = word.index
            if hs:
                codelens[effective_words[0]] = word.code_len
                codes[effective_words[0]] = word.code
                points[effective_words[0]] = word.point
            effective_words[0] += 1
            if effective_words[0] == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences[0] += 1
        sentence_idx[effective_sentences[0]] = effective_words[0]

        if effective_words[0] == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    for i in range(effective_words[0]):
        reduced_windows[i] = random_int32(next_random) % window


def iterate_batches_from_pystream(input_stream):
    job_batch = []
    data = None
    batch_size = 0
    data_length = 0

    for data in input_stream:
        data = [any2utf8(x) for x in data]
        data_length = len(data)

        # can we fit this sentence into the existing job batch?
        if batch_size + data_length <= MAX_SENTENCE_LEN:
            # yes => add it to the current job
            job_batch.append(data)
            batch_size += data_length
        else:
            yield job_batch

            job_batch = [data]

            batch_size = data_length

    # add the last job too (may be significantly smaller than batch_words)
    if job_batch:
        yield job_batch


cdef REAL_t get_alpha(REAL_t alpha, REAL_t end_alpha, int cur_epoch, int num_epochs) nogil:
    return alpha - ((alpha - end_alpha) * (<REAL_t> cur_epoch) / num_epochs)


cdef REAL_t get_next_alpha(REAL_t start_alpha, REAL_t end_alpha, int total_examples, int total_words,
                           int expected_examples, int expected_words, int cur_epoch, int num_epochs) nogil:
    cdef REAL_t epoch_progress

    if expected_examples != -1:
        # examples-based decay
        epoch_progress = (<REAL_t> total_examples) / expected_examples
    else:
        # word-based decay
        epoch_progress = (<REAL_t> total_words) / expected_words

    cdef REAL_t progress = (cur_epoch + epoch_progress) / num_epochs
    cdef REAL_t next_alpha = start_alpha - (start_alpha - end_alpha) * progress
    return max(end_alpha, next_alpha)


def train_epoch_sg(model, corpus_file, offset, _cur_epoch, _expected_examples, _expected_words, _work, _neu1,
                   compute_loss):
    """Train Skipgram model for one epoch by training on an input stream. This function is used only in multistream mode.

    Called internally from :meth:`~gensim.models.word2vec.Word2Vec.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.word2vec.Word2Vec`
        The Word2Vec model instance to train.
    input_stream : iterable of list of str
        The corpus used to train the model.
    _cur_epoch : int
        Current epoch number. Used for calculating and decaying learning rate.
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
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int workers = model.workers

    # For learning rate updates
    cdef int cur_epoch = _cur_epoch
    cdef int num_epochs = model.epochs
    cdef int expected_examples = (-1 if _expected_examples is None else _expected_examples)
    cdef int expected_words = (-1 if _expected_words is None else _expected_words)
    cdef REAL_t start_alpha = model.alpha
    cdef REAL_t end_alpha = model.min_alpha
    cdef REAL_t _alpha = get_alpha(model.alpha, end_alpha, cur_epoch, num_epochs)

    cdef CythonLineSentence input_stream = CythonLineSentence(corpus_file, offset)

    cdef int _compute_loss = (1 if compute_loss == True else 0)
    cdef REAL_t _running_training_loss = model.running_training_loss

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_lockf))
    cdef REAL_t *work
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int total_effective_words = 0, total_sentences = 0, total_words = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)

    # for preparing batches & training
    cdef unordered_map[string, VocabItem] vocab
    cdef vector[vector[string]] sentences
    cdef unsigned long long random_number
    cdef VocabItem word

    # prepare C structures so we can go "full C" and release the Python GIL
    for py_token, vocab_item in iteritems(model.wv.vocab):
        token = any2utf8(py_token)
        word.index = vocab_item.index
        word.sample_int = vocab_item.sample_int

        if hs:
            word.code = <np.uint8_t *>np.PyArray_DATA(vocab_item.code)
            word.code_len = <int>len(vocab_item.code)
            word.point = <np.uint32_t *>np.PyArray_DATA(vocab_item.point)

        vocab[token] = word

    with nogil:
        input_stream.reset()
        while not (input_stream.is_eof() or total_words > expected_words / workers):
            effective_sentences = 0
            effective_words = 0

            sentences = input_stream.next_batch()

            prepare_c_structures_for_batch(sentences, sample, hs, window, &total_words, &effective_words,
                                           &effective_sentences, &next_random, vocab, sentence_idx, indexes,
                                           codelens, codes, points, reduced_windows)

            for sent_idx in range(effective_sentences):
                idx_start = sentence_idx[sent_idx]
                idx_end = sentence_idx[sent_idx + 1]
                for i in range(idx_start, idx_end):
                    j = i - window + reduced_windows[i]
                    if j < idx_start:
                        j = idx_start
                    k = i + window + 1 - reduced_windows[i]
                    if k > idx_end:
                        k = idx_end
                    for j in range(j, k):
                        if j == i:
                            continue
                        if hs:
                            fast_sentence_sg_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], _alpha, work, word_locks, _compute_loss, &_running_training_loss)
                        if negative:
                            next_random = fast_sentence_sg_neg(negative, cum_table, cum_table_len, syn0, syn1neg, size, indexes[i], indexes[j], _alpha, work, next_random, word_locks, _compute_loss, &_running_training_loss)

            total_sentences += sentences.size()
            total_effective_words += effective_words

            _alpha = get_next_alpha(start_alpha, end_alpha, total_sentences, total_words,
                                    expected_examples, expected_words, cur_epoch, num_epochs)

    model.running_training_loss = _running_training_loss
    return total_sentences, total_effective_words, total_words


def train_epoch_cbow(model, corpus_file, offset, _cur_epoch, _expected_examples, _expected_words, _work, _neu1,
                     compute_loss):
    """Train CBOW model for one epoch by training on an input stream. This function is used only in multistream mode.

    Called internally from :meth:`~gensim.models.word2vec.Word2Vec.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.word2vec.Word2Vec`
        The Word2Vec model instance to train.
    input_stream : iterable of list of str
        The corpus used to train the model.
    _cur_epoch : int
        Current epoch number. Used for calculating and decaying learning rate.
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
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int cbow_mean = model.cbow_mean

    # For learning rate updates
    cdef int cur_epoch = _cur_epoch
    cdef int num_epochs = model.epochs
    cdef int expected_examples = (-1 if _expected_examples is None else _expected_examples)
    cdef int expected_words = (-1 if _expected_words is None else _expected_words)
    cdef REAL_t start_alpha = model.alpha
    cdef REAL_t end_alpha = model.min_alpha
    cdef REAL_t _alpha = get_alpha(model.alpha, end_alpha, cur_epoch, num_epochs)

    cdef CythonLineSentence input_stream = CythonLineSentence(corpus_file, offset)

    cdef int _compute_loss = (1 if compute_loss == True else 0)
    cdef REAL_t _running_training_loss = model.running_training_loss

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_lockf))
    cdef REAL_t *work
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int total_effective_words = 0, total_sentences = 0, total_words = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    # for preparing batches & training
    cdef unordered_map[string, VocabItem] vocab
    cdef vector[vector[string]] sentences
    cdef unsigned long long random_number
    cdef VocabItem word

    # prepare C structures so we can go "full C" and release the Python GIL
    for py_token, vocab_item in iteritems(model.wv.vocab):
        token = any2utf8(py_token)
        word.index = vocab_item.index
        word.sample_int = vocab_item.sample_int

        if hs:
            word.code = <np.uint8_t *>np.PyArray_DATA(vocab_item.code)
            word.code_len = <int>len(vocab_item.code)
            word.point = <np.uint32_t *>np.PyArray_DATA(vocab_item.point)

        vocab[token] = word

    with nogil:
        input_stream.reset()
        while not (input_stream.is_eof() or total_words > expected_words / workers):
            effective_sentences = 0
            effective_words = 0

            sentences = input_stream.next_batch()

            prepare_c_structures_for_batch(sentences, sample, hs, window, &total_words, &effective_words,
                                           &effective_sentences, &next_random, vocab, sentence_idx, indexes,
                                           codelens, codes, points, reduced_windows)

            for sent_idx in range(effective_sentences):
                idx_start = sentence_idx[sent_idx]
                idx_end = sentence_idx[sent_idx + 1]
                for i in range(idx_start, idx_end):
                    j = i - window + reduced_windows[i]
                    if j < idx_start:
                        j = idx_start
                    k = i + window + 1 - reduced_windows[i]
                    if k > idx_end:
                        k = idx_end
                    if hs:
                        fast_sentence_cbow_hs(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, _alpha, work, i, j, k, cbow_mean, word_locks, _compute_loss, &_running_training_loss)
                    if negative:
                        next_random = fast_sentence_cbow_neg(negative, cum_table, cum_table_len, codelens, neu1, syn0, syn1neg, size, indexes, _alpha, work, i, j, k, cbow_mean, next_random, word_locks, _compute_loss, &_running_training_loss)

            total_sentences += sentences.size()
            total_effective_words += effective_words

            _alpha = get_next_alpha(start_alpha, end_alpha, total_sentences, total_words,
                                    expected_examples, expected_words, cur_epoch, num_epochs)

    model.running_training_loss = _running_training_loss
    return total_sentences, total_effective_words, total_words


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

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *work
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)

    vlookup = model.wv.vocab
    i = 0
    for token in sentence:
        word = vlookup[token] if token in vlookup else None
        if word is None:
            continue  # should drop the
        indexes[i] = word.index
        codelens[i] = <int>len(word.code)
        codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
        points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
        result += 1
        i += 1
        if i == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?
    sentence_len = i

    # release GIL & train on the sentence
    work[0] = 0.0

    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window
            if j < 0:
                j = 0
            k = i + window + 1
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                score_pair_sg_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], work)

    return work[0]

cdef void score_pair_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, REAL_t *work) nogil:

    cdef long long b
    cdef long long row1 = word2_index * size, row2, sgn
    cdef REAL_t f

    for b in range(codelen):
        row2 = word_point[b] * size
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
    cdef int cbow_mean = model.cbow_mean

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *work
    cdef REAL_t *neu1
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    vlookup = model.wv.vocab
    i = 0
    for token in sentence:
        word = vlookup[token] if token in vlookup else None
        if word is None:
            continue  # for score, should this be a default negative value?
        indexes[i] = word.index
        codelens[i] = <int>len(word.code)
        codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
        points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
        result += 1
        i += 1
        if i == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?
    sentence_len = i

    # release GIL & train on the sentence
    work[0] = 0.0
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window
            if j < 0:
                j = 0
            k = i + window + 1
            if k > sentence_len:
                k = sentence_len
            score_pair_cbow_hs(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, work, i, j, k, cbow_mean)

    return work[0]

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
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)

    for b in range(codelens[i]):
        row2 = word_point[b] * size
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
