#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import cython
import numpy as np
from numpy import zeros, float32 as REAL
cimport numpy as np

from libc.math cimport exp
from libc.string cimport memset, memcpy

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

from scipy.linalg.blas import fblas

REAL = np.float32
ctypedef np.float32_t REAL_t

DEF MAX_SENTENCE_LEN = 10000

ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

# function implementations swapped based on BLAS detected
ctypedef REAL_t (*our_dot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef void (*our_saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil

cdef our_dot_ptr our_dot
cdef our_saxpy_ptr our_saxpy

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


cdef void fast_sentence_dbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *context_vectors, REAL_t *syn1, const int size,
    const np.uint32_t context_index, const REAL_t alpha, REAL_t *work, int learn_context, int learn_hidden, 
    REAL_t *context_locks) nogil:

    cdef long long a, b
    cdef long long row1 = context_index * size, row2
    cdef REAL_t f, g

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f = our_dot(&size, &context_vectors[row1], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&size, &g, &context_vectors[row1], &ONE, &syn1[row2], &ONE)
    if learn_context:
        our_saxpy(&size, &context_locks[context_index], work, &ONE, &context_vectors[row1], &ONE)


cdef unsigned long long fast_sentence_dbow_neg(
    const int negative, np.uint32_t *table, unsigned long long table_len,
    REAL_t *context_vectors, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t context_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, int learn_context, int learn_hidden, REAL_t *context_locks) nogil:

    cdef long long a
    cdef long long row1 = context_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0
        row2 = target_index * size
        f = our_dot(&size, &context_vectors[row1], &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&size, &g, &context_vectors[row1], &ONE, &syn1neg[row2], &ONE)
    if learn_context:
        our_saxpy(&size, &context_locks[context_index], work, &ONE, &context_vectors[row1], &ONE)

    return next_random


cdef void fast_sentence_dm_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int word_code_len,
    REAL_t *neu1, REAL_t *syn1, const REAL_t alpha, REAL_t *work,
    const int size, int learn_hidden) nogil:

    cdef long long b
    cdef long long row2
    cdef REAL_t f, g

    # l1 already composed by caller, passed in as neu1
    # work (also passed in)  will accumulate l1 error
    for b in range(word_code_len):
        row2 = word_point[b] * size
        f = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)


cdef unsigned long long fast_sentence_dm_neg(
    const int negative, np.uint32_t *table, unsigned long long table_len, unsigned long long next_random,
    REAL_t *neu1, REAL_t *syn1neg, const int predict_word_index, const REAL_t alpha, REAL_t *work,
    const int size, int learn_hidden) nogil:

    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    # l1 already composed by caller, passed in as neu1
    # work (also passsed in) will accumulate l1 error for outside application
    for d in range(negative+1):
        if d == 0:
            target_index = predict_word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == predict_word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    return next_random

cdef void fast_sentence_dmc_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int word_code_len,
    REAL_t *neu1, REAL_t *syn1, const REAL_t alpha, REAL_t *work, 
    const int layer1_size, const int vector_size, int learn_hidden) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g
    cdef int m

    # l1 already composed by caller, passed in as neu1
    # work accumulates net l1 error; eventually applied by caller
    for b in range(word_code_len):
        row2 = word_point[b] * layer1_size
        f = our_dot(&layer1_size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        our_saxpy(&layer1_size, &g, &syn1[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&layer1_size, &g, neu1, &ONE, &syn1[row2], &ONE)


cdef unsigned long long fast_sentence_dmc_neg(
    const int negative, np.uint32_t *table, unsigned long long table_len, unsigned long long next_random,
    REAL_t *neu1, REAL_t *syn1neg, const int predict_word_index, const REAL_t alpha, REAL_t *work, 
    const int layer1_size, const int vector_size, int learn_hidden) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d, m

    # l1 already composed by caller, passed in as neu1
    # work accumulates net l1 error; eventually applied by caller    
    for d in range(negative+1):
        if d == 0:
            target_index = predict_word_index
            label = ONEF
        else:
            target_index = table[(next_random >> 16) % table_len]
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == predict_word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * layer1_size
        f = our_dot(&layer1_size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&layer1_size, &g, &syn1neg[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&layer1_size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    return next_random


def train_sentence_dbow(model, word_vocabs, doclbl_indexes, alpha, work=None,
                        train_words=False, learn_doclbls=True, learn_words=True, learn_hidden=True,
                        word_vectors=None, word_locks=None, doclbl_vectors=None, doclbl_locks=None):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int _train_words = train_words
    cdef int _learn_words = learn_words
    cdef int _learn_hidden = learn_hidden
    cdef int _learn_doclbls = learn_doclbls

    cdef REAL_t *_word_vectors
    cdef REAL_t *_doclbl_vectors
    cdef REAL_t *_word_locks
    cdef REAL_t *_doclbl_locks
    cdef REAL_t *_work
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t _doclbl_indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int doclbl_len
    cdef int window = model.window

    cdef int i, j
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    # default vectors, locks from syn0/doclbl_syn0
    if word_vectors is None:
       word_vectors = model.syn0
    _word_vectors = <REAL_t *>(np.PyArray_DATA(word_vectors))
    if doclbl_vectors is None:
       doclbl_vectors = model.docvecs.doclbl_syn0
    _doclbl_vectors = <REAL_t *>(np.PyArray_DATA(doclbl_vectors))
    if word_locks is None:
       word_locks = model.syn0_lockf
    _word_locks = <REAL_t *>(np.PyArray_DATA(word_locks))
    if doclbl_locks is None:
       doclbl_locks = model.docvecs.doclbl_syn0_lockf
    _doclbl_locks = <REAL_t *>(np.PyArray_DATA(doclbl_locks))

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    if work is None:
       work = zeros(model.layer1_size, dtype=REAL)
    _work = <REAL_t *>np.PyArray_DATA(work)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(word_vocabs))
    doclbl_len = <int>min(MAX_SENTENCE_LEN, len(doclbl_indexes))

    for i in range(sentence_len):
        predict_word = word_vocabs[i]
        if predict_word is None:
            codelens[i] = 0
        else:
            indexes[i] = predict_word.index
            if hs:
                codelens[i] = <int>len(predict_word.code)
                codes[i] = <np.uint8_t *>np.PyArray_DATA(predict_word.code)
                points[i] = <np.uint32_t *>np.PyArray_DATA(predict_word.point)
            else:
                codelens[i] = 1
            result += 1
    if _train_words:
        # single randint() call avoids a big thread-synchronization slowdown
        for i, item in enumerate(np.random.randint(0, window, sentence_len)):
            reduced_windows[i] = item
    for i in range(doclbl_len):
        _doclbl_indexes[i] = doclbl_indexes[i]
        result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            if _train_words:  # simultaneous skip-gram wordvec-training
                j = i - window + reduced_windows[i]
                if j < 0:
                    j = 0
                k = i + window + 1 - reduced_windows[i]
                if k > sentence_len:
                    k = sentence_len
                for j in range(j, k):
                    if j == i or codelens[j] == 0:
                        continue
                    if hs:
                        # we reuse the DBOW function, as it is equivalent to skip-gram for this purpose
                        fast_sentence_dbow_hs(points[i], codes[i], codelens[i], _word_vectors, syn1, size, indexes[j],
                                              _alpha, _work, _learn_words, _learn_hidden, _word_locks)
                    if negative:
                        # we reuse the DBOW function, as it is equivalent to skip-gram for this purpose
                        next_random = fast_sentence_dbow_neg(negative, table, table_len, _word_vectors, syn1neg, size,
                                                             indexes[i], indexes[j], _alpha, _work, next_random,
                                                             _learn_words, _learn_hidden, _word_locks)

            # docvec-training
            for j in range(doclbl_len):
                if hs:
                    fast_sentence_dbow_hs(points[i], codes[i], codelens[i], _doclbl_vectors, syn1, size, _doclbl_indexes[j],
                                          _alpha, _work, _learn_doclbls, _learn_hidden, _doclbl_locks)
                if negative:
                    next_random = fast_sentence_dbow_neg(negative, table, table_len, _doclbl_vectors, syn1neg, size,
                                                             indexes[i], _doclbl_indexes[j], _alpha, _work, next_random,
                                                             _learn_doclbls, _learn_hidden, _doclbl_locks)

    return result


def train_sentence_dm(model, word_vocabs, doclbl_indexes, alpha, work=None, neu1=None,
                      learn_doclbls=True, learn_words=True, learn_hidden=True,
                      word_vectors=None, word_locks=None, doclbl_vectors=None, doclbl_locks=None):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int _learn_doclbls = learn_doclbls
    cdef int _learn_words = learn_words
    cdef int _learn_hidden = learn_hidden
    cdef int cbow_mean = model.cbow_mean
    cdef REAL_t count, inv_count = 1.0

    cdef REAL_t *_word_vectors
    cdef REAL_t *_doclbl_vectors
    cdef REAL_t *_word_locks
    cdef REAL_t *_doclbl_locks
    cdef REAL_t *_work
    cdef REAL_t *_neu1
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t _doclbl_indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int doclbl_len
    cdef int window = model.window

    cdef int i, j, k, m
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    # default vectors, locks from syn0/doclbl_syn0
    if word_vectors is None:
       word_vectors = model.syn0
    _word_vectors = <REAL_t *>(np.PyArray_DATA(word_vectors))
    if doclbl_vectors is None:
       doclbl_vectors = model.docvecs.doclbl_syn0
    _doclbl_vectors = <REAL_t *>(np.PyArray_DATA(doclbl_vectors))
    if word_locks is None:
       word_locks = model.syn0_lockf
    _word_locks = <REAL_t *>(np.PyArray_DATA(word_locks))
    if doclbl_locks is None:
       doclbl_locks = model.docvecs.doclbl_syn0_lockf
    _doclbl_locks = <REAL_t *>(np.PyArray_DATA(doclbl_locks))

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24)*np.random.randint(0,2**24) + np.random.randint(0,2**24)

    # convert Python structures to primitive types, so we can release the GIL
    if work is None:
       work = zeros(model.layer1_size, dtype=REAL)
    _work = <REAL_t *>np.PyArray_DATA(work)
    if neu1 is None:
       neu1 = zeros(model.layer1_size, dtype=REAL)
    _neu1 = <REAL_t *>np.PyArray_DATA(neu1)

    sentence_len = <int>min(MAX_SENTENCE_LEN, len(word_vocabs))
    j = 0
    for i in range(sentence_len):
        word = word_vocabs[i]
        if word is None:
            # shrink sentence to leave out word
            sentence_len = sentence_len - 1
            continue  # leaving j unchanged
        else:
            indexes[j] = word.index
            if hs:
                codelens[j] = <int>len(word.code)
                codes[j] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[j] = <np.uint32_t *>np.PyArray_DATA(word.point)
            result += 1
            j = j + 1
    # single randint() call avoids a big thread-sync slowdown
    for i, item in enumerate(np.random.randint(0, window, sentence_len)):
        reduced_windows[i] = item

    doclbl_len = <int>min(MAX_SENTENCE_LEN, len(doclbl_indexes))
    for i in range(doclbl_len):
        _doclbl_indexes[i] = doclbl_indexes[i]
        result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > sentence_len:
                k = sentence_len

            # compose l1 (in _neu1) & clear _work
            memset(_neu1, 0, size * cython.sizeof(REAL_t))
            count = <REAL_t>0.0
            for m in range(j, k):
                if m == i:
                    continue
                else:
                    count += ONEF
                    our_saxpy(&size, &ONEF, &_word_vectors[indexes[m] * size], &ONE, _neu1, &ONE)
            for m in range(doclbl_len):
                count += ONEF
                our_saxpy(&size, &ONEF, &_doclbl_vectors[_doclbl_indexes[m] * size], &ONE, _neu1, &ONE)
            if count > (<REAL_t>0.5):
                inv_count = ONEF/count
            if cbow_mean:
                sscal(&size, &inv_count, _neu1, &ONE)  # (does this need BLAS-variants like saxpy?)
            memset(_work, 0, size * cython.sizeof(REAL_t))  # work to accumulate l1 error
            
            if hs:
                fast_sentence_dm_hs(points[i], codes[i], codelens[i],
                                    _neu1, syn1, _alpha, _work,
                                    size, _learn_hidden)
            if negative:
                next_random = fast_sentence_dm_neg(negative, table, table_len, next_random,
                                                   _neu1, syn1neg, indexes[i], _alpha, _work,
                                                   size, _learn_hidden)

            if not cbow_mean:
                sscal(&size, &inv_count, _work, &ONE)  # (does this need BLAS-variants like saxpy?)
            # apply accumulated error in work
            if _learn_doclbls:
                for m in range(doclbl_len):
                    our_saxpy(&size, &_doclbl_locks[_doclbl_indexes[m]], _work,
                              &ONE, &_doclbl_vectors[_doclbl_indexes[m] * size], &ONE)
            if _learn_words:
                for m in range(j, k):
                    if m == i:
                        continue
                    else:
                         our_saxpy(&size, &_word_locks[indexes[m]], _work, &ONE,
                                   &_word_vectors[indexes[m] * size], &ONE)

    return result


def train_sentence_dm_concat(model, word_vocabs, doclbl_indexes, alpha, work=None, neu1=None,
                             learn_doclbls=True, learn_words=True, learn_hidden=True,
                             word_vectors=None, word_locks=None, doclbl_vectors=None, doclbl_locks=None):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int _learn_doclbls = learn_doclbls
    cdef int _learn_words = learn_words
    cdef int _learn_hidden = learn_hidden

    cdef REAL_t *_word_vectors
    cdef REAL_t *_doclbl_vectors
    cdef REAL_t *_word_locks
    cdef REAL_t *_doclbl_locks
    cdef REAL_t *_work
    cdef REAL_t *_neu1
    cdef REAL_t _alpha = alpha
    cdef int layer1_size = model.layer1_size
    cdef int vector_size = model.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t _doclbl_indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t window_indexes[MAX_SENTENCE_LEN] 
    cdef int sentence_len
    cdef int doclbl_len
    cdef int window = model.window
    cdef int expected_doclbl_len = model.dm_lbl_count

    cdef int i, j, k, m, n
    cdef long result = 0
    cdef int null_word_index = model.vocab['\0'].index

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    doclbl_len = <int>min(MAX_SENTENCE_LEN, len(doclbl_indexes))
    if doclbl_len != expected_doclbl_len:
        return 0  # skip doc without expected nmber of lbls

    # default vectors, locks from syn0/doclbl_syn0
    if word_vectors is None:
       word_vectors = model.syn0
    _word_vectors = <REAL_t *>(np.PyArray_DATA(word_vectors))
    if doclbl_vectors is None:
       doclbl_vectors = model.docvecs.doclbl_syn0
    _doclbl_vectors = <REAL_t *>(np.PyArray_DATA(doclbl_vectors))
    if word_locks is None:
       word_locks = model.syn0_lockf
    _word_locks = <REAL_t *>(np.PyArray_DATA(word_locks))
    if doclbl_locks is None:
       doclbl_locks = model.docvecs.doclbl_syn0_lockf
    _doclbl_locks = <REAL_t *>(np.PyArray_DATA(doclbl_locks))

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24)*np.random.randint(0,2**24) + np.random.randint(0,2**24)

    # convert Python structures to primitive types, so we can release the GIL
    if work is None:
       work = zeros(model.layer1_size, dtype=REAL)
    _work = <REAL_t *>np.PyArray_DATA(work)
    if neu1 is None:
       neu1 = zeros(model.layer1_size, dtype=REAL)
    _neu1 = <REAL_t *>np.PyArray_DATA(neu1)

    sentence_len = <int>min(MAX_SENTENCE_LEN, len(word_vocabs))
    j = 0
    for i in range(sentence_len):
        word = word_vocabs[i]
        if word is None:
            # shrink sentence to leave out word
            sentence_len = sentence_len - 1
            continue  # leaving j unchanged
        else:
            indexes[j] = word.index
            if hs:
                codelens[j] = <int>len(word.code)
                codes[j] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[j] = <np.uint32_t *>np.PyArray_DATA(word.point)
            else:
                codelens[j] = 1
            result += 1
            j = j + 1

    for i in range(doclbl_len):
        _doclbl_indexes[i] = doclbl_indexes[i]
        result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            j = i - window      # negative OK: will pad with null word
            k = i + window + 1  # past sentence end OK: will pad with null word

            # compose l1 & clear work
            for m in range(doclbl_len):
                # doc vector(s)
                memcpy(&_neu1[m * vector_size], &_doclbl_vectors[_doclbl_indexes[m] * vector_size],
                       vector_size * cython.sizeof(REAL_t))
            n = 0
            for m in range(j, k):
                # word vectors in window
                if m == i:
                    continue
                if m < 0 or m >= sentence_len:
                    window_indexes[n] =  null_word_index
                else:
                    window_indexes[n] = indexes[m]
                n = n + 1
            for m in range(2 * window):
                memcpy(&_neu1[(doclbl_len + m) * vector_size], &_word_vectors[window_indexes[m] * vector_size],
                       vector_size * cython.sizeof(REAL_t))
            memset(_work, 0, layer1_size * cython.sizeof(REAL_t))  # work to accumulate l1 error

            if hs:
                fast_sentence_dmc_hs(points[i], codes[i], codelens[i],
                                     _neu1, syn1, _alpha, _work,
                                     layer1_size, vector_size, _learn_hidden)
            if negative:
                next_random = fast_sentence_dmc_neg(negative, table, table_len, next_random,
                                                    _neu1, syn1neg, indexes[i], _alpha, _work, 
                                                   layer1_size, vector_size, _learn_hidden)

            if _learn_doclbls:
                for m in range(doclbl_len):
                    our_saxpy(&vector_size, &_doclbl_locks[_doclbl_indexes[m]], &_work[m * vector_size],
                              &ONE, &_doclbl_vectors[_doclbl_indexes[m] * vector_size], &ONE)
            if _learn_words:
                for m in range(2 * window):
                    our_saxpy(&vector_size, &_word_locks[window_indexes[m]], &_work[(doclbl_len + m) * vector_size],
                              &ONE, &_word_vectors[window_indexes[m] * vector_size], &ONE)

    return result


def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.

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

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        our_dot = our_dot_double
        our_saxpy = saxpy
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
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
