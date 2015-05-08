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
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, int train_hidden, int train_inputs,
    REAL_t *syn0locks) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        if train_hidden:
            our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)
    if train_inputs:
        our_saxpy(&size, &syn0locks[word2_index], work, &ONE, &syn0[row1], &ONE)


cdef unsigned long long fast_sentence_dbow_neg(
    const int negative, np.uint32_t *table, unsigned long long table_len,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, int train_hidden, int train_inputs, REAL_t *syn0locks) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
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
        f = our_dot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        if train_hidden:
            our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
    if train_inputs:
        our_saxpy(&size, &syn0locks[word2_index], work, &ONE, &syn0[row1], &ONE)

    return next_random


cdef void fast_sentence_dm_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN], 
    int lbl_codelens[MAX_SENTENCE_LEN], REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const np.uint32_t lbl_indexes[MAX_SENTENCE_LEN], 
    const REAL_t alpha, REAL_t *work, int i, int j, int k, int lbl_length, int learn_hidden,
    int learn_lbls, int learn_words, REAL_t *syn0locks) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count
    cdef int m

    # l1 already composed by caller, passed in as neu1
    memset(work, 0, size * cython.sizeof(REAL_t))  # work accumulates net l1 error
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)
    if learn_words:
        for m in range(j, k):
            if m == i or codelens[m] == 0:
                continue
            else:
                our_saxpy(&size, &syn0locks[indexes[m]], work, &ONE, &syn0[indexes[m] * size], &ONE)
    if learn_lbls:
        for m in range(lbl_length):
            if lbl_codelens[m] == 0:
                continue
            else:
                our_saxpy(&size, &syn0locks[lbl_indexes[m]], work, &ONE, &syn0[lbl_indexes[m]*size], &ONE)


cdef unsigned long long fast_sentence_dm_neg(
    const int negative, np.uint32_t *table, unsigned long long table_len, int codelens[MAX_SENTENCE_LEN], 
    int lbl_codelens[MAX_SENTENCE_LEN], REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, const int size,
    np.uint32_t indexes[MAX_SENTENCE_LEN], np.uint32_t lbl_indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, unsigned long long next_random, int lbl_length, int learn_hidden, int learn_lbls,
    int learn_words, REAL_t *syn0locks) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count, label
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    # l1 already composed by caller, passed in as neu1
    memset(work, 0, size * cython.sizeof(REAL_t))  # work accumulates net l1 error
    word_index = indexes[i]
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
        f = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)
    if learn_words:
        for m in range(j,k):
            if m == i or codelens[m] == 0:
                continue
            else:
                our_saxpy(&size, &syn0locks[indexes[m]], work, &ONE, &syn0[indexes[m]*size], &ONE)
    if learn_lbls:
        for m in range(lbl_length):
            if lbl_codelens[m] == 0:
                continue
            else:
                our_saxpy(&size, &syn0locks[lbl_indexes[m]], work, &ONE, &syn0[lbl_indexes[m]*size], &ONE)

    return next_random

cdef void fast_sentence_dmc_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int word_code_len,
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int layer1_size, const int vector_size,
    const np.uint32_t window_indexes[MAX_SENTENCE_LEN],
    const REAL_t alpha, REAL_t *work, const int lbl_length, const int window,
    int learn_hidden, int learn_lbls, int learn_words, REAL_t *syn0locks) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g
    cdef int m

    # l1 already composed by caller, passed in as neu1
    memset(work, 0, layer1_size * cython.sizeof(REAL_t))  # work accumulates net l1 error
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
    if learn_lbls:
        for m in range(lbl_length):
            our_saxpy(&vector_size, &syn0locks[window_indexes[m]], &work[m * vector_size], &ONE,
                      &syn0[window_indexes[m] * vector_size], &ONE)
    if learn_words:
        for m in range(lbl_length, lbl_length + (2 * window)):
            our_saxpy(&vector_size, &syn0locks[window_indexes[m]], &work[m*vector_size], &ONE,
                      &syn0[window_indexes[m] * vector_size], &ONE)


cdef unsigned long long fast_sentence_dmc_neg(
    const int negative, np.uint32_t *table, unsigned long long table_len,
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, const int layer1_size, const int vector_size,
    np.uint32_t window_indexes[MAX_SENTENCE_LEN],
    const REAL_t alpha, REAL_t *work, const int predict_word_index,
    const int lbl_length, const int window, unsigned long long next_random,
    int learn_hidden, int learn_lbls, int learn_words, REAL_t *syn0locks) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d, m

    # l1 already composed by caller, passed in as neu1
    memset(work, 0, layer1_size * cython.sizeof(REAL_t))  # work accumulates net l1 error
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
    if learn_lbls:
        for m in range(lbl_length):
            our_saxpy(&vector_size, &syn0locks[window_indexes[m]], &work[m * vector_size], &ONE,
                      &syn0[window_indexes[m] * vector_size], &ONE)
    if learn_words:
        for m in range(lbl_length, lbl_length + (2 * window)):
            our_saxpy(&vector_size, &syn0locks[window_indexes[m]], &work[m*vector_size], &ONE,
                      &syn0[window_indexes[m] * vector_size], &ONE)

    return next_random


def train_sentence_dbow(model, sentence, lbls, alpha, _work, train_words, train_lbls):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int tw = train_words
    cdef int tl = train_lbls

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef int lbl_codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t lbl_indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int lbl_length
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

    # lock some of syn0 against training
    cdef REAL_t *syn0locks

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))
    lbl_length = <int>min(MAX_SENTENCE_LEN, len(lbls))

    syn0locks = <REAL_t *>np.PyArray_DATA(model.syn0locks)

    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            reduced_windows[i] = np.random.randint(window)
            if hs:
                codelens[i] = <int>len(word.code)
                codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            else:
                codelens[i] = 1
            result += 1
    for i in range(lbl_length):
        word = lbls[i]
        if word is None:
            lbl_codelens[i] = 0
        else:
            lbl_indexes[i] = word.index
            if hs:
                lbl_codelens[i] = <int>len(word.code)
            else:
                lbl_codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            if tw:  # simultaneous skip-gram wordvec-training
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
                        fast_sentence_dbow_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j],
                                              _alpha, work, 1, 1, syn0locks)
                    if negative:
                        # we reuse the DBOW function, as it is equivalent to skip-gram for this purpose
                        next_random = fast_sentence_dbow_neg(negative, table, table_len, syn0, syn1neg, size,
                                                             indexes[i], indexes[j], _alpha, work, next_random,
                                                             1, 1, syn0locks)

            if tl:  # docvec-training
                for j in range(lbl_length):
                    if lbl_codelens[j] == 0:
                        continue
                    if hs:
                        fast_sentence_dbow_hs(points[i], codes[i], codelens[i], syn0, syn1, size, lbl_indexes[j],
                                              _alpha, work, 1, 1, syn0locks)
                    if negative:
                        next_random = fast_sentence_dbow_neg(negative, table, table_len, syn0, syn1neg, size,
                                                             indexes[i], lbl_indexes[j], _alpha, work, next_random,
                                                             1, 1, syn0locks)

    return result


def train_sentence_dm(model, sentence, lbls, alpha, _work, _neu1, _train_words, _train_lbls):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int learn_words = _train_words
    cdef int learn_lbls = _train_lbls
    cdef int learn_hidden = True
    cdef int cbow_mean = model.cbow_mean
    cdef REAL_t count, inv_count

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *work
    cdef REAL_t *neu1
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef int lbl_codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t lbl_indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int lbl_length
    cdef int window = model.window

    cdef int i, j, k, m
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]
    cdef np.uint32_t *lbl_points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *lbl_codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24)*np.random.randint(0,2**24) + np.random.randint(0,2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))

    syn0locks = <REAL_t *>np.PyArray_DATA(model.syn0locks)

    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            reduced_windows[i] = np.random.randint(window)
            if hs:
                codelens[i] = <int>len(word.code)
                codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            else:
                codelens[i] = 1
            result += 1

    lbl_length = <int>min(MAX_SENTENCE_LEN, len(lbls))
    for i in range(lbl_length):
        word = lbls[i]
        if word is None:
            lbl_codelens[i] = 0
        else:
            lbl_indexes[i] = word.index
            reduced_windows[i] = np.random.randint(window)
            if hs:
                lbl_codelens[i] = <int>len(word.code)
            else:
                lbl_codelens[i] = 1
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > sentence_len:
                k = sentence_len

            # compose l1 (in neu1)
            memset(neu1, 0, size * cython.sizeof(REAL_t))
            count = <REAL_t>0.0
            for m in range(j, k):
                if m == i or codelens[m] == 0:
                    continue
                else:
                    count += ONEF
                    our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
            for m in range(lbl_length):
                if lbl_codelens[m] == 0:
                    continue
                else:
                    count += ONEF
                    our_saxpy(&size, &ONEF, &syn0[lbl_indexes[m] * size], &ONE, neu1, &ONE)
            if cbow_mean and count > (<REAL_t>0.5):
                inv_count = ONEF/count
                sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

            if hs:
                fast_sentence_dm_hs(points[i], codes[i], codelens, lbl_codelens, neu1, syn0, syn1,
                                    size, indexes, lbl_indexes, _alpha, work, i, j, k, lbl_length,
                                    learn_hidden, learn_lbls, learn_words, syn0locks)
            if negative:
                next_random = fast_sentence_dm_neg(negative, table, table_len, codelens, lbl_codelens, neu1, syn0,
                                                   syn1neg, size, indexes, lbl_indexes, _alpha, work, i, j, k,
                                                   next_random, lbl_length, learn_hidden, learn_lbls, learn_words, syn0locks)

    return result


def train_sentence_dm_concat(model, sentence, lbls, alpha, _work, _neu1, _learn_words, _learn_lbls):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int learn_hidden = True
    cdef int learn_lbls = _learn_lbls
    cdef int learn_words = _learn_words

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *work
    cdef REAL_t *neu1
    cdef REAL_t _alpha = alpha
    cdef int layer1_size = model.layer1_size
    cdef int vector_size = model.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef int lbl_codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t window_indexes[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int lbl_length
    cdef int window = model.window
    cdef int expected_lbl_length = model.dm_lbl_count

    cdef int i, j, k, m, n
    cdef long result = 0
    cdef int null_word_index = model.vocab['\0'].index

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]
    cdef np.uint32_t *lbl_points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *lbl_codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    lbl_length = <int>min(MAX_SENTENCE_LEN, len(lbls))
    if lbl_length != expected_lbl_length:
        return 0  # skip doc without expected nmber of lbls

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.table))
        table_len = len(model.table)
        next_random = (2**24)*np.random.randint(0,2**24) + np.random.randint(0,2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    # optional locking of some vactors against backprop-learnind
    syn0locks = <REAL_t *>np.PyArray_DATA(model.syn0locks)

    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))
    j = 0
    for i in range(sentence_len):
        word = sentence[i]
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

    for i in range(lbl_length):
        word = lbls[i]
        if word is None:
            # no support for missing lbls where expected; skip sentence
            return 0
        else:
            window_indexes[i] = word.index
            result += 1

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            j = i - window      # negative OK: will pad with null word
            k = i + window + 1  # past sentence end OK: will pad with null word

            # compose l1 & clear work
            n = lbl_length
            for m in range(j, k):
                if m == i:
                    continue
                if m < 0 or m >= sentence_len:
                    window_indexes[n] =  null_word_index
                else:
                    window_indexes[n] = indexes[m]
                n = n + 1
            for m in range(lbl_length + (2 * window)):
                memcpy(&neu1[m * vector_size], &syn0[window_indexes[m] * vector_size], vector_size * cython.sizeof(REAL_t))
            memset(work, 0, layer1_size * cython.sizeof(REAL_t))

            if hs:
                fast_sentence_dmc_hs(points[i], codes[i], codelens[i], neu1, syn0, syn1,
                                     layer1_size, vector_size, window_indexes, _alpha,
                                     work, lbl_length, window,
                                     learn_hidden, learn_lbls, learn_words, syn0locks)
            if negative:
                next_random = fast_sentence_dmc_neg(negative, table, table_len, neu1, syn0, syn1neg,
                                                   layer1_size, vector_size, window_indexes, _alpha,
                                                   work, indexes[i], lbl_length, window,
                                                   next_random, learn_hidden, learn_lbls, learn_words, syn0locks)

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
