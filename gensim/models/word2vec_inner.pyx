#!/usr/bin/env python
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
from libc.string cimport memset

from cpython cimport PyCObject_AsVoidPtr
from scipy.linalg.blas import cblas

# y = x
ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil

# y += alpha * x
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil

# dot(x, y); return value should be `float`, but it only works with `double` (?!)
ctypedef double (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil

# sqrt(x*x); again, only works with `double`
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil


cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(cblas.scopy._cpointer)
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(cblas.saxpy._cpointer)
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(cblas.sdot._cpointer)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(cblas.snrm2._cpointer)


REAL = np.float32
ctypedef np.float32_t REAL_t


DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE

def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.

    """
    cdef int i
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = EXP_TABLE[i] / (EXP_TABLE[i] + 1)

init()  # initialize the module


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void fast_sentence(
    np.uint32_t[::1] word_point, np.uint8_t[::1] word_code, unsigned long int codelen,
    REAL_t[:, ::1] _syn0, REAL_t[:, ::1] _syn1, int size,
    np.uint32_t word2_index, REAL_t alpha, REAL_t[::1] _work) nogil:
    cdef int one = 1
    cdef REAL_t onef = <REAL_t>1.0
    cdef long long a, b
    cdef long long row1 = word2_index * size, row2
    cdef REAL_t f, g
    cdef REAL_t *syn0 = &_syn0[0, 0]
    cdef REAL_t *syn1 = &_syn1[0, 0]
    cdef REAL_t *work = &_work[0]

    # plain cython
    # for a in range(size):
    #     work[a] = <REAL_t>0.0
    # for b in range(codelen):
    #     row2 = word_point[b] * size
    #     f = <REAL_t>0.0
    #     for a in range(size):
    #         f += syn0[row1 + a] * syn1[row2 + a]
    #     g = (1 - word_code[b] - <REAL_t>1.0 / (<REAL_t>1.0 + exp(-f))) * alpha
    #     for a in range(size):
    #         work[a] += g * syn1[row2 + a]
    #     for a in range(size):
    #         syn1[row2 + a] += g * syn0[row1 + a]
    # for a in range(size):
    #     syn0[row1 + a] += work[a]

    # cython + EXP_TABLE
    # for a in range(size):
    #     work[a] = <REAL_t>0.0
    # for b in range(codelen):
    #     row2 = word_point[b] * size
    #     f = <REAL_t>0.0
    #     for a in range(size):
    #         f += syn0[row1 + a] * syn1[row2 + a]
    #     if f <= -MAX_EXP or f >= MAX_EXP:
    #         continue
    #     f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
    #     g = (1 - word_code[b] - f) * alpha
    #     for a in range(size):
    #         work[a] += g * syn1[row2 + a]
    #     for a in range(size):
    #         syn1[row2 + a] += g * syn0[row1 + a]
    # for a in range(size):
    #     syn0[row1 + a] += work[a]

    # cython + BLAS
    # memset(work, 0, size * cython.sizeof(REAL_t))
    # for b in range(codelen):
    #     row2 = word_point[b] * size
    #     f = sdot(&size, &syn0[row1], &one, &syn1[row2], &one)
    #     g = (1 - word_code[b] - <REAL_t>1.0 / (<REAL_t>1.0 + exp(-f))) * alpha
    #     saxpy(&size, &g, &syn1[row2], &one, work, &one)
    #     saxpy(&size, &g, &syn0[row1], &one, &syn1[row2], &one)
    # saxpy(&size, &onef, work, &one, &syn0[row1], &one)

    # cython + BLAS + EXP_TABLE
    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f = sdot(&size, &syn0[row1], &one, &syn1[row2], &one)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        saxpy(&size, &g, &syn1[row2], &one, work, &one)
        saxpy(&size, &g, &syn0[row1], &one, &syn1[row2], &one)
    saxpy(&size, &onef, work, &one, &syn0[row1], &one)


def train_sentence(model, sentence, alpha):
    """
    Update skip-gram hierarchical softmax model by training on a single sentence,
    where `sentence` is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary). Called internally from `train_model())`.

    """
    work = np.empty(model.layer1_size, dtype=REAL)  # each thread must have its own work memory
    for pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip
        reduced_window = np.random.randint(model.window)  # `b` in the original word2vec code

        # now go over all words from the (reduced) window, predicting each one in turn
        start = max(0, pos - model.window + reduced_window)
        for pos2, word2 in enumerate(sentence[start : pos + model.window + 1 - reduced_window], start):
            if pos2 == pos or word2 is None:
                # don't train on OOV words and on the `word` itself
                continue
            fast_sentence(word.point, word.code, len(word.point), model.syn0, model.syn1, model.layer1_size, word2.index, alpha, work)
