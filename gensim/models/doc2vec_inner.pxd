#!/usr/bin/env cython
# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8
#
# shared type definitions for doc2vec_inner
# used from doc2vec_corpusfile
#
# Copyright (C) 2018 Dmitry Persiyanov <dmitry.persiyanov@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import numpy as np
cimport numpy as np

from word2vec_inner cimport REAL_t

DEF MAX_DOCUMENT_LEN = 10000


cdef struct Doc2VecConfig:
    int hs, negative, sample, learn_doctags, learn_words, learn_hidden, train_words, cbow_mean
    int document_len, doctag_len, window, expected_doctag_len, null_word_index, workers, docvecs_count

    REAL_t *word_vectors
    REAL_t *doctag_vectors
    REAL_t *word_locks
    REAL_t *doctag_locks
    REAL_t *work
    REAL_t *neu1
    REAL_t alpha
    int layer1_size, vector_size

    int codelens[MAX_DOCUMENT_LEN]
    np.uint32_t indexes[MAX_DOCUMENT_LEN]
    np.uint32_t doctag_indexes[MAX_DOCUMENT_LEN]
    np.uint32_t window_indexes[MAX_DOCUMENT_LEN]
    np.uint32_t reduced_windows[MAX_DOCUMENT_LEN]

    # For hierarchical softmax
    REAL_t *syn1
    np.uint32_t *points[MAX_DOCUMENT_LEN]
    np.uint8_t *codes[MAX_DOCUMENT_LEN]

    # For negative sampling
    REAL_t *syn1neg
    np.uint32_t *cum_table
    unsigned long long cum_table_len, next_random


cdef void fast_document_dbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *context_vectors, REAL_t *syn1, const int size,
    const np.uint32_t context_index, const REAL_t alpha, REAL_t *work, int learn_context, int learn_hidden,
    REAL_t *context_locks) nogil


cdef unsigned long long fast_document_dbow_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *context_vectors, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t context_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, int learn_context, int learn_hidden, REAL_t *context_locks) nogil


cdef void fast_document_dm_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int word_code_len,
    REAL_t *neu1, REAL_t *syn1, const REAL_t alpha, REAL_t *work,
    const int size, int learn_hidden) nogil


cdef unsigned long long fast_document_dm_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, unsigned long long next_random,
    REAL_t *neu1, REAL_t *syn1neg, const int predict_word_index, const REAL_t alpha, REAL_t *work,
    const int size, int learn_hidden) nogil


cdef void fast_document_dmc_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int word_code_len,
    REAL_t *neu1, REAL_t *syn1, const REAL_t alpha, REAL_t *work,
    const int layer1_size, const int vector_size, int learn_hidden) nogil


cdef unsigned long long fast_document_dmc_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, unsigned long long next_random,
    REAL_t *neu1, REAL_t *syn1neg, const int predict_word_index, const REAL_t alpha, REAL_t *work,
    const int layer1_size, const int vector_size, int learn_hidden) nogil


cdef init_d2v_config(Doc2VecConfig *c, model, alpha, learn_doctags, learn_words, learn_hidden, train_words=*, work=*,
                     neu1=*, word_vectors=*, word_locks=*, doctag_vectors=*, doctag_locks=*, docvecs_count=*)
