#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8
#
# shared type definitions for fasttext_inner
# used from fasttext_corpusfile
#


import numpy as np
cimport numpy as np

from word2vec_inner cimport REAL_t


DEF MAX_SENTENCE_LEN = 10000


cdef struct FastTextConfig:
    int hs, negative, sample, size, window, cbow_mean, workers
    REAL_t alpha

    REAL_t *syn0_vocab
    REAL_t *word_locks_vocab
    REAL_t *syn0_ngrams
    REAL_t *word_locks_ngrams

    REAL_t *work
    REAL_t *neu1

    int codelens[MAX_SENTENCE_LEN]
    np.uint32_t indexes[MAX_SENTENCE_LEN]
    np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    int sentence_idx[MAX_SENTENCE_LEN + 1]

    # For hierarchical softmax
    REAL_t *syn1
    np.uint32_t *points[MAX_SENTENCE_LEN]
    np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    REAL_t *syn1neg
    np.uint32_t *cum_table
    unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    unsigned long long next_random

    # For passing subwords information as C objects for nogil
    int subwords_idx_len[MAX_SENTENCE_LEN]
    np.uint32_t *subwords_idx[MAX_SENTENCE_LEN]


cdef void fasttext_fast_sentence_sg_neg(FastTextConfig *c, int i, int j) nogil


cdef void fasttext_fast_sentence_sg_hs(FastTextConfig *c, int i, int j) nogil


cdef void fasttext_fast_sentence_cbow_neg(FastTextConfig *c, int i, int j, int k) nogil


cdef void fasttext_fast_sentence_cbow_hs(FastTextConfig *c, int i, int j, int k) nogil


cdef void fasttext_train_any(FastTextConfig *c, int num_sentences, int sg) nogil


cdef init_ft_config(FastTextConfig *c, model, alpha, _work, _neu1)
