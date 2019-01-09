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


cdef unsigned long long fasttext_fast_sentence_sg_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1neg, const int size,
    const np.uint32_t word_index, const np.uint32_t word2_index, const np.uint32_t *subwords_index,
    const np.uint32_t subwords_len, const REAL_t alpha, REAL_t *work, REAL_t *l1, unsigned long long next_random,
    REAL_t *word_locks_vocab, REAL_t *word_locks_ngrams) nogil


cdef void fasttext_fast_sentence_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const np.uint32_t *subwords_index, const np.uint32_t subwords_len,
    const REAL_t alpha, REAL_t *work, REAL_t *l1, REAL_t *word_locks_vocab,
    REAL_t *word_locks_ngrams) nogil


cdef unsigned long long fasttext_fast_sentence_cbow_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], np.uint32_t *subwords_idx[MAX_SENTENCE_LEN],
    const int subwords_idx_len[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random, REAL_t *word_locks_vocab, REAL_t *word_locks_ngrams) nogil


cdef void fasttext_fast_sentence_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], np.uint32_t *subwords_idx[MAX_SENTENCE_LEN],
    const int subwords_idx_len[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, REAL_t *word_locks_vocab, REAL_t *word_locks_ngrams) nogil


cdef init_ft_config(FastTextConfig *c, model, alpha, _work, _neu1)