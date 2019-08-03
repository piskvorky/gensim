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
    #
    # Contains model parameters and indices required for training.
    #
    # This struct performs two main roles.  First, it offers a lower-level
    # abstraction over the gensim.models.fasttext.FastText model class, keeping
    # some of its attributes as C types.
    #
    # The second role is to index batches of the corpus in a way that is
    # productive for FastText training.  More specifically, this index is flat:
    # it arranges all tokens in a conceptually one-dimensional array, bypassing
    # OOV terms and empty sentences.
    #
    # Once this struct is fully initialized, it is sufficient for training.
    # Because it consists of entirely C-level data types, it can exist without
    # the GIL, enabling faster processing and parallelization.
    #
    # Example usage:
    #
    #   1) init_ft_config: initialize the struct, allocate working memory
    #   2) populate_ft_config: populate the indices
    #   3) fasttext_train_any: perform actual training
    #

    #
    # Model parameters.  These get copied as-is from the Python model.
    #
    int hs, negative, sample, size, window, cbow_mean, workers
    REAL_t alpha

    #
    # The syn0_vocab and syn0_ngrams arrays store vectors for vocabulary terms
    # and ngrams, respectively, as 1D arrays in scanline order. For example,
    # syn0_vocab[i * size : (i + 1) * size] contains the elements for the ith
    # vocab term.
    #
    REAL_t *syn0_vocab
    REAL_t *syn0_ngrams

    #
    # The arrays below selectively enable/disable training for specific vocab
    # terms and ngrams.  If word_locks_vocab[i] is 0, training is disabled;
    # if it is 1, training is enabled.
    #
    REAL_t *word_locks_vocab
    REAL_t *word_locks_ngrams

    #
    # Working memory.  These are typically large enough to hold a single
    # vector each.
    #
    REAL_t *work
    REAL_t *neu1

    #
    # Most of the arrays are indexed by the ordinal number of a word
    # (also known as terms or tokens).  For example:
    #
    #   - indexes[N]: the index of the Nth token within the vocabulary
    #   - reduced_windows[N]: a random integer by which to resize the window around the Nth token
    #
    np.uint32_t indexes[MAX_SENTENCE_LEN]
    np.uint32_t reduced_windows[MAX_SENTENCE_LEN]

    #
    # We keep track of sentence boundaries here.  The tokens of the Xth
    # sentence will be between [sentence_idx[X], sentence_idx[X + 1]).
    #
    int sentence_idx[MAX_SENTENCE_LEN + 1]

    # For hierarchical softmax
    REAL_t *syn1
    np.uint32_t *points[MAX_SENTENCE_LEN]

    #
    # Each vocabulary term has a binary code, with frequent terms having
    # shorter codes.  This gets assigned in the _assign_binary_codes function
    # in gensim.models.word2vec.py.  Since the lengths of the codes vary, and
    # this is C, we need to keep the lengths of each code as well as the codes
    # themselves.
    #
    np.uint8_t *codes[MAX_SENTENCE_LEN]
    int codelens[MAX_SENTENCE_LEN]

    # For negative sampling
    REAL_t *syn1neg
    np.uint32_t *cum_table
    unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    unsigned long long next_random

    #
    # For passing subwords information as C objects for nogil.  More
    # specifically, subwords_idx[i] is an array that contains the buckets in
    # which the ith subword (ngram) occurs.  Since this is C, we also need to
    # store the length of that array separately: that's what subwords_idx_len
    # is for.
    #
    int subwords_idx_len[MAX_SENTENCE_LEN]
    np.uint32_t *subwords_idx[MAX_SENTENCE_LEN]


#
# See fasttext_inner.pyx for documentation on the functions below.
#
cdef void init_ft_config(FastTextConfig *c, model, alpha, _work, _neu1)


cdef object populate_ft_config(FastTextConfig *c, vocab, buckets_word, sentences)


cdef void fasttext_fast_sentence_sg_neg(FastTextConfig *c, int i, int j) nogil


cdef void fasttext_fast_sentence_sg_hs(FastTextConfig *c, int i, int j) nogil


cdef void fasttext_fast_sentence_cbow_neg(FastTextConfig *c, int i, int j, int k) nogil


cdef void fasttext_fast_sentence_cbow_hs(FastTextConfig *c, int i, int j, int k) nogil


cdef void fasttext_train_any(FastTextConfig *c, int num_sentences, int sg) nogil
