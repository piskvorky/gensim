#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

"""Optimized cython functions for training :class:`~gensim.models.fasttext.FastText` model."""

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.math cimport log
from libc.string cimport memset

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

from word2vec_inner cimport bisect_left, random_int32, scopy, saxpy, dsdot, sscal, \
     REAL_t, EXP_TABLE, our_dot, our_saxpy, our_dot_double, our_dot_float, our_dot_noblas, our_saxpy_noblas

REAL = np.float32

DEF MAX_SENTENCE_LEN = 10000
DEF MAX_SUBWORDS = 1000

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

cdef unsigned long long fasttext_fast_sentence_sg_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1neg, const int size,
    const np.uint32_t word_index, const np.uint32_t word2_index, const np.uint32_t *subwords_index,
    const np.uint32_t subwords_len, const REAL_t alpha, REAL_t *work, REAL_t *l1, unsigned long long next_random,
    REAL_t *word_locks_vocab, REAL_t *word_locks_ngrams) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))
    memset(l1, 0, size * cython.sizeof(REAL_t))

    scopy(&size, &syn0_vocab[row1], &ONE, l1, &ONE)
    for d in range(subwords_len):
        our_saxpy(&size, &ONEF, &syn0_ngrams[subwords_index[d] * size], &ONE, l1, &ONE)
    cdef REAL_t norm_factor = ONEF / subwords_len
    sscal(&size, &norm_factor, l1 , &ONE)

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
        f_dot = our_dot(&size, l1, &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, l1, &ONE, &syn1neg[row2], &ONE)
    our_saxpy(&size, &word_locks_vocab[word2_index], work, &ONE, &syn0_vocab[row1], &ONE)
    for d in range(subwords_len):
        our_saxpy(&size, &word_locks_ngrams[subwords_index[d]], work, &ONE, &syn0_ngrams[subwords_index[d]*size], &ONE)
    return next_random


cdef void fasttext_fast_sentence_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const np.uint32_t *subwords_index, const np.uint32_t subwords_len,
    const REAL_t alpha, REAL_t *work, REAL_t *l1, REAL_t *word_locks_vocab,
    REAL_t *word_locks_ngrams) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2, sgn
    cdef REAL_t f, g, f_dot, lprob

    memset(work, 0, size * cython.sizeof(REAL_t))
    memset(l1, 0, size * cython.sizeof(REAL_t))

    scopy(&size, &syn0_vocab[row1], &ONE, l1, &ONE)
    for d in range(subwords_len):
        our_saxpy(&size, &ONEF, &syn0_ngrams[subwords_index[d] * size], &ONE, l1, &ONE)
    cdef REAL_t norm_factor = ONEF / subwords_len
    sscal(&size, &norm_factor, l1 , &ONE)

    for b in range(codelen):
        row2 = word_point[b] * size
        f_dot = our_dot(&size, l1, &ONE, &syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha

        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, l1, &ONE, &syn1[row2], &ONE)

    our_saxpy(&size, &word_locks_vocab[word2_index], work, &ONE, &syn0_vocab[row1], &ONE)
    for d in range(subwords_len):
        our_saxpy(&size, &word_locks_ngrams[subwords_index[d]], work, &ONE, &syn0_ngrams[subwords_index[d]*size], &ONE)


cdef unsigned long long fasttext_fast_sentence_cbow_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], np.uint32_t *subwords_idx[MAX_SENTENCE_LEN],
    const int subwords_idx_len[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random, REAL_t *word_locks_vocab, REAL_t *word_locks_ngrams) nogil:

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
        count += ONEF
        our_saxpy(&size, &ONEF, &syn0_vocab[indexes[m] * size], &ONE, neu1, &ONE)
        for d in range(subwords_idx_len[m]):
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0_ngrams[subwords_idx[m][d] * size], &ONE, neu1, &ONE)

    if count > (<REAL_t>0.5):
        inv_count = ONEF / count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)

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

        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)

    for m in range(j,k):
        if m == i:
            continue
        our_saxpy(&size, &word_locks_vocab[indexes[m]], work, &ONE, &syn0_vocab[indexes[m]*size], &ONE)
        for d in range(subwords_idx_len[m]):
            our_saxpy(&size, &word_locks_ngrams[subwords_idx[m][d]], work, &ONE, &syn0_ngrams[subwords_idx[m][d]*size], &ONE)

    return next_random


cdef void fasttext_fast_sentence_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], np.uint32_t *subwords_idx[MAX_SENTENCE_LEN],
    const int subwords_idx_len[MAX_SENTENCE_LEN],const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, REAL_t *word_locks_vocab, REAL_t *word_locks_ngrams) nogil:

    cdef long long a, b
    cdef long long row2, sgn
    cdef REAL_t f, g, count, inv_count = 1.0, f_dot, lprob
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        count += ONEF
        our_saxpy(&size, &ONEF, &syn0_vocab[indexes[m] * size], &ONE, neu1, &ONE)
        for d in range(subwords_idx_len[m]):
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0_ngrams[subwords_idx[m][d] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF / count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f_dot = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha

        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)

    for m in range(j,k):
        if m == i:
            continue
        our_saxpy(&size, &word_locks_vocab[indexes[m]], work, &ONE, &syn0_vocab[indexes[m]*size], &ONE)
        for d in range(subwords_idx_len[m]):
            our_saxpy(&size, &word_locks_ngrams[subwords_idx[m][d]], work, &ONE, &syn0_ngrams[subwords_idx[m][d]*size], &ONE)


cdef init_ft_config(FastTextConfig *c, model, alpha, _work, _neu1):
    c[0].hs = model.hs
    c[0].negative = model.negative
    c[0].sample = (model.vocabulary.sample != 0)
    c[0].cbow_mean = model.cbow_mean
    c[0].window = model.window
    c[0].workers = model.workers

    c[0].syn0_vocab = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_vocab))
    c[0].word_locks_vocab = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_vocab_lockf))
    c[0].syn0_ngrams = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_ngrams))
    c[0].word_locks_ngrams = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_ngrams_lockf))

    c[0].alpha = alpha
    c[0].size = model.wv.vector_size

    if c[0].hs:
        c[0].syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if c[0].negative:
        c[0].syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        c[0].cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        c[0].cum_table_len = len(model.vocabulary.cum_table)
    if c[0].negative or c[0].sample:
        c[0].next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    c[0].work = <REAL_t *>np.PyArray_DATA(_work)
    c[0].neu1 = <REAL_t *>np.PyArray_DATA(_neu1)


def train_batch_sg(model, sentences, alpha, _work, _l1):
    """Update skip-gram model by training on a sequence of sentences.

    Each sentence is a list of string tokens, which are looked up in the model's
    vocab dictionary. Called internally from :meth:`gensim.models.fasttext.FastText.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.fasttext.FastText`
        Model to be trained.
    sentences : iterable of list of str
        Corpus streamed directly from disk/network.
    alpha : float
        Learning rate.
    _work : np.ndarray
        Private working memory for each worker.
    _l1 : np.ndarray
        Private working memory for each worker.

    Returns
    -------
    int
        Effective number of words trained.

    """
    cdef FastTextConfig c

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    init_ft_config(&c, model, alpha, _work, _l1)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    c.sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if c.sample and word.sample_int < random_int32(&c.next_random):
                continue
            c.indexes[effective_words] = word.index

            c.subwords_idx_len[effective_words] = <int>(len(model.wv.buckets_word[word.index]))
            c.subwords_idx[effective_words] = <np.uint32_t *>np.PyArray_DATA(model.wv.buckets_word[word.index])

            if c.hs:
                c.codelens[effective_words] = <int>len(word.code)
                c.codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                c.points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)

            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        c.sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, c.window, effective_words)):
        c.reduced_windows[i] = item

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
                        fasttext_fast_sentence_sg_hs(
                            c.points[j], c.codes[j], c.codelens[j], c.syn0_vocab, c.syn0_ngrams, c.syn1, c.size,
                            c.indexes[i], c.subwords_idx[i], c.subwords_idx_len[i], c.alpha, c.work, c.neu1,
                            c.word_locks_vocab, c.word_locks_ngrams)
                    if c.negative:
                        c.next_random = fasttext_fast_sentence_sg_neg(
                            c.negative, c.cum_table, c.cum_table_len, c.syn0_vocab, c.syn0_ngrams, c.syn1neg, c.size,
                            c.indexes[j], c.indexes[i], c.subwords_idx[i], c.subwords_idx_len[i], c.alpha, c.work,
                            c.neu1, c.next_random, c.word_locks_vocab, c.word_locks_ngrams)

    return effective_words


def train_batch_cbow(model, sentences, alpha, _work, _neu1):
    """Update the CBOW model by training on a sequence of sentences.

    Each sentence is a list of string tokens, which are looked up in the model's
    vocab dictionary. Called internally from :meth:`gensim.models.fasttext.FastText.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.fasttext.FastText`
        Model to be trained.
    sentences : iterable of list of str
        Corpus streamed directly from disk/network.
    alpha : float
        Learning rate.
    _work : np.ndarray
        Private working memory for each worker.
    _neu1 : np.ndarray
        Private working memory for each worker.
    Returns
    -------
    int
        Effective number of words trained.

    """
    cdef FastTextConfig c

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    init_ft_config(&c, model, alpha, _work, _neu1)

    if c.hs:
        c.syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if c.negative:
        c.syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        c.cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        c.cum_table_len = len(model.vocabulary.cum_table)
    if c.negative or c.sample:
        c.next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    c.sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if c.sample and word.sample_int < random_int32(&c.next_random):
                continue
            c.indexes[effective_words] = word.index

            c.subwords_idx_len[effective_words] = <int>len(model.wv.buckets_word[word.index])
            c.subwords_idx[effective_words] = <np.uint32_t *>np.PyArray_DATA(model.wv.buckets_word[word.index])

            if c.hs:
                c.codelens[effective_words] = <int>len(word.code)
                c.codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                c.points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        c.sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, c.window, effective_words)):
        c.reduced_windows[i] = item

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
                    fasttext_fast_sentence_cbow_hs(
                        c.points[i], c.codes[i], c.codelens, c.neu1, c.syn0_vocab, c.syn0_ngrams, c.syn1, c.size,
                        c.indexes, c.subwords_idx, c.subwords_idx_len, c.alpha, c.work, i, j, k, c.cbow_mean,
                        c.word_locks_vocab, c.word_locks_ngrams)
                if c.negative:
                    c.next_random = fasttext_fast_sentence_cbow_neg(
                        c.negative, c.cum_table, c.cum_table_len, c.codelens, c.neu1, c.syn0_vocab, c.syn0_ngrams,
                        c.syn1neg, c.size, c.indexes, c.subwords_idx, c.subwords_idx_len, c.alpha, c.work, i, j, k,
                        c.cbow_mean, c.next_random, c.word_locks_vocab, c.word_locks_ngrams)

    return effective_words


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
