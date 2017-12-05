#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

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

from word2vec_inner cimport bisect_left, random_int32, \
     scopy, saxpy, sdot, dsdot, snrm2, sscal, \
     REAL_t, EXP_TABLE, \
     our_dot, our_saxpy, \
     our_dot_double, our_dot_float, our_dot_noblas, our_saxpy_noblas

REAL = np.float32

DEF MAX_SENTENCE_LEN = 10000
DEF MAX_SUBWORDS = 1000

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0


cdef unsigned long long fast_sentence_sg_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1neg, const int size,
    const np.uint32_t word_index, const np.uint32_t *subwords_index, const np.uint32_t subwords_len,
    const REAL_t alpha, REAL_t *work, REAL_t *l1, unsigned long long next_random, REAL_t *word_locks_vocab,
    REAL_t *word_locks_ngrams) nogil:

    cdef long long a
    cdef np.uint32_t word2_index = subwords_index[0]
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))
    memset(l1, 0, size * cython.sizeof(REAL_t))

    scopy(&size, &syn0_vocab[row1], &ONE, l1, &ONE)
    for d in range(1, subwords_len):
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
    for d in range(1, subwords_len):
        our_saxpy(&size, &word_locks_ngrams[subwords_index[d]], work, &ONE, &syn0_ngrams[subwords_index[d]*size], &ONE)
    return next_random


cdef void fast_sentence_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1, const int size,
    const np.uint32_t *subwords_index, const np.uint32_t subwords_len, 
    const REAL_t alpha, REAL_t *work, REAL_t *l1, REAL_t *word_locks_vocab,
    REAL_t *word_locks_ngrams) nogil:

    cdef long long a, b
    cdef np.uint32_t word2_index = subwords_index[0]
    cdef long long row1 = word2_index * size, row2, sgn
    cdef REAL_t f, g, f_dot, lprob

    memset(work, 0, size * cython.sizeof(REAL_t))
    memset(l1, 0, size * cython.sizeof(REAL_t))

    scopy(&size, &syn0_vocab[row1], &ONE, l1, &ONE)
    for d in range(1, subwords_len):
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
    for d in range(1, subwords_len):
        our_saxpy(&size, &word_locks_ngrams[subwords_index[d]], work, &ONE, &syn0_ngrams[subwords_index[d]*size], &ONE)


cdef unsigned long long fast_sentence_cbow_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const np.uint32_t *subwords_idx[MAX_SENTENCE_LEN],
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


cdef void fast_sentence_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const np.uint32_t *subwords_idx[MAX_SENTENCE_LEN],
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


def train_batch_sg(model, sentences, alpha, _work, _l1):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.sample != 0)

    cdef REAL_t *syn0_vocab = <REAL_t *>(np.PyArray_DATA(model.wv.syn0_vocab))
    cdef REAL_t *word_locks_vocab = <REAL_t *>(np.PyArray_DATA(model.syn0_vocab_lockf))
    cdef REAL_t *syn0_ngrams = <REAL_t *>(np.PyArray_DATA(model.wv.syn0_ngrams))
    cdef REAL_t *word_locks_ngrams = <REAL_t *>(np.PyArray_DATA(model.syn0_ngrams_lockf))

    cdef REAL_t *work
    cdef REAL_t *l1
    
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

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

    # For passing subwords information as C objects for nogil
    cdef int subwords_idx_len[MAX_SENTENCE_LEN]
    cdef np.uint32_t *subwords_idx[MAX_SENTENCE_LEN]
    # dummy dictionary to ensure that the memory locations that subwords_idx point to
    # are referenced throughout so that it isn't put back to free memory pool by Python's memory manager
    subword_arrays = {} 

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
        cum_table_len = len(model.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    l1 = <REAL_t *>np.PyArray_DATA(_l1)

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

            subwords = [model.wv.ngrams[subword_i] for subword_i in model.wv.ngrams_word[model.wv.index2word[word.index]]]
            word_subwords = np.array([word.index] + subwords, dtype=np.uint32)
            subwords_idx_len[effective_words] = <int>(len(subwords) + 1)
            subwords_idx[effective_words] = <np.uint32_t *>np.PyArray_DATA(word_subwords)
            # ensures reference count of word_subwords doesn't reach 0
            subword_arrays[effective_words] = word_subwords

            if hs:
                codelens[effective_words] = <int>len(word.code)
                codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)

            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, window, effective_words)):
        reduced_windows[i] = item

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
                        fast_sentence_sg_hs(
                            points[j], codes[j], codelens[j], syn0_vocab, syn0_ngrams, syn1, size,
                            subwords_idx[i], subwords_idx_len[i], _alpha, work, l1, word_locks_vocab,
                            word_locks_ngrams)
                    if negative:
                        next_random = fast_sentence_sg_neg(
                            negative, cum_table, cum_table_len, syn0_vocab, syn0_ngrams, syn1neg, size,
                            indexes[j], subwords_idx[i], subwords_idx_len[i], _alpha, work, l1,
                            next_random, word_locks_vocab, word_locks_ngrams)

    return effective_words


def train_batch_cbow(model, sentences, alpha, _work, _neu1):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.sample != 0)
    cdef int cbow_mean = model.cbow_mean

    cdef REAL_t *syn0_vocab = <REAL_t *>(np.PyArray_DATA(model.wv.syn0_vocab))
    cdef REAL_t *word_locks_vocab = <REAL_t *>(np.PyArray_DATA(model.syn0_vocab_lockf))
    cdef REAL_t *syn0_ngrams = <REAL_t *>(np.PyArray_DATA(model.wv.syn0_ngrams))
    cdef REAL_t *word_locks_ngrams = <REAL_t *>(np.PyArray_DATA(model.syn0_ngrams_lockf))

    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

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

    # For passing subwords information as C objects for nogil
    cdef int subwords_idx_len[MAX_SENTENCE_LEN]
    cdef np.uint32_t *subwords_idx[MAX_SENTENCE_LEN]
    # dummy dictionary to ensure that the memory locations that subwords_idx point to
    # are referenced throughout so that it isn't put back to free memory pool by Python's memory manager
    subword_arrays = {}

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
        cum_table_len = len(model.cum_table)
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

            subwords = [model.wv.ngrams[subword_i] for subword_i in model.wv.ngrams_word[model.wv.index2word[word.index]]]
            word_subwords = np.array(subwords, dtype=np.uint32)
            subwords_idx_len[effective_words] = <int>len(subwords)
            subwords_idx[effective_words] = <np.uint32_t *>np.PyArray_DATA(word_subwords)
            # ensures reference count of word_subwords doesn't reach 0
            subword_arrays[effective_words] = word_subwords

            if hs:
                codelens[effective_words] = <int>len(word.code)
                codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break

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
                    fast_sentence_cbow_hs(
                        points[i], codes[i], codelens, neu1, syn0_vocab, syn0_ngrams, syn1, size,indexes,
                        subwords_idx,subwords_idx_len,_alpha, work, i, j, k, cbow_mean, word_locks_vocab,
                        word_locks_ngrams)
                if negative:
                    next_random = fast_sentence_cbow_neg(
                        negative, cum_table, cum_table_len, codelens, neu1, syn0_vocab, syn0_ngrams,
                        syn1neg, size, indexes, subwords_idx, subwords_idx_len, _alpha, work, i, j, k,
                        cbow_mean, next_random, word_locks_vocab, word_locks_ngrams)

    return effective_words


def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.  Also calculate log(sigmoid(x)) into LOG_TABLE.
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
MAX_WORDS_IN_BATCH = MAX_SENTENCE_LEN

