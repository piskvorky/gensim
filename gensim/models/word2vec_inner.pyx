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
from libc.math cimport log
from libc.string cimport memset

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
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, REAL_t *neu1,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, const REAL_t doc2vecC, int idx_start, int idx_end, REAL_t *_running_training_loss_param) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot
    cdef np.uint32_t target_index
    cdef int d

    cdef unsigned long long sen_sample[MAX_SENTENCE_LEN]
    cdef int already_sampled
    cdef REAL_t corruption_constant =  (1.0 / doc2vecC / (idx_end - idx_start) if doc2vecC else 0.0)    # w in doc2vecC
    cdef REAL_t inv_count = 0.5

    memset(work, 0, size * cython.sizeof(REAL_t))
    memset(neu1, 0, size * cython.sizeof(REAL_t))

    our_saxpy(&size, &ONEF, &syn0[row1], &ONE, neu1, &ONE)

    # handle the sentence using Doc2VecC
    if doc2vecC:
        memset(sen_sample, -1, size * cython.sizeof(REAL_t))
        already_sampled = 0

        for t in range(idx_start,idx_end):
            # randomly sample rp_sample ((0, 1]) of the words in the sentence to represent one sentence
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo

            if (next_random & 0xFFFF) / <REAL_t>65536 > doc2vecC:
                continue

            sen_sample[already_sampled] = t
            already_sampled += 1
            if (already_sampled >= MAX_SENTENCE_LEN):
                break

            our_saxpy(&size, &corruption_constant, &syn0[indexes[t] * size], &ONE, neu1, &ONE)

            sscal(&size, &inv_count, neu1, &ONE) #In original: neu1 /= 2

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

    our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn0[row1], &ONE)

    if doc2vecC:
        # doc2vecC script error is also divided by 2: syn0[l1] += neu1e * w/2;
        sscal(&size, &inv_count, work, &ONE)
        sscal(&size, &corruption_constant, work, &ONE)

        for m in range(already_sampled):
            our_saxpy(&size, &word_locks[indexes[sen_sample[m]]], work, &ONE, &syn0[indexes[sen_sample[m]]*size], &ONE)

        return next_random


cdef void fast_sentence_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param) nogil:

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
    const int _compute_loss, const REAL_t doc2vecC, int idx_start, int idx_end, REAL_t *_running_training_loss_param) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count = 1.0, label, log_e_f_dot, f_dot
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    cdef unsigned long long sen_sample[MAX_SENTENCE_LEN]
    cdef int already_sampled
    cdef REAL_t corruption_constant =  1.0 / doc2vecC / (idx_end - idx_start)     # w in doc2vecC

    word_index = indexes[i]

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)

    # handle the sentence using Doc2VecC
    if doc2vecC:
        memset(sen_sample, -1, size * cython.sizeof(REAL_t))
        already_sampled = 0
        for t in range(idx_start,idx_end):
            # randomly sample rp_sample ((0, 1]) of the words in the sentence to represent one sentence
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo

            if (next_random & 0xFFFF) / <REAL_t>65536 > doc2vecC:
                continue

            sen_sample[already_sampled] = t
            already_sampled += 1
            if (already_sampled >= MAX_SENTENCE_LEN):
                break

            our_saxpy(&size, &corruption_constant, &syn0[indexes[t] * size], &ONE, neu1, &ONE)
        count += ONEF

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

    if doc2vecC:
        # doc2vecC script error is also divided by wordCount: syn0[c + l1] += neu1e[c] * w / cw;
        sscal(&size, &corruption_constant, work, &ONE) # neu1e *= w

        for m in range(already_sampled):
            our_saxpy(&size, &word_locks[indexes[sen_sample[m]]], work, &ONE, &syn0[indexes[sen_sample[m]]*size], &ONE)

    return next_random


def train_batch_sg(model, sentences, alpha, _work, compute_loss, _doc2vecC=None):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)

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

    # for doc2vecC Corruption
    cdef REAL_t doc2vecC = _doc2vecC
    cdef REAL_t *neu1

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
    neu1 = <REAL_t *>np.PyArray_DATA(_work)

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
                        next_random = fast_sentence_sg_neg(negative, cum_table, cum_table_len, neu1, syn0,
                        syn1neg, size, indexes[i], indexes[j], indexes, _alpha, work, next_random,
                        word_locks, _compute_loss, doc2vecC, idx_start, idx_end, &_running_training_loss)

    model.running_training_loss = _running_training_loss
    return effective_words


def train_batch_cbow(model, sentences, alpha, _work, _neu1, compute_loss, _doc2vecC=None):
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

    # for doc2vecC Corruption
    cdef REAL_t doc2vecC = _doc2vecC

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
                    next_random = fast_sentence_cbow_neg(negative, cum_table, cum_table_len, codelens, neu1,
                    syn0, syn1neg, size, indexes, _alpha, work, i, j, k, cbow_mean, next_random,
                     word_locks, _compute_loss, doc2vecC, idx_start, idx_end, &_running_training_loss)

    model.running_training_loss = _running_training_loss
    return effective_words


# Score is only implemented for hierarchical softmax
def score_sentence_sg(model, sentence, _work):

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
