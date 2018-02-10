#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
# distutils : language = c++
# distutils : extra_compile_args = -std=c++11

import cython
import numpy as np
from numpy import zeros, float32 as REAL
cimport numpy as np
from libcpp.vector cimport vector

# scipy <= 0.15
try:
     from scipy.linalg.blas import fblas
except ImportError:
     # in scipy > 0.15, fblas function has been removed
     import scipy.linalg.blas as fblas

from word2vec_inner cimport bisect_left, random_int32, \
     scopy, saxpy, sdot, dsdot, snrm2, sscal, \
     REAL_t, EXP_TABLE, \
     LOG_TABLE, our_dot, our_saxpy, \
     our_dot_double, our_dot_float, our_dot_noblas, our_saxpy_noblas

from word2vec import FAST_VERSION

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t negative_sampling(const int target, const REAL_t lr, REAL_t *grad, REAL_t *wo,
                              REAL_t *hidden, const int vector_size, const int neg, int *negatives,
                              int *negpos, const int negatives_len)nogil:

    cdef REAL_t loss = <REAL_t> 0.0
    cdef int label_true, label_false
    label_true = <int> 1
    label_false = <int> 0
    cdef int temp, new_negpos
    for i from 0 <= i <= neg:
        if i == 0:
            loss += binary_logistic(target, label_true, lr, vector_size, wo, grad, hidden)
        else:
            temp = get_negative(target, negatives, negpos, negatives_len)
            loss += binary_logistic(temp, label_false, lr, vector_size, wo, grad, hidden)
    return loss


cdef REAL_t sigmoid(const REAL_t val)nogil:

    if val < -MAX_EXP:
        return 0.0
    elif val > MAX_EXP:
        return 1.0
    else:
        return EXP_TABLE[<int>((val + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
    

cdef REAL_t log(const REAL_t val)nogil:

    if val > 1.0:
        return 0.0
    else:
        return LOG_TABLE[<int>(val * EXP_TABLE_SIZE)]


cdef REAL_t binary_logistic(const int target, const int label, const REAL_t lr,
                            const int vector_size, REAL_t *wo, REAL_t *grad,
                            REAL_t *hidden)nogil:

    cdef REAL_t temp = our_dot(&vector_size, &wo[target], &ONE, hidden, &ONE)
    cdef REAL_t score = sigmoid(temp)
    cdef REAL_t alpha = lr * (<REAL_t>label - score)
    our_saxpy(&vector_size, &alpha, &wo[target], &ONE, grad, &ONE)
    our_saxpy(&vector_size, &alpha, hidden, &ONE, &wo[target], &ONE)
    if label == 1:
        return -log(score)
    else:
        return -log(ONEF - score)
    

cdef int get_negative(const int target, int *negatives,
                              int *negpos, const int negatives_len)nogil:

    cdef int negative
    while True:
        negative = negatives[negpos[0]]
        negpos[0] = (negpos[0] + ONE) % negatives_len
        if target != negative:
            break
    return negative


cdef REAL_t update(vector[int] &context, int target, REAL_t lr, REAL_t *hidden, REAL_t *grad,
                  int vector_size, int *negpos, int neg, int negatives_len,
                  REAL_t *wi, REAL_t *wo, int *negatives)nogil:

    cdef REAL_t alpha = ONEF / <REAL_t>(context.size())
    cdef REAL_t loss
    cdef int i

    for i from 0 <= i < context.size():
        our_saxpy(&vector_size, &ONEF, &wi[context[i]], &ONE, hidden, &ONE)
    sscal(&vector_size, &alpha, hidden, &ONE)
    loss = negative_sampling(target, lr, grad, wo, hidden, vector_size, neg, negatives, negpos, negatives_len)
    sscal(&vector_size, &alpha, grad, &ONE)
    for i from 0 <= i < context.size():
        our_saxpy(&vector_size, &ONEF, grad, &ONE, &wi[context[i]], &ONE)
    return loss


cdef extern from "<random>" namespace "std":

    cdef cppclass minstd_rand nogil:
        minstd_rand()
        minstd_rand(int seed)

    cdef cppclass uniform_real_distribution[T] nogil:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(minstd_rand gen)

    cdef cppclass uniform_int_distribution[T] nogil:
        uniform_int_distribution()
        uniform_int_distribution(int, int)
        T operator()(minstd_rand gen)


cdef int get_line(vector[int] &wids, vector[int] &words, int max_line_size)nogil:

    cdef int ntokens = <int> 0
    cdef int i

    for i from 0 <= i < wids.size():
        if wids[i] < 0:
            continue
        ntokens += 1
        words.push_back(wids[i])
        if ntokens > max_line_size:
            break
    return ntokens


cdef void add_ngrams_train(vector[int] &line, int n, int k, int bucket, int size, minstd_rand gen)nogil:

    cdef int num_discarded = 0
    cdef vector[int] discard
    cdef int line_size = line.size()
    cdef int token_to_discard
    cdef int i, j, h
    cdef uniform_int_distribution[int] dist = uniform_int_distribution[int](0, line_size - 1)

    for i from 0<= i < line.size():
        discard.push_back(0)

    while num_discarded < k and line_size - num_discarded > 2:
        token_to_discard = dist(gen)
        if discard[token_to_discard] == 0:
            discard[token_to_discard] = 1
            num_discarded += 1

    for i from 0 <= i < line_size:
        if discard[i] == 1:
            continue
        h = line[i]
        for j from i + 1 <= j < line_size:
            if j >= i + n or discard[j] == 1:
                break
            h = ((h * (116049371 % bucket)) % bucket + line[j]) % bucket
            line.push_back(size + h)


cdef (int, int, REAL_t) _do_train_job_util(vector[vector[int]] &word_ids, REAL_t *pdiscard, int max_line_size,
                             int word_ngrams, int dropout_k, REAL_t lr, REAL_t *hidden, REAL_t *grad,
                             int vector_size, int *negpos, int neg, int negatives_len,
                             REAL_t *wi, REAL_t *wo, int *negatives, int bucket, int size, minstd_rand gen,
                             uniform_real_distribution[REAL_t] dist)nogil:

    cdef int local_token_count = 0
    cdef int nexamples = 0
    cdef REAL_t loss = <REAL_t> 0.0
    cdef vector[int] words, context
    cdef int i, j, ntokens_temp

    for i from 0 <= i < word_ids.size():
        ntokens_temp = get_line(word_ids[i], words, max_line_size)
        local_token_count += ntokens_temp
        words_size = words.size()
        if words_size > 0:
            for j from 0 <= j < words_size:
                if dist(gen) > pdiscard[words[j]]:
                    continue
                nexamples += 1
                context.assign(words.begin(), words.end())
                context[j] = 0
                add_ngrams_train(context, word_ngrams, dropout_k, bucket, size, gen)
                loss += update(context, words[j], lr, hidden, grad, vector_size,
                               negpos, neg, negatives_len, wi, wo, negatives)
                context.clear()
        words.clear()

    return local_token_count, nexamples, loss


def _do_train_job_fast(model, sentences_, lr_, hidden_, grad_):

    cdef REAL_t lr = <REAL_t> lr_
    cdef int vector_size = <int> model.vector_size
    cdef int *negatives = <int *> np.PyArray_DATA(model.negatives)
    cdef int negpos = <int> model.negpos
    cdef int neg = <int> model.neg
    cdef REAL_t *wi = <REAL_t *> np.PyArray_DATA(model.wi)
    cdef REAL_t *wo = <REAL_t *> np.PyArray_DATA(model.wo)
    cdef int negatives_len = <int> len(model.negatives)
    cdef REAL_t *hidden = <REAL_t *> np.PyArray_DATA(hidden_)
    cdef REAL_t *grad = <REAL_t *> np.PyArray_DATA(grad_)
    cdef REAL_t *pdiscard = <REAL_t *> np.PyArray_DATA(np.array(model.dict.pdiscard))
    cdef int max_line_size = <int> (model.dict.max_line_size)
    cdef int size = <int> (model.dict.size)
    cdef int bucket = <int> (model.dict.bucket)
    cdef int word_ngrams = <int> (model.word_ngrams)
    cdef int dropout_k = <int> (model.dropout_k)
    cdef minstd_rand gen = minstd_rand(model.seed)
    cdef uniform_real_distribution[REAL_t] dist = uniform_real_distribution[REAL_t](0.0, 0.1)

    cdef vector[vector[int]] word_ids
    cdef vector[int] ids
    cdef int i, local_token_count, nexamples
    cdef REAL_t loss

    for sentence in sentences_:
        for word in sentence:
            h = model.dict.find(word)
            ids.push_back(<int> model.dict.word2int[h])
        word_ids.push_back(ids)
        ids.clear()

    with nogil:
        local_token_count, nexamples, loss = _do_train_job_util(word_ids, pdiscard,
                                                                max_line_size, word_ngrams,
                                                                dropout_k, lr, hidden,
                                                                grad, vector_size, &negpos,
                                                                neg, negatives_len, wi, wo,
                                                                negatives, bucket, size, gen, dist)
    model.negpos = negpos
    return local_token_count, nexamples, loss
