#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

import cython
import numpy as np
from numpy import zeros, float32 as REAL
cimport numpy as np
#from libc.math cimport log, exp
from libc.string cimport memset
from libc.stdlib cimport calloc, free

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
cdef float ONEF = <float>1.0
DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef float negative_sampling(const int target, const float lr, float *grad, float *wo, float *hidden,
                              const int vector_size, const int neg, int *negatives,
                              int *negpos, const int negatives_len)nogil:

    cdef float loss
    loss = <float> 0.0
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


cdef float sigmoid(const float val)nogil:
    if val < -MAX_EXP:
        return 0.0
    elif val > MAX_EXP:
        return 1.0
    else:
        return EXP_TABLE[<int>((val + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
    

cdef float log(const float val)nogil:
    if val > 1.0:
        return 0.0
    else:
        return LOG_TABLE[<int>(val * EXP_TABLE_SIZE)]


cdef float binary_logistic(const int target, const int label, const float lr,
                            const int vector_size, float *wo, float *grad, float *hidden)nogil:

    cdef float temp = our_dot(&vector_size, &wo[target], &ONE, hidden, &ONE)
    cdef float score = sigmoid(temp)
    cdef float alpha = lr * (<float>label - score)
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


def update(model, context_, target_, lr_, hidden_, grad_):

    cdef float loss
    cdef float lr = <float> lr_
    cdef int vector_size = <int> model.vector_size
    cdef int *context = <int *> np.PyArray_DATA(context_)
    cdef int target = <int> target_
    cdef int *negatives = <int *> np.PyArray_DATA(model.negatives)
    cdef int negpos = <int> model.negpos
    cdef int neg = <int> model.neg
    cdef REAL_t *wi = <REAL_t *> np.PyArray_DATA(model.wi)
    cdef REAL_t *wo = <REAL_t *> np.PyArray_DATA(model.wo)
    cdef int negatives_len = <int> len(model.negatives)
    cdef int context_len = <int> len(context_)
    #hidden_ = zeros(vector_size, dtype=REAL)
    #grad_ = zeros(vector_size, dtype=REAL)
    cdef REAL_t *hidden = <REAL_t *> np.PyArray_DATA(hidden_)
    cdef REAL_t *grad = <REAL_t *> np.PyArray_DATA(grad_)
    memset(hidden, 0, vector_size)
    memset(grad, 0, vector_size)
    #cdef REAL_t *hidden, *grad
    #hidden = <REAL_t *>calloc(vector_size, cython.sizeof(REAL_t))
    #grad = <REAL_t *>calloc(vector_size, cython.sizeof(REAL_t))

    cdef np.int32_t i
    cdef float alpha = ONEF / <float>(context_len)

    with nogil:
        for i from 0 <= i < context_len:
            our_saxpy(&vector_size, &ONEF, &wi[context[i]], &ONE, hidden, &ONE)
        sscal(&vector_size, &alpha, hidden, &ONE)
        loss = negative_sampling(target, lr, grad, wo, hidden, vector_size, neg, negatives, &negpos, negatives_len)
        sscal(&vector_size, &alpha, grad, &ONE)
        for i from 0 <= i < context_len:
            our_saxpy(&vector_size, &ONEF, grad, &ONE, &wi[context[i]], &ONE)

    model.negpos = negpos
    #free(hidden)
    #free(grad)
    return loss