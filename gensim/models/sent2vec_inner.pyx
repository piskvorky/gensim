#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

import cython
import numpy as np
from numpy import zeros, float32 as REAL
cimport numpy as np
from libc.math cimport log, exp
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

from word2vec import FAST_VERSION

cdef int ONE = 1
cdef float ONEF = <float>1.0


cdef float negative_sampling(const int target, const float lr, float *grad, float *wo, float *hidden,
                              const int vector_size, const int neg, int *negatives,
                              int *negpos, const int negatives_len)nogil:

    cdef float loss
    loss = <float> 0.0
    cdef int label_true, label_false
    label_true = <int> 1
    label_false = <int> 0
    cdef int temp
    for i from 0 <= i <= neg:
        if i == 0:
            loss += binary_logistic(target, label_true, lr, vector_size, wo, grad, hidden)
        else:
            temp = get_negative(target, negatives, negpos, negatives_len)
            loss += binary_logistic(temp, label_false, lr, vector_size, wo, grad, hidden)
    return loss


cdef float binary_logistic(const int target, const int label, const float lr,
                            const int vector_size, float *wo, float *grad, float *hidden)nogil:

    cdef float score = exp(our_dot(&vector_size, &wo[<int>target], &ONE, hidden, &ONE))
    cdef float alpha = lr * (<float>label - score)
    our_saxpy(&vector_size, &alpha, &wo[<int>target], &ONE, grad, &ONE)
    our_saxpy(&vector_size, &alpha, hidden, &ONE, &wo[<int>target], &ONE)
    if label is True:
        #return -LOG_TABLE[<int>score]
        return -log(score)
    else:
        #return -LOG_TABLE[<int>(ONEF - score)]
        return -log(ONEF - score)
    

cdef int get_negative(const int target, int *negatives,
                              int *negpos, const int negatives_len)nogil:

    cdef int negative
    while True:
        negative = negatives[<int>negpos[0]]
        negpos[0] = (negpos[0] + <int>ONE) % negatives_len
        if target != negative:
            break
    return negative


def update(model, context_, target_, lr_):

    cdef float loss = <float> model.loss
    cdef float lr = <float> lr_
    cdef int vector_size = <int> model.vector_size
    cdef int *context = <int *>np.PyArray_DATA(np.array(context_))
    cdef int target = <int> target_
    cdef int *negatives = <int *>np.PyArray_DATA(np.array(model.negatives))
    cdef int negpos = <int> model.negpos
    cdef int neg = <int> model.neg
    cdef float *wi = <float *>np.PyArray_DATA(model.wi)
    cdef float *wo = <float *>np.PyArray_DATA(model.wo)
    cdef int negatives_len = <int>len(model.negatives)
    cdef int context_len = <int>len(context_)

    hidden_ = zeros(vector_size, dtype=REAL)
    grad_ = zeros(vector_size, dtype=REAL)
    cdef float *hidden = <float *>np.PyArray_DATA(hidden_)
    cdef float *grad = <float *>np.PyArray_DATA(grad_)
    
    cdef np.int32_t i
    for i from 0 <= i < context_len:
        our_saxpy(&vector_size, &ONEF, &wi[i], &ONE, hidden, &ONE)
        
    cdef float alpha = ONEF / <float>(len(context_))
    sscal(&vector_size, &alpha, hidden, &ONE)
    loss += negative_sampling(target, lr, grad, wo, hidden, vector_size, neg, negatives, &negpos, negatives_len)
    model.nexamples += 1
    sscal(&vector_size, &alpha, grad, &ONE)
    for i from 0 <= i < context_len:
        our_saxpy(&vector_size, &ONEF, grad, &ONE, &wi[i], &ONE)