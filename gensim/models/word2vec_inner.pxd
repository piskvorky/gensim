# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8
#
# shared type definitions for word2vec_inner
# used by both word2vec_inner.pyx (automatically) and doc2vec_inner.pyx (by explicit cimport)
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool as bool_t

cimport numpy as np


cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

cdef extern from "fast_line_sentence.cpp":
    pass

cdef extern from "fast_line_sentence.h":
    cdef cppclass FastLineSentence:
        FastLineSentence() except +
        FastLineSentence(string&, size_t) except +
        vector[string] ReadSentence() nogil except +
        bool_t IsEof() nogil
        void Reset() nogil


cdef class CythonLineSentence:
    cdef FastLineSentence* _thisptr
    cdef public bytes source
    cdef public size_t max_sentence_length, max_words_in_batch, offset
    cdef vector[vector[string]] buf_data

    cpdef bool_t is_eof(self) nogil
    cpdef vector[string] read_sentence(self) nogil except *
    cpdef vector[vector[string]] _read_chunked_sentence(self) nogil except *
    cpdef vector[vector[string]] _chunk_sentence(self, vector[string] sent) nogil
    cpdef void reset(self) nogil
    cpdef vector[vector[string]] next_batch(self) nogil except *


cdef struct VocabItem:
    long long sample_int
    np.uint32_t index
    np.uint8_t *code
    int code_len
    np.uint32_t *point

ctypedef unordered_map[string, VocabItem] cvocab_t

cdef class CythonVocab:
    cdef cvocab_t vocab
    cdef cvocab_t* get_vocab_ptr(self) nogil except *


ctypedef np.float32_t REAL_t

# BLAS routine signatures
ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil

cdef scopy_ptr scopy
cdef saxpy_ptr saxpy
cdef sdot_ptr sdot
cdef dsdot_ptr dsdot
cdef snrm2_ptr snrm2
cdef sscal_ptr sscal

# precalculated sigmoid table
DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6
cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE

# function implementations swapped based on BLAS detected in word2vec_inner.pyx init()
ctypedef REAL_t (*our_dot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef void (*our_saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil

cdef our_dot_ptr our_dot
cdef our_saxpy_ptr our_saxpy

# for when fblas.sdot returns a double
cdef REAL_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil

# for when fblas.sdot returns a float
cdef REAL_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil

# for when no blas available
cdef REAL_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil

# to support random draws from negative-sampling cum_table
cdef unsigned long long bisect_left(np.uint32_t *a, unsigned long long x, unsigned long long lo, unsigned long long hi) nogil

cdef unsigned long long random_int32(unsigned long long *next_random) nogil
