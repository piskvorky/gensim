# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8
#
# shared type definitions for word2vec_corpusfile
# also used from fasttext_corpusfile and doc2vec_corpusfile
#
# Copyright (C) 2018 Dmitry Persiyanov <dmitry.persiyanov@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool as bool_t

cimport numpy as np

ctypedef np.float32_t REAL_t


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

    # for FastText
    int subword_idx_len
    np.uint32_t *subword_idx


ctypedef unordered_map[string, VocabItem] cvocab_t

cdef class CythonVocab:
    cdef cvocab_t vocab
    cdef subword_arrays
    cdef cvocab_t* get_vocab_ptr(self) nogil except *


cdef REAL_t get_alpha(REAL_t alpha, REAL_t end_alpha, int cur_epoch, int num_epochs) nogil
cdef REAL_t get_next_alpha(REAL_t start_alpha, REAL_t end_alpha, long long total_examples, long long total_words,
                           long long expected_examples, long long expected_words, int cur_epoch, int num_epochs) nogil
