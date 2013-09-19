#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp

REAL = np.float32
ctypedef np.float32_t REAL_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void fast_sentence(np.uint32_t[::1] word_point, np.uint8_t[::1] word_code, np.uint32_t word_index,
    REAL_t[:, ::1] _syn0, REAL_t[:, ::1] _syn1, REAL_t alpha, unsigned int size, REAL_t[::1] _work, unsigned long int codelen):
    cdef long long a, b
    cdef long long row1 = word_index * size, row2
    cdef REAL_t f, g
    cdef REAL_t *syn0 = &_syn0[0, 0]
    cdef REAL_t *syn1 = &_syn1[0, 0]
    cdef REAL_t *work = &_work[0]

    for a in range(size):
        work[a] = <REAL_t>0.0
    for b in range(codelen):
        row2 = word_point[b] * size
        f = <REAL_t>0.0
        for a in range(size):
            f += syn0[row1 + a] * syn1[row2 + a]
        g = (1 - word_code[b] - <REAL_t>1.0 / (<REAL_t>1.0 + exp(-f))) * alpha
        for a in range(size):
            work[a] += g * syn1[row2 + a]
        for a in range(size):
            syn1[row2 + a] += g * syn0[row1 + a]
    for a in range(size):
        syn0[row1 + a] += work[a]


def train_sentence(model, sentence, alpha):
    """
    Update skip-gram hierarchical softmax model by training on a single sentence,
    where `sentence` is a list of Vocab objects (or None, where the corresponding
    word is not in the vocabulary). Called internally from `train_model())`.

    """
    cdef int pos, pos2
    work = np.empty(model.layer1_size, dtype=REAL)
    for pos, word in enumerate(sentence):
        if word is None:
            continue  # OOV word in the input sentence => skip
        reduced_window = np.random.randint(model.window)  # `b` in the original word2vec code

        # now go over all words from the (reduced) window, predicting each one in turn
        start = max(0, pos - model.window + reduced_window)
        for pos2, word2 in enumerate(sentence[start : pos + model.window + 1 - reduced_window], start):
            if pos2 == pos or word2 is None:
                # don't train on OOV words and on the `word` itself
                continue
            fast_sentence(word.point, word.code, word2.index, model.syn0, model.syn1, alpha, model.layer1_size, work, len(word.point))
