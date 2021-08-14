#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Optimized cython functions for training :class:`~gensim.models.doc2vec.Doc2Vec` model."""

import cython
import numpy as np
from numpy import zeros, float32 as REAL
cimport numpy as np

from libc.string cimport memset, memcpy

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

from word2vec_inner cimport bisect_left, random_int32, sscal, REAL_t, EXP_TABLE, our_dot, our_saxpy

DEF MAX_DOCUMENT_LEN = 10000

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef void fast_document_dbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *context_vectors, REAL_t *syn1, const int size,
    const np.uint32_t context_index, const REAL_t alpha, REAL_t *work, int learn_context, int learn_hidden,
    REAL_t *contexts_lockf, const np.uint32_t contexts_lockf_len) nogil:

    cdef long long a, b
    cdef long long row1 = context_index * size, row2
    cdef REAL_t f, g

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f = our_dot(&size, &context_vectors[row1], &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&size, &g, &context_vectors[row1], &ONE, &syn1[row2], &ONE)
    if learn_context:
        our_saxpy(&size, &contexts_lockf[context_index % contexts_lockf_len],
                  work, &ONE, &context_vectors[row1], &ONE)


cdef unsigned long long fast_document_dbow_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *context_vectors, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t context_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, int learn_context, int learn_hidden, REAL_t *contexts_lockf,
    const np.uint32_t contexts_lockf_len) nogil:

    cdef long long a
    cdef long long row1 = context_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

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
        f = our_dot(&size, &context_vectors[row1], &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&size, &g, &context_vectors[row1], &ONE, &syn1neg[row2], &ONE)
    if learn_context:
        our_saxpy(&size, &contexts_lockf[context_index % contexts_lockf_len],
                  work, &ONE, &context_vectors[row1], &ONE)

    return next_random


cdef void fast_document_dm_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int word_code_len,
    REAL_t *neu1, REAL_t *syn1, const REAL_t alpha, REAL_t *work,
    const int size, int learn_hidden) nogil:

    cdef long long b
    cdef long long row2
    cdef REAL_t f, g

    # l1 already composed by caller, passed in as neu1
    # work (also passed in)  will accumulate l1 error
    for b in range(word_code_len):
        row2 = word_point[b] * size
        f = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)


cdef unsigned long long fast_document_dm_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, unsigned long long next_random,
    REAL_t *neu1, REAL_t *syn1neg, const int predict_word_index, const REAL_t alpha, REAL_t *work,
    const int size, int learn_hidden) nogil:

    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    # l1 already composed by caller, passed in as neu1
    # work (also passsed in) will accumulate l1 error for outside application
    for d in range(negative+1):
        if d == 0:
            target_index = predict_word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == predict_word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    return next_random

cdef void fast_document_dmc_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int word_code_len,
    REAL_t *neu1, REAL_t *syn1, const REAL_t alpha, REAL_t *work,
    const int layer1_size, const int vector_size, int learn_hidden) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g
    cdef int m

    # l1 already composed by caller, passed in as neu1
    # work accumulates net l1 error; eventually applied by caller
    for b in range(word_code_len):
        row2 = word_point[b] * layer1_size
        f = our_dot(&layer1_size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        our_saxpy(&layer1_size, &g, &syn1[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&layer1_size, &g, neu1, &ONE, &syn1[row2], &ONE)


cdef unsigned long long fast_document_dmc_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, unsigned long long next_random,
    REAL_t *neu1, REAL_t *syn1neg, const int predict_word_index, const REAL_t alpha, REAL_t *work,
    const int layer1_size, const int vector_size, int learn_hidden) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d, m

    # l1 already composed by caller, passed in as neu1
    # work accumulates net l1 error; eventually applied by caller
    for d in range(negative+1):
        if d == 0:
            target_index = predict_word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == predict_word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * layer1_size
        f = our_dot(&layer1_size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&layer1_size, &g, &syn1neg[row2], &ONE, work, &ONE)
        if learn_hidden:
            our_saxpy(&layer1_size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    return next_random


cdef init_d2v_config(Doc2VecConfig *c, model, alpha, learn_doctags, learn_words, learn_hidden,
                     train_words=False, work=None, neu1=None, word_vectors=None, words_lockf=None,
                     doctag_vectors=None, doctags_lockf=None, docvecs_count=0):
    c[0].hs = model.hs
    c[0].negative = model.negative
    c[0].sample = (model.sample != 0)
    c[0].cbow_mean = model.cbow_mean
    c[0].train_words = train_words
    c[0].learn_doctags = learn_doctags
    c[0].learn_words = learn_words
    c[0].learn_hidden = learn_hidden
    c[0].alpha = alpha
    c[0].layer1_size = model.layer1_size
    c[0].vector_size = model.dv.vector_size
    c[0].workers = model.workers
    c[0].docvecs_count = docvecs_count

    c[0].window = model.window
    c[0].expected_doctag_len = model.dm_tag_count

    if '\0' in model.wv:
        c[0].null_word_index = model.wv.get_index('\0')

    # default vectors, locks from syn0/doctag_syn0
    if word_vectors is None:
       word_vectors = model.wv.vectors
    c[0].word_vectors = <REAL_t *>(np.PyArray_DATA(word_vectors))
    if doctag_vectors is None:
       doctag_vectors = model.dv.vectors
    c[0].doctag_vectors = <REAL_t *>(np.PyArray_DATA(doctag_vectors))
    if words_lockf is None:
       words_lockf = model.wv.vectors_lockf
    c[0].words_lockf = <REAL_t *>(np.PyArray_DATA(words_lockf))
    c[0].words_lockf_len = len(words_lockf)
    if doctags_lockf is None:
       doctags_lockf = model.dv.vectors_lockf
    c[0].doctags_lockf = <REAL_t *>(np.PyArray_DATA(doctags_lockf))
    c[0].doctags_lockf_len = len(doctags_lockf)

    if c[0].hs:
        c[0].syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if c[0].negative:
        c[0].syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        c[0].cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
        c[0].cum_table_len = len(model.cum_table)
    if c[0].negative or c[0].sample:
        c[0].next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    if work is None:
       work = zeros(model.layer1_size, dtype=REAL)
    c[0].work = <REAL_t *>np.PyArray_DATA(work)
    if neu1 is None:
       neu1 = zeros(model.layer1_size, dtype=REAL)
    c[0].neu1 = <REAL_t *>np.PyArray_DATA(neu1)



def train_document_dbow(model, doc_words, doctag_indexes, alpha, work=None,
                        train_words=False, learn_doctags=True, learn_words=True, learn_hidden=True,
                        word_vectors=None, words_lockf=None, doctag_vectors=None, doctags_lockf=None):
    """Update distributed bag of words model ("PV-DBOW") by training on a single document.

    Called internally from :meth:`~gensim.models.doc2vec.Doc2Vec.train` and
    :meth:`~gensim.models.doc2vec.Doc2Vec.infer_vector`.

    Parameters
    ----------
    model : :class:`~gensim.models.doc2vec.Doc2Vec`
        The model to train.
    doc_words : list of str
        The input document as a list of words to be used for training. Each word will be looked up in
        the model's vocabulary.
    doctag_indexes : list of int
        Indices into `doctag_vectors` used to obtain the tags of the document.
    alpha : float
        Learning rate.
    work : list of float, optional
        Updates to be performed on each neuron in the hidden layer of the underlying network.
    train_words : bool, optional
        Word vectors will be updated exactly as per Word2Vec skip-gram training only if **both** `learn_words`
        and `train_words` are set to True.
    learn_doctags : bool, optional
        Whether the tag vectors should be updated.
    learn_words : bool, optional
        Word vectors will be updated exactly as per Word2Vec skip-gram training only if **both**
        `learn_words` and `train_words` are set to True.
    learn_hidden : bool, optional
        Whether or not the weights of the hidden layer will be updated.
    word_vectors : numpy.ndarray, optional
        The vector representation for each word in the vocabulary. If None, these will be retrieved from the model.
    words_lockf : numpy.ndarray, optional
        EXPERIMENTAL. A learning lock factor for each word-vector; value 0.0 completely blocks updates, a value
        of 1.0 allows normal updates to word-vectors.
    doctag_vectors : numpy.ndarray, optional
        Vector representations of the tags. If None, these will be retrieved from the model.
    doctags_lockf : numpy.ndarray, optional
        EXPERIMENTAL. The lock factors for each tag, same as `words_lockf`, but for document-vectors.

    Returns
    -------
    int
        Number of words in the input document that were actually used for training.

    """
    cdef Doc2VecConfig c

    cdef int i, j
    cdef long result = 0
    cdef np.uint32_t *vocab_sample_ints

    init_d2v_config(&c, model, alpha, learn_doctags, learn_words, learn_hidden, train_words=train_words, work=work,
                    neu1=None, word_vectors=word_vectors, words_lockf=words_lockf,
                    doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
    c.doctag_len = <int>min(MAX_DOCUMENT_LEN, len(doctag_indexes))
    if c.sample:
        vocab_sample_ints = <np.uint32_t *>np.PyArray_DATA(model.wv.expandos['sample_int'])
    if c.hs:
        vocab_codes = model.wv.expandos['code']
        vocab_points = model.wv.expandos['point']

    i = 0
    for token in doc_words:
        word_index = model.wv.key_to_index.get(token, None)
        if word_index is None:  # shrink document to leave out word
            continue  # leaving i unchanged
        if c.sample and vocab_sample_ints[word_index] < random_int32(&c.next_random):
            continue
        c.indexes[i] = word_index
        if c.hs:
            c.codelens[i] = <int>len(vocab_codes[word_index])
            c.codes[i] = <np.uint8_t *>np.PyArray_DATA(vocab_codes[word_index])
            c.points[i] = <np.uint32_t *>np.PyArray_DATA(vocab_points[word_index])
        result += 1
        i += 1
        if i == MAX_DOCUMENT_LEN:
            break  # TODO: log warning, tally overflow?
    c.document_len = i

    if c.train_words:
        # single randint() call avoids a big thread-synchronization slowdown
        if model.shrink_windows:
            for i, item in enumerate(model.random.randint(0, c.window, c.document_len)):
                c.reduced_windows[i] = item
        else:
            for i in range(c.document_len):
                c.reduced_windows[i] = 0

    for i in range(c.doctag_len):
        c.doctag_indexes[i] = doctag_indexes[i]
        result += 1

    # release GIL & train on the document
    with nogil:
        for i in range(c.document_len):
            if c.train_words:  # simultaneous skip-gram wordvec-training
                j = i - c.window + c.reduced_windows[i]
                if j < 0:
                    j = 0
                k = i + c.window + 1 - c.reduced_windows[i]
                if k > c.document_len:
                    k = c.document_len
                for j in range(j, k):
                    if j == i:
                        continue
                    if c.hs:
                        # we reuse the DBOW function, as it is equivalent to skip-gram for this purpose
                        fast_document_dbow_hs(c.points[i], c.codes[i], c.codelens[i], c.word_vectors, c.syn1, c.layer1_size,
                                              c.indexes[j], c.alpha, c.work, c.learn_words, c.learn_hidden, c.words_lockf,
                                              c.words_lockf_len)
                    if c.negative:
                        # we reuse the DBOW function, as it is equivalent to skip-gram for this purpose
                        c.next_random = fast_document_dbow_neg(c.negative, c.cum_table, c.cum_table_len, c.word_vectors,
                                                               c.syn1neg, c.layer1_size, c.indexes[i], c.indexes[j],
                                                               c.alpha, c.work, c.next_random, c.learn_words,
                                                               c.learn_hidden, c.words_lockf, c.words_lockf_len)

            # docvec-training
            for j in range(c.doctag_len):
                if c.hs:
                    fast_document_dbow_hs(c.points[i], c.codes[i], c.codelens[i], c.doctag_vectors, c.syn1, c.layer1_size,
                                          c.doctag_indexes[j], c.alpha, c.work, c.learn_doctags, c.learn_hidden, c.doctags_lockf,
                                          c.doctags_lockf_len)
                if c.negative:
                    c.next_random = fast_document_dbow_neg(c.negative, c.cum_table, c.cum_table_len, c.doctag_vectors,
                                                           c.syn1neg, c.layer1_size, c.indexes[i], c.doctag_indexes[j],
                                                           c.alpha, c.work, c.next_random, c.learn_doctags,
                                                           c.learn_hidden, c.doctags_lockf, c.doctags_lockf_len)

    return result


def train_document_dm(model, doc_words, doctag_indexes, alpha, work=None, neu1=None,
                      learn_doctags=True, learn_words=True, learn_hidden=True,
                      word_vectors=None, words_lockf=None, doctag_vectors=None, doctags_lockf=None):
    """Update distributed memory model ("PV-DM") by training on a single document.
    This method implements the DM model with a projection (input) layer that is either the sum or mean of the context
    vectors, depending on the model's `dm_mean` configuration field.

    Called internally from :meth:`~gensim.models.doc2vec.Doc2Vec.train` and
    :meth:`~gensim.models.doc2vec.Doc2Vec.infer_vector`.

    Parameters
    ----------
    model : :class:`~gensim.models.doc2vec.Doc2Vec`
        The model to train.
    doc_words : list of str
        The input document as a list of words to be used for training. Each word will be looked up in
        the model's vocabulary.
    doctag_indexes : list of int
        Indices into `doctag_vectors` used to obtain the tags of the document.
    alpha : float
        Learning rate.
    work : np.ndarray, optional
        Private working memory for each worker.
    neu1 : np.ndarray, optional
        Private working memory for each worker.
    learn_doctags : bool, optional
        Whether the tag vectors should be updated.
    learn_words : bool, optional
        Word vectors will be updated exactly as per Word2Vec skip-gram training only if **both**
        `learn_words` and `train_words` are set to True.
    learn_hidden : bool, optional
        Whether or not the weights of the hidden layer will be updated.
    word_vectors : numpy.ndarray, optional
        The vector representation for each word in the vocabulary. If None, these will be retrieved from the model.
    words_lockf : numpy.ndarray, optional
        EXPERIMENTAL. A learning lock factor for each word-vector; value 0.0 completely blocks updates, a value
        of 1.0 allows normal updates to word-vectors.
    doctag_vectors : numpy.ndarray, optional
        Vector representations of the tags. If None, these will be retrieved from the model.
    doctags_lockf : numpy.ndarray, optional
        EXPERIMENTAL. The lock factors for each tag, same as `words_lockf`, but for document-vectors.

    Returns
    -------
    int
        Number of words in the input document that were actually used for training.

    """
    cdef Doc2VecConfig c

    cdef REAL_t count, inv_count = 1.0
    cdef int i, j, k, m
    cdef long result = 0
    cdef np.uint32_t *vocab_sample_ints

    init_d2v_config(&c, model, alpha, learn_doctags, learn_words, learn_hidden, train_words=False,
                    work=work, neu1=neu1, word_vectors=word_vectors, words_lockf=words_lockf,
                    doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
    c.doctag_len = <int>min(MAX_DOCUMENT_LEN, len(doctag_indexes))
    if c.sample:
        vocab_sample_ints = <np.uint32_t *>np.PyArray_DATA(model.wv.expandos['sample_int'])
#        vocab_sample_ints = model.wv.expandos['sample_int']  # this variant noticeably slower
    if c.hs:
        vocab_codes = model.wv.expandos['code']
        vocab_points = model.wv.expandos['point']

    i = 0
    for token in doc_words:
        word_index = model.wv.key_to_index.get(token, None)
        if word_index is None:  # shrink document to leave out word
            continue  # leaving i unchanged
        if c.sample and vocab_sample_ints[word_index] < random_int32(&c.next_random):
            continue
        c.indexes[i] = word_index
        if c.hs:
            c.codelens[i] = <int>len(vocab_codes[word_index])
            c.codes[i] = <np.uint8_t *>np.PyArray_DATA(vocab_codes[word_index])
            c.points[i] = <np.uint32_t *>np.PyArray_DATA(vocab_points[word_index])
        result += 1
        i += 1
        if i == MAX_DOCUMENT_LEN:
            break  # TODO: log warning, tally overflow?
    c.document_len = i

    # single randint() call avoids a big thread-sync slowdown
    if model.shrink_windows:
        for i, item in enumerate(model.random.randint(0, c.window, c.document_len)):
            c.reduced_windows[i] = item
    else:
        for i in range(c.document_len):
            c.reduced_windows[i] = 0

    for i in range(c.doctag_len):
        c.doctag_indexes[i] = doctag_indexes[i]
        result += 1

    # release GIL & train on the document
    with nogil:
        for i in range(c.document_len):
            j = i - c.window + c.reduced_windows[i]
            if j < 0:
                j = 0
            k = i + c.window + 1 - c.reduced_windows[i]
            if k > c.document_len:
                k = c.document_len

            # compose l1 (in _neu1) & clear _work
            memset(c.neu1, 0, c.layer1_size * cython.sizeof(REAL_t))
            count = <REAL_t>0.0
            for m in range(j, k):
                if m == i:
                    continue
                else:
                    count += ONEF
                    our_saxpy(&c.layer1_size, &ONEF, &c.word_vectors[c.indexes[m] * c.layer1_size], &ONE, c.neu1, &ONE)
            for m in range(c.doctag_len):
                count += ONEF
                our_saxpy(&c.layer1_size, &ONEF, &c.doctag_vectors[c.doctag_indexes[m] * c.layer1_size], &ONE, c.neu1, &ONE)
            if count > (<REAL_t>0.5):
                inv_count = ONEF/count
            if c.cbow_mean:
                sscal(&c.layer1_size, &inv_count, c.neu1, &ONE)  # (does this need BLAS-variants like saxpy?)
            memset(c.work, 0, c.layer1_size * cython.sizeof(REAL_t))  # work to accumulate l1 error
            if c.hs:
                fast_document_dm_hs(c.points[i], c.codes[i], c.codelens[i], c.neu1, c.syn1, c.alpha, c.work,
                                    c.layer1_size, c.learn_hidden)
            if c.negative:
                c.next_random = fast_document_dm_neg(c.negative, c.cum_table, c.cum_table_len, c.next_random,
                                                     c.neu1, c.syn1neg, c.indexes[i], c.alpha, c.work, c.layer1_size,
                                                     c.learn_hidden)

            if not c.cbow_mean:
                sscal(&c.layer1_size, &inv_count, c.work, &ONE)  # (does this need BLAS-variants like saxpy?)
            # apply accumulated error in work
            if c.learn_doctags:
                for m in range(c.doctag_len):
                    our_saxpy(&c.layer1_size, &c.doctags_lockf[c.doctag_indexes[m] % c.doctags_lockf_len], c.work,
                              &ONE, &c.doctag_vectors[c.doctag_indexes[m] * c.layer1_size], &ONE)
            if c.learn_words:
                for m in range(j, k):
                    if m == i:
                        continue
                    else:
                         our_saxpy(&c.layer1_size, &c.words_lockf[c.indexes[m] % c.doctags_lockf_len], c.work, &ONE,
                                   &c.word_vectors[c.indexes[m] * c.layer1_size], &ONE)

    return result


def train_document_dm_concat(model, doc_words, doctag_indexes, alpha, work=None, neu1=None,
                             learn_doctags=True, learn_words=True, learn_hidden=True,
                             word_vectors=None, words_lockf=None, doctag_vectors=None, doctags_lockf=None):
    """Update distributed memory model ("PV-DM") by training on a single document, using a concatenation of the
     context window word vectors (rather than a sum or average).
     This will be slower since the input at each batch will be significantly larger.

    Called internally from :meth:`~gensim.models.doc2vec.Doc2Vec.train` and
    :meth:`~gensim.models.doc2vec.Doc2Vec.infer_vector`.

    Parameters
    ----------
    model : :class:`~gensim.models.doc2vec.Doc2Vec`
        The model to train.
    doc_words : list of str
        The input document as a list of words to be used for training. Each word will be looked up in
        the model's vocabulary.
    doctag_indexes : list of int
        Indices into `doctag_vectors` used to obtain the tags of the document.
    alpha : float, optional
        Learning rate.
    work : np.ndarray, optional
        Private working memory for each worker.
    neu1 : np.ndarray, optional
        Private working memory for each worker.
    learn_doctags : bool, optional
        Whether the tag vectors should be updated.
    learn_words : bool, optional
        Word vectors will be updated exactly as per Word2Vec skip-gram training only if **both**
        `learn_words` and `train_words` are set to True.
    learn_hidden : bool, optional
        Whether or not the weights of the hidden layer will be updated.
    word_vectors : numpy.ndarray, optional
        The vector representation for each word in the vocabulary. If None, these will be retrieved from the model.
    words_lockf : numpy.ndarray, optional
        EXPERIMENTAL. A learning lock factor for each word-vector, value 0.0 completely blocks updates, a value
        of 1.0 allows normal updates to word-vectors.
    doctag_vectors : numpy.ndarray, optional
        Vector representations of the tags. If None, these will be retrieved from the model.
    doctags_lockf : numpy.ndarray, optional
        EXPERIMENTAL. The lock factors for each tag, same as `words_lockf`, but for document-vectors.

    Returns
    -------
    int
        Number of words in the input document that were actually used for training.

    """
    cdef Doc2VecConfig c

    cdef int i, j, k, m, n
    cdef long result = 0
    cdef np.uint32_t *vocab_sample_ints

    init_d2v_config(&c, model, alpha, learn_doctags, learn_words, learn_hidden, train_words=False, work=work, neu1=neu1,
                    word_vectors=word_vectors, words_lockf=words_lockf, doctag_vectors=doctag_vectors, doctags_lockf=doctags_lockf)
    c.doctag_len = <int>min(MAX_DOCUMENT_LEN, len(doctag_indexes))
    if c.sample:
        vocab_sample_ints = <np.uint32_t *>np.PyArray_DATA(model.wv.expandos['sample_int'])
    if c.hs:
        vocab_codes = model.wv.expandos['code']
        vocab_points = model.wv.expandos['point']

    if c.doctag_len != c.expected_doctag_len:
        return 0  # skip doc without expected number of tags

    i = 0
    for token in doc_words:
        word_index = model.wv.key_to_index.get(token, None)
        if word_index is None:  # shrink document to leave out word
            continue  # leaving i unchanged
        if c.sample and vocab_sample_ints[word_index] < random_int32(&c.next_random):
            continue
        c.indexes[i] = word_index
        if c.hs:
            c.codelens[i] = <int>len(vocab_codes[word_index])
            c.codes[i] = <np.uint8_t *>np.PyArray_DATA(vocab_codes[word_index])
            c.points[i] = <np.uint32_t *>np.PyArray_DATA(vocab_points[word_index])
        result += 1
        i += 1
        if i == MAX_DOCUMENT_LEN:
            break  # TODO: log warning, tally overflow?
    c.document_len = i

    for i in range(c.doctag_len):
        c.doctag_indexes[i] = doctag_indexes[i]
        result += 1

    # release GIL & train on the document
    with nogil:
        for i in range(c.document_len):
            j = i - c.window      # negative OK: will pad with null word
            k = i + c.window + 1  # past document end OK: will pad with null word

            # compose l1 & clear work
            for m in range(c.doctag_len):
                # doc vector(s)
                memcpy(&c.neu1[m * c.vector_size], &c.doctag_vectors[c.doctag_indexes[m] * c.vector_size],
                       c.vector_size * cython.sizeof(REAL_t))
            n = 0
            for m in range(j, k):
                # word vectors in window
                if m == i:
                    continue
                if m < 0 or m >= c.document_len:
                    c.window_indexes[n] = c.null_word_index
                else:
                    c.window_indexes[n] = c.indexes[m]
                n += 1
            for m in range(2 * c.window):
                memcpy(&c.neu1[(c.doctag_len + m) * c.vector_size], &c.word_vectors[c.window_indexes[m] * c.vector_size],
                       c.vector_size * cython.sizeof(REAL_t))
            memset(c.work, 0, c.layer1_size * cython.sizeof(REAL_t))  # work to accumulate l1 error

            if c.hs:
                fast_document_dmc_hs(c.points[i], c.codes[i], c.codelens[i],
                                     c.neu1, c.syn1, c.alpha, c.work,
                                     c.layer1_size, c.vector_size, c.learn_hidden)
            if c.negative:
                c.next_random = fast_document_dmc_neg(c.negative, c.cum_table, c.cum_table_len, c.next_random,
                                                      c.neu1, c.syn1neg, c.indexes[i], c.alpha, c.work,
                                                      c.layer1_size, c.vector_size, c.learn_hidden)

            if c.learn_doctags:
                for m in range(c.doctag_len):
                    our_saxpy(&c.vector_size, &c.doctags_lockf[c.doctag_indexes[m] % c.doctags_lockf_len], &c.work[m * c.vector_size],
                              &ONE, &c.doctag_vectors[c.doctag_indexes[m] * c.vector_size], &ONE)
            if c.learn_words:
                for m in range(2 * c.window):
                    our_saxpy(&c.vector_size, &c.words_lockf[c.window_indexes[m] % c.words_lockf_len], &c.work[(c.doctag_len + m) * c.vector_size],
                              &ONE, &c.word_vectors[c.window_indexes[m] * c.vector_size], &ONE)

    return result
