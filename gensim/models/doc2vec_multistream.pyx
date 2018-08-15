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

from libcpp.string cimport string
from libcpp.vector cimport vector

from libc.string cimport memset, memcpy

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

from gensim.models.doc2vec_inner cimport (
    fast_document_dbow_hs,
    fast_document_dbow_neg,
    fast_document_dm_hs,
    fast_document_dm_neg,
    fast_document_dmc_hs,
    fast_document_dmc_neg
)

from gensim.models.word2vec_inner cimport bisect_left, random_int32, sscal, REAL_t, EXP_TABLE, our_dot, our_saxpy

from gensim.models.word2vec_multistream cimport (
    VocabItem,
    CythonVocab,
    CythonLineSentence,
    get_alpha,
    get_next_alpha,
    cvocab_t
)

DEF MAX_DOCUMENT_LEN = 10000

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0


cdef void prepare_c_structures_for_batch(vector[string] &doc_words, int sample, int hs, int window, int *total_words,
                                         int *effective_words, unsigned long long *next_random, cvocab_t *vocab,
                                         np.uint32_t *indexes, int *codelens, np.uint8_t **codes, np.uint32_t **points,
                                         np.uint32_t *reduced_windows, int *document_len, int train_words,
                                         int docvecs_count, int doc_tag, np.uint32_t *doctag_indexes) nogil:
    cdef VocabItem predict_word
    cdef string token
    cdef int i = 0

    total_words[0] += doc_words.size()

    for token in doc_words:
        if vocab[0].find(token) == vocab[0].end():  # shrink document to leave out word
            continue  # leaving i unchanged

        predict_word = vocab[0][token]
        if sample and predict_word.sample_int < random_int32(next_random):
            continue
        indexes[i] = predict_word.index
        if hs:
            codelens[i] = predict_word.code_len
            codes[i] = predict_word.code
            points[i] = predict_word.point

        effective_words[0] += 1
        i += 1
        if i == MAX_DOCUMENT_LEN:
            break  # TODO: log warning, tally overflow?
    document_len[0] = i

    if train_words and reduced_windows != NULL:
        for i in range(document_len[0]):
            reduced_windows[i] = random_int32(next_random) % window

    if doc_tag < docvecs_count:
        doctag_indexes[i] = doc_tag
        effective_words[0] += 1



def d2v_train_epoch_dbow(model, corpus_file, offset, _cython_vocab, _cur_epoch, _expected_examples, _expected_words,
                         work, neu1, docvecs_count, word_vectors=None, word_locks=None, train_words=False, learn_doctags=True,
                         learn_words=True, learn_hidden=True, doctag_vectors=None, doctag_locks=None):
    """Train distributed bag of words model ("PV-DBOW") by training on a corpus file.

    Called internally from :meth:`~gensim.models.doc2vec.Doc2Vec.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.doc2vec.Doc2Vec`
        The FastText model instance to train.
    corpus_file : str
        Path to corpus file.
    _cur_epoch : int
        Current epoch number. Used for calculating and decaying learning rate.
    work : np.ndarray
        Private working memory for each worker.
    neu1 : np.ndarray
        Private working memory for each worker.
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
    word_locks : numpy.ndarray, optional
        A learning lock factor for each weight in the hidden layer for words, value 0 completely blocks updates,
        a value of 1 allows to update word-vectors.
    doctag_vectors : numpy.ndarray, optional
        Vector representations of the tags. If None, these will be retrieved from the model.
    doctag_locks : numpy.ndarray, optional
        The lock factors for each tag, same as `word_locks`, but for document-vectors.

    Returns
    -------
    int
        Number of words in the input document that were actually used for training.

    """
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int workers = model.workers
    cdef int _train_words = train_words
    cdef int _learn_words = learn_words
    cdef int _learn_hidden = learn_hidden
    cdef int _learn_doctags = learn_doctags
    cdef int _docvecs_count = docvecs_count

    cdef int cur_epoch = _cur_epoch
    cdef int num_epochs = model.epochs
    cdef int expected_examples = (-1 if _expected_examples is None else _expected_examples)
    cdef int expected_words = (-1 if _expected_words is None else _expected_words)
    cdef REAL_t start_alpha = model.alpha
    cdef REAL_t end_alpha = model.min_alpha
    cdef REAL_t _alpha = get_alpha(model.alpha, end_alpha, cur_epoch, num_epochs)

    cdef CythonLineSentence input_stream = CythonLineSentence(corpus_file, offset)
    cdef CythonVocab vocab = _cython_vocab

    cdef REAL_t *_word_vectors
    cdef REAL_t *_doctag_vectors
    cdef REAL_t *_word_locks
    cdef REAL_t *_doctag_locks
    cdef REAL_t *_work
    cdef int size = model.trainables.layer1_size

    cdef int codelens[MAX_DOCUMENT_LEN]
    cdef np.uint32_t indexes[MAX_DOCUMENT_LEN]
    cdef np.uint32_t _doctag_indexes[MAX_DOCUMENT_LEN]
    cdef np.uint32_t reduced_windows[MAX_DOCUMENT_LEN]
    cdef int document_len
    cdef int window = model.window

    cdef int i, j
    cdef unsigned long long r
    cdef int effective_words = 0
    cdef int total_effective_words = 0, total_documents = 0, total_words = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_DOCUMENT_LEN]
    cdef np.uint8_t *codes[MAX_DOCUMENT_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    cdef unsigned long long next_random

    # default vectors, locks from syn0/doctag_syn0
    if word_vectors is None:
       word_vectors = model.wv.vectors
    _word_vectors = <REAL_t *>(np.PyArray_DATA(word_vectors))
    if doctag_vectors is None:
       doctag_vectors = model.docvecs.vectors_docs
    _doctag_vectors = <REAL_t *>(np.PyArray_DATA(doctag_vectors))
    if word_locks is None:
       word_locks = model.trainables.vectors_lockf
    _word_locks = <REAL_t *>(np.PyArray_DATA(word_locks))
    if doctag_locks is None:
       doctag_locks = model.trainables.vectors_docs_lockf
    _doctag_locks = <REAL_t *>(np.PyArray_DATA(doctag_locks))

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    if work is None:
       work = zeros(model.trainables.layer1_size, dtype=REAL)
    _work = <REAL_t *>np.PyArray_DATA(work)

    # for preparing batches & training
    cdef vector[string] document
    cdef unsigned long long random_number
    cdef VocabItem word
    cdef int _doc_tag

    # release GIL & train on the full corpus, document by document
    with nogil:
        input_stream.reset()
        # Dummy read_sentence, to start from a new line (because offset is given in bytes).
        input_stream.read_sentence()
        while not (input_stream.is_eof() or total_words > expected_words / workers):
            effective_words = 0

            doc_words = input_stream.read_sentence()
            _doc_tag = total_documents

            if doc_words.empty():
                continue

            prepare_c_structures_for_batch(doc_words, sample, hs, window, &total_words, &effective_words,
                                           &next_random, vocab.get_vocab_ptr(), indexes,
                                           codelens, codes, points, reduced_windows, &document_len, _train_words,
                                           _docvecs_count, _doc_tag, _doctag_indexes)

            for i in range(document_len):
                if _train_words:  # simultaneous skip-gram wordvec-training
                    j = i - window + reduced_windows[i]
                    if j < 0:
                        j = 0
                    k = i + window + 1 - reduced_windows[i]
                    if k > document_len:
                        k = document_len
                    for j in range(j, k):
                        if j == i:
                            continue
                        if hs:
                            # we reuse the DBOW function, as it is equivalent to skip-gram for this purpose
                            fast_document_dbow_hs(points[i], codes[i], codelens[i], _word_vectors, syn1, size,
                                                  indexes[j], _alpha, _work, _learn_words, _learn_hidden, _word_locks)
                        if negative:
                            # we reuse the DBOW function, as it is equivalent to skip-gram for this purpose
                            next_random = fast_document_dbow_neg(negative, cum_table, cum_table_len, _word_vectors,
                                                                 syn1neg, size, indexes[i], indexes[j], _alpha, _work,
                                                                 next_random, _learn_words, _learn_hidden, _word_locks)

                # docvec-training
                if _doc_tag < _docvecs_count:
                    if hs:
                        fast_document_dbow_hs(points[i], codes[i], codelens[i], _doctag_vectors, syn1, size,
                                              _doctag_indexes[0], _alpha, _work, _learn_doctags, _learn_hidden,
                                              _doctag_locks)
                    if negative:
                        next_random = fast_document_dbow_neg(negative, cum_table, cum_table_len, _doctag_vectors,
                                                             syn1neg, size, indexes[i], _doctag_indexes[0], _alpha,
                                                             _work, next_random, _learn_doctags, _learn_hidden,
                                                             _doctag_locks)

            total_documents += 1
            total_effective_words += effective_words

            _alpha = get_next_alpha(start_alpha, end_alpha, total_documents, total_words, expected_examples,
                                    expected_words, cur_epoch, num_epochs)

    return total_documents, total_effective_words, total_words


def d2v_train_epoch_dm(model, corpus_file, offset, _cython_vocab, _cur_epoch, _expected_examples, _expected_words,
                         work, neu1, docvecs_count, word_vectors=None, word_locks=None, learn_doctags=True,
                         learn_words=True, learn_hidden=True, doctag_vectors=None, doctag_locks=None):
    """Train distributed memory model ("PV-DM") by training on a corpus file.
    This method implements the DM model with a projection (input) layer that is either the sum or mean of the context
    vectors, depending on the model's `dm_mean` configuration field.

    Called internally from :meth:`~gensim.models.doc2vec.Doc2Vec.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.doc2vec.Doc2Vec`
        The FastText model instance to train.
    corpus_file : str
        Path to corpus file.
    _cur_epoch : int
        Current epoch number. Used for calculating and decaying learning rate.
    work : np.ndarray
        Private working memory for each worker.
    neu1 : np.ndarray
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
    word_locks : numpy.ndarray, optional
        A learning lock factor for each weight in the hidden layer for words, value 0 completely blocks updates,
        a value of 1 allows to update word-vectors.
    doctag_vectors : numpy.ndarray, optional
        Vector representations of the tags. If None, these will be retrieved from the model.
    doctag_locks : numpy.ndarray, optional
        The lock factors for each tag, same as `word_locks`, but for document-vectors.

    Returns
    -------
    int
        Number of words in the input document that were actually used for training.

    """
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int workers = model.workers
    cdef int _train_words = 0
    cdef int _learn_words = learn_words
    cdef int _learn_hidden = learn_hidden
    cdef int _learn_doctags = learn_doctags
    cdef int _docvecs_count = docvecs_count
    cdef int cbow_mean = model.cbow_mean
    cdef REAL_t count, inv_count = 1.0

    cdef int cur_epoch = _cur_epoch
    cdef int num_epochs = model.epochs
    cdef int expected_examples = (-1 if _expected_examples is None else _expected_examples)
    cdef int expected_words = (-1 if _expected_words is None else _expected_words)
    cdef REAL_t start_alpha = model.alpha
    cdef REAL_t end_alpha = model.min_alpha
    cdef REAL_t _alpha = get_alpha(model.alpha, end_alpha, cur_epoch, num_epochs)

    cdef CythonLineSentence input_stream = CythonLineSentence(corpus_file, offset)
    cdef CythonVocab vocab = _cython_vocab

    cdef REAL_t *_word_vectors
    cdef REAL_t *_doctag_vectors
    cdef REAL_t *_word_locks
    cdef REAL_t *_doctag_locks
    cdef REAL_t *_work
    cdef REAL_t *_neu1
    cdef int size = model.trainables.layer1_size

    cdef int codelens[MAX_DOCUMENT_LEN]
    cdef np.uint32_t indexes[MAX_DOCUMENT_LEN]
    cdef np.uint32_t _doctag_indexes[MAX_DOCUMENT_LEN]
    cdef np.uint32_t reduced_windows[MAX_DOCUMENT_LEN]
    cdef int document_len
    cdef int window = model.window

    cdef int i, j, k, m
    cdef int effective_words = 0
    cdef int total_effective_words = 0, total_documents = 0, total_words = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_DOCUMENT_LEN]
    cdef np.uint8_t *codes[MAX_DOCUMENT_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    cdef unsigned long long next_random

    # default vectors, locks from syn0/doctag_syn0
    if word_vectors is None:
       word_vectors = model.wv.vectors
    _word_vectors = <REAL_t *>(np.PyArray_DATA(word_vectors))
    if doctag_vectors is None:
       doctag_vectors = model.docvecs.vectors_docs
    _doctag_vectors = <REAL_t *>(np.PyArray_DATA(doctag_vectors))
    if word_locks is None:
       word_locks = model.trainables.vectors_lockf
    _word_locks = <REAL_t *>(np.PyArray_DATA(word_locks))
    if doctag_locks is None:
       doctag_locks = model.trainables.vectors_docs_lockf
    _doctag_locks = <REAL_t *>(np.PyArray_DATA(doctag_locks))

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    # convert Python structures to primitive types, so we can release the GIL
    if work is None:
       work = zeros(model.trainables.layer1_size, dtype=REAL)
    _work = <REAL_t *>np.PyArray_DATA(work)
    if neu1 is None:
       neu1 = zeros(model.trainables.layer1_size, dtype=REAL)
    _neu1 = <REAL_t *>np.PyArray_DATA(neu1)

    # for preparing batches & training
    cdef vector[string] document
    cdef unsigned long long random_number
    cdef VocabItem word
    cdef int _doc_tag

    # release GIL & train on the full corpus, document by document
    with nogil:
        input_stream.reset()
        # Dummy read_sentence, to start from a new line (because offset is given in bytes).
        input_stream.read_sentence()
        while not (input_stream.is_eof() or total_words > expected_words / workers):
            effective_words = 0

            doc_words = input_stream.read_sentence()
            _doc_tag = total_documents

            if doc_words.empty():
                continue

            prepare_c_structures_for_batch(doc_words, sample, hs, window, &total_words, &effective_words,
                                           &next_random, vocab.get_vocab_ptr(), indexes,
                                           codelens, codes, points, reduced_windows, &document_len, _train_words,
                                           _docvecs_count, _doc_tag, _doctag_indexes)

            for i in range(document_len):
                j = i - window + reduced_windows[i]
                if j < 0:
                    j = 0
                k = i + window + 1 - reduced_windows[i]
                if k > document_len:
                    k = document_len

                # compose l1 (in _neu1) & clear _work
                memset(_neu1, 0, size * cython.sizeof(REAL_t))
                count = <REAL_t>0.0
                for m in range(j, k):
                    if m == i:
                        continue
                    else:
                        count += ONEF
                        our_saxpy(&size, &ONEF, &_word_vectors[indexes[m] * size], &ONE, _neu1, &ONE)

                if _doc_tag < _docvecs_count:
                    count += ONEF
                    our_saxpy(&size, &ONEF, &_doctag_vectors[_doctag_indexes[0] * size], &ONE, _neu1, &ONE)
                if count > (<REAL_t>0.5):
                    inv_count = ONEF/count
                if cbow_mean:
                    sscal(&size, &inv_count, _neu1, &ONE)  # (does this need BLAS-variants like saxpy?)
                memset(_work, 0, size * cython.sizeof(REAL_t))  # work to accumulate l1 error
                if hs:
                    fast_document_dm_hs(points[i], codes[i], codelens[i],
                                        _neu1, syn1, _alpha, _work,
                                        size, _learn_hidden)
                if negative:
                    next_random = fast_document_dm_neg(negative, cum_table, cum_table_len, next_random,
                                                       _neu1, syn1neg, indexes[i], _alpha, _work,
                                                       size, _learn_hidden)

                if not cbow_mean:
                    sscal(&size, &inv_count, _work, &ONE)  # (does this need BLAS-variants like saxpy?)
                # apply accumulated error in work
                if _learn_doctags and _doc_tag < _docvecs_count:
                    our_saxpy(&size, &_doctag_locks[_doctag_indexes[0]], _work,
                              &ONE, &_doctag_vectors[_doctag_indexes[0] * size], &ONE)
                if _learn_words:
                    for m in range(j, k):
                        if m == i:
                            continue
                        else:
                             our_saxpy(&size, &_word_locks[indexes[m]], _work, &ONE,
                                       &_word_vectors[indexes[m] * size], &ONE)

            total_documents += 1
            total_effective_words += effective_words

            _alpha = get_next_alpha(start_alpha, end_alpha, total_documents, total_words, expected_examples,
                                    expected_words, cur_epoch, num_epochs)

    return total_documents, total_effective_words, total_words


def d2v_train_epoch_dm_concat(model, corpus_file, offset, _cython_vocab, _cur_epoch, _expected_examples, _expected_words,
                         work, neu1, docvecs_count, word_vectors=None, word_locks=None, learn_doctags=True,
                         learn_words=True, learn_hidden=True, doctag_vectors=None, doctag_locks=None):
    """Train distributed memory model ("PV-DM") by training on a corpus file, using a concatenation of the context
     window word vectors (rather than a sum or average).
     This might be slower since the input at each batch will be significantly larger.

    Called internally from :meth:`~gensim.models.doc2vec.Doc2Vec.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.doc2vec.Doc2Vec`
        The FastText model instance to train.
    corpus_file : str
        Path to corpus file.
    _cur_epoch : int
        Current epoch number. Used for calculating and decaying learning rate.
    work : np.ndarray
        Private working memory for each worker.
    neu1 : np.ndarray
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
    word_locks : numpy.ndarray, optional
        A learning lock factor for each weight in the hidden layer for words, value 0 completely blocks updates,
        a value of 1 allows to update word-vectors.
    doctag_vectors : numpy.ndarray, optional
        Vector representations of the tags. If None, these will be retrieved from the model.
    doctag_locks : numpy.ndarray, optional
        The lock factors for each tag, same as `word_locks`, but for document-vectors.

    Returns
    -------
    int
        Number of words in the input document that were actually used for training.

    """
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int workers = model.workers
    cdef int _train_words = 0
    cdef int _learn_words = learn_words
    cdef int _learn_hidden = learn_hidden
    cdef int _learn_doctags = learn_doctags
    cdef int _docvecs_count = docvecs_count

    cdef int cur_epoch = _cur_epoch
    cdef int num_epochs = model.epochs
    cdef int expected_examples = (-1 if _expected_examples is None else _expected_examples)
    cdef int expected_words = (-1 if _expected_words is None else _expected_words)
    cdef REAL_t start_alpha = model.alpha
    cdef REAL_t end_alpha = model.min_alpha
    cdef REAL_t _alpha = get_alpha(model.alpha, end_alpha, cur_epoch, num_epochs)

    cdef CythonLineSentence input_stream = CythonLineSentence(corpus_file, offset)
    cdef CythonVocab vocab = _cython_vocab

    cdef REAL_t *_word_vectors
    cdef REAL_t *_doctag_vectors
    cdef REAL_t *_word_locks
    cdef REAL_t *_doctag_locks
    cdef REAL_t *_work
    cdef REAL_t *_neu1
    cdef int layer1_size = model.trainables.layer1_size
    cdef int vector_size = model.docvecs.vector_size

    cdef int codelens[MAX_DOCUMENT_LEN]
    cdef np.uint32_t indexes[MAX_DOCUMENT_LEN]
    cdef np.uint32_t _doctag_indexes[MAX_DOCUMENT_LEN]
    cdef np.uint32_t window_indexes[MAX_DOCUMENT_LEN]
    cdef int document_len
    cdef int doctag_len
    cdef int window = model.window
    cdef int expected_doctag_len = model.dm_tag_count

    cdef int i, j, k, m, n
    cdef int null_word_index = model.wv.vocab['\0'].index
    cdef int effective_words = 0
    cdef int total_effective_words = 0, total_documents = 0, total_words = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_DOCUMENT_LEN]
    cdef np.uint8_t *codes[MAX_DOCUMENT_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    cdef unsigned long long next_random

    # default vectors, locks from syn0/doctag_syn0
    if word_vectors is None:
       word_vectors = model.wv.vectors
    _word_vectors = <REAL_t *>(np.PyArray_DATA(word_vectors))
    if doctag_vectors is None:
       doctag_vectors = model.docvecs.vectors_docs
    _doctag_vectors = <REAL_t *>(np.PyArray_DATA(doctag_vectors))
    if word_locks is None:
       word_locks = model.trainables.vectors_lockf
    _word_locks = <REAL_t *>(np.PyArray_DATA(word_locks))
    if doctag_locks is None:
       doctag_locks = model.trainables.vectors_docs_lockf
    _doctag_locks = <REAL_t *>(np.PyArray_DATA(doctag_locks))

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    # convert Python structures to primitive types, so we can release the GIL
    if work is None:
       work = zeros(model.trainables.layer1_size, dtype=REAL)
    _work = <REAL_t *>np.PyArray_DATA(work)
    if neu1 is None:
       neu1 = zeros(model.trainables.layer1_size, dtype=REAL)
    _neu1 = <REAL_t *>np.PyArray_DATA(neu1)

    # for preparing batches & training
    cdef vector[string] document
    cdef unsigned long long random_number
    cdef VocabItem word
    cdef int _doc_tag

    # release GIL & train on the full corpus, document by document
    with nogil:
        input_stream.reset()
        # Dummy read_sentence, to start from a new line (because offset is given in bytes).
        input_stream.read_sentence()
        while not (input_stream.is_eof() or total_words > expected_words / workers):
            effective_words = 0

            doc_words = input_stream.read_sentence()
            _doc_tag = total_documents
            doctag_len = _doc_tag < _docvecs_count

             # skip doc either empty or without expected number of tags
            if doc_words.empty() or expected_doctag_len != doctag_len:
                continue

            prepare_c_structures_for_batch(doc_words, sample, hs, window, &total_words, &effective_words,
                                           &next_random, vocab.get_vocab_ptr(), indexes,
                                           codelens, codes, points, NULL, &document_len, _train_words,
                                           _docvecs_count, _doc_tag, _doctag_indexes)

            for i in range(document_len):
                j = i - window      # negative OK: will pad with null word
                k = i + window + 1  # past document end OK: will pad with null word

                # compose l1 & clear work
                if _doc_tag < _docvecs_count:
                    # doc vector(s)
                    memcpy(&_neu1[0], &_doctag_vectors[_doctag_indexes[0] * vector_size],
                           vector_size * cython.sizeof(REAL_t))
                n = 0
                for m in range(j, k):
                    # word vectors in window
                    if m == i:
                        continue
                    if m < 0 or m >= document_len:
                        window_indexes[n] =  null_word_index
                    else:
                        window_indexes[n] = indexes[m]
                    n += 1
                for m in range(2 * window):
                    memcpy(&_neu1[(doctag_len + m) * vector_size], &_word_vectors[window_indexes[m] * vector_size],
                           vector_size * cython.sizeof(REAL_t))
                memset(_work, 0, layer1_size * cython.sizeof(REAL_t))  # work to accumulate l1 error

                if hs:
                    fast_document_dmc_hs(points[i], codes[i], codelens[i],
                                         _neu1, syn1, _alpha, _work,
                                         layer1_size, vector_size, _learn_hidden)
                if negative:
                    next_random = fast_document_dmc_neg(negative, cum_table, cum_table_len, next_random,
                                                        _neu1, syn1neg, indexes[i], _alpha, _work,
                                                       layer1_size, vector_size, _learn_hidden)

                if _learn_doctags:
                    for m in range(doctag_len):
                        our_saxpy(&vector_size, &_doctag_locks[_doctag_indexes[m]], &_work[m * vector_size],
                                  &ONE, &_doctag_vectors[_doctag_indexes[m] * vector_size], &ONE)
                if _learn_words:
                    for m in range(2 * window):
                        our_saxpy(&vector_size, &_word_locks[window_indexes[m]], &_work[(doctag_len + m) * vector_size],
                                  &ONE, &_word_vectors[window_indexes[m] * vector_size], &ONE)

            total_documents += 1
            total_effective_words += effective_words

            _alpha = get_next_alpha(start_alpha, end_alpha, total_documents, total_words, expected_examples,
                                    expected_words, cur_epoch, num_epochs)

    return total_documents, total_effective_words, total_words


MULTISTREAM_VERSION = 1
