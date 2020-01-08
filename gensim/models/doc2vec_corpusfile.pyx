#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8
#
# Copyright (C) 2018 Dmitry Persiyanov <dmitry.persiyanov@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Optimized cython functions for file-based training :class:`~gensim.models.doc2vec.Doc2Vec` model."""

import cython
import numpy as np
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
    fast_document_dmc_neg,
    init_d2v_config,
    Doc2VecConfig
)

from gensim.models.word2vec_inner cimport random_int32, sscal, REAL_t, our_saxpy

from gensim.models.word2vec_corpusfile cimport (
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


cdef void prepare_c_structures_for_batch(vector[string] &doc_words, int sample, int hs, int window, long long *total_words,
                                         int *effective_words, unsigned long long *next_random, cvocab_t *vocab,
                                         np.uint32_t *indexes, int *codelens, np.uint8_t **codes, np.uint32_t **points,
                                         np.uint32_t *reduced_windows, int *document_len, int train_words,
                                         int docvecs_count, int doc_tag) nogil:
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
        effective_words[0] += 1


def d2v_train_epoch_dbow(model, corpus_file, offset, start_doctag, _cython_vocab, _cur_epoch, _expected_examples,
                         _expected_words, work, neu1, docvecs_count, word_vectors=None, word_locks=None,
                         train_words=False, learn_doctags=True, learn_words=True, learn_hidden=True,
                         doctag_vectors=None, doctag_locks=None):
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
    cdef Doc2VecConfig c

    cdef int cur_epoch = _cur_epoch
    cdef int num_epochs = model.epochs
    cdef long long expected_examples = (-1 if _expected_examples is None else _expected_examples)
    cdef long long expected_words = (-1 if _expected_words is None else _expected_words)
    cdef REAL_t start_alpha = model.alpha
    cdef REAL_t end_alpha = model.min_alpha
    cdef REAL_t _alpha = get_alpha(model.alpha, end_alpha, cur_epoch, num_epochs)

    cdef CythonLineSentence input_stream = CythonLineSentence(corpus_file, offset)
    cdef CythonVocab vocab = _cython_vocab

    cdef int i, j, document_len
    cdef int effective_words = 0
    cdef long long total_documents = 0
    cdef long long total_effective_words = 0, total_words = 0
    cdef int sent_idx, idx_start, idx_end

    cdef vector[string] doc_words
    cdef int _doc_tag = start_doctag

    init_d2v_config(
        &c, model, _alpha, learn_doctags, learn_words, learn_hidden, train_words=train_words,
        work=work, neu1=neu1, word_vectors=word_vectors, word_locks=word_locks,
        doctag_vectors=doctag_vectors, doctag_locks=doctag_locks, docvecs_count=docvecs_count)

    # release GIL & train on the full corpus, document by document
    with nogil:
        input_stream.reset()
        while not (input_stream.is_eof() or total_words > expected_words / c.workers):
            effective_words = 0

            doc_words = input_stream.read_sentence()

            if doc_words.empty():
                continue

            prepare_c_structures_for_batch(
                doc_words, c.sample, c.hs, c.window, &total_words, &effective_words,
                &c.next_random, vocab.get_vocab_ptr(), c.indexes, c.codelens,  c.codes, c.points,
                c.reduced_windows, &document_len, c.train_words, c.docvecs_count, _doc_tag)

            for i in range(document_len):
                if c.train_words:  # simultaneous skip-gram wordvec-training
                    j = i - c.window + c.reduced_windows[i]
                    if j < 0:
                        j = 0
                    k = i + c.window + 1 - c.reduced_windows[i]
                    if k > document_len:
                        k = document_len
                    for j in range(j, k):
                        if j == i:
                            continue
                        if c.hs:
                            # we reuse the DBOW function, as it is equivalent to skip-gram for this purpose
                            fast_document_dbow_hs(
                                c.points[i], c.codes[i], c.codelens[i], c.word_vectors, c.syn1, c.layer1_size,
                                c.indexes[j], c.alpha, c.work, c.learn_words, c.learn_hidden, c.word_locks)

                        if c.negative:
                            # we reuse the DBOW function, as it is equivalent to skip-gram for this purpose
                            c.next_random = fast_document_dbow_neg(
                                c.negative, c.cum_table, c.cum_table_len, c.word_vectors, c.syn1neg,
                                c.layer1_size, c.indexes[i], c.indexes[j], c.alpha, c.work,
                                c.next_random, c.learn_words, c.learn_hidden, c.word_locks)

                # docvec-training
                if _doc_tag < c.docvecs_count:
                    if c.hs:
                        fast_document_dbow_hs(
                            c.points[i], c.codes[i], c.codelens[i], c.doctag_vectors, c.syn1, c.layer1_size,
                            _doc_tag, c.alpha, c.work, c.learn_doctags, c.learn_hidden, c.doctag_locks)

                    if c.negative:
                        c.next_random = fast_document_dbow_neg(
                            c.negative, c.cum_table, c.cum_table_len, c.doctag_vectors, c.syn1neg,
                            c.layer1_size, c.indexes[i], _doc_tag, c.alpha, c.work, c.next_random,
                            c.learn_doctags, c.learn_hidden, c.doctag_locks)

            total_documents += 1
            total_effective_words += effective_words
            _doc_tag += 1

            c.alpha = get_next_alpha(
                start_alpha, end_alpha, total_documents, total_words,
                expected_examples, expected_words, cur_epoch, num_epochs)

    return total_documents, total_effective_words, total_words


def d2v_train_epoch_dm(model, corpus_file, offset, start_doctag, _cython_vocab, _cur_epoch, _expected_examples,
                       _expected_words, work, neu1, docvecs_count, word_vectors=None, word_locks=None,
                       learn_doctags=True, learn_words=True, learn_hidden=True, doctag_vectors=None, doctag_locks=None):
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
    cdef Doc2VecConfig c

    cdef int cur_epoch = _cur_epoch
    cdef int num_epochs = model.epochs
    cdef long long expected_examples = (-1 if _expected_examples is None else _expected_examples)
    cdef long long expected_words = (-1 if _expected_words is None else _expected_words)
    cdef REAL_t start_alpha = model.alpha
    cdef REAL_t end_alpha = model.min_alpha
    cdef REAL_t _alpha = get_alpha(model.alpha, end_alpha, cur_epoch, num_epochs)

    cdef CythonLineSentence input_stream = CythonLineSentence(corpus_file, offset)
    cdef CythonVocab vocab = _cython_vocab

    cdef int i, j, k, m, document_len
    cdef int effective_words = 0
    cdef long long total_documents = 0
    cdef long long total_effective_words = 0, total_words = 0
    cdef int sent_idx, idx_start, idx_end
    cdef REAL_t count, inv_count = 1.0

    cdef vector[string] doc_words
    cdef int _doc_tag = start_doctag

    init_d2v_config(
        &c, model, _alpha, learn_doctags, learn_words, learn_hidden, train_words=False,
        work=work, neu1=neu1, word_vectors=word_vectors, word_locks=word_locks,
        doctag_vectors=doctag_vectors, doctag_locks=doctag_locks, docvecs_count=docvecs_count)

    # release GIL & train on the full corpus, document by document
    with nogil:
        input_stream.reset()
        while not (input_stream.is_eof() or total_words > expected_words / c.workers):
            effective_words = 0

            doc_words = input_stream.read_sentence()

            if doc_words.empty():
                continue

            prepare_c_structures_for_batch(
                doc_words, c.sample, c.hs, c.window, &total_words, &effective_words, &c.next_random,
                vocab.get_vocab_ptr(), c.indexes, c.codelens, c.codes, c.points, c.reduced_windows,
                &document_len, c.train_words, c.docvecs_count, _doc_tag)

            for i in range(document_len):
                j = i - c.window + c.reduced_windows[i]
                if j < 0:
                    j = 0
                k = i + c.window + 1 - c.reduced_windows[i]
                if k > document_len:
                    k = document_len

                # compose l1 (in _neu1) & clear _work
                memset(c.neu1, 0, c.layer1_size * cython.sizeof(REAL_t))
                count = <REAL_t>0.0
                for m in range(j, k):
                    if m == i:
                        continue
                    else:
                        count += ONEF
                        our_saxpy(&c.layer1_size, &ONEF, &c.word_vectors[c.indexes[m] * c.layer1_size], &ONE, c.neu1, &ONE)

                if _doc_tag < c.docvecs_count:
                    count += ONEF
                    our_saxpy(&c.layer1_size, &ONEF, &c.doctag_vectors[_doc_tag * c.layer1_size], &ONE, c.neu1, &ONE)
                if count > (<REAL_t>0.5):
                    inv_count = ONEF/count
                if c.cbow_mean:
                    sscal(&c.layer1_size, &inv_count, c.neu1, &ONE)  # (does this need BLAS-variants like saxpy?)
                memset(c.work, 0, c.layer1_size * cython.sizeof(REAL_t))  # work to accumulate l1 error
                if c.hs:
                    fast_document_dm_hs(
                        c.points[i], c.codes[i], c.codelens[i], c.neu1,
                        c.syn1, c.alpha, c.work, c.layer1_size, c.learn_hidden)

                if c.negative:
                    c.next_random = fast_document_dm_neg(
                        c.negative, c.cum_table, c.cum_table_len, c.next_random, c.neu1,
                        c.syn1neg, c.indexes[i], c.alpha, c.work, c.layer1_size, c.learn_hidden)

                if not c.cbow_mean:
                    sscal(&c.layer1_size, &inv_count, c.work, &ONE)  # (does this need BLAS-variants like saxpy?)
                # apply accumulated error in work
                if c.learn_doctags and _doc_tag < c.docvecs_count:
                    our_saxpy(&c.layer1_size, &c.doctag_locks[_doc_tag], c.work,
                              &ONE, &c.doctag_vectors[_doc_tag * c.layer1_size], &ONE)
                if c.learn_words:
                    for m in range(j, k):
                        if m == i:
                            continue
                        else:
                             our_saxpy(&c.layer1_size, &c.word_locks[c.indexes[m]], c.work, &ONE,
                                       &c.word_vectors[c.indexes[m] * c.layer1_size], &ONE)

            total_documents += 1
            total_effective_words += effective_words
            _doc_tag += 1

            c.alpha = get_next_alpha(start_alpha, end_alpha, total_documents, total_words, expected_examples,
                                    expected_words, cur_epoch, num_epochs)

    return total_documents, total_effective_words, total_words


def d2v_train_epoch_dm_concat(model, corpus_file, offset, start_doctag, _cython_vocab, _cur_epoch, _expected_examples,
                              _expected_words, work, neu1, docvecs_count, word_vectors=None, word_locks=None,
                              learn_doctags=True, learn_words=True, learn_hidden=True, doctag_vectors=None,
                              doctag_locks=None):
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
    cdef Doc2VecConfig c

    cdef int cur_epoch = _cur_epoch
    cdef int num_epochs = model.epochs
    cdef long long expected_examples = (-1 if _expected_examples is None else _expected_examples)
    cdef long long expected_words = (-1 if _expected_words is None else _expected_words)
    cdef REAL_t start_alpha = model.alpha
    cdef REAL_t end_alpha = model.min_alpha
    cdef REAL_t _alpha = get_alpha(model.alpha, end_alpha, cur_epoch, num_epochs)

    cdef CythonLineSentence input_stream = CythonLineSentence(corpus_file, offset)
    cdef CythonVocab vocab = _cython_vocab

    cdef int i, j, k, m, n, document_len
    cdef int effective_words = 0
    cdef long long total_documents = 0
    cdef long long total_effective_words = 0, total_words = 0
    cdef int sent_idx, idx_start, idx_end

    cdef vector[string] doc_words
    cdef int _doc_tag = start_doctag

    init_d2v_config(
        &c, model, _alpha, learn_doctags, learn_words, learn_hidden, train_words=False,
        work=work, neu1=neu1, word_vectors=word_vectors, word_locks=word_locks,
        doctag_vectors=doctag_vectors, doctag_locks=doctag_locks, docvecs_count=docvecs_count)

    # release GIL & train on the full corpus, document by document
    with nogil:
        input_stream.reset()
        while not (input_stream.is_eof() or total_words > expected_words / c.workers):
            effective_words = 0

            doc_words = input_stream.read_sentence()
            _doc_tag = total_documents
            c.doctag_len = _doc_tag < c.docvecs_count

             # skip doc either empty or without expected number of tags
            if doc_words.empty() or c.expected_doctag_len != c.doctag_len:
                continue

            prepare_c_structures_for_batch(
                doc_words, c.sample, c.hs, c.window, &total_words, &effective_words,
                &c.next_random, vocab.get_vocab_ptr(), c.indexes, c.codelens, c.codes,
                c.points, NULL, &document_len, c.train_words, c.docvecs_count, _doc_tag)

            for i in range(document_len):
                j = i - c.window      # negative OK: will pad with null word
                k = i + c.window + 1  # past document end OK: will pad with null word

                # compose l1 & clear work
                if _doc_tag < c.docvecs_count:
                    # doc vector(s)
                    memcpy(&c.neu1[0], &c.doctag_vectors[_doc_tag * c.vector_size],
                           c.vector_size * cython.sizeof(REAL_t))
                n = 0
                for m in range(j, k):
                    # word vectors in window
                    if m == i:
                        continue
                    if m < 0 or m >= document_len:
                        c.window_indexes[n] = c.null_word_index
                    else:
                        c.window_indexes[n] = c.indexes[m]
                    n += 1
                for m in range(2 * c.window):
                    memcpy(&c.neu1[(c.doctag_len + m) * c.vector_size], &c.word_vectors[c.window_indexes[m] * c.vector_size],
                           c.vector_size * cython.sizeof(REAL_t))
                memset(c.work, 0, c.layer1_size * cython.sizeof(REAL_t))  # work to accumulate l1 error

                if c.hs:
                    fast_document_dmc_hs(
                        c.points[i], c.codes[i], c.codelens[i], c.neu1, c.syn1,
                        c.alpha, c.work, c.layer1_size, c.vector_size, c.learn_hidden)

                if c.negative:
                    c.next_random = fast_document_dmc_neg(
                        c.negative, c.cum_table, c.cum_table_len, c.next_random, c.neu1, c.syn1neg,
                        c.indexes[i], c.alpha, c.work, c.layer1_size, c.vector_size, c.learn_hidden)

                if c.learn_doctags and _doc_tag < c.docvecs_count:
                    our_saxpy(&c.vector_size, &c.doctag_locks[_doc_tag], &c.work[m * c.vector_size],
                              &ONE, &c.doctag_vectors[_doc_tag * c.vector_size], &ONE)
                if c.learn_words:
                    for m in range(2 * c.window):
                        our_saxpy(&c.vector_size, &c.word_locks[c.window_indexes[m]], &c.work[(c.doctag_len + m) * c.vector_size],
                                  &ONE, &c.word_vectors[c.window_indexes[m] * c.vector_size], &ONE)

            total_documents += 1
            total_effective_words += effective_words
            _doc_tag += 1

            c.alpha = get_next_alpha(start_alpha, end_alpha, total_documents, total_words, expected_examples,
                                    expected_words, cur_epoch, num_epochs)

    return total_documents, total_effective_words, total_words


CORPUSFILE_VERSION = 1
