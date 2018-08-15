#!/usr/bin/env cython
# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Optimized cython functions for training :class:`~gensim.models.fasttext.FastText` model."""

import numpy as np
cimport numpy as np

from libcpp.string cimport string
from libcpp.vector cimport vector

from gensim.models.fasttext_inner cimport (
    fasttext_fast_sentence_sg_hs,
    fasttext_fast_sentence_sg_neg,
    fasttext_fast_sentence_cbow_hs,
    fasttext_fast_sentence_cbow_neg,
    init
)

from gensim.models.word2vec_inner cimport random_int32

from gensim.models.word2vec_corpusfile cimport (
    VocabItem,
    CythonVocab,
    CythonLineSentence,
    get_alpha,
    get_next_alpha,
    cvocab_t
)

ctypedef np.float32_t REAL_t
DEF MAX_SENTENCE_LEN = 10000
DEF MAX_SUBWORDS = 1000


cdef void prepare_c_structures_for_batch(vector[vector[string]] &sentences, int sample, int hs, int window, int *total_words,
                                         int *effective_words, int *effective_sentences, unsigned long long *next_random,
                                         cvocab_t *vocab, int *sentence_idx, np.uint32_t *indexes,
                                         int *codelens, np.uint8_t **codes, np.uint32_t **points,
                                         np.uint32_t *reduced_windows, int *subwords_idx_len, np.uint32_t **subwords_idx) nogil:
    cdef VocabItem word
    cdef string token
    cdef vector[string] sent

    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if sent.empty():
            continue # ignore empty sentences; leave effective_sentences unchanged
        total_words[0] += sent.size()

        for token in sent:
            # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if vocab[0].find(token) == vocab[0].end():
                continue

            word = vocab[0][token]
            if sample and word.sample_int < random_int32(next_random):
                continue
            indexes[effective_words[0]] = word.index
            subwords_idx_len[effective_words[0]] = word.subword_idx_len
            subwords_idx[effective_words[0]] = word.subword_idx

            if hs:
                codelens[effective_words[0]] = word.code_len
                codes[effective_words[0]] = word.code
                points[effective_words[0]] = word.point

            effective_words[0] += 1
            if effective_words[0] == MAX_SENTENCE_LEN:
                break

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences[0] += 1
        sentence_idx[effective_sentences[0]] = effective_words[0]

        if effective_words[0] == MAX_SENTENCE_LEN:
            break

    # precompute "reduced window" offsets in a single randint() call
    for i in range(effective_words[0]):
        reduced_windows[i] = random_int32(next_random) % window


def train_epoch_sg(model, corpus_file, offset, _cython_vocab, _cur_epoch, _expected_examples, _expected_words, _work,
                   _l1, compute_loss):
    """Train Skipgram model for one epoch by training on an input stream. This function is used only in multistream mode.

    Called internally from :meth:`~gensim.models.fasttext.FastText.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.fasttext.FastText`
        The FastText model instance to train.
    corpus_file : str
        Path to corpus file.
    _cur_epoch : int
        Current epoch number. Used for calculating and decaying learning rate.
    _work : np.ndarray
        Private working memory for each worker.
    _l1 : np.ndarray
        Private working memory for each worker.
    compute_loss : bool
        Whether or not the training loss should be computed in this batch.

    Returns
    -------
    int
        Number of words in the vocabulary actually used for training (They already existed in the vocabulary
        and were not discarded by negative sampling).
    """
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int workers = model.workers

    # For learning rate updates
    cdef int cur_epoch = _cur_epoch
    cdef int num_epochs = model.epochs
    cdef int expected_examples = (-1 if _expected_examples is None else _expected_examples)
    cdef int expected_words = (-1 if _expected_words is None else _expected_words)
    cdef REAL_t start_alpha = model.alpha
    cdef REAL_t end_alpha = model.min_alpha
    cdef REAL_t _alpha = get_alpha(model.alpha, end_alpha, cur_epoch, num_epochs)

    cdef CythonLineSentence input_stream = CythonLineSentence(corpus_file, offset)
    cdef CythonVocab vocab = _cython_vocab

    cdef int _compute_loss = (1 if compute_loss == True else 0)
    cdef REAL_t _running_training_loss = model.running_training_loss

    cdef REAL_t *syn0_vocab = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_vocab))
    cdef REAL_t *word_locks_vocab = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_vocab_lockf))
    cdef REAL_t *syn0_ngrams = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_ngrams))
    cdef REAL_t *word_locks_ngrams = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_ngrams_lockf))

    cdef REAL_t *work
    cdef REAL_t *l1
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int total_effective_words = 0, total_sentences = 0, total_words = 0
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

    # For passing subwords information as C objects for nogil
    cdef int subwords_idx_len[MAX_SENTENCE_LEN]
    cdef np.uint32_t *subwords_idx[MAX_SENTENCE_LEN]

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
    l1 = <REAL_t *>np.PyArray_DATA(_l1)

    # for preparing batches & training
    cdef vector[vector[string]] sentences
    cdef unsigned long long random_number
    cdef VocabItem word

    with nogil:
        input_stream.reset()
        while not (input_stream.is_eof() or total_words > expected_words / workers):
            effective_sentences = 0
            effective_words = 0

            sentences = input_stream.next_batch()

            prepare_c_structures_for_batch(sentences, sample, hs, window, &total_words, &effective_words,
                                           &effective_sentences, &next_random, vocab.get_vocab_ptr(), sentence_idx, indexes,
                                           codelens, codes, points, reduced_windows, subwords_idx_len, subwords_idx)

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
                            fasttext_fast_sentence_sg_hs(
                                points[j], codes[j], codelens[j], syn0_vocab, syn0_ngrams, syn1, size,
                                subwords_idx[i], subwords_idx_len[i], _alpha, work, l1, word_locks_vocab,
                                word_locks_ngrams)
                        if negative:
                            next_random = fasttext_fast_sentence_sg_neg(
                                negative, cum_table, cum_table_len, syn0_vocab, syn0_ngrams, syn1neg, size,
                                indexes[j], subwords_idx[i], subwords_idx_len[i], _alpha, work, l1,
                                next_random, word_locks_vocab, word_locks_ngrams)

            total_sentences += sentences.size()
            total_effective_words += effective_words

            _alpha = get_next_alpha(start_alpha, end_alpha, total_sentences, total_words,
                                    expected_examples, expected_words, cur_epoch, num_epochs)

    model.running_training_loss = _running_training_loss
    return total_sentences, total_effective_words, total_words


def train_epoch_cbow(model, corpus_file, offset, _cython_vocab, _cur_epoch, _expected_examples, _expected_words, _work,
                     _neu1, compute_loss):
    """Train CBOW model for one epoch by training on an input stream. This function is used only in multistream mode.

    Called internally from :meth:`~gensim.models.fasttext.FastText.train`.

    Parameters
    ----------
    model : :class:`~gensim.models.fasttext.FastText`
        The FastText model instance to train.
    corpus_file : str
        Path to a corpus file.
    _cur_epoch : int
        Current epoch number. Used for calculating and decaying learning rate.
    _work : np.ndarray
        Private working memory for each worker.
    _neu1 : np.ndarray
        Private working memory for each worker.
    compute_loss : bool
        Whether or not the training loss should be computed in this batch.

    Returns
    -------
    int
        Number of words in the vocabulary actually used for training (They already existed in the vocabulary
        and were not discarded by negative sampling).
    """
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int cbow_mean = model.cbow_mean
    cdef int workers = model.workers

    # For learning rate updates
    cdef int cur_epoch = _cur_epoch
    cdef int num_epochs = model.epochs
    cdef int expected_examples = (-1 if _expected_examples is None else _expected_examples)
    cdef int expected_words = (-1 if _expected_words is None else _expected_words)
    cdef REAL_t start_alpha = model.alpha
    cdef REAL_t end_alpha = model.min_alpha
    cdef REAL_t _alpha = get_alpha(model.alpha, end_alpha, cur_epoch, num_epochs)

    cdef CythonLineSentence input_stream = CythonLineSentence(corpus_file, offset)
    cdef CythonVocab vocab = _cython_vocab

    cdef int _compute_loss = (1 if compute_loss == True else 0)
    cdef REAL_t _running_training_loss = model.running_training_loss

    cdef REAL_t *syn0_vocab = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_vocab))
    cdef REAL_t *word_locks_vocab = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_vocab_lockf))
    cdef REAL_t *syn0_ngrams = <REAL_t *>(np.PyArray_DATA(model.wv.vectors_ngrams))
    cdef REAL_t *word_locks_ngrams = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_ngrams_lockf))

    cdef REAL_t *work
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int total_effective_words = 0, total_sentences = 0, total_words = 0
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

    # For passing subwords information as C objects for nogil
    cdef int subwords_idx_len[MAX_SENTENCE_LEN]
    cdef np.uint32_t *subwords_idx[MAX_SENTENCE_LEN]

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

    # for preparing batches & training
    cdef vector[vector[string]] sentences
    cdef unsigned long long random_number
    cdef VocabItem word

    with nogil:
        input_stream.reset()
        while not (input_stream.is_eof() or total_words > expected_words / workers):
            effective_sentences = 0
            effective_words = 0

            sentences = input_stream.next_batch()

            prepare_c_structures_for_batch(sentences, sample, hs, window, &total_words, &effective_words,
                                           &effective_sentences, &next_random, vocab.get_vocab_ptr(), sentence_idx, indexes,
                                           codelens, codes, points, reduced_windows, subwords_idx_len, subwords_idx)

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
                        fasttext_fast_sentence_cbow_hs(
                            points[i], codes[i], codelens, neu1, syn0_vocab, syn0_ngrams, syn1, size, indexes,
                            subwords_idx, subwords_idx_len, _alpha, work, i, j, k, cbow_mean, word_locks_vocab,
                            word_locks_ngrams)
                    if negative:
                        next_random = fasttext_fast_sentence_cbow_neg(
                            negative, cum_table, cum_table_len, codelens, neu1, syn0_vocab, syn0_ngrams,
                            syn1neg, size, indexes, subwords_idx, subwords_idx_len, _alpha, work, i, j, k,
                            cbow_mean, next_random, word_locks_vocab, word_locks_ngrams)

            total_sentences += sentences.size()
            total_effective_words += effective_words

            _alpha = get_next_alpha(start_alpha, end_alpha, total_sentences, total_words,
                                    expected_examples, expected_words, cur_epoch, num_epochs)

    model.running_training_loss = _running_training_loss
    return total_sentences, total_effective_words, total_words


CORPUSFILE_VERSION = 1