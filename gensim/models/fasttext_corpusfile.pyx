#!/usr/bin/env cython
# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8
#
# Copyright (C) 2018 Dmitry Persiyanov <dmitry.persiyanov@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Optimized cython functions for file-based training :class:`~gensim.models.fasttext.FastText` model."""

import numpy as np
cimport numpy as np

from libcpp.string cimport string
from libcpp.vector cimport vector

from gensim.models.fasttext_inner cimport (
    fasttext_fast_sentence_sg_hs,
    fasttext_fast_sentence_sg_neg,
    fasttext_fast_sentence_cbow_hs,
    fasttext_fast_sentence_cbow_neg,
    init_ft_config,
    FastTextConfig
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


cdef void prepare_c_structures_for_batch(
        vector[vector[string]] &sentences, int sample, int hs, int window, long long *total_words,
        int *effective_words, int *effective_sentences, unsigned long long *next_random, cvocab_t *vocab,
        int *sentence_idx, np.uint32_t *indexes, int *codelens, np.uint8_t **codes, np.uint32_t **points,
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


def train_epoch_sg(
        model, corpus_file, offset, _cython_vocab, _cur_epoch, _expected_examples, _expected_words, _work, _l1):
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

    Returns
    -------
    int
        Number of words in the vocabulary actually used for training (They already existed in the vocabulary
        and were not discarded by negative sampling).
    """
    cdef FastTextConfig c

    # For learning rate updates
    cdef int cur_epoch = _cur_epoch
    cdef int num_epochs = model.epochs
    cdef int expected_examples = (-1 if _expected_examples is None else _expected_examples)
    cdef long long expected_words = (-1 if _expected_words is None else _expected_words)
    cdef REAL_t start_alpha = model.alpha
    cdef REAL_t end_alpha = model.min_alpha
    cdef REAL_t _alpha = get_alpha(model.alpha, end_alpha, cur_epoch, num_epochs)

    cdef CythonLineSentence input_stream = CythonLineSentence(corpus_file, offset)
    cdef CythonVocab vocab = _cython_vocab

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int total_sentences = 0
    cdef long long total_effective_words = 0, total_words = 0
    cdef int sent_idx, idx_start, idx_end

    init_ft_config(&c, model, _alpha, _work, _l1)

    # for preparing batches & training
    cdef vector[vector[string]] sentences

    with nogil:
        input_stream.reset()
        while not (input_stream.is_eof() or total_words > expected_words / c.workers):
            effective_sentences = 0
            effective_words = 0

            sentences = input_stream.next_batch()

            prepare_c_structures_for_batch(
                sentences, c.sample, c.hs, c.window, &total_words, &effective_words, &effective_sentences,
                &c.next_random, vocab.get_vocab_ptr(), c.sentence_idx, c.indexes, c.codelens,
                c.codes, c.points, c.reduced_windows, c.subwords_idx_len, c.subwords_idx)

            for sent_idx in range(effective_sentences):
                idx_start = c.sentence_idx[sent_idx]
                idx_end = c.sentence_idx[sent_idx + 1]
                for i in range(idx_start, idx_end):
                    j = i - c.window + c.reduced_windows[i]
                    if j < idx_start:
                        j = idx_start
                    k = i + c.window + 1 - c.reduced_windows[i]
                    if k > idx_end:
                        k = idx_end
                    for j in range(j, k):
                        if j == i:
                            continue
                        if c.hs:
                            fasttext_fast_sentence_sg_hs(
                                c.points[j], c.codes[j], c.codelens[j], c.syn0_vocab, c.syn0_ngrams, c.syn1, c.size,
                                c.indexes[i], c.subwords_idx[i], c.subwords_idx_len[i], c.alpha, c.work, c.neu1,
                                c.word_locks_vocab, c.word_locks_ngrams)
                        if c.negative:
                            c.next_random = fasttext_fast_sentence_sg_neg(
                                c.negative, c.cum_table, c.cum_table_len, c.syn0_vocab, c.syn0_ngrams, c.syn1neg, c.size,
                                c.indexes[j], c.indexes[i], c.subwords_idx[i], c.subwords_idx_len[i], c.alpha, c.work,
                                c.neu1, c.next_random, c.word_locks_vocab, c.word_locks_ngrams)

            total_sentences += sentences.size()
            total_effective_words += effective_words

            c.alpha = get_next_alpha(start_alpha, end_alpha, total_sentences, total_words,
                                     expected_examples, expected_words, cur_epoch, num_epochs)

    return total_sentences, total_effective_words, total_words


def train_epoch_cbow(model, corpus_file, offset, _cython_vocab, _cur_epoch, _expected_examples, _expected_words, _work,
                     _neu1):
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

    Returns
    -------
    int
        Number of words in the vocabulary actually used for training (They already existed in the vocabulary
        and were not discarded by negative sampling).
    """
    cdef FastTextConfig c

    # For learning rate updates
    cdef int cur_epoch = _cur_epoch
    cdef int num_epochs = model.epochs
    cdef int expected_examples = (-1 if _expected_examples is None else _expected_examples)
    cdef long long expected_words = (-1 if _expected_words is None else _expected_words)
    cdef REAL_t start_alpha = model.alpha
    cdef REAL_t end_alpha = model.min_alpha
    cdef REAL_t _alpha = get_alpha(model.alpha, end_alpha, cur_epoch, num_epochs)

    cdef CythonLineSentence input_stream = CythonLineSentence(corpus_file, offset)
    cdef CythonVocab vocab = _cython_vocab

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int total_sentences = 0
    cdef long long total_effective_words = 0, total_words = 0
    cdef int sent_idx, idx_start, idx_end

    init_ft_config(&c, model, _alpha, _work, _neu1)

    # for preparing batches & training
    cdef vector[vector[string]] sentences

    with nogil:
        input_stream.reset()
        while not (input_stream.is_eof() or total_words > expected_words / c.workers):
            effective_sentences = 0
            effective_words = 0

            sentences = input_stream.next_batch()

            prepare_c_structures_for_batch(
                sentences, c.sample, c.hs, c.window, &total_words, &effective_words, &effective_sentences,
                &c.next_random, vocab.get_vocab_ptr(), c.sentence_idx, c.indexes, c.codelens,
                c.codes, c.points, c.reduced_windows, c.subwords_idx_len, c.subwords_idx)

            for sent_idx in range(effective_sentences):
                idx_start = c.sentence_idx[sent_idx]
                idx_end = c.sentence_idx[sent_idx + 1]
                for i in range(idx_start, idx_end):
                    j = i - c.window + c.reduced_windows[i]
                    if j < idx_start:
                        j = idx_start
                    k = i + c.window + 1 - c.reduced_windows[i]
                    if k > idx_end:
                        k = idx_end

                    if c.hs:
                        fasttext_fast_sentence_cbow_hs(
                            c.points[i], c.codes[i], c.codelens, c.neu1, c.syn0_vocab, c.syn0_ngrams, c.syn1, c.size,
                            c.indexes, c.subwords_idx, c.subwords_idx_len, c.alpha, c.work, i, j, k, c.cbow_mean,
                            c.word_locks_vocab, c.word_locks_ngrams)
                    if c.negative:
                        c.next_random = fasttext_fast_sentence_cbow_neg(
                            c.negative, c.cum_table, c.cum_table_len, c.codelens, c.neu1, c.syn0_vocab, c.syn0_ngrams,
                            c.syn1neg, c.size, c.indexes, c.subwords_idx, c.subwords_idx_len, c.alpha, c.work, i, j, k,
                            c.cbow_mean, c.next_random, c.word_locks_vocab, c.word_locks_ngrams)

            total_sentences += sentences.size()
            total_effective_words += effective_words

            c.alpha = get_next_alpha(start_alpha, end_alpha, total_sentences, total_words,
                                     expected_examples, expected_words, cur_epoch, num_epochs)

    return total_sentences, total_effective_words, total_words


CORPUSFILE_VERSION = 1
