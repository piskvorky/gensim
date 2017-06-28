#!/usr/bin/env cython
# cython: boundscheck=False
# cython: cdivision=True
# coding: utf-8

# Author: Prakhar Pratyush (er.prakhar2b@gmail.com)

import sys
import os
import logging
import warnings

from gensim import utils
from collections import defaultdict

import cython
import numpy as np
cimport numpy as np

logger = logging.getLogger(__name__)

def learn_vocab(model, sentences):

    cdef np.uint32_t sentence_no = -1
    cdef np.uint32_t total_words = 0
    cdef np.uint32_t _progress_per = model.progress_per
    cdef np.uint32_t _max_vocab_size = model.max_vocab_size
    delimiter = model.delimiter

    logger.info("collecting all words and their counts")
    cdef vocab = defaultdict(int)

    cdef np.uint32_t min_reduce = -1

    for sentence_no, sentence in enumerate(sentences):
        if sentence_no % _progress_per == 0:
                logger.info(
                    "PROGRESS: at sentence #%i, processed %i words and %i word types",
                    sentence_no, total_words, len(vocab))

        if model.recode_to_utf8:
                sentence = [utils.any2utf8(w) for w in sentence]

        for bigram in zip(sentence, sentence[1:]):
            vocab[bigram[0]] += 1
            vocab[delimiter.join(bigram)] += 1
            total_words += 1

        if sentence:  # add last word skipped by previous loop
            word = sentence[-1]
            vocab[word] += 1

        if len(vocab) > _max_vocab_size:
            utils.prune_vocab(vocab, min_reduce)
            min_reduce += 1

    logger.info("collected %i word types from a corpus of %i words (unigram + bigrams) and %i sentences" %
                (len(vocab), total_words, sentence_no + 1))

    return min_reduce, vocab


