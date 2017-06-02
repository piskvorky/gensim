#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

# Author: Prakhar Pratyush (er.prakhar2b@gmail.com)

import cython
import numpy as np
cimport numpy as np

import sys
import os
import logging
import warnings

from gensim import utils
from collections import defaultdict

logger = logging.getLogger(__name__)

def learn_vocab(sentences, max_vocab_size, delimiter=b'_', progress_per=10000):
    logger.info("STARTED CYTHONIZING from .pyx")

    cdef int sentence_no = -1
    cdef int total_words = 0

    logger.info("collecting all words and their counts")
    vocab = defaultdict(int)

    cdef int min_reduce = 1
    #cdef int l = -1

    for sentence_no, sentence in enumerate(sentences):
        if sentence_no % progress_per == 0:
            logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
                        (sentence_no, total_words, len(vocab)))
        sentence = [utils.any2utf8(w) for w in sentence]
        for bigram in zip(sentence, sentence[1:]):
            vocab[bigram[0]] += 1
            vocab[delimiter.join(bigram)] += 1
            total_words += 1

        if sentence:  # add last word skipped by previous loop
            #l=len(sentences)
            word = sentence[-1]   # error about negative index
            vocab[word] += 1

        if len(vocab) > max_vocab_size:
            utils.prune_vocab(vocab, min_reduce)
            min_reduce += 1

    logger.info("collected %i word types from a corpus of %i words (unigram + bigrams) and %i sentences" %
                (len(vocab), total_words, sentence_no + 1))
    return min_reduce, vocab
