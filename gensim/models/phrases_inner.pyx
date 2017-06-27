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
from gensim.models import Phrases
from collections import defaultdict

import cython
import numpy as np

logger = logging.getLogger(__name__)

cdef bytes any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')

class Phrases_inner(Phrases):

    


    def learn_vocab(self, sentences, max_vocab_size, delimiter=b'_', progress_per=10000):

        cdef np.uint32_t sentence_no = -1
        cdef np.uint32_t total_words = 0
        cdef np.uint32_t _progress_per = progress_per
        cdef np.uint32_t _max_vocab_size = max_vocab_size

        logger.info("collecting all words and their counts")
        cdef vocab = defaultdict(int)

        cdef np.uint32_t min_reduce = -1

        cdef bytes w
        cdef np.uint32_t len_s = -1

        for sentence_no, sentence in enumerate(sentences):
            len_s = len(sentence)
            if sentence_no % _progress_per == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
                            (sentence_no, total_words, len(vocab)))

            if sentence and isinstance(sentence[0], bytes):
                sentence = [w for w in (any2utf8(b'_;_'.join(sentence)).split(b'_;_'))]
            else:
                sentence = [w for w in (any2utf8(u'_;_'.join(sentence)).split(b'_;_'))]

            assert len_s == len(sentence), 'mismatch between number of tokens after utf8 conversion'

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


