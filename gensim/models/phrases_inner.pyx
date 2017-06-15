#!/usr/bin/env cython
# cython: boundscheck=False
# cython: cdivision=True
# coding: utf-8

# Author: Prakhar Pratyush (er.prakhar2b@gmail.com)

import cython
import numpy as np
cimport numpy as np
#ctypedef np.float32_t REAL_t

REAL = np.float32

import sys
import os
import logging
import warnings

from gensim import utils
from collections import defaultdict

logger = logging.getLogger(__name__)

@staticmethod
def learn_vocab(model, sentences, max_vocab_size, delimiter=b'_', progress_per=10000):
    logger.info("STARTED CYTHONIZING from .pyx file")

    cdef np.uint32_t sentence_no = -1
    cdef np.uint32_t total_words = 0
    cdef np.uint32_t _progress_per = progress_per
    cdef np.uint32_t _max_vocab_size = max_vocab_size

    logger.info("collecting all words and their counts")
    cdef vocab = defaultdict(int)
    
    cdef int min_reduce = -1

    cdef list sentence = []
    cdef np.uint32_t len_vocab = -1

    for sentence_no, sentence in enumerate(sentences):
        if sentence_no % _progress_per == 0:
            logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
                        (sentence_no, total_words, len(vocab)))
        #sentence = [utils.any2utf8(w) for w in sentence]
        for bigram in zip(sentence, sentence[1:]):
            vocab[bigram[0]] += 1
            vocab[delimiter.join(bigram)] += 1
            total_words += 1

        if sentence:  # add last word skipped by previous loop
            word = sentence[-1] 
            vocab[word] += 1

        len_vocab = len(vocab)

        if len_vocab > _max_vocab_size:
            utils.prune_vocab(vocab, min_reduce)
            min_reduce += 1

    logger.info("collected %i word types from a corpus of %i words (unigram + bigrams) and %i sentences" %
                (len_vocab, total_words, sentence_no + 1))

    return min_reduce, vocab


def export_phrases(model, sentences, out_delimiter=b' ', as_tuples=False):
    cdef list s = []
    cdef vocab = defaultdict(int)
    vocab = model.vocab

    cdef np.uint32_t threshold = model.threshold
    #cdef bool last_bigram = False
    delimiter = model.delimiter

    cdef np.uint32_t min_count = model.min_count

    cdef np.uint32_t len_vocab = -1

    cdef REAL_t score = -1

    cdef REAL_t pa = -1
    cdef REAL_t pb = -1
    cdef REAL_t pab = -1

    for s in sentences:
        #s = [utils.any2utf8(w) for w in sentence]
        last_bigram = False
        #vocab = self.vocab
        #threshold = self.threshold
        #delimiter = self.delimiter  # delimiter used for lookup
        #min_count = self.min_count

        for word_a, word_b in zip(s, s[1:]):
            if word_a in vocab and word_b in vocab:
                bigram_word = delimiter.join((word_a, word_b))
                if bigram_word in vocab and not last_bigram:
                    pa = float(vocab[word_a])
                    pb = float(vocab[word_b])
                    pab = float(vocab[bigram_word])
                    len_vocab = len(vocab)
                    score = (pab - min_count) / pa / pb * len_vocab
                    # logger.debug("score for %s: (pab=%s - min_count=%s) / pa=%s / pb=%s * vocab_size=%s = %s",
                    #     bigram_word, pab, self.min_count, pa, pb, len(self.vocab), score)
                    if score > threshold:
                        if as_tuples:
                            yield ((word_a, word_b), score)
                        else:
                            yield (out_delimiter.join((word_a, word_b)), score)
                        last_bigram = True
                        continue
                last_bigram = False
