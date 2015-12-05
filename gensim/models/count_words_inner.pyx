#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Contributed by Matthew Honnibal <matt@spacy.io>
# Copyright (C) 2015 ceded to Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


from cpython cimport PyUnicode_AS_DATA
from cpython cimport PyUnicode_GET_DATA_SIZE
from libc.stdint cimport uint64_t

from murmurhash.mrmr cimport hash64
from preshed.counter cimport PreshCounter

from collections import defaultdict
from six import iteritems


cpdef uint64_t _hash_string(unicode string) except 0:
    # TODO: Is it acceptable to have the argument strictly typed?? It's not, right?
    # That seems against the way Gensim is set out --- the library seems to
    # follow an "input agnostic", duck-typey sort of design, right?
    #
    # This code is copied from spacy.strings. The implementation took some thought,
    # and consultation with Stefan Behnel. Do not change blindly. Interaction
    # with Python 2/3 is subtle.
    chars = <char*>PyUnicode_AS_DATA(string)
    size = PyUnicode_GET_DATA_SIZE(string)
    return hash64(chars, size, 1)


def count_words_fast(sentences, int min_freq, int progress_per, log_progress):
    cdef PreshCounter counts = PreshCounter()
    strings = {}
    sentence_no = -1
    total_words = 0
    cdef unicode word
    cdef uint64_t key
    for sentence_no, sentence in enumerate(sentences):
        if sentence_no % progress_per == 0:
            log_progress(sentence_no, total_words, len(strings))
        
        for word in sentence:
            key = _hash_string(word)
            counts.inc(key, 1)
            # TODO: Why doesn't .inc return this? =/
            count = counts[key]
            # Remember the string when we exceed min count
            if count == min_freq:
                 strings[key] = word
        total_words += len(sentence)
    
    # Use defaultdict to match the pure Python version of the function
    vocab = defaultdict(int)
    for key, word in iteritems(strings):
        vocab[word] = counts[key]
    return vocab, sentence_no

