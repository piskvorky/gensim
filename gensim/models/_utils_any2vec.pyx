#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

"""General functions used for any2vec models."""

#
# This is here to support older versions of the MSVC compiler that don't have stdint.h.
#
cdef extern from "stdint_wrapper.h":
    ctypedef unsigned int uint32_t
    ctypedef signed char int8_t

from six import PY2
import numpy as np
cimport numpy as np


cpdef ft_hash_bytes(bytes bytez):
    """Calculate hash based on `bytez`.
    Reproduce `hash method from Facebook fastText implementation
    <https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc>`_.

    Parameters
    ----------
    bytez : bytes
        The string whose hash needs to be calculated, encoded as UTF-8.

    Returns
    -------
    unsigned int
        The hash of the string.

    """
    cdef uint32_t h = 2166136261
    cdef char b

    for b in bytez:
        h = h ^ <uint32_t>(<int8_t>b)
        h = h * 16777619
    return h


cpdef ft_hash_broken(unicode string):
    """Calculate hash based on `string`.

    This implementation is broken, see https://github.com/RaRe-Technologies/gensim/issues/2059.
    It is here only for maintaining backwards compatibility with older models.

    Parameters
    ----------
    string : unicode
        The string whose hash needs to be calculated.

    Returns
    -------
    unsigned int
        The hash of the string.

    """
    cdef unsigned int h = 2166136261
    for c in string:
        h ^= ord(c)
        h *= 16777619
    return h


cpdef compute_ngrams(word, unsigned int min_n, unsigned int max_n):
    """Get the list of all possible ngrams for a given word.

    Parameters
    ----------
    word : str
        The word whose ngrams need to be computed.
    min_n : unsigned int
        Minimum character length of the ngrams.
    max_n : unsigned int
        Maximum character length of the ngrams.

    Returns
    -------
    list of str
        Sequence of character ngrams.

    """
    cdef unicode extended_word = f'<{word}>'
    ngrams = []
    for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
        for i in range(0, len(extended_word) - ngram_length + 1):
            ngrams.append(extended_word[i:i + ngram_length])
    return ngrams

#
# UTF-8 bytes that begin with 10 are subsequent bytes of a multi-byte sequence,
# as opposed to a new character.
#
cdef unsigned char _MB_MASK = 0xC0
cdef unsigned char _MB_START = 0x80


cpdef compute_ngrams_bytes(word, unsigned int min_n, unsigned int max_n):
    """Computes ngrams for a word.

    Ported from the original FB implementation.

    Parameters
    ----------
    word : str
        A unicode string.
    min_n : unsigned int
        The minimum ngram length.
    max_n : unsigned int
        The maximum ngram length.

    Returns:
    --------
    list of str
        A list of ngrams, where each ngram is a list of **bytes**.

    See Also
    --------
    `Original implementation <https://github.com/facebookresearch/fastText/blob/7842495a4d64c7a3bb4339d45d6e64321d002ed8/src/dictionary.cc#L172>`__

    """
    cdef bytes utf8_word = ('<%s>' % word).encode("utf-8")
    cdef const unsigned char *bytez = utf8_word
    cdef size_t num_bytes = len(utf8_word)
    cdef size_t j, i, n

    ngrams = []
    for i in range(num_bytes):
        if bytez[i] & _MB_MASK == _MB_START:
            continue

        j, n = i, 1
        while j < num_bytes and n <= max_n:
            j += 1
            while j < num_bytes and (bytez[j] & _MB_MASK) == _MB_START:
                j += 1
            if n >= min_n and not (n == 1 and (i == 0 or j == num_bytes)):
                ngram = bytes(bytez[i:j])
                ngrams.append(ngram)
            n += 1
    return ngrams
