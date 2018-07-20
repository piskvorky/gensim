#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: embedsignature=True
# coding: utf-8

"""General functions used for any2vec models."""

cpdef ft_hash(unicode string):
    """Calculate hash based on `string`.
    Reproduce `hash method from Facebook fastText implementation
    <https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc>`_.

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
