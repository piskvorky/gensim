#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Code adapted from TinyFastSS (public domain), https://github.com/fujimotos/TinyFastSS

"""Create and query FastSS index for fast approximate string similarity search."""

import struct
import itertools

ENCODING = 'utf-8'
DELIMITER = b'\x00'


def editdist(s1, s2):
    """Return the Levenshtein distance between two strings.

    >>> editdist('aiu', 'aie')
    1
    """
    matrix = {}

    for i in range(len(s1)+1):
        matrix[(i, 0)] = i
    for j in range(len(s2)+1):
        matrix[(0, j)] = j

    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            if s1[i-1] == s2[j-1]:
                matrix[(i, j)] = matrix[(i-1, j-1)]
            else:
                matrix[(i, j)] = min(
                    matrix[(i-1, j)],
                    matrix[(i, j-1)],
                    matrix[(i-1, j-1)]
                ) + 1

    return matrix[(i, j)]


def indexkeys(word, max_dist):
    """Return the set of index keys ("variants") of a word.

    >>> indexkeys('aiu', 1)
    {'aiu', 'iu', 'au', 'ai'}
    """
    res = set()
    wordlen = len(word)
    limit = min(max_dist, wordlen) + 1

    for dist in range(limit):
        variants = itertools.combinations(word, wordlen-dist)

        for variant in variants:
            res.add(''.join(variant))

    return res


def int2byte(i):
    """Encode a positive int (<= 256) into a 8-bit byte.

    >>> int2byte(1)
    b'\x01'
    """
    return struct.pack('B', i)


def byte2int(b):
    """Decode a 8-bit byte into an integer.

    >>> byte2int(b'\x01')
    1
    """
    return struct.unpack('B', b)[0]


def set2bytes(s):
    """Serialize a set of unicode strings into bytes.

    >>> set2byte({u'a', u'b', u'c')
    b'a\x00b\x00c'
    """
    lis = []
    for uword in sorted(s):
        bword = uword.encode(ENCODING)
        lis.append(bword)
    return DELIMITER.join(lis)


def bytes2set(b):
    """Deserialize bytes into a set of unicode strings.

    >>> int2byte(b'a\x00b\x00c')
    {u'a', u'b', u'c'}
    """
    if not b:
        return set()

    lis = b.split(DELIMITER)
    return set(bword.decode(ENCODING) for bword in lis)


class FastSS:
    """Open a FastSS index."""

    def __init__(self, max_dist=2):
        """max_dist: the upper threshold of edit distance of works from the index."""
        self.db = {}
        self.max_dist = max_dist

    def __str__(self):
        return "%s<max_dist=%s, db_size=%i>" % (self.__class__.__name__, self.max_dist, len(self.db), )

    def __contains__(self, word):
        bkey = word.encode(ENCODING)
        if bkey in self.db:
            return word in bytes2set(self.db[bkey])
        return False

    def add(self, word):
        """Add a string to the index."""
        for key in indexkeys(word, self.max_dist):
            bkey = key.encode(ENCODING)
            wordset = {word}

            if bkey in self.db:
                wordset |= bytes2set(self.db[bkey])

            self.db[bkey] = set2bytes(wordset)

    def query(self, word, max_dist=None):
        """Find all words from the index that are within max_dist of `word`."""
        if max_dist is None:
            max_dist = self.max_dist
        if max_dist > self.max_dist:
            raise ValueError("query max_dist cannot be greater than max_dist specified in the constructor")

        res = {d: [] for d in range(max_dist+1)}
        cands = set()

        for key in indexkeys(word, max_dist):
            bkey = key.encode(ENCODING)

            if bkey in self.db:
                cands.update(bytes2set(self.db[bkey]))

        for cand in cands:
            dist = editdist(word, cand)
            if dist <= max_dist:
                res[dist].append(cand)

        return res
