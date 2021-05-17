#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Code adapted from TinyFastSS (public domain), https://github.com/fujimotos/TinyFastSS

"""Create and query FastSS index for fast approximate string similarity search."""

import struct
import itertools
import logging

logger = logging.getLogger(__name__)


def editdist(s1, s2, maximum=None):
    """Return the Levenshtein distance between two strings, or maximum+1 if the distance is larger than `maximum`."""
    # TODO: rewrite in C; big impact on query performance!
    if s1 == s2:
        return 0

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    if maximum is None:
        maximum = len(s1)

    if len(s2) - len(s1) > maximum:
        return maximum + 1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        all_bad = i2 > maximum
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                val = distances[i1]
            else:
                val = 1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
            distances_.append(val)
            if all_bad and val <= maximum:
                all_bad = False
        if all_bad:
            return maximum + 1
        distances = distances_
    return distances[-1]


def indexkeys(word, max_dist):
    """Return the set of index keys ("variants") of a word.

    >>> indexkeys('aiu', 1)
    {'aiu', 'iu', 'au', 'ai'}
    """
    res = set()
    wordlen = len(word)
    limit = min(max_dist, wordlen) + 1

    for dist in range(limit):
        variants = itertools.combinations(word, wordlen - dist)

        for variant in variants:
            res.add(''.join(variant))

    return res


def int2byte(i):
    """Encode an int <0, 255> into an 8-bit unsigned byte.

    >>> int2byte(1)
    b'\x01'
    """
    return struct.pack('B', i)


def byte2int(b):
    """Decode an 8-bit unsigned byte into an int.

    >>> byte2int(b'\x01')
    1
    """
    return struct.unpack('B', b)[0]


def set2bytes(s):
    """Serialize a set of unicode strings into bytes.

    >>> set2byte({u'a', u'b', u'c'})
    b'a\x00b\x00c'
    """
    return '\x00'.join(s).encode('utf8')


def bytes2set(b):
    """Deserialize bytes into a set of unicode strings.

    >>> int2byte(b'a\x00b\x00c')
    {u'a', u'b', u'c'}
    """
    if not b:
        return set()

    return set(b.decode('utf8').split('\x00'))


class FastSS:

    def __init__(self, words=None, max_dist=2):
        """
        Create a FastSS index. The index will contain encoded variants of all
        indexed words.

        max_dist: maximum allowed edit distance of an indexed word to a query word. Keep
        max_dist<=3 for sane performance.

        """
        self.db = {}
        self.max_dist = max_dist
        if words:
            for word in words:
                self.add(word)

    def __str__(self):
        return "%s<max_dist=%s, db_size=%i>" % (self.__class__.__name__, self.max_dist, len(self.db), )

    def __contains__(self, word):
        bkey = word.encode('utf8')
        if bkey in self.db:
            return word in bytes2set(self.db[bkey])
        return False

    def add(self, word):
        """Add a string to the index."""
        for key in indexkeys(word, self.max_dist):
            bkey = key.encode('utf8')
            wordset = {word}

            if bkey in self.db:
                wordset |= bytes2set(self.db[bkey])

            self.db[bkey] = set2bytes(wordset)

    def query(self, word, max_dist=None):
        """Find all words from the index that are within max_dist of `word`."""
        if max_dist is None:
            max_dist = self.max_dist
        if max_dist > self.max_dist:
            raise ValueError(
                f"query max_dist={max_dist} cannot be greater than "
                f"max_dist={self.max_dist} specified in the constructor"
            )

        res = {d: [] for d in range(max_dist + 1)}
        cands = set()

        for key in indexkeys(word, max_dist):
            bkey = key.encode('utf8')

            if bkey in self.db:
                cands.update(bytes2set(self.db[bkey]))

        for cand in cands:
            dist = editdist(word, cand, max_dist)
            if dist <= max_dist:
                res[dist].append(cand)

        return res
