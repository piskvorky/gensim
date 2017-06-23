#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from collections import Counter

from libc.stdint cimport int64_t, uint64_t

cimport preshed.counter


cdef uint64_t chash(obj):
    # TODO use something faster, can assume string
    return <uint64_t>hash(obj)


class FastCounterCython(object):
    """
    Fast counting of item frequency frequency across large, streamed iterables.
    """

    def __init__(self, doc2items=None, max_size=None):
        self.doc2items = doc2items
        self.max_size = max_size
        self.min_reduce = 0
        self.hash2cnt = Counter()

    def update(self, documents):
        """
        Update the relevant ngram counters from the iterable `documents`.

        If the memory structures get too large, clip them (then the internal counts may be only approximate).
        """
        cdef int idx, l
        cdef uint64_t h1, h2
        hash2cnt = self.hash2cnt
        for document in documents:
            l = len(document)
            if l:
                h1 = chash(document[0])
                hash2cnt[h1] += 1
                for idx in range(1, l):
                    h2 = chash(document[idx])
                    hash2cnt[h2] += 1
                    hash2cnt[h1 ^ h2] += 1
                    h1 = h2

            # FIXME: add optimized prune

        return self  # for easier chaining

    def prune_items(self):
        """Trim data structures to fit in memory, if too large."""
        # XXX: Or use a fixed-size data structure to start with (hyperloglog?)
        pass

    def get(self, item, default=None):
        """Return the item frequency of `item` (or `default` if item not present)."""
        return self.hash2cnt.get(chash(item), default)

    def merge(self, other):
        """
        Merge counts from another FastCounter into self, in-place.
        """
        self.hash2cnt.update(other.hash2cnt)
        self.min_reduce = max(self.min_reduce, other.min_reduce)
        self.prune_items()

    def __len__(self):
        return len(self.hash2cnt)

    def __str__(self):
        return "%s<%i items>" % (self.__class__.__name__, len(self))


class FastCounterPreshed(object):
    """
    Fast counting of item frequency frequency across large, streamed iterables.
    """

    def __init__(self, doc2items=None, max_size=None):
        self.doc2items = doc2items
        self.max_size = max_size
        self.min_reduce = 0
        self.hash2cnt = preshed.counter.PreshCounter()  # TODO replace by some GIL-free low-level struct

    def update(self, documents):
        """
        Update the relevant ngram counters from the iterable `documents`.

        If the memory structures get too large, clip them (then the internal counts may be only approximate).
        """
        cdef int idx, l
        cdef uint64_t h1, h2
        cdef preshed.counter.PreshCounter hash2cnt = self.hash2cnt
        for document in documents:
            l = len(document)
            if l:
                h1 = chash(document[0])
                hash2cnt.inc(h1, 1)
                for idx in range(1, l):
                    h2 = chash(document[idx])
                    hash2cnt.inc(h2, 1)
                    hash2cnt.inc(h1 ^ h2, 1)
                    h1 = h2

            # FIXME: add optimized prune

        return self  # for easier chaining

    def prune_items(self):
        """Trim data structures to fit in memory, if too large."""
        # XXX: Or use a fixed-size data structure to start with (hyperloglog?)
        pass

    def get(self, item, default=None):
        """Return the item frequency of `item` (or `default` if item not present)."""
        return self.hash2cnt.get(chash(item), default)

    def merge(self, other):
        """
        Merge counts from another FastCounter into self, in-place.
        """
        self.hash2cnt.update(other.hash2cnt)
        self.min_reduce = max(self.min_reduce, other.min_reduce)
        self.prune_items()

    def __len__(self):
        return len(self.hash2cnt)

    def __str__(self):
        return "%s<%i items>" % (self.__class__.__name__, len(self))
