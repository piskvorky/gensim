#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from collections import Counter

cimport preshed.counter


class FastCounterCython(object):
    """
    Fast counting of item frequency frequency across large, streamed iterables.
    """

    def __init__(self, doc2items=None, max_size=None):
        self.doc2items = doc2items
        self.max_size = max_size
        self.min_reduce = 0
        self.hash2cnt = Counter()  # TODO replace by some GIL-free low-level struct

    def update(self, documents):
        """
        Update the relevant ngram counters from the iterable `documents`.

        If the memory structures get too large, clip them (then the internal counts may be only approximate).
        """
        hash2cnt = self.hash2cnt
        for document in documents:
            # TODO: release GIL, so we can run update() in parallel threads.
            # Or maybe not needed, if we create multiple FastCounters from multiple input streams using
            # multiprocessing, and only .merge() them at the end.
            if document:
                hash2cnt[hash(document[0])] += 1
                for idx in range(len(document) - 1):
                    hash2cnt[hash(document[idx + 1])] += 1
                    hash2cnt[hash((document[idx], document[idx + 1]))] += 1

            # FIXME: add optimized prune

        return self  # for easier chaining

    def prune_items(self):
        """Trim data structures to fit in memory, if too large."""
        # XXX: Or use a fixed-size data structure to start with (hyperloglog?)
        pass

    def get(self, item, default=None):
        """Return the item frequency of `item` (or `default` if item not present)."""
        return self.hash2cnt.get(hash(item), default)

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
