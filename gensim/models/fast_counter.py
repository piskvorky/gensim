#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Fast & memory efficient counting of things (and n-grams of things).

This module is designed to count item frequencies over large, streamed corpora (lazy iteration).

Such counts are useful in various other modules, such as Dictionary, TfIdf, Phrases etc.

"""

from collections import Counter
import logging

from six import iterkeys, iteritems

logger = logging.getLogger(__name__)


def iter_ngrams(document, ngrams):
    assert ngrams[0] <= ngrams[1]

    for n in range(ngrams[0], ngrams[1] + 1):
        for ngram in zip(*[document[i:] for i in range(n)]):
            logger.debug("yielding ngram %r", ngram)
            yield ngram

def iter_gram1(document):
    return iter_ngrams(document, (1, 1))

def iter_gram2(document):
    return iter_ngrams(document, (2, 2))

def iter_gram12(document):
    return iter_ngrams(document, (1, 2))


class FastCounter(object):
    """
    Fast counting of item frequency and document frequency across large, streamed iterables.
    """

    def __init__(self, doc2items=iter_gram1):
        self.doc2items = doc2items
        self.hash2cnt = Counter()  # TODO replace by some GIL-free low-level struct

    def hash(self, item):
        return hash(item)

    def update(self, documents):
        """
        Update the relevant ngram counters from the iterable `documents`.

        If the memory structures get too large, clip them (then the internal counts may be only approximate).
        """
        for document in documents:
            # TODO: release GIL, so we can run update() in parallel threads.
            # Or maybe not needed, if we create multiple FastCounters from multiple input streams using
            # multiprocessing, and only .merge() them at the end.
            self.hash2cnt.update(self.hash(ngram) for ngram in self.doc2items(document))

            # self.prune_vocab()

        return self  # for easier chaining

    def prune_vocab(self):
        # Trim data structures to fit in memory, if too large.
        # Or use a fixed-size data structure to start with (hyperloglog?)
        raise NotImplementedError

    def get(self, item, default=None):
        """Return the item frequency of `item` (or `default` if item not present)."""
        return self.hash2cnt.get(self.hash(item), default)

    def merge(self, other):
        """
        Merge counts from other into self, in-place.
        """
        # rare operation, no need to optimize too much
        raise NotImplementedError

    def __len__(self):
        return len(self.hash2cnt)

    def __str__(self):
        return "%s<%i items>" % (self.__class__.__name__, len(self))


class Phrases(object):
    def __init__(self, min_count=5, threshold=10.0, max_vocab_size=40000000):
        self.threshold = threshold
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.counter = FastCounter(iter_gram12)

    def add_documents(self, documents):
        self.counter.update(documents)

        return self  # for easier chaining

    def export_phrases(self, document):
        """
        Yield all collocations (pairs of adjacent closely related tokens) from the
        input `document`, as 2-tuples `(score, bigram)`.
        """
        if not self.counter:
            return
        norm = 1.0 * len(self.counter)
        for bigram in iter_gram2(document):
            pa, pb, pab = self.counter.get((bigram[0],)), self.counter.get((bigram[1],)), self.counter.get(bigram, 0)
            if pa and pb:
                score = norm / pa / pb * (pab - self.min_count)
                if score > self.threshold:
                    yield score, bigram
