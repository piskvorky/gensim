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

import sys
import os
from collections import defaultdict
import logging

from gensim import utils
from gensim.models.fast_counter_cython import FastCounterCython, FastCounterPreshed

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
    Fast counting of item frequency frequency across large, streamed iterables.
    """

    def __init__(self, doc2items=iter_gram1, max_size=None):
        self.doc2items = doc2items
        self.max_size = max_size
        self.min_reduce = 0
        self.hash2cnt = defaultdict(int)

    def hash(self, item):
        return hash(item)

    def update(self, documents):
        """
        Update the relevant ngram counters from the iterable `documents`.

        If the memory structures get too large, clip them (then the internal counts may be only approximate).
        """
        for document in documents:
            for item in self.doc2items(document):
                self.hash2cnt[self.hash(item)] += 1
            self.prune_items()

        return self  # for easier chaining

    def prune_items(self):
        """Trim data structures to fit in memory, if too large."""
        # XXX: Or use a fixed-size data structure to start with (hyperloglog?)
        while self.max_size and len(self) > self.max_size:
            self.min_reduce += 1
            utils.prune_vocab(self.hash2cnt, self.min_reduce)

    def get(self, item, default=None):
        """Return the item frequency of `item` (or `default` if item not present)."""
        return self.hash2cnt.get(self.hash(item), default)

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


class Phrases(object):
    def __init__(self, min_count=5, threshold=10.0, max_vocab_size=40000000):
        self.threshold = threshold
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        # self.counter = FastCounter(iter_gram12, max_size=max_vocab_size)
        self.counter = FastCounterCython()
        # self.counter = FastCounterPreshed()

    def add_documents(self, documents):
        self.counter.update(documents)

        return self  # for easier chaining

    def export_phrases(self, document):
        """
        Yield all collocations (pairs of adjacent closely related tokens) from the
        input `document`, as 2-tuples `(score, bigram)`.
        """
        norm = 1.0 * len(self.counter)
        for bigram in iter_gram2(document):
            pa, pb, pab = self.counter.get((bigram[0],)), self.counter.get((bigram[1],)), self.counter.get(bigram, 0)
            if pa and pb:
                score = norm / pa / pb * (pab - self.min_count)
                if score > self.threshold:
                    yield score, bigram


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s", " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    infile = sys.argv[1]

    from gensim.models.word2vec import Text8Corpus
    documents = Text8Corpus(infile)

    logger.info("training phrases")
    bigram = Phrases(min_count=5, threshold=100).add_documents(documents)
    logger.info("finished training phrases")
    print(bigram.counter)
    # for doc in documents:
    #     s = u' '.join(doc)
    #     for _, bigram in bigram.export_phrases(doc):
    #         s = s.replace(u' '.join(bigram), u'_'.join(bigram))
    #     print(utils.to_utf8(s))

    logger.info("finished running %s", " ".join(sys.argv))
