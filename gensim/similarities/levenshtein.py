#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Vit Novotny <witiko@mail.muni.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module provides a namespace for functions that use the Levenshtein distance.
"""

from heapq import heappush, heappop
import logging
from multiprocessing import Pool

from numpy import float32 as REAL

# If python-Levenshtein is available, import it.
# If python-Levenshtein is unavailable, ImportError will be raised in levsim.
try:
    from Levenshtein import distance
    LEVENSHTEIN_EXT = True
except ImportError:
    LEVENSHTEIN_EXT = False

from gensim.similarities.termsim import TermSimilarityIndex, SparseTermSimilarityMatrix
from gensim.utils import deprecated

logger = logging.getLogger(__name__)


def levsim(t1, t2, alpha=1.8, beta=5.0):
    """Get the Levenshtein similarity between two terms.

    Return the Levenshtein similarity between two terms. The similarity is a
    number between <0.0, 1.0>, higher is more similar.

    Parameters
    ----------
    t1 : {bytes, str, unicode}
        The first compared term.
    t2 : {bytes, str, unicode}
        The second compared term.
    alpha : float
        The multiplicative factor alpha defined by Charlet and Damnati (2017).
    beta : float
        The exponential factor beta defined by Charlet and Damnati (2017).

    Returns
    -------
    float
        The Levenshtein similarity between `t1` and `t2`.

    Notes
    -----
    This notion of Levenshtein similarity was first defined in section 2.2 of
    `Delphine Charlet and Geraldine Damnati, "SimBow at SemEval-2017 Task 3:
    Soft-Cosine Semantic Similarity between Questions for Community Question
    Answering", 2017 <http://www.aclweb.org/anthology/S/S17/S17-2051.pdf>`__.

    """
    if not LEVENSHTEIN_EXT:
        raise ImportError("Please install python-Levenshtein Python package to compute the Levenshtein distance.")
    return alpha * (1 - distance(t1, t2) * 1.0 / max(len(t1), len(t2)))**beta


def _levsim_worker(args):
    _, t2, _, _ = args
    return (t2, levsim(*args))


class LevenshteinSimilarityIndex(TermSimilarityIndex):
    """
    Computes Levenshtein similarities between terms and retrieves most similar
    terms for a given term.

    Parameters
    ----------
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        A dictionary that specifies the considered terms.
    workers : int, optional
        The number of workers to use when computing the Levenshtein similarities.
    alpha : float, optional
        The multiplicative factor alpha defined by Charlet and Damnati (2017).
    beta : float, optional
        The exponential factor beta defined by Charlet and Damnati (2017).
    threshold : float, optional
        Only terms more similar than `threshold` are considered when retrieving
        the most similar terms for a given term.

    See Also
    --------
    :func:`gensim.similarities.levenshtein.levsim`
        The Levenshtein similarity.
    :class:`~gensim.similarities.termsim.SparseTermSimilarityMatrix`
        Build a term similarity matrix and compute the Soft Cosine Measure.

    """
    CHUNK_SIZE = 5000  # the number of similarities a single worker computes in one batch

    def __init__(self, dictionary, workers=1, alpha=1.8, beta=5.0, threshold=0.0):
        self.dictionary = dictionary
        self.pool = Pool(workers)
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        super(LevenshteinSimilarityIndex, self).__init__()

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def most_similar(self, t1, topn=10):
        heap = []
        for t2, similarity in self.pool.imap_unordered(
                _levsim_worker, (
                    (t1, t2, self.alpha, self.beta)
                    for t2_index, t2 in self.dictionary.items()
                    if t1 != t2),
                self.CHUNK_SIZE):
            heappush(heap, (-similarity, t2))
        for _, (t2, similarity) in zip(
                range(topn),
                (
                    (t2, -_similarity) for _similarity, t2 in (
                        heappop(heap) for _ in range(len(heap)))
                    if similarity > self.threshold)):
            yield (t2, similarity)
