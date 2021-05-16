#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Vit Novotny <witiko@mail.muni.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module provides a namespace for functions that use the Levenshtein distance.
"""

import logging
from math import floor

from gensim.similarities import fastss

from gensim.similarities.termsim import TermSimilarityIndex

logger = logging.getLogger(__name__)


class LevenshteinSimilarityIndex(TermSimilarityIndex):
    """
    Computes Levenshtein similarities between terms and retrieves most similar
    terms for a given term.

    Notes
    -----
    This is a naive implementation that iteratively computes pointwise Levenshtein similarities
    between individual terms. Using this implementation to compute the similarity of all terms in
    real-world dictionaries such as the English Wikipedia will take years.

    Parameters
    ----------
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        A dictionary that specifies the considered terms.
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

    def __init__(self, dictionary, alpha=1.8, beta=5.0, threshold=0.0, max_distance=1):
        self.dictionary = dictionary
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

        self.index = fastss.FastSS(max_dist=max_distance)
        for word in self.dictionary.values():
            self.index.add(word)

        super(LevenshteinSimilarityIndex, self).__init__()

    def levsim(self, t1, t2, distance):
        """
        Turn Levenshtein distance into a similarity score.
        The similarity is a number between <0.0, 1.0>, higher means more similar.

        Notes
        -----
        This notion of Levenshtein similarity was first defined in section 2.2 of
        `Delphine Charlet and Geraldine Damnati, "SimBow at SemEval-2017 Task 3:
        Soft-Cosine Semantic Similarity between Questions for Community Question
        Answering", 2017 <http://www.aclweb.org/anthology/S/S17/S17-2051.pdf>`_.

        """
        max_lengths = max(len(t1), len(t2))
        if max_lengths == 0:
            return 1.0

        return self.alpha * (1 - distance * 1.0 / max_lengths)**self.beta

    def most_similar(self, t1, topn=10):
        result = []
        for error, terms in sorted(self.index.query(t1).items()):
            for t2 in terms:
                if t1 == t2:
                    continue

                similarity = self.levsim(t1, t2, distance=error)
                if similarity > 0:
                    result.append((t2, similarity))
        result.sort(key=lambda item: (-item[1], item[0]))
        return result[ : int(topn)]
