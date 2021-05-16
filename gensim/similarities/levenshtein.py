#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Vit Novotny <witiko@mail.muni.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module provides a namespace for functions that use the Levenshtein distance.
"""

import logging

from gensim.similarities.termsim import TermSimilarityIndex
from gensim.similarities.fastss import FastSS

logger = logging.getLogger(__name__)


class LevenshteinSimilarityIndex(TermSimilarityIndex):
    r"""
    Compute Levenshtein similarities between a static set of terms
    ("dictionary") and retrieve the most similar terms for any given
    query term.

    Notes
    -----
    This implementation uses a Directed Acyclic Word Graph (DAWG)
    for fast nearest-neighbor retrieval of the most similar terms.

    Parameters
    ----------
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        A dictionary that specifies the considered terms.
    alpha : float, optional
        Multiplicative factor `alpha` for Levenshtein similarity defined in [charletetal17]_.
    beta : float, optional
        The exponential factor `beta` for Levenshtein similarity defined in [charletetal17]_.
    max_distance : int, optional
        Do not consider terms with Levenshtein distance larger than this as
        "similar".  This is for performance reasons: keep this value below 3
        for reasonable retrieval performance. Default is 1.

    Attributes
    ----------
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        A dictionary that specifies the considered terms.
    alpha : float
        The multiplicative factor alpha defined by [charletetal17]_.
    beta : float
        The exponential factor beta defined by [charletetal17]_.
    index : :class:`lexpy.dawg.DAWG`
        The DAWG nearest-neighbor search index.
    max_distance : int
        The maximum Levenshtein distance of the most similar terms.
        Keeping this value below 3 has a significant impact on the
        retrieval performance.

    See Also
    --------
    :class:`~gensim.similarities.termsim.WordEmbeddingSimilarityIndex`
        Retrieve most similar terms for a given term using the cosine
        similarity over word embeddings.
    :class:`~gensim.similarities.termsim.SparseTermSimilarityMatrix`
        Build a term similarity matrix and compute the Soft Cosine Measure.

    References
    ----------
    The Levenshtein similarity, as a function of Levenshtein distance, was
    defined in the context of term similarity by [charletetal17]_.

    .. [charletetal17] Delphine Charlet and Geraldine Damnati, "SimBow at SemEval-2017 Task 3:
       Soft-Cosine Semantic Similarity between Questions for Community Question Answering", 2017,
       https://www.aclweb.org/anthology/S17-2051/.

    """
    def __init__(self, dictionary, alpha=1.8, beta=5.0, max_distance=1):
        self.dictionary = dictionary
        self.alpha = alpha
        self.beta = beta
        self.max_distance = max_distance

        self.index = FastSS(self.max_distance)
        for term in self.dictionary.values():
            self.index.add(term)

        super(LevenshteinSimilarityIndex, self).__init__()

    def _levsim(self, t1, t2, distance):
        max_lengths = max(len(t1), len(t2)) or 1
        similarity = self.alpha * (1.0 - distance * 1.0 / max_lengths)**self.beta
        return similarity

    def most_similar(self, t1, topn=10):
        most_similar = []
        if self.max_distance > 0:
            effective_topn = topn + 1 if t1 in self.dictionary.token2id else topn
            effective_topn = min(len(self.dictionary), effective_topn)
            for distance in range(1, self.max_distance + 1):
                terms = self.index.query(t1, distance)[distance]
                for t2 in terms:
                    if t1 == t2:
                        continue
                    similarity = self._levsim(t1, t2, distance)
                    most_similar.append((t2, similarity))
                if len(most_similar) >= effective_topn:
                    break
        most_similar = ((t2, similarity) for t2, similarity in most_similar if similarity > 0.0)
        most_similar = sorted(most_similar, key=lambda x: (-x[1], x[0]))
        return most_similar[:topn]
