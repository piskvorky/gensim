#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Vit Novotny <witiko@mail.muni.cz>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""
This module allows fast fuzzy search between strings, using kNN queries with Levenshtein similarity.
"""

import logging

from gensim.similarities.termsim import TermSimilarityIndex
from gensim import utils
try:
    from gensim.similarities.fastss import FastSS, editdist  # noqa:F401
except ImportError:
    raise utils.NO_CYTHON


logger = logging.getLogger(__name__)


class LevenshteinSimilarityIndex(TermSimilarityIndex):
    r"""
    Retrieve the most similar terms from a static set of terms ("dictionary")
    given a query term, using Levenshtein similarity.

    "Levenshtein similarity" is a modification of the Levenshtein (edit) distance,
    defined in [charletetal17]_.

    This implementation uses the :class:`~gensim.similarities.fastss.FastSS` algorithm
    for fast kNN nearest-neighbor retrieval.

    Parameters
    ----------
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        A dictionary that specifies the considered terms.
    alpha : float, optional
        Multiplicative factor `alpha` for the Levenshtein similarity. See [charletetal17]_.
    beta : float, optional
        The exponential factor `beta` for the Levenshtein similarity. See [charletetal17]_.
    max_distance : int, optional
        Do not consider terms with Levenshtein distance larger than this as
        "similar". This is done for performance reasons: keep this value below 3
        for reasonable retrieval performance. Default is 1.

    See Also
    --------
    :class:`~gensim.similarities.termsim.WordEmbeddingSimilarityIndex`
        Retrieve most similar terms for a given term using the cosine
        similarity over word embeddings.
    :class:`~gensim.similarities.termsim.SparseTermSimilarityMatrix`
        Build a term similarity matrix and compute the Soft Cosine Measure.

    References
    ----------

    .. [charletetal17] Delphine Charlet and Geraldine Damnati, "SimBow at SemEval-2017 Task 3:
       Soft-Cosine Semantic Similarity between Questions for Community Question Answering", 2017,
       https://www.aclweb.org/anthology/S17-2051/.

    """
    def __init__(self, dictionary, alpha=1.8, beta=5.0, max_distance=2):
        self.dictionary = dictionary
        self.alpha = alpha
        self.beta = beta
        self.max_distance = max_distance
        logger.info("creating FastSS index from %s", dictionary)
        self.index = FastSS(words=self.dictionary.values(), max_dist=max_distance)
        super(LevenshteinSimilarityIndex, self).__init__()

    def levsim(self, t1, t2, distance):
        """Calculate the Levenshtein similarity between two terms given their Levenshtein distance."""
        max_lengths = max(len(t1), len(t2)) or 1
        return self.alpha * (1.0 - distance * 1.0 / max_lengths)**self.beta

    def most_similar(self, t1, topn=10):
        """kNN fuzzy search: find the `topn` most similar terms from `self.dictionary` to `t1`."""
        result = {}  # map of {dictionary term => its levenshtein similarity to t1}
        if self.max_distance > 0:
            effective_topn = topn + 1 if t1 in self.dictionary.token2id else topn
            effective_topn = min(len(self.dictionary), effective_topn)

            # Implement a "distance backoff" algorithm:
            # Start with max_distance=1, for performance. And if that doesn't return enough results,
            # continue with max_distance=2 etc, all the way until self.max_distance which
            # is a hard cutoff.
            # At that point stop searching, even if we don't have topn results yet.
            #
            # We use the backoff algo to speed up queries for short terms. These return enough results already
            # with max_distance=1.
            #
            # See the discussion at https://github.com/RaRe-Technologies/gensim/pull/3146
            for distance in range(1, self.max_distance + 1):
                for t2 in self.index.query(t1, distance).get(distance, []):
                    if t1 == t2:
                        continue
                    similarity = self.levsim(t1, t2, distance)
                    if similarity > 0:
                        result[t2] = similarity
                if len(result) >= effective_topn:
                    break

        return sorted(result.items(), key=lambda x: (-x[1], x[0]))[:topn]
