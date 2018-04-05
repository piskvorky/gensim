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

from Levenshtein import distance
from numpy import float32 as REAL

from gensim.models.term_similarity import TermSimilarityIndex, SparseTermSimilarityMatrix
from gensim.utils import deprecated

logger = logging.getLogger(__name__)


def levsim(alpha, beta, t1, t2):
    """Get the Levenshtein similarity between two terms.

    Return the Levenshtein similarity between two terms. The similarity is a
    number between <0.0, 1.0>, higher is more similar.

    Parameters
    ----------
    alpha : float
        The multiplicative factor alpha defined by Charlet and Damnati (2017).
    beta : float
        The exponential factor beta defined by Charlet and Damnati (2017).
    t1 : {bytes, str, unicode}
        The first compared term.
    t2 : {bytes, str, unicode}
        The second compared term.

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
    return alpha * (1 - distance(t1, t2) * 1.0 / max(len(t1), len(t2)))**beta


def _levsim_worker(args):
    _, _, _, t2 = args
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
    :func:`gensim.models.levenshtein.levsim`
        The Levenshtein similarity.
    :class:`~gensim.models.term_similarity.SparseTermSimilarityMatrix`
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
                    (self.alpha, self.beta, t1, self.dictionary[t2_index])
                    for t2_index in range(len(self.dictionary))
                    if t1 != self.dictionary[t2_index]),
                self.CHUNK_SIZE):
            heappush(heap, (-similarity, t2))
        for _, (t2, similarity) in zip(
                range(topn),
                (
                    (t2, -_similarity) for _similarity, t2 in (
                        heappop(heap) for _ in range(len(heap)))
                    if similarity > self.threshold)):
            yield (t2, similarity)


@deprecated(
    "Function will be deprecated in 4.0.0, use " +
    "gensim.models.levenshtein.LevenshteinSimilarityIndex instead")
def similarity_matrix(dictionary, tfidf=None, threshold=0.0, alpha=1.8, beta=5.0,
                      nonzero_limit=100, workers=1, dtype=REAL):
    """Constructs a term similarity matrix for computing Soft Cosine Measure.
    Constructs a sparse term similarity matrix in the :class:`scipy.sparse.csc_matrix` format
    for computing Soft Cosine Measure between documents.
    Parameters
    ----------
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        A dictionary that specifies a mapping between terms and the indices of rows and columns
        of the resulting term similarity matrix.
    tfidf : :class:`gensim.models.tfidfmodel.TfidfModel` or None, optional
        A model that specifies the relative importance of the terms in the dictionary. The columns
        of the term similarity matrix will be build in a decreasing order of importance of
        terms, or in the order of term identifiers if None.
    threshold : float, optional
        Only terms more similar than `threshold` are considered when retrieving
        the most similar terms for a given term.
    alpha : float, optional
        The multiplicative factor alpha defined by Charlet and Damnati (2017).
    beta : float, optional
        The exponential factor beta defined by Charlet and Damnati (2017).
    nonzero_limit : int, optional
        The maximum number of non-zero elements outside the diagonal in a single column of the
        sparse term similarity matrix.
    workers : int, optional
        The number of workers to use when computing the Levenshtein similarities.
    dtype : numpy.dtype, optional
        Data-type of the sparse term similarity matrix.

    Returns
    -------
    :class:`scipy.sparse.csc_matrix`
        Term similarity matrix.
    See Also
    --------
    :func:`gensim.matutils.softcossim`
        The Soft Cosine Measure.
    :class:`gensim.similarities.docsim.SoftCosineSimilarity`
        A class for performing corpus-based similarity queries with Soft Cosine Measure.
    :meth:`gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similarity_matrix`
        A term similarity matrix produced from term embeddings.

    Notes
    -----
    The constructed matrix corresponds to the matrix Mlev defined in section 2.2 of
    `Delphine Charlet and Geraldine Damnati, "SimBow at SemEval-2017 Task 3: Soft-Cosine
    Semantic Similarity between Questions for Community Question Answering", 2017
    <http://www.aclweb.org/anthology/S/S17/S17-2051.pdf>`__.

    """
    index = LevenshteinSimilarityIndex(
        dictionary, workers=workers, alpha=alpha, beta=beta, threshold=threshold)
    similarity_matrix = SparseTermSimilarityMatrix(
        index, dictionary, tfidf=tfidf, nonzero_limit=nonzero_limit, dtype=dtype)
    return similarity_matrix.matrix
