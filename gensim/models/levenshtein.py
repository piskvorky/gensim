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
from scipy import sparse

logger = logging.getLogger('gensim.models.levenshtein')

CHUNKSIZE = 500


def levsim_worker(args):
    alpha, beta, w1, w2, w2_index = args
    return (w2_index, levsim(alpha, beta, w1, w2))


def levsim(alpha, beta, w1, w2):
    """Get the Levenshtein similarity of two terms.

    Return the Levenshtein similarity of two terms. The similarity is a number between
    <0.0, 1.0>, higher is more similar.

    Parameters
    ----------
    w1 : {string, unicode}
        The first compared term.
    w2 : {string, unicode}
        The second compared term.

    Returns
    -------
    float
        The Levenshtein similarity of `w1` and `w2`.
    """
    return alpha * (1 - distance(w1, w2) / max(len(w1), len(w2)))**beta


def similarity_matrix(dictionary, tfidf=None, threshold=0.0, alpha=1.8, beta=5.0,
                      nonzero_limit=100, workers=1, dtype=REAL):
    """Constructs a term similarity matrix for computing Soft Cosine Measure.

    Constructs a sparse term similarity matrix in the :class:`scipy.sparse.csc_matrix` format
    for computing Soft Cosine Measure between documents.

    Parameters
    ----------
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        A dictionary that specifies a mapping between words and the indices of rows and columns
        of the resulting term similarity matrix.
    tfidf : :class:`gensim.models.tfidfmodel.TfidfModel`, optional
        A model that specifies the relative importance of the terms in the dictionary. The columns
        of the term similarity matrix will be build in a decreasing order of importance of
        terms, or in the order of term identifiers if None.
    threshold : float, optional
        Only pairs of words whose embeddings are more similar than `threshold` are considered
        when building the sparse term similarity matrix.
    alpha : float, optional
        The multiplicative factor alpha from the definition of the matrix Mlev.
    beta : float, optional
        The exponent beta from the definition of the matrix Mlev.
    nonzero_limit : int, optional
        The maximum number of non-zero elements outside the diagonal in a single column of the term
        similarity matrix. Setting `nonzero_limit` to a constant ensures that the time complexity of
        computing the Soft Cosine Measure will be linear in the document length rather than
        quadratic.
    workers : int, optional
        The number of workers to use when computing the Levenshtein similarities.
    dtype : numpy.dtype, optional
        Data-type of the term similarity matrix.

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
    logger.info("constructing a term similarity matrix")
    matrix_order = len(dictionary)
    matrix_nonzero = [1] * matrix_order
    matrix = sparse.identity(matrix_order, dtype=dtype, format="dok")
    pool = Pool(workers)
    # Decide the order of columns.
    if tfidf is None:
        word_indices = range(matrix_order)
    else:
        assert max(tfidf.idfs) < matrix_order
        word_indices = [
            index for index, _
            in sorted(tfidf.idfs.items(), key=lambda x: (x[1], -x[0]), reverse=True)
        ]

    # Traverse columns.
    for column_number, w1_index in enumerate(word_indices):
        if column_number % 1000 == 0:
            logger.info(
                "PROGRESS: at %.02f%% columns (%d / %d, %.06f%% density)",
                100.0 * (column_number + 1) / matrix_order, column_number + 1, matrix_order,
                100.0 * matrix.getnnz() / matrix_order**2)
        w1 = dictionary[w1_index]

        # Traverse rows.
        heap = []
        for w2_index, similarity in pool.imap_unordered(
                levsim_worker, (
                    (alpha, beta, w1, dictionary[w2_index], w2_index)
                    for w2_index in range(matrix_order)
                    if w1_index != w2_index), CHUNKSIZE):
            heappush(heap, (-similarity, w2_index))
        num_nonzero = matrix_nonzero[w1_index] - 1
        rows = (
            (w2_index, -similarity_) for similarity_, w2_index in (
                heappop(heap) for _ in range(min(matrix_order - 1, nonzero_limit - num_nonzero))))

        for w2_index, similarity in rows:
            # Ensure that we don't exceed `nonzero_limit` by mirroring the elements.
            if similarity > threshold and matrix_nonzero[w2_index] <= nonzero_limit:
                matrix[w1_index, w2_index] = similarity
                matrix_nonzero[w1_index] += 1
                matrix[w2_index, w1_index] = similarity
                matrix_nonzero[w2_index] += 1

    logger.info(
        "constructed a term similarity matrix with %0.6f %% nonzero elements",
        100.0 * matrix.getnnz() / matrix_order**2
    )
    pool.close()
    return matrix.T.tocsc()
