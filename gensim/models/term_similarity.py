#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Vit Novotny <witiko@mail.muni.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module provides classes that deal with term similarities.
"""

from itertools import chain
import logging
from math import sqrt

import numpy as np
from scipy import sparse

from gensim.utils import SaveLoad, is_corpus

logger = logging.getLogger(__name__)


class TermSimilarityIndex(SaveLoad):
    """
    Retrieves most similar terms for a given term.

    See Also
    --------
    :class:`~gensim.models.term_similarity.SparseTermSimilarityMatrix`
        Build a term similarity matrix and compute the Soft Cosine Measure.

    """
    def most_similar(self, term, topn=10):
        """Get most similar terms for a given term.

        Return most similar terms for a given term along with the similarities.

        Parameters
        ----------
        term : str
            Tne term for which we are retrieving `topn` most similar terms.
        topn : int, optional
            The maximum number of most similar terms to `term` that will be retrieved.

        Returns
        -------
        iterable of (str, float)
            Most similar terms along with their similarities to `term`. Only terms distinct `term`
            must be returned.

        """
        raise NotImplementedError


class UniformTermSimilarityIndex(TermSimilarityIndex):
    """
    Retrieves most similar terms for a given term under the hypothesis that the similarities between
    distinct terms are uniform.

    Parameters
    ----------
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        A dictionary that specifies the considered terms.
    term_similarity : float, optional
        The uniform similarity between distinct terms.

    See Also
    --------
    :class:`~gensim.models.term_similarity.SparseTermSimilarityMatrix`
        Build a term similarity matrix and compute the Soft Cosine Measure.

    Notes
    -----
    This class is mainly intended for testing SparseTermSimilarityMatrix and other classes that
    depend on the TermSimilarityIndex.

    """
    def __init__(self, dictionary, term_similarity=0.5):
        self.dictionary = dictionary
        self.term_similarity = term_similarity

    def most_similar(self, t1, topn=10):
        for t2_index in range(min(topn, len(self.dictionary))):
            t2 = self.dictionary[t2_index]
            if t1 != t2:
                yield (t2, self.term_similarity)


class SparseTermSimilarityMatrix(SaveLoad):
    """
    Builds a sparse term similarity matrix using a term similarity index.

    Parameters
    ----------
    source : :class:`~gensim.models.term_similarity.TermSimilarityIndex` or :class:`scipy.sparse.spmatrix`
        The source of the term similarity. Either a term similarity index that will be used for
        building the term similarity matrix, or an existing sparse term similarity matrix that will
        be encapsulated and stored in the matrix attribute.
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary` or None, optional
        A dictionary that specifies a mapping between terms and the indices of rows and columns
        of the resulting term similarity matrix. The dictionary may only be `None` when `source` is
        a :class:`scipy.sparse.spmatrix`.
    tfidf : :class:`gensim.models.tfidfmodel.TfidfModel` or None, optional
        A model that specifies the relative importance of the terms in the dictionary. The columns
        of the term similarity matrix will be build in a decreasing order of importance of
        terms, or in the order of term identifiers if None.
    symmetric : bool, optional
        Whether the symmetry of the term similarity matrix will be enforced. This parameter only has
        an effect when `source` is a :class:`scipy.sparse.spmatrix`.
    nonzero_limit : int, optional
        The maximum number of non-zero elements outside the diagonal in a single column of the
        sparse term similarity matrix.
    dtype : numpy.dtype, optional
        Data-type of the sparse term similarity matrix.

    Attributes
    ----------
    matrix : :class:`scipy.sparse.csc_matrix`
        The encapsulated sparse term similarity matrix.
    """
    PROGRESS_MESSAGE_PERIOD = 1000  # how many columns are processed between progress messages

    def __init__(self, source, dictionary=None, tfidf=None, symmetric=True, nonzero_limit=100, dtype=np.float32):
        if sparse.issparse(source):
            self.matrix = source.tocsc()  # encapsulate the passed sparse matrix
            return

        index = source
        assert isinstance(index, TermSimilarityIndex)
        assert dictionary is not None
        matrix_order = len(dictionary)

        logger.info("constructing a sparse term similarity matrix using %s", index)

        if tfidf is None:
            logger.info("iterating over columns in dictionary order")
            columns = range(matrix_order)
        else:
            assert max(tfidf.idfs) == matrix_order - 1
            logger.info("iterating over columns in tf-idf order")
            columns = [
                term_index for term_index, _
                in sorted(tfidf.idfs.items(), key=lambda x: (x[1], -x[0]), reverse=True)]

        matrix_nonzero = [1] * matrix_order
        matrix = sparse.identity(matrix_order, dtype=dtype, format="dok")

        for column_number, t1_index in enumerate(columns):
            if column_number % self.PROGRESS_MESSAGE_PERIOD == 0:
                logger.info(
                    "PROGRESS: at %.02f%% columns (%d / %d, %.06f%% density, "
                    "%.06f%% projected density)",
                    100.0 * (column_number + 1) / matrix_order, column_number + 1, matrix_order,
                    100.0 * matrix.getnnz() / matrix_order**2,
                    100.0 * np.clip(
                        (1.0 * (matrix.getnnz() - matrix_order) / matrix_order**2)
                        * (1.0 * matrix_order / (column_number + 1))
                        + (1.0 / matrix_order),  # add density correspoding to the main diagonal
                        0.0, 1.0))

            t1 = dictionary[t1_index]
            num_nonzero = matrix_nonzero[t1_index] - 1
            rows = index.most_similar(t1, nonzero_limit - num_nonzero)
            for t2, similarity in rows:
                if t2 not in dictionary.token2id:
                    logger.debug('an out-of-dictionary term "%s"', t2)
                    continue
                t2_index = dictionary.token2id[t2]
                if (not symmetric or matrix_nonzero[t2_index] <= nonzero_limit):
                    if not matrix.has_key((t1_index, t2_index)):
                        matrix[t1_index, t2_index] = similarity
                        matrix_nonzero[t1_index] += 1
                        if symmetric:
                            matrix[t2_index, t1_index] = similarity
                            matrix_nonzero[t2_index] += 1

        logger.info(
            "constructed a sparse term similarity matrix with %0.06f%% density",
            100.0 * matrix.getnnz() / matrix_order**2)

        matrix = matrix.T
        assert sparse.issparse(matrix)
        self.__init__(matrix)

    def inner_product(self, vec1, vec2, normalized=False):
        """Get the inner product between real vectors vec1 and vec2.

        Return the inner product between real vectors vec1 and vec2 expressed in a non-orthogonal
        normalized basis, where the dot product between the basis vectors is given by the sparse
        term similarity matrix.

        Parameters
        ----------
        vec1 : list of (int, float)
            A query vector in the BoW format.
        vec2 : list of (int, float)
            A document vector in the BoW format.
        normalized : bool, optional
            Whether the inner product should be L2-normalized. The normalized inner product
            corresponds to the Soft Cosine Measure (SCM). SCM is a number between <-1.0, 1.0>,
            higher is more similar.

        Returns
        -------
        `self.matrix.dtype`
            The inner product between `vec1` and `vec2`.

        References
        ----------
        The soft cosine measure was perhaps first described by [sidorovetal14]_.

        .. [sidorovetal14] Grigori Sidorov et al., "Soft Similarity and Soft Cosine Measure: Similarity
           of Features in Vector Space Model", 2014, http://www.cys.cic.ipn.mx/ojs/index.php/CyS/article/view/2043/1921.

        """
        if not vec1 or not vec2:
            return 0.0

        vec1 = dict(vec1)
        vec2 = dict(vec2)
        word_indices = sorted(set(chain(vec1, vec2)))
        dtype = self.matrix.dtype
        vec1 = np.array([vec1[i] if i in vec1 else 0 for i in word_indices], dtype=dtype)
        vec2 = np.array([vec2[i] if i in vec2 else 0 for i in word_indices], dtype=dtype)
        dense_matrix = self.matrix[[[i] for i in word_indices], word_indices].todense()
        result = vec1.T.dot(dense_matrix).dot(vec2)[0, 0]

        if normalized:
            vec1len = vec1.T.dot(dense_matrix).dot(vec1)[0, 0]
            vec2len = vec2.T.dot(dense_matrix).dot(vec2)[0, 0]

            assert \
                vec1len > 0.0 and vec2len > 0.0, \
                u"sparse documents must not contain any explicit zero entries and the similarity matrix S " \
                u"must satisfy x^T * S * x > 0 for any nonzero bag-of-words vector x."

            result /= sqrt(vec1len) * sqrt(vec2len)
            result = np.clip(result, -1.0, 1.0)

        return result
