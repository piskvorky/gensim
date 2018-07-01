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

from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad, is_corpus

logger = logging.getLogger(__name__)


class TermSimilarityIndex(SaveLoad):
    """
    Retrieves most similar terms for a given term.

    See Also
    --------
    :class:`~gensim.similarities.termsim.SparseTermSimilarityMatrix`
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
            Most similar terms along with their similarities to `term`. Only terms distinct from
            `term` must be returned.

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
    :class:`~gensim.similarities.termsim.SparseTermSimilarityMatrix`
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
        for __, (t2_index, t2) in zip(range(topn), (
                (t2_index, t2) for t2_index, t2 in sorted(self.dictionary.items()) if t2 != t1)):
            yield (t2, self.term_similarity)


def _shortest_uint_dtype(max_value):
    """Get the shortest unsingned integer data-type required for representing values up to a given
    maximum value.

    Returns the shortest unsingned integer data-type required for representing values up to a given
    maximum value.

    Parameters
    ----------
    max_value : int
        The maximum value we wish to represent.

    Returns
    -------
    data-type
        The shortest unsigned integer data-type required for representing values up to a given
        maximum value.
    """
    if max_value < 2**8:
        return np.uint8
    elif max_value < 2**16:
        return np.uint16
    elif max_value < 2**32:
        return np.uint32
    return np.uint64


class SparseTermSimilarityMatrix(SaveLoad):
    """
    Builds a sparse term similarity matrix using a term similarity index.

    Parameters
    ----------
    source : :class:`~gensim.similarities.termsim.TermSimilarityIndex` or :class:`scipy.sparse.spmatrix`
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
            columns = sorted(dictionary.keys())
        else:
            assert max(tfidf.idfs) == matrix_order - 1
            logger.info("iterating over columns in tf-idf order")
            columns = [
                term_index for term_index, _
                in sorted(tfidf.idfs.items(), key=lambda x: (x[1], -x[0]), reverse=True)]

        matrix_nonzero = np.array([1] * matrix_order, dtype=_shortest_uint_dtype(nonzero_limit))
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
            num_rows = nonzero_limit - num_nonzero
            rows = index.most_similar(t1, num_rows)
            for row_number, (t2, similarity) in zip(range(num_rows), rows):
                if t2 not in dictionary.token2id:
                    logger.debug('an out-of-dictionary term "%s"', t2)
                    continue
                t2_index = dictionary.token2id[t2]
                if symmetric:
                    if matrix_nonzero[t2_index] <= nonzero_limit and not (t1_index, t2_index) in matrix:
                        matrix[t1_index, t2_index] = similarity
                        matrix_nonzero[t1_index] += 1
                        matrix[t2_index, t1_index] = similarity
                        matrix_nonzero[t2_index] += 1
                else:
                    matrix[t1_index, t2_index] = similarity

        logger.info(
            "constructed a sparse term similarity matrix with %0.06f%% density",
            100.0 * matrix.getnnz() / matrix_order**2)

        matrix = matrix.T
        assert sparse.issparse(matrix)
        self.__init__(matrix)

    def inner_product(self, X, Y, normalized=False):
        """Get the inner product(s) between real vectors / corpora X and Y.

        Return the inner product(s) between real vectors / corpora vec1 and vec2 expressed in a
        non-orthogonal normalized basis, where the dot product between the basis vectors is given by
        the sparse term similarity matrix.

        Parameters
        ----------
        vec1 : list of (int, float) or iterable of list of (int, float)
            A query vector / corpus in the sparse bag-of-words format.
        vec2 : list of (int, float) or iterable of list of (int, float)
            A document vector / corpus in the sparse bag-of-words format.
        normalized : bool, optional
            Whether the inner product should be L2-normalized. The normalized inner product
            corresponds to the Soft Cosine Measure (SCM). SCM is a number between <-1.0, 1.0>,
            where higher is more similar.

        Returns
        -------
        `self.matrix.dtype`,  `scipy.sparse.csr_matrix`, or :class:`numpy.matrix`
            The inner product(s) between `X` and `Y`.

        References
        ----------
        The soft cosine measure was perhaps first described by [sidorovetal14]_.

        .. [sidorovetal14] Grigori Sidorov et al., "Soft Similarity and Soft Cosine Measure: Similarity
           of Features in Vector Space Model", 2014, http://www.cys.cic.ipn.mx/ojs/index.php/CyS/article/view/2043/1921.

        """
        if not X or not Y:
            return self.matrix.dtype.type(0.0)

        is_corpus_X, X = is_corpus(X)
        is_corpus_Y, Y = is_corpus(Y)

        if not is_corpus_X and not is_corpus_Y:
            X = dict(X)
            Y = dict(Y)
            word_indices = sorted(set(chain(X, Y)))
            dtype = self.matrix.dtype
            X = np.array([X[i] if i in X else 0 for i in word_indices], dtype=dtype)
            Y = np.array([Y[i] if i in Y else 0 for i in word_indices], dtype=dtype)
            matrix = self.matrix[word_indices].T[word_indices].T.todense()

            result = X.T.dot(matrix).dot(Y)

            if normalized:
                X_norm = X.T.dot(matrix).dot(X)[0, 0]
                Y_norm = Y.T.dot(matrix).dot(Y)[0, 0]

                assert \
                    X_norm > 0.0 and Y_norm > 0.0, \
                    u"sparse documents must not contain any explicit zero entries and the similarity matrix S " \
                    u"must satisfy x^T * S * x > 0 for any nonzero bag-of-words vector x."

                result /= sqrt(X_norm) * sqrt(Y_norm)
                result = np.clip(result, -1.0, 1.0)

            return result[0, 0]
        elif not is_corpus_X or not is_corpus_Y:
            if is_corpus_X and not is_corpus_Y:
                is_corpus_X, X, is_corpus_Y, Y = is_corpus_Y, Y, is_corpus_X, X  # make Y the corpus
                transposed = True
            else:
                transposed = False

            dtype = self.matrix.dtype
            expanded_X = corpus2csc([X], num_terms=self.matrix.shape[0], dtype=dtype).T.dot(self.matrix)
            word_indices = sorted(expanded_X.nonzero()[1])
            del expanded_X

            X = dict(X)
            X = np.array([X[i] if i in X else 0 for i in word_indices], dtype=dtype)
            Y = corpus2csc(Y, num_terms=self.matrix.shape[0], dtype=dtype)[word_indices, :].todense()
            matrix = self.matrix[word_indices].T[word_indices].T.todense()
            if normalized:
                # use the following equality: np.diag(A.T.dot(B).dot(A)) == A.T.dot(B).multiply(A.T).sum(axis=1).T
                X_norm = np.multiply(X.T.dot(matrix), X.T).sum(axis=1).T
                Y_norm = np.multiply(Y.T.dot(matrix), Y.T).sum(axis=1).T

                assert \
                    X_norm.min() > 0.0 and Y_norm.min() >= 0.0, \
                    u"sparse documents must not contain any explicit zero entries and the similarity matrix S " \
                    u"must satisfy x^T * S * x > 0 for any nonzero bag-of-words vector x."

                X = np.multiply(X, 1 / np.sqrt(X_norm)).T
                Y = np.multiply(Y, 1 / np.sqrt(Y_norm))
                Y = np.nan_to_num(Y)  # Account for division by zero when Y_norm.min() == 0.0

            result = X.T.dot(matrix).dot(Y)

            if normalized:
                result = np.clip(result, -1.0, 1.0)

            if transposed:
                result = result.T

            return result
        else:  # if is_corpus_X and is_corpus_Y:
            dtype = self.matrix.dtype
            X = corpus2csc(X if is_corpus_X else [X], num_terms=self.matrix.shape[0], dtype=dtype)
            Y = corpus2csc(Y if is_corpus_Y else [Y], num_terms=self.matrix.shape[0], dtype=dtype)
            matrix = self.matrix

            if normalized:
                # use the following equality: np.diag(A.T.dot(B).dot(A)) == A.T.dot(B).multiply(A.T).sum(axis=1).T
                X_norm = X.T.dot(matrix).multiply(X.T).sum(axis=1).T
                Y_norm = Y.T.dot(matrix).multiply(Y.T).sum(axis=1).T

                assert \
                    X_norm.min() > 0.0 and Y_norm.min() >= 0.0, \
                    u"sparse documents must not contain any explicit zero entries and the similarity matrix S " \
                    u"must satisfy x^T * S * x > 0 for any nonzero bag-of-words vector x."

                X = X.multiply(sparse.csr_matrix(1 / np.sqrt(X_norm)))
                Y = Y.multiply(sparse.csr_matrix(1 / np.sqrt(Y_norm)))
                Y[Y == np.inf] = 0  # Account for division by zero when Y_norm.min() == 0.0

            result = X.T.dot(matrix).dot(Y)

            if normalized:
                result.data = np.clip(result.data, -1.0, 1.0)

            return result
