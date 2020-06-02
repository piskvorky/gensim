#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Vit Novotny <witiko@mail.muni.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module provides classes that deal with term similarities.
"""

from array import array
from itertools import chain
import logging
from math import sqrt
import warnings

import numpy as np
from six.moves import range
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

    def __str__(self):
        return "%s(%s)" % (
            self.__class__.__name__,
            ', '.join(
                '%s=%s' % (key, value)
                for key, value
                in vars(self).items()
            )
        )


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
        self.dictionary = sorted(dictionary.items())
        self.term_similarity = term_similarity

    def most_similar(self, t1, topn=10):
        for __, (t2_index, t2) in zip(range(topn), (
                (t2_index, t2) for t2_index, t2 in self.dictionary if t2 != t1)):
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

    Notes
    -----
    Building a DOK matrix, and converting it to a CSC matrix carries a significant memory overhead.
    Future work should switch to building arrays of rows, columns, and non-zero elements and
    directly passing these arrays to the CSC matrix constructor without copying.

    Examples
    --------
    >>> from gensim.test.utils import common_texts
    >>> from gensim.corpora import Dictionary
    >>> from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
    >>> from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
    >>> from gensim.similarities.index import AnnoyIndexer
    >>> from scikits.sparse.cholmod import cholesky
    >>>
    >>> model = Word2Vec(common_texts, size=20, min_count=1)  # train word-vectors
    >>> annoy = AnnoyIndexer(model, num_trees=2)  # use annoy for faster word similarity lookups
    >>> termsim_index = WordEmbeddingSimilarityIndex(model.wv, kwargs={'indexer': annoy})
    >>> dictionary = Dictionary(common_texts)
    >>> bow_corpus = [dictionary.doc2bow(document) for document in common_texts]
    >>> similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, symmetric=True, dominant=True)
    >>> docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=10)
    >>>
    >>> query = 'graph trees computer'.split()  # make a query
    >>> sims = docsim_index[dictionary.doc2bow(query)]  # calculate similarity of query to each doc from bow_corpus
    >>>
    >>> word_embeddings = cholesky(similarity_matrix.matrix).L()  # obtain word embeddings from similarity matrix

    Check out `Tutorial Notebook
    <https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb>`_
    for more examples.

    Parameters
    ----------
    source : :class:`~gensim.similarities.termsim.TermSimilarityIndex` or :class:`scipy.sparse.spmatrix`
        The source of the term similarity. Either a term similarity index that will be used for
        building the term similarity matrix, or an existing sparse term similarity matrix that will
        be encapsulated and stored in the matrix attribute. When a matrix is specified as the
        source, any other parameters will be ignored.
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary` or None, optional
        A dictionary that specifies a mapping between terms and the indices of rows and columns
        of the resulting term similarity matrix. The dictionary may only be None when source is
        a :class:`scipy.sparse.spmatrix`.
    tfidf : :class:`gensim.models.tfidfmodel.TfidfModel` or None, optional
        A model that specifies the relative importance of the terms in the dictionary. The columns
        of the term similarity matrix will be build in a decreasing order of importance of
        terms, or in the order of term identifiers if None.
    symmetric : bool, optional
        Whether the symmetry of the term similarity matrix will be enforced. Symmetry is a necessary
        precondition for positive definiteness, which is necessary if you later wish to derive a
        unique change-of-basis matrix from the term similarity matrix using Cholesky factorization.
        Setting symmetric to False will significantly reduce memory usage during matrix construction.
    dominant: bool, optional
        Whether the strict column diagonal dominance of the term similarity matrix will be enforced.
        Strict diagonal dominance and symmetry are sufficient preconditions for positive
        definiteness, which is necessary if you later wish to derive a change-of-basis matrix from
        the term similarity matrix using Cholesky factorization.
    nonzero_limit : int or None, optional
        The maximum number of non-zero elements outside the diagonal in a single column of the
        sparse term similarity matrix. If None, then no limit will be imposed.
    dtype : numpy.dtype, optional
        The data type of the sparse term similarity matrix.
    positive_definite: bool or None, optional
        A deprecated alias for dominant.

    Attributes
    ----------
    matrix : :class:`scipy.sparse.csc_matrix`
        The encapsulated sparse term similarity matrix.

    """
    def __init__(self, source, dictionary=None, tfidf=None, symmetric=True, dominant=False,
                 nonzero_limit=100, dtype=np.float32, positive_definite=None):
        if sparse.issparse(source):
            self.matrix = source.tocsc()  # encapsulate the passed sparse matrix
            return

        if positive_definite is not None:
            warnings.warn(
                'Parameter positive_definite will be removed in 4.0.0, use dominant instead',
                category=DeprecationWarning,
            )
            dominant = positive_definite

        index = source
        assert isinstance(index, TermSimilarityIndex)
        assert dictionary is not None
        matrix_order = len(dictionary)

        logger.info("constructing a sparse term similarity matrix using %s", index)

        if nonzero_limit is None:
            nonzero_limit = matrix_order

        if tfidf is None:
            logger.info("iterating over columns in dictionary order")
            columns = sorted(dictionary.keys())
        else:
            assert max(tfidf.idfs) == matrix_order - 1
            logger.info("iterating over columns in tf-idf order")
            columns = [
                term_index for term_index, _
                in sorted(
                    tfidf.idfs.items(),
                    key=lambda x: (lambda term_index, term_idf: (term_idf, -term_index))(*x),
                    reverse=True,
                )
            ]

        if dtype is np.float16 or dtype is np.float32:
            similarity_type_code = 'f'
        elif dtype is np.float64:
            similarity_type_code = 'd'
        else:
            raise ValueError('Dtype %s is unsupported, use numpy.float16, float32, or float64.' % dtype)
        nonzero_counter_dtype = _shortest_uint_dtype(nonzero_limit)

        column_nonzero = np.array([0] * matrix_order, dtype=nonzero_counter_dtype)
        if dominant:
            column_sum = np.zeros(matrix_order, dtype=dtype)
        if symmetric:
            assigned_cells = set()
        row_buffer = array('Q')
        column_buffer = array('Q')
        data_buffer = array(similarity_type_code)

        try:
            from tqdm import tqdm as progress_bar
        except ImportError:
            def progress_bar(iterable):
                return iterable

        for column_number, t1_index in enumerate(progress_bar(columns)):
            column_buffer.append(column_number)
            row_buffer.append(column_number)
            data_buffer.append(1.0)

            if nonzero_limit <= 0:
                continue

            t1 = dictionary[t1_index]
            num_nonzero = column_nonzero[t1_index]
            num_rows = nonzero_limit - num_nonzero
            most_similar = [
                (dictionary.token2id[term], similarity)
                for term, similarity in index.most_similar(t1, topn=num_rows)
                if term in dictionary.token2id
            ] if num_rows > 0 else []

            if tfidf is None:
                rows = sorted(most_similar)
            else:
                rows = sorted(
                    most_similar,
                    key=lambda x: (lambda term_index, _: (tfidf.idfs[term_index], -term_index))(*x),
                    reverse=True,
                )

            for row_number, (t2_index, similarity) in zip(range(num_rows), rows):
                if dominant and column_sum[t1_index] + abs(similarity) >= 1.0:
                    break
                if symmetric:
                    if column_nonzero[t2_index] < nonzero_limit \
                            and (not dominant or column_sum[t2_index] + abs(similarity) < 1.0) \
                            and (t1_index, t2_index) not in assigned_cells:
                        assigned_cells.add((t1_index, t2_index))
                        column_buffer.append(t1_index)
                        row_buffer.append(t2_index)
                        data_buffer.append(similarity)
                        column_nonzero[t1_index] += 1

                        assigned_cells.add((t2_index, t1_index))
                        column_buffer.append(t2_index)
                        row_buffer.append(t1_index)
                        data_buffer.append(similarity)
                        column_nonzero[t2_index] += 1

                        if dominant:
                            column_sum[t1_index] += abs(similarity)
                            column_sum[t2_index] += abs(similarity)
                else:
                    column_buffer.append(t1_index)
                    row_buffer.append(t2_index)
                    data_buffer.append(similarity)
                    column_nonzero[t1_index] += 1

                    if dominant:
                        column_sum[t1_index] += abs(similarity)

        data_buffer = np.frombuffer(data_buffer, dtype=dtype)
        row_buffer = np.frombuffer(row_buffer, dtype=np.uint64)
        column_buffer = np.frombuffer(column_buffer, dtype=np.uint64)
        matrix = sparse.coo_matrix((data_buffer, (row_buffer, column_buffer)), shape=(matrix_order, matrix_order))

        logger.info(
            "constructed a sparse term similarity matrix with %0.06f%% density",
            100.0 * matrix.getnnz() / matrix_order**2,
        )

        assert sparse.issparse(matrix)
        self.__init__(matrix)

    def inner_product(self, X, Y, normalized=(False, False)):
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
        normalized : tuple of {True, False, 'maintain'}, optional
            First/second value specifies whether the query/document vectors in the inner product
            will be L2-normalized (True; corresponds to the soft cosine measure), maintain their
            L2-norm during change of basis ('maintain'; corresponds to query expansion with partial
            membership), or kept as-is (False; corresponds to query expansion; default).

        Returns
        -------
        `self.matrix.dtype`,  `scipy.sparse.csr_matrix`, or :class:`numpy.matrix`
            The inner product(s) between `X` and `Y`.

        References
        ----------
        The soft cosine measure was perhaps first described by [sidorovetal14]_.
        Further notes on the efficient implementation of the soft cosine measure are described by
        [novotny18]_.

        .. [sidorovetal14] Grigori Sidorov et al., "Soft Similarity and Soft Cosine Measure: Similarity
           of Features in Vector Space Model", 2014, http://www.cys.cic.ipn.mx/ojs/index.php/CyS/article/view/2043/1921.

        .. [novotny18] Vít Novotný, "Implementation Notes for the Soft Cosine Measure", 2018,
           http://dx.doi.org/10.1145/3269206.3269317.

        """
        if not X or not Y:
            return self.matrix.dtype.type(0.0)

        if normalized in (True, False):
            warnings.warn(
                'Boolean parameter normalized will be removed in 4.0.0, use '
                'normalized=({normalized}, {normalized}) instead of '
                'normalized={normalized}'.format(normalized=normalized),
                category=DeprecationWarning,
            )
            normalized = (normalized, normalized)

        normalized_X, normalized_Y = normalized
        valid_normalized_values = (True, False, 'maintain')

        if normalized_X not in valid_normalized_values:
            raise ValueError('{} is not a valid value of normalize'.format(normalized_X))
        if normalized_Y not in valid_normalized_values:
            raise ValueError('{} is not a valid value of normalize'.format(normalized_Y))

        is_corpus_X, X = is_corpus(X)
        is_corpus_Y, Y = is_corpus(Y)

        non_negative_norm_assertion_message = u"sparse documents must not contain any explicit " \
            u"zero entries and the similarity matrix S must satisfy x^T * S * x >= 0 for any " \
            u"nonzero bag-of-words vector x."

        if not is_corpus_X and not is_corpus_Y:
            X = dict(X)
            Y = dict(Y)
            word_indices = np.array(sorted(set(chain(X, Y))))
            dtype = self.matrix.dtype
            X = np.array([X[i] if i in X else 0 for i in word_indices], dtype=dtype)
            Y = np.array([Y[i] if i in Y else 0 for i in word_indices], dtype=dtype)
            matrix = self.matrix[word_indices[:, None], word_indices].todense()

            result = X.T.dot(matrix).dot(Y)

            norm = 1.0

            if normalized_X:
                X_norm = X.T.dot(matrix).dot(X)[0, 0]
                assert X_norm >= 0.0, non_negative_norm_assertion_message
                if normalized_X is 'maintain' and X_norm > 0.0:
                    X_norm /= X.T.dot(X)
                X_norm = sqrt(X_norm)
                if X_norm > 0.0:
                    norm *= X_norm

            if normalized_Y:
                Y_norm = Y.T.dot(matrix).dot(Y)[0, 0]
                assert Y_norm >= 0.0, non_negative_norm_assertion_message
                if normalized_Y is 'maintain' and Y_norm > 0.0:
                    Y_norm /= Y.T.dot(Y)
                Y_norm = sqrt(Y_norm)
                if Y_norm > 0.0:
                    norm *= Y_norm

            if normalized_X or normalized_Y:
                result *= 1.0 / norm

            if normalized_X is True and normalized_Y is True:
                result = np.clip(result, -1.0, 1.0)

            return result[0, 0]
        elif not is_corpus_X or not is_corpus_Y:
            if is_corpus_X and not is_corpus_Y:
                is_corpus_X, X, is_corpus_Y, Y, normalized_X, normalized_Y = \
                    is_corpus_Y, Y, is_corpus_X, X, normalized_Y, normalized_X  # make Y the corpus
                transposed = True
            else:
                transposed = False

            dtype = self.matrix.dtype
            expanded_X = corpus2csc([X], num_terms=self.matrix.shape[0], dtype=dtype).T.dot(self.matrix)
            word_indices = np.array(sorted(expanded_X.nonzero()[1]))
            del expanded_X

            X = dict(X)
            X = np.array([X[i] if i in X else 0 for i in word_indices], dtype=dtype)
            Y = corpus2csc(Y, num_terms=self.matrix.shape[0], dtype=dtype)[word_indices, :].todense()
            matrix = self.matrix[word_indices[:, None], word_indices].todense()

            # use the following equality: np.diag(A.T.dot(B).dot(A)) == A.T.dot(B).multiply(A.T).sum(axis=1).T

            if normalized_X:
                X_norm = np.multiply(X.T.dot(matrix), X.T).sum(axis=1).T
                assert X_norm.min() >= 0.0, non_negative_norm_assertion_message
                if normalized_X is 'maintain':
                    X_norm /= X.T.dot(X)
                X_norm = np.sqrt(X_norm)
                X = np.multiply(X, 1.0 / X_norm).T
                X = np.nan_to_num(X)  # account for division by zero

            if normalized_Y:
                Y_norm = np.multiply(Y.T.dot(matrix), Y.T).sum(axis=1).T
                assert Y_norm.min() >= 0.0, non_negative_norm_assertion_message
                if normalized_Y is 'maintain':
                    Y_norm /= np.multiply(Y.T, Y.T).sum(axis=1).T
                Y_norm = np.sqrt(Y_norm)
                Y = np.multiply(Y, 1.0 / Y_norm)
                Y = np.nan_to_num(Y)  # account for division by zero

            result = X.T.dot(matrix).dot(Y)

            if normalized_X is True and normalized_Y is True:
                result = np.clip(result, -1.0, 1.0)

            if transposed:
                result = result.T

            return result
        else:  # if is_corpus_X and is_corpus_Y:
            dtype = self.matrix.dtype
            X = corpus2csc(X if is_corpus_X else [X], num_terms=self.matrix.shape[0], dtype=dtype)
            Y = corpus2csc(Y if is_corpus_Y else [Y], num_terms=self.matrix.shape[0], dtype=dtype)
            matrix = self.matrix

            # use the following equality: np.diag(A.T.dot(B).dot(A)) == A.T.dot(B).multiply(A.T).sum(axis=1).T

            if normalized_X:
                X_norm = X.T.dot(matrix).multiply(X.T).sum(axis=1).T
                assert X_norm.min() >= 0.0, non_negative_norm_assertion_message
                if normalized_X is 'maintain':
                    X_norm /= X.T.multiply(X.T).sum(axis=1).T
                X_norm = np.sqrt(X_norm)
                X = X.multiply(sparse.csr_matrix(1.0 / X_norm))
                X[X == np.inf] = 0  # account for division by zero

            if normalized_Y:
                Y_norm = Y.T.dot(matrix).multiply(Y.T).sum(axis=1).T
                assert Y_norm.min() >= 0.0, non_negative_norm_assertion_message
                if normalized_Y is 'maintain':
                    Y_norm /= Y.T.multiply(Y.T).sum(axis=1).T
                Y_norm = np.sqrt(Y_norm)
                Y = Y.multiply(sparse.csr_matrix(1.0 / Y_norm))
                Y[Y == np.inf] = 0  # account for division by zero

            result = X.T.dot(matrix).dot(Y)

            if normalized_X is True and normalized_Y is True:
                result.data = np.clip(result.data, -1.0, 1.0)

            return result
