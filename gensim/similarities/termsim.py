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

import numpy as np
from scipy import sparse

from gensim.matutils import corpus2csc
from gensim.utils import SaveLoad, is_corpus

logger = logging.getLogger(__name__)

NON_NEGATIVE_NORM_ASSERTION_MESSAGE = (
    u"sparse documents must not contain any explicit "
    u"zero entries and the similarity matrix S must satisfy x^T * S * x >= 0 for any "
    u"nonzero bag-of-words vector x."
)


class TermSimilarityIndex(SaveLoad):
    """
    Base class = common interface for retrieving the most similar terms for a given term.

    See Also
    --------
    :class:`~gensim.similarities.termsim.SparseTermSimilarityMatrix`
        A sparse term similarity matrix built using a term similarity index.

    """
    def most_similar(self, term, topn=10):
        """Get most similar terms for a given term.

        Return the most similar terms for a given term along with their similarities.

        Parameters
        ----------
        term : str
            The term for which we are retrieving `topn` most similar terms.
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
        members = ', '.join('%s=%s' % pair for pair in vars(self).items())
        return '%s(%s)' % (self.__class__.__name__, members)


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
        A sparse term similarity matrix built using a term similarity index.

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


class WordEmbeddingSimilarityIndex(TermSimilarityIndex):
    """
    Computes cosine similarities between word embeddings and retrieves most
    similar terms for a given term.

    Notes
    -----
    By fitting the word embeddings to a vocabulary that you will be using, you
    can eliminate all out-of-vocabulary (OOV) words that you would otherwise
    receive from the `most_similar` method. In subword models such as fastText,
    this procedure will also infer word-vectors for words from your vocabulary
    that previously had no word-vector.

    >>> from gensim.test.utils import common_texts, datapath
    >>> from gensim.corpora import Dictionary
    >>> from gensim.models import FastText
    >>> from gensim.models.word2vec import LineSentence
    >>> from gensim.similarities import WordEmbeddingSimilarityIndex
    >>>
    >>> model = FastText(common_texts, vector_size=20, min_count=1)  # train word-vectors on a corpus
    >>> different_corpus = LineSentence(datapath('lee_background.cor'))
    >>> dictionary = Dictionary(different_corpus)  # construct a vocabulary on a different corpus
    >>> words = [word for word, count in dictionary.most_common()]
    >>> word_vectors = model.wv.vectors_for_all(words)  # remove OOV word-vectors and infer word-vectors for new words
    >>> assert len(dictionary) == len(word_vectors)  # all words from our vocabulary received their word-vectors
    >>> termsim_index = WordEmbeddingSimilarityIndex(word_vectors)

    Parameters
    ----------
    keyedvectors : :class:`~gensim.models.keyedvectors.KeyedVectors`
        The word embeddings.
    threshold : float, optional
        Only embeddings more similar than `threshold` are considered when retrieving word embeddings
        closest to a given word embedding.
    exponent : float, optional
        Take the word embedding similarities larger than `threshold` to the power of `exponent`.
    kwargs : dict or None
        A dict with keyword arguments that will be passed to the
        :meth:`~gensim.models.keyedvectors.KeyedVectors.most_similar` method
        when retrieving the word embeddings closest to a given word embedding.

    See Also
    --------
    :class:`~gensim.similarities.levenshtein.LevenshteinSimilarityIndex`
        Retrieve most similar terms for a given term using the Levenshtein distance.
    :class:`~gensim.similarities.termsim.SparseTermSimilarityMatrix`
        Build a term similarity matrix and compute the Soft Cosine Measure.

    """
    def __init__(self, keyedvectors, threshold=0.0, exponent=2.0, kwargs=None):
        self.keyedvectors = keyedvectors
        self.threshold = threshold
        self.exponent = exponent
        self.kwargs = kwargs or {}
        super(WordEmbeddingSimilarityIndex, self).__init__()

    def most_similar(self, t1, topn=10):
        if t1 not in self.keyedvectors:
            logger.debug('an out-of-dictionary term "%s"', t1)
        else:
            most_similar = self.keyedvectors.most_similar(positive=[t1], topn=topn, **self.kwargs)
            for t2, similarity in most_similar:
                if similarity > self.threshold:
                    yield (t2, similarity**self.exponent)


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


def _create_source(index, dictionary, tfidf, symmetric, dominant, nonzero_limit, dtype):
    """Build a sparse term similarity matrix using a term similarity index.

    Returns
    -------
    matrix : :class:`scipy.sparse.coo_matrix`
        The sparse term similarity matrix.

    """
    assert isinstance(index, TermSimilarityIndex)
    assert dictionary is not None
    matrix_order = len(dictionary)

    if matrix_order == 0:
        raise ValueError('Dictionary provided to SparseTermSimilarityMatrix must not be empty')

    logger.info("constructing a sparse term similarity matrix using %s", index)

    if nonzero_limit is None:
        nonzero_limit = matrix_order

    def tfidf_sort_key(term_index):
        if isinstance(term_index, tuple):
            term_index, *_ = term_index
        term_idf = tfidf.idfs[term_index]
        return (-term_idf, term_index)

    if tfidf is None:
        columns = sorted(dictionary.keys())
        logger.info("iterating over %i columns in dictionary order", len(columns))
    else:
        assert max(tfidf.idfs) == matrix_order - 1
        columns = sorted(tfidf.idfs.keys(), key=tfidf_sort_key)
        logger.info("iterating over %i columns in tf-idf order", len(columns))

    nonzero_counter_dtype = _shortest_uint_dtype(nonzero_limit)

    column_nonzero = np.array([0] * matrix_order, dtype=nonzero_counter_dtype)
    if dominant:
        column_sum = np.zeros(matrix_order, dtype=dtype)
    if symmetric:
        assigned_cells = set()
    row_buffer = array('Q')
    column_buffer = array('Q')
    if dtype is np.float16 or dtype is np.float32:
        data_buffer = array('f')
    elif dtype is np.float64:
        data_buffer = array('d')
    else:
        raise ValueError('Dtype %s is unsupported, use numpy.float16, float32, or float64.' % dtype)

    def cell_full(t1_index, t2_index, similarity):
        if dominant and column_sum[t1_index] + abs(similarity) >= 1.0:
            return True  # after adding the similarity, the matrix would cease to be strongly diagonally dominant
        assert column_nonzero[t1_index] <= nonzero_limit
        if column_nonzero[t1_index] == nonzero_limit:
            return True  # after adding the similarity, the column would contain more than nonzero_limit elements
        if symmetric and (t1_index, t2_index) in assigned_cells:
            return True  # a similarity has already been assigned to this cell
        return False

    def populate_buffers(t1_index, t2_index, similarity):
        column_buffer.append(t1_index)
        row_buffer.append(t2_index)
        data_buffer.append(similarity)
        column_nonzero[t1_index] += 1
        if symmetric:
            assigned_cells.add((t1_index, t2_index))
        if dominant:
            column_sum[t1_index] += abs(similarity)

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
            rows = sorted(most_similar, key=tfidf_sort_key)

        for t2_index, similarity in rows:
            if cell_full(t1_index, t2_index, similarity):
                continue
            if not symmetric:
                populate_buffers(t1_index, t2_index, similarity)
            elif not cell_full(t2_index, t1_index, similarity):
                populate_buffers(t1_index, t2_index, similarity)
                populate_buffers(t2_index, t1_index, similarity)

    data_buffer = np.frombuffer(data_buffer, dtype=dtype)
    row_buffer = np.frombuffer(row_buffer, dtype=np.uint64)
    column_buffer = np.frombuffer(column_buffer, dtype=np.uint64)
    matrix = sparse.coo_matrix((data_buffer, (row_buffer, column_buffer)), shape=(matrix_order, matrix_order))

    logger.info(
        "constructed a sparse term similarity matrix with %0.06f%% density",
        100.0 * matrix.getnnz() / matrix_order**2,
    )

    return matrix


def _normalize_dense_vector(vector, matrix, normalization):
    """Normalize a dense vector after a change of basis.

    Parameters
    ----------
    vector : 1xN ndarray
        A dense vector.
    matrix : NxN ndarray
        A change-of-basis matrix.
    normalization : {True, False, 'maintain'}
        Whether the vector will be L2-normalized (True; corresponds to the soft
        cosine measure), maintain its L2-norm during the change of basis
        ('maintain'; corresponds to query expansion with partial membership),
        or kept as-is (False; corresponds to query expansion).

    Returns
    -------
    vector : ndarray
        The normalized dense vector.

    """
    if not normalization:
        return vector

    vector_norm = vector.T.dot(matrix).dot(vector)[0, 0]
    assert vector_norm >= 0.0, NON_NEGATIVE_NORM_ASSERTION_MESSAGE
    if normalization == 'maintain' and vector_norm > 0.0:
        vector_norm /= vector.T.dot(vector)
    vector_norm = sqrt(vector_norm)

    normalized_vector = vector
    if vector_norm > 0.0:
        normalized_vector /= vector_norm

    return normalized_vector


def _normalize_dense_corpus(corpus, matrix, normalization):
    """Normalize a dense corpus after a change of basis.

    Parameters
    ----------
    corpus : MxN ndarray
        A dense corpus.
    matrix : NxN ndarray
        A change-of-basis matrix.
    normalization : {True, False, 'maintain'}
        Whether the vector will be L2-normalized (True; corresponds to the soft
        cosine measure), maintain its L2-norm during the change of basis
        ('maintain'; corresponds to query expansion with partial membership),
        or kept as-is (False; corresponds to query expansion).

    Returns
    -------
    normalized_corpus : ndarray
        The normalized dense corpus.

    """
    if not normalization:
        return corpus

    # use the following equality: np.diag(A.T.dot(B).dot(A)) == A.T.dot(B).multiply(A.T).sum(axis=1).T
    corpus_norm = np.multiply(corpus.T.dot(matrix), corpus.T).sum(axis=1).T
    assert corpus_norm.min() >= 0.0, NON_NEGATIVE_NORM_ASSERTION_MESSAGE
    if normalization == 'maintain':
        corpus_norm /= np.multiply(corpus.T, corpus.T).sum(axis=1).T
    corpus_norm = np.sqrt(corpus_norm)

    normalized_corpus = np.multiply(corpus, 1.0 / corpus_norm)
    normalized_corpus = np.nan_to_num(normalized_corpus)  # account for division by zero
    return normalized_corpus


def _normalize_sparse_corpus(corpus, matrix, normalization):
    """Normalize a sparse corpus after a change of basis.

    Parameters
    ----------
    corpus : MxN :class:`scipy.sparse.csc_matrix`
        A sparse corpus.
    matrix : NxN :class:`scipy.sparse.csc_matrix`
        A change-of-basis matrix.
    normalization : {True, False, 'maintain'}
        Whether the vector will be L2-normalized (True; corresponds to the soft
        cosine measure), maintain its L2-norm during the change of basis
        ('maintain'; corresponds to query expansion with partial membership),
        or kept as-is (False; corresponds to query expansion).

    Returns
    -------
    normalized_corpus : :class:`scipy.sparse.csc_matrix`
        The normalized sparse corpus.

    """
    if not normalization:
        return corpus

    # use the following equality: np.diag(A.T.dot(B).dot(A)) == A.T.dot(B).multiply(A.T).sum(axis=1).T
    corpus_norm = corpus.T.dot(matrix).multiply(corpus.T).sum(axis=1).T
    assert corpus_norm.min() >= 0.0, NON_NEGATIVE_NORM_ASSERTION_MESSAGE
    if normalization == 'maintain':
        corpus_norm /= corpus.T.multiply(corpus.T).sum(axis=1).T
    corpus_norm = np.sqrt(corpus_norm)

    normalized_corpus = corpus.multiply(sparse.csr_matrix(1.0 / corpus_norm))
    normalized_corpus[normalized_corpus == np.inf] = 0  # account for division by zero
    return normalized_corpus


class SparseTermSimilarityMatrix(SaveLoad):
    """
    Builds a sparse term similarity matrix using a term similarity index.

    Examples
    --------
    >>> from gensim.test.utils import common_texts as corpus, datapath
    >>> from gensim.corpora import Dictionary
    >>> from gensim.models import Word2Vec
    >>> from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
    >>> from gensim.similarities.index import AnnoyIndexer
    >>>
    >>> model_corpus_file = datapath('lee_background.cor')
    >>> model = Word2Vec(corpus_file=model_corpus_file, vector_size=20, min_count=1)  # train word-vectors
    >>>
    >>> dictionary = Dictionary(corpus)
    >>> tfidf = TfidfModel(dictionary=dictionary)
    >>> words = [word for word, count in dictionary.most_common()]
    >>> word_vectors = model.wv.vectors_for_all(words, allow_inference=False)  # produce vectors for words in corpus
    >>>
    >>> indexer = AnnoyIndexer(word_vectors, num_trees=2)  # use Annoy for faster word similarity lookups
    >>> termsim_index = WordEmbeddingSimilarityIndex(word_vectors, kwargs={'indexer': indexer})
    >>> similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)  # compute word similarities
    >>>
    >>> tfidf_corpus = tfidf[[dictionary.doc2bow(document) for document in common_texts]]
    >>> docsim_index = SoftCosineSimilarity(tfidf_corpus, similarity_matrix, num_best=10)  # index tfidf_corpus
    >>>
    >>> query = 'graph trees computer'.split()  # make a query
    >>> sims = docsim_index[dictionary.doc2bow(query)]  # find the ten closest documents from tfidf_corpus

    Check out `the Gallery <https://radimrehurek.com/gensim/auto_examples/tutorials/run_scm.html>`_
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

    Attributes
    ----------
    matrix : :class:`scipy.sparse.csc_matrix`
        The encapsulated sparse term similarity matrix.

    Raises
    ------
    ValueError
        If `dictionary` is empty.

    See Also
    --------
    :class:`~gensim.similarities.docsim.SoftCosineSimilarity`
        A document similarity index using the soft cosine similarity over the term similarity matrix.
    :class:`~gensim.similarities.termsim.LevenshteinSimilarityIndex`
        A term similarity index that computes Levenshtein similarities between terms.
    :class:`~gensim.similarities.termsim.WordEmbeddingSimilarityIndex`
        A term similarity index that computes cosine similarities between word embeddings.

    """
    def __init__(self, source, dictionary=None, tfidf=None, symmetric=True, dominant=False,
            nonzero_limit=100, dtype=np.float32):

        if not sparse.issparse(source):
            index = source
            args = (index, dictionary, tfidf, symmetric, dominant, nonzero_limit, dtype)
            source = _create_source(*args)
            assert sparse.issparse(source)

        self.matrix = source.tocsc()

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

        normalized_X, normalized_Y = normalized
        valid_normalized_values = (True, False, 'maintain')

        if normalized_X not in valid_normalized_values:
            raise ValueError('{} is not a valid value of normalize'.format(normalized_X))
        if normalized_Y not in valid_normalized_values:
            raise ValueError('{} is not a valid value of normalize'.format(normalized_Y))

        is_corpus_X, X = is_corpus(X)
        is_corpus_Y, Y = is_corpus(Y)

        if not is_corpus_X and not is_corpus_Y:
            X = dict(X)
            Y = dict(Y)
            word_indices = np.array(sorted(set(chain(X, Y))))
            dtype = self.matrix.dtype
            X = np.array([X[i] if i in X else 0 for i in word_indices], dtype=dtype)
            Y = np.array([Y[i] if i in Y else 0 for i in word_indices], dtype=dtype)
            matrix = self.matrix[word_indices[:, None], word_indices].todense()

            X = _normalize_dense_vector(X, matrix, normalized_X)
            Y = _normalize_dense_vector(Y, matrix, normalized_Y)
            result = X.T.dot(matrix).dot(Y)

            if normalized_X is True and normalized_Y is True:
                result = np.clip(result, -1.0, 1.0)

            return result[0, 0]
        elif not is_corpus_X or not is_corpus_Y:
            if is_corpus_X and not is_corpus_Y:
                X, Y = Y, X  # make Y the corpus
                is_corpus_X, is_corpus_Y = is_corpus_Y, is_corpus_X
                normalized_X, normalized_Y = normalized_Y, normalized_X
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

            X = _normalize_dense_vector(X, matrix, normalized_X)
            Y = _normalize_dense_corpus(Y, matrix, normalized_Y)
            result = X.dot(matrix).dot(Y)

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

            X = _normalize_sparse_corpus(X, matrix, normalized_X)
            Y = _normalize_sparse_corpus(Y, matrix, normalized_Y)
            result = X.T.dot(matrix).dot(Y)

            if normalized_X is True and normalized_Y is True:
                result.data = np.clip(result.data, -1.0, 1.0)

            return result
