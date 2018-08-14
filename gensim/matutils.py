#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Math helper functions."""

from __future__ import with_statement


from itertools import chain
import logging
import math

from gensim import utils

import numpy as np
import scipy.sparse
from scipy.stats import entropy
import scipy.linalg
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.special_matrices import triu
from scipy.special import psi  # gamma function utils

from six import iteritems, itervalues, string_types
from six.moves import xrange, zip as izip


logger = logging.getLogger(__name__)


def blas(name, ndarray):
    """Helper for getting the appropriate BLAS function, using :func:`scipy.linalg.get_blas_funcs`.

    Parameters
    ----------
    name : str
        Name(s) of BLAS functions, without the type prefix.
    ndarray : numpy.ndarray
        Arrays can be given to determine optimal prefix of BLAS routines.

    Returns
    -------
    object
        BLAS function for the needed operation on the given data type.

    """
    return scipy.linalg.get_blas_funcs((name,), (ndarray,))[0]


def argsort(x, topn=None, reverse=False):
    """Efficiently calculate indices of the `topn` smallest elements in array `x`.

    Parameters
    ----------
    x : array_like
        Array to get the smallest element indices from.
    topn : int, optional
        Number of indices of the smallest (greatest) elements to be returned.
        If not given, indices of all elements will be returned in ascending (descending) order.
    reverse : bool, optional
        Return the `topn` greatest elements in descending order,
        instead of smallest elements in ascending order?

    Returns
    -------
    numpy.ndarray
        Array of `topn` indices that sort the array in the requested order.

    """
    x = np.asarray(x)  # unify code path for when `x` is not a np array (list, tuple...)
    if topn is None:
        topn = x.size
    if topn <= 0:
        return []
    if reverse:
        x = -x
    if topn >= x.size or not hasattr(np, 'argpartition'):
        return np.argsort(x)[:topn]
    # np >= 1.8 has a fast partial argsort, use that!
    most_extreme = np.argpartition(x, topn)[:topn]
    return most_extreme.take(np.argsort(x.take(most_extreme)))  # resort topn into order


def corpus2csc(corpus, num_terms=None, dtype=np.float64, num_docs=None, num_nnz=None, printprogress=0):
    """Convert a streamed corpus in bag-of-words format into a sparse matrix `scipy.sparse.csc_matrix`,
    with documents as columns.

    Notes
    -----
    If the number of terms, documents and non-zero elements is known, you can pass
    them here as parameters and a (much) more memory efficient code path will be taken.

    Parameters
    ----------
    corpus : iterable of iterable of (int, number)
        Input corpus in BoW format
    num_terms : int, optional
        Number of terms in `corpus`. If provided, the `corpus.num_terms` attribute (if any) will be ignored.
    dtype : data-type, optional
        Data type of output CSC matrix.
    num_docs : int, optional
        Number of documents in `corpus`. If provided, the `corpus.num_docs` attribute (in any) will be ignored.
    num_nnz : int, optional
        Number of non-zero elements in `corpus`. If provided, the `corpus.num_nnz` attribute (if any) will be ignored.
    printprogress : int, optional
        Log a progress message at INFO level once every `printprogress` documents. 0 to turn off progress logging.

    Returns
    -------
    scipy.sparse.csc_matrix
        `corpus` converted into a sparse CSC matrix.

    See Also
    --------
    :class:`~gensim.matutils.Sparse2Corpus`
        Convert sparse format to Gensim corpus format.

    """
    try:
        # if the input corpus has the `num_nnz`, `num_docs` and `num_terms` attributes
        # (as is the case with MmCorpus for example), we can use a more efficient code path
        if num_terms is None:
            num_terms = corpus.num_terms
        if num_docs is None:
            num_docs = corpus.num_docs
        if num_nnz is None:
            num_nnz = corpus.num_nnz
    except AttributeError:
        pass  # not a MmCorpus...
    if printprogress:
        logger.info("creating sparse matrix from corpus")
    if num_terms is not None and num_docs is not None and num_nnz is not None:
        # faster and much more memory-friendly version of creating the sparse csc
        posnow, indptr = 0, [0]
        indices = np.empty((num_nnz,), dtype=np.int32)  # HACK assume feature ids fit in 32bit integer
        data = np.empty((num_nnz,), dtype=dtype)
        for docno, doc in enumerate(corpus):
            if printprogress and docno % printprogress == 0:
                logger.info("PROGRESS: at document #%i/%i", docno, num_docs)
            posnext = posnow + len(doc)
            indices[posnow: posnext] = [feature_id for feature_id, _ in doc]
            data[posnow: posnext] = [feature_weight for _, feature_weight in doc]
            indptr.append(posnext)
            posnow = posnext
        assert posnow == num_nnz, "mismatch between supplied and computed number of non-zeros"
        result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, num_docs), dtype=dtype)
    else:
        # slower version; determine the sparse matrix parameters during iteration
        num_nnz, data, indices, indptr = 0, [], [], [0]
        for docno, doc in enumerate(corpus):
            if printprogress and docno % printprogress == 0:
                logger.info("PROGRESS: at document #%i", docno)
            indices.extend([feature_id for feature_id, _ in doc])
            data.extend([feature_weight for _, feature_weight in doc])
            num_nnz += len(doc)
            indptr.append(num_nnz)
        if num_terms is None:
            num_terms = max(indices) + 1 if indices else 0
        num_docs = len(indptr) - 1
        # now num_docs, num_terms and num_nnz contain the correct values
        data = np.asarray(data, dtype=dtype)
        indices = np.asarray(indices)
        result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, num_docs), dtype=dtype)
    return result


def pad(mat, padrow, padcol):
    """Add additional rows/columns to `mat`. The new rows/columns will be initialized with zeros.

    Parameters
    ----------
    mat : numpy.ndarray
        Input 2D matrix
    padrow : int
        Number of additional rows
    padcol : int
        Number of additional columns

    Returns
    -------
    numpy.matrixlib.defmatrix.matrix
        Matrix with needed padding.

    """
    if padrow < 0:
        padrow = 0
    if padcol < 0:
        padcol = 0
    rows, cols = mat.shape
    return np.bmat([
        [mat, np.matrix(np.zeros((rows, padcol)))],
        [np.matrix(np.zeros((padrow, cols + padcol)))],
    ])


def zeros_aligned(shape, dtype, order='C', align=128):
    """Get array aligned at `align` byte boundary in memory.

    Parameters
    ----------
    shape : int or (int, int)
        Shape of array.
    dtype : data-type
        Data type of array.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous (row- or column-wise) order in memory.
    align : int, optional
        Boundary for alignment in bytes.

    Returns
    -------
    numpy.ndarray
        Aligned array.

    """
    nbytes = np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize
    buffer = np.zeros(nbytes + align, dtype=np.uint8)  # problematic on win64 ("maximum allowed dimension exceeded")
    start_index = -buffer.ctypes.data % align
    return buffer[start_index: start_index + nbytes].view(dtype).reshape(shape, order=order)


def ismatrix(m):
    """Check whether `m` is a 2D `numpy.ndarray` or `scipy.sparse` matrix.

    Parameters
    ----------
    m : object
        Object to check.

    Returns
    -------
    bool
        Is `m` a 2D `numpy.ndarray` or `scipy.sparse` matrix.

    """
    return isinstance(m, np.ndarray) and m.ndim == 2 or scipy.sparse.issparse(m)


def any2sparse(vec, eps=1e-9):
    """Convert a numpy.ndarray or `scipy.sparse` vector into the Gensim bag-of-words format.

    Parameters
    ----------
    vec : {`numpy.ndarray`, `scipy.sparse`}
        Input vector
    eps : float, optional
        Value used for threshold, all coordinates less than `eps` will not be presented in result.

    Returns
    -------
    list of (int, float)
        Vector in BoW format.

    """
    if isinstance(vec, np.ndarray):
        return dense2vec(vec, eps)
    if scipy.sparse.issparse(vec):
        return scipy2sparse(vec, eps)
    return [(int(fid), float(fw)) for fid, fw in vec if np.abs(fw) > eps]


def scipy2scipy_clipped(matrix, topn, eps=1e-9):
    """Get the 'topn' elements of the greatest magnitude (absolute value) from a `scipy.sparse` vector or matrix.

    Parameters
    ----------
    matrix : `scipy.sparse`
        Input vector or matrix (1D or 2D sparse array).
    topn : int
        Number of greatest elements, in absolute value, to return.
    eps : float
        Ignored.

    Returns
    -------
    `scipy.sparse.csr.csr_matrix`
        Clipped matrix.

    """
    if not scipy.sparse.issparse(matrix):
        raise ValueError("'%s' is not a scipy sparse vector." % matrix)
    if topn <= 0:
        return scipy.sparse.csr_matrix([])
    # Return clipped sparse vector if input is a sparse vector.
    if matrix.shape[0] == 1:
        # use np.argpartition/argsort and only form tuples that are actually returned.
        biggest = argsort(abs(matrix.data), topn, reverse=True)
        indices, data = matrix.indices.take(biggest), matrix.data.take(biggest)
        return scipy.sparse.csr_matrix((data, indices, [0, len(indices)]))
    # Return clipped sparse matrix if input is a matrix, processing row by row.
    else:
        matrix_indices = []
        matrix_data = []
        matrix_indptr = [0]
        # calling abs() on entire matrix once is faster than calling abs() iteratively for each row
        matrix_abs = abs(matrix)
        for i in range(matrix.shape[0]):
            v = matrix.getrow(i)
            v_abs = matrix_abs.getrow(i)
            # Sort and clip each row vector first.
            biggest = argsort(v_abs.data, topn, reverse=True)
            indices, data = v.indices.take(biggest), v.data.take(biggest)
            # Store the topn indices and values of each row vector.
            matrix_data.append(data)
            matrix_indices.append(indices)
            matrix_indptr.append(matrix_indptr[-1] + min(len(indices), topn))
        matrix_indices = np.concatenate(matrix_indices).ravel()
        matrix_data = np.concatenate(matrix_data).ravel()
        # Instantiate and return a sparse csr_matrix which preserves the order of indices/data.
        return scipy.sparse.csr.csr_matrix(
            (matrix_data, matrix_indices, matrix_indptr),
            shape=(matrix.shape[0], np.max(matrix_indices) + 1)
        )


def scipy2sparse(vec, eps=1e-9):
    """Convert a scipy.sparse vector into the Gensim bag-of-words format.

    Parameters
    ----------
    vec : `scipy.sparse`
        Sparse vector.

    eps : float, optional
        Value used for threshold, all coordinates less than `eps` will not be presented in result.

    Returns
    -------
    list of (int, float)
        Vector in Gensim bag-of-words format.

    """
    vec = vec.tocsr()
    assert vec.shape[0] == 1
    return [(int(pos), float(val)) for pos, val in zip(vec.indices, vec.data) if np.abs(val) > eps]


class Scipy2Corpus(object):
    """Convert a sequence of dense/sparse vectors into a streamed Gensim corpus object.

    See Also
    --------
    :func:`~gensim.matutils.corpus2csc`
        Convert corpus in Gensim format to `scipy.sparse.csc` matrix.

    """
    def __init__(self, vecs):
        """

        Parameters
        ----------
        vecs : iterable of {`numpy.ndarray`, `scipy.sparse`}
            Input vectors.

        """
        self.vecs = vecs

    def __iter__(self):
        for vec in self.vecs:
            if isinstance(vec, np.ndarray):
                yield full2sparse(vec)
            else:
                yield scipy2sparse(vec)

    def __len__(self):
        return len(self.vecs)


def sparse2full(doc, length):
    """Convert a document in Gensim bag-of-words format into a dense numpy array.

    Parameters
    ----------
    doc : list of (int, number)
        Document in BoW format.
    length : int
        Vector dimensionality. This cannot be inferred from the BoW, and you must supply it explicitly.
        This is typically the vocabulary size or number of topics, depending on how you created `doc`.

    Returns
    -------
    numpy.ndarray
        Dense numpy vector for `doc`.

    See Also
    --------
    :func:`~gensim.matutils.full2sparse`
        Convert dense array to gensim bag-of-words format.

    """
    result = np.zeros(length, dtype=np.float32)  # fill with zeroes (default value)
    # convert indices to int as numpy 1.12 no longer indexes by floats
    doc = ((int(id_), float(val_)) for (id_, val_) in doc)

    doc = dict(doc)
    # overwrite some of the zeroes with explicit values
    result[list(doc)] = list(itervalues(doc))
    return result


def full2sparse(vec, eps=1e-9):
    """Convert a dense numpy array into the Gensim bag-of-words format.

    Parameters
    ----------
    vec : numpy.ndarray
        Dense input vector.
    eps : float
        Feature weight threshold value. Features with `abs(weight) < eps` are considered sparse and
        won't be included in the BOW result.

    Returns
    -------
    list of (int, float)
        BoW format of `vec`, with near-zero values omitted (sparse vector).

    See Also
    --------
    :func:`~gensim.matutils.sparse2full`
        Convert a document in Gensim bag-of-words format into a dense numpy array.

    """
    vec = np.asarray(vec, dtype=float)
    nnz = np.nonzero(abs(vec) > eps)[0]
    return list(zip(nnz, vec.take(nnz)))


dense2vec = full2sparse


def full2sparse_clipped(vec, topn, eps=1e-9):
    """Like :func:`~gensim.matutils.full2sparse`, but only return the `topn` elements of the greatest magnitude (abs).

    This is more efficient that sorting a vector and then taking the greatest values, especially
    where `len(vec) >> topn`.

    Parameters
    ----------
    vec : numpy.ndarray
        Input dense vector
    topn : int
        Number of greatest (abs) elements that will be presented in result.
    eps : float
        Threshold value, if coordinate in `vec` < eps, this will not be presented in result.

    Returns
    -------
    list of (int, float)
        Clipped vector in BoW format.

    See Also
    --------
    :func:`~gensim.matutils.full2sparse`
        Convert dense array to gensim bag-of-words format.

    """
    # use np.argpartition/argsort and only form tuples that are actually returned.
    # this is about 40x faster than explicitly forming all 2-tuples to run sort() or heapq.nlargest() on.
    if topn <= 0:
        return []
    vec = np.asarray(vec, dtype=float)
    nnz = np.nonzero(abs(vec) > eps)[0]
    biggest = nnz.take(argsort(abs(vec).take(nnz), topn, reverse=True))
    return list(zip(biggest, vec.take(biggest)))


def corpus2dense(corpus, num_terms, num_docs=None, dtype=np.float32):
    """Convert corpus into a dense numpy 2D array, with documents as columns.

    Parameters
    ----------
    corpus : iterable of iterable of (int, number)
        Input corpus in the Gensim bag-of-words format.
    num_terms : int
        Number of terms in the dictionary. X-axis of the resulting matrix.
    num_docs : int, optional
        Number of documents in the corpus. If provided, a slightly more memory-efficient code path is taken.
        Y-axis of the resulting matrix.
    dtype : data-type, optional
        Data type of the output matrix.

    Returns
    -------
    numpy.ndarray
        Dense 2D array that presents `corpus`.

    See Also
    --------
    :class:`~gensim.matutils.Dense2Corpus`
        Convert dense matrix to Gensim corpus format.

    """
    if num_docs is not None:
        # we know the number of documents => don't bother column_stacking
        docno, result = -1, np.empty((num_terms, num_docs), dtype=dtype)
        for docno, doc in enumerate(corpus):
            result[:, docno] = sparse2full(doc, num_terms)
        assert docno + 1 == num_docs
    else:
        result = np.column_stack(sparse2full(doc, num_terms) for doc in corpus)
    return result.astype(dtype)


class Dense2Corpus(object):
    """Treat dense numpy array as a streamed Gensim corpus in the bag-of-words format.

    Notes
    -----
    No data copy is made (changes to the underlying matrix imply changes in the streamed corpus).

    See Also
    --------
    :func:`~gensim.matutils.corpus2dense`
        Convert Gensim corpus to dense matrix.
    :class:`~gensim.matutils.Sparse2Corpus`
        Convert sparse matrix to Gensim corpus format.

    """
    def __init__(self, dense, documents_columns=True):
        """

        Parameters
        ----------
        dense : numpy.ndarray
            Corpus in dense format.
        documents_columns : bool, optional
            Documents in `dense` represented as columns, as opposed to rows?

        """
        if documents_columns:
            self.dense = dense.T
        else:
            self.dense = dense

    def __iter__(self):
        """Iterate over the corpus.

        Yields
        ------
        list of (int, float)
            Document in BoW format.

        """
        for doc in self.dense:
            yield full2sparse(doc.flat)

    def __len__(self):
        return len(self.dense)


class Sparse2Corpus(object):
    """Convert a matrix in scipy.sparse format into a streaming Gensim corpus.

    See Also
    --------
    :func:`~gensim.matutils.corpus2csc`
        Convert gensim corpus format to `scipy.sparse.csc` matrix
    :class:`~gensim.matutils.Dense2Corpus`
        Convert dense matrix to gensim corpus.

    """
    def __init__(self, sparse, documents_columns=True):
        """

        Parameters
        ----------
        sparse : `scipy.sparse`
            Corpus scipy sparse format
        documents_columns : bool, optional
            Documents will be column?

        """
        if documents_columns:
            self.sparse = sparse.tocsc()
        else:
            self.sparse = sparse.tocsr().T  # make sure shape[1]=number of docs (needed in len())

    def __iter__(self):
        """

        Yields
        ------
        list of (int, float)
            Document in BoW format.

        """
        for indprev, indnow in izip(self.sparse.indptr, self.sparse.indptr[1:]):
            yield list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))

    def __len__(self):
        return self.sparse.shape[1]

    def __getitem__(self, document_index):
        """Retrieve a document vector from the corpus by its index.

        Parameters
        ----------
        document_index : int
            Index of document

        Returns
        -------
        list of (int, number)
            Document in BoW format.

        """
        indprev = self.sparse.indptr[document_index]
        indnow = self.sparse.indptr[document_index + 1]
        return list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))


def veclen(vec):
    """Calculate L2 (euclidean) length of a vector.

    Parameters
    ----------
    vec : list of (int, number)
        Input vector in sparse bag-of-words format.

    Returns
    -------
    float
        Length of `vec`.

    """
    if len(vec) == 0:
        return 0.0
    length = 1.0 * math.sqrt(sum(val**2 for _, val in vec))
    assert length > 0.0, "sparse documents must not contain any explicit zero entries"
    return length


def ret_normalized_vec(vec, length):
    """Normalize a vector in L2 (Euclidean unit norm).

    Parameters
    ----------
    vec : list of (int, number)
        Input vector in BoW format.
    length : float
        Length of vector

    Returns
    -------
    list of (int, number)
        L2-normalized vector in BoW format.

    """
    if length != 1.0:
        return [(termid, val / length) for termid, val in vec]
    else:
        return list(vec)


def ret_log_normalize_vec(vec, axis=1):
    log_max = 100.0
    if len(vec.shape) == 1:
        max_val = np.max(vec)
        log_shift = log_max - np.log(len(vec) + 1.0) - max_val
        tot = np.sum(np.exp(vec + log_shift))
        log_norm = np.log(tot) - log_shift
        vec -= log_norm
    else:
        if axis == 1:  # independently normalize each sample
            max_val = np.max(vec, 1)
            log_shift = log_max - np.log(vec.shape[1] + 1.0) - max_val
            tot = np.sum(np.exp(vec + log_shift[:, np.newaxis]), 1)
            log_norm = np.log(tot) - log_shift
            vec = vec - log_norm[:, np.newaxis]
        elif axis == 0:  # normalize each feature
            k = ret_log_normalize_vec(vec.T)
            return k[0].T, k[1]
        else:
            raise ValueError("'%s' is not a supported axis" % axis)
    return vec, log_norm


blas_nrm2 = blas('nrm2', np.array([], dtype=float))
blas_scal = blas('scal', np.array([], dtype=float))


def unitvec(vec, norm='l2', return_norm=False):
    """Scale a vector to unit length.

    Parameters
    ----------
    vec : {numpy.ndarray, scipy.sparse, list of (int, float)}
        Input vector in any format
    norm : {'l1', 'l2'}, optional
        Metric to normalize in.
    return_norm : bool, optional
        Return the length of vector `vec`, in addition to the normalized vector itself?

    Returns
    -------
    numpy.ndarray, scipy.sparse, list of (int, float)}
        Normalized vector in same format as `vec`.
    float
        Length of `vec` before normalization, if `return_norm` is set.

    Notes
    -----
    Zero-vector will be unchanged.

    """
    if norm not in ('l1', 'l2'):
        raise ValueError("'%s' is not a supported norm. Currently supported norms are 'l1' and 'l2'." % norm)

    if scipy.sparse.issparse(vec):
        vec = vec.tocsr()
        if norm == 'l1':
            veclen = np.sum(np.abs(vec.data))
        if norm == 'l2':
            veclen = np.sqrt(np.sum(vec.data ** 2))
        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.int):
                vec = vec.astype(np.float)
            vec /= veclen
            if return_norm:
                return vec, veclen
            else:
                return vec
        else:
            if return_norm:
                return vec, 1.
            else:
                return vec

    if isinstance(vec, np.ndarray):
        if norm == 'l1':
            veclen = np.sum(np.abs(vec))
        if norm == 'l2':
            veclen = blas_nrm2(vec)
        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.int):
                vec = vec.astype(np.float)
            if return_norm:
                return blas_scal(1.0 / veclen, vec).astype(vec.dtype), veclen
            else:
                return blas_scal(1.0 / veclen, vec).astype(vec.dtype)
        else:
            if return_norm:
                return vec, 1
            else:
                return vec

    try:
        first = next(iter(vec))  # is there at least one element?
    except StopIteration:
        return vec

    if isinstance(first, (tuple, list)) and len(first) == 2:  # gensim sparse format
        if norm == 'l1':
            length = float(sum(abs(val) for _, val in vec))
        if norm == 'l2':
            length = 1.0 * math.sqrt(sum(val ** 2 for _, val in vec))
        assert length > 0.0, "sparse documents must not contain any explicit zero entries"
        if return_norm:
            return ret_normalized_vec(vec, length), length
        else:
            return ret_normalized_vec(vec, length)
    else:
        raise ValueError("unknown input type")


def cossim(vec1, vec2):
    """Get cosine similarity between two sparse vectors.

    Cosine similarity is a number between `<-1.0, 1.0>`, higher means more similar.

    Parameters
    ----------
    vec1 : list of (int, float)
        Vector in BoW format.
    vec2 : list of (int, float)
        Vector in BoW format.

    Returns
    -------
    float
        Cosine similarity between `vec1` and `vec2`.

    """
    vec1, vec2 = dict(vec1), dict(vec2)
    if not vec1 or not vec2:
        return 0.0
    vec1len = 1.0 * math.sqrt(sum(val * val for val in itervalues(vec1)))
    vec2len = 1.0 * math.sqrt(sum(val * val for val in itervalues(vec2)))
    assert vec1len > 0.0 and vec2len > 0.0, "sparse documents must not contain any explicit zero entries"
    if len(vec2) < len(vec1):
        vec1, vec2 = vec2, vec1  # swap references so that we iterate over the shorter vector
    result = sum(value * vec2.get(index, 0.0) for index, value in iteritems(vec1))
    result /= vec1len * vec2len  # rescale by vector lengths
    return result


def softcossim(vec1, vec2, similarity_matrix):
    """Get Soft Cosine Measure between two vectors given a term similarity matrix.

    Return Soft Cosine Measure between two sparse vectors given a sparse term similarity matrix
    in the :class:`scipy.sparse.csc_matrix` format. The similarity is a number between `<-1.0, 1.0>`,
    higher is more similar.

    Notes
    -----
    Soft Cosine Measure was perhaps first defined by `Grigori Sidorov et al.,
    "Soft Similarity and Soft Cosine Measure: Similarity of Features in Vector Space Model"
    <http://www.cys.cic.ipn.mx/ojs/index.php/CyS/article/view/2043/1921>`_.

    Parameters
    ----------
    vec1 : list of (int, float)
        A query vector in the BoW format.
    vec2 : list of (int, float)
        A document vector in the BoW format.
    similarity_matrix : {:class:`scipy.sparse.csc_matrix`, :class:`scipy.sparse.csr_matrix`}
        A term similarity matrix, typically produced by
        :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similarity_matrix`.

    Returns
    -------
    `similarity_matrix.dtype`
        The Soft Cosine Measure between `vec1` and `vec2`.

    Raises
    ------
    ValueError
        When the term similarity matrix is in an unknown format.

    See Also
    --------
    :meth:`gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similarity_matrix`
        A term similarity matrix produced from term embeddings.
    :class:`gensim.similarities.docsim.SoftCosineSimilarity`
        A class for performing corpus-based similarity queries with Soft Cosine Measure.

    """
    if not isinstance(similarity_matrix, scipy.sparse.csc_matrix):
        if isinstance(similarity_matrix, scipy.sparse.csr_matrix):
            similarity_matrix = similarity_matrix.T
        else:
            raise ValueError('unknown similarity matrix format')

    if not vec1 or not vec2:
        return 0.0

    vec1 = dict(vec1)
    vec2 = dict(vec2)
    word_indices = sorted(set(chain(vec1, vec2)))
    dtype = similarity_matrix.dtype
    vec1 = np.array([vec1[i] if i in vec1 else 0 for i in word_indices], dtype=dtype)
    vec2 = np.array([vec2[i] if i in vec2 else 0 for i in word_indices], dtype=dtype)
    dense_matrix = similarity_matrix[[[i] for i in word_indices], word_indices].todense()
    vec1len = vec1.T.dot(dense_matrix).dot(vec1)[0, 0]
    vec2len = vec2.T.dot(dense_matrix).dot(vec2)[0, 0]

    assert \
        vec1len > 0.0 and vec2len > 0.0, \
        u"sparse documents must not contain any explicit zero entries and the similarity matrix S " \
        u"must satisfy x^T * S * x > 0 for any nonzero bag-of-words vector x."

    result = vec1.T.dot(dense_matrix).dot(vec2)[0, 0]
    result /= math.sqrt(vec1len) * math.sqrt(vec2len)  # rescale by vector lengths
    return np.clip(result, -1.0, 1.0)


def isbow(vec):
    """Checks if a vector is in the sparse Gensim bag-of-words format.

    Parameters
    ----------
    vec : object
        Object to check.

    Returns
    -------
    bool
        Is `vec` in BoW format.

    """
    if scipy.sparse.issparse(vec):
        vec = vec.todense().tolist()
    try:
        id_, val_ = vec[0]  # checking first value to see if it is in bag of words format by unpacking
        int(id_), float(val_)
    except IndexError:
        return True  # this is to handle the empty input case
    except (ValueError, TypeError):
        return False
    return True


def _convert_vec(vec1, vec2, num_features=None):
    if scipy.sparse.issparse(vec1):
        vec1 = vec1.toarray()
    if scipy.sparse.issparse(vec2):
        vec2 = vec2.toarray()  # converted both the vectors to dense in case they were in sparse matrix
    if isbow(vec1) and isbow(vec2):  # if they are in bag of words format we make it dense
        if num_features is not None:  # if not None, make as large as the documents drawing from
            dense1 = sparse2full(vec1, num_features)
            dense2 = sparse2full(vec2, num_features)
            return dense1, dense2
        else:
            max_len = max(len(vec1), len(vec2))
            dense1 = sparse2full(vec1, max_len)
            dense2 = sparse2full(vec2, max_len)
            return dense1, dense2
    else:
        # this conversion is made because if it is not in bow format, it might be a list within a list after conversion
        # the scipy implementation of Kullback fails in such a case so we pick up only the nested list.
        if len(vec1) == 1:
            vec1 = vec1[0]
        if len(vec2) == 1:
            vec2 = vec2[0]
        return vec1, vec2


def kullback_leibler(vec1, vec2, num_features=None):
    """Calculate Kullback-Leibler distance between two probability distributions using `scipy.stats.entropy`.

    Parameters
    ----------
    vec1 : {scipy.sparse, numpy.ndarray, list of (int, float)}
        Distribution vector.
    vec2 : {scipy.sparse, numpy.ndarray, list of (int, float)}
        Distribution vector.
    num_features : int, optional
        Number of features in the vectors.

    Returns
    -------
    float
        Kullback-Leibler distance between `vec1` and `vec2`.
        Value in range [0, +∞) where values closer to 0 mean less distance (higher similarity).

    """
    vec1, vec2 = _convert_vec(vec1, vec2, num_features=num_features)
    return entropy(vec1, vec2)


def jensen_shannon(vec1, vec2, num_features=None):
    """Calculate Jensen-Shannon distance between two probability distributions using `scipy.stats.entropy`.

    Parameters
    ----------
    vec1 : {scipy.sparse, numpy.ndarray, list of (int, float)}
        Distribution vector.
    vec2 : {scipy.sparse, numpy.ndarray, list of (int, float)}
        Distribution vector.
    num_features : int, optional
        Number of features in the vectors.

    Returns
    -------
    float
        Jensen-Shannon distance between `vec1` and `vec2`.

    Notes
    -----
    This is a symmetric and finite "version" of :func:`gensim.matutils.kullback_leibler`.

    """
    vec1, vec2 = _convert_vec(vec1, vec2, num_features=num_features)
    avg_vec = 0.5 * (vec1 + vec2)
    return 0.5 * (entropy(vec1, avg_vec) + entropy(vec2, avg_vec))


def hellinger(vec1, vec2):
    """Calculate Hellinger distance between two probability distributions.

    Parameters
    ----------
    vec1 : {scipy.sparse, numpy.ndarray, list of (int, float)}
        Distribution vector.
    vec2 : {scipy.sparse, numpy.ndarray, list of (int, float)}
        Distribution vector.

    Returns
    -------
    float
        Hellinger distance between `vec1` and `vec2`.
        Value in range `[0, 1]`, where 0 is min distance (max similarity) and 1 is max distance (min similarity).

    """
    if scipy.sparse.issparse(vec1):
        vec1 = vec1.toarray()
    if scipy.sparse.issparse(vec2):
        vec2 = vec2.toarray()
    if isbow(vec1) and isbow(vec2):
        # if it is a BoW format, instead of converting to dense we use dictionaries to calculate appropriate distance
        vec1, vec2 = dict(vec1), dict(vec2)
        indices = set(list(vec1.keys()) + list(vec2.keys()))
        sim = np.sqrt(
            0.5 * sum((np.sqrt(vec1.get(index, 0.0)) - np.sqrt(vec2.get(index, 0.0)))**2 for index in indices)
        )
        return sim
    else:
        sim = np.sqrt(0.5 * ((np.sqrt(vec1) - np.sqrt(vec2))**2).sum())
        return sim


def jaccard(vec1, vec2):
    """Calculate Jaccard distance between two vectors.

    Parameters
    ----------
    vec1 : {scipy.sparse, numpy.ndarray, list of (int, float)}
        Distribution vector.
    vec2 : {scipy.sparse, numpy.ndarray, list of (int, float)}
        Distribution vector.

    Returns
    -------
    float
        Jaccard distance between `vec1` and `vec2`.
        Value in range `[0, 1]`, where 0 is min distance (max similarity) and 1 is max distance (min similarity).

    """

    # converting from sparse for easier manipulation
    if scipy.sparse.issparse(vec1):
        vec1 = vec1.toarray()
    if scipy.sparse.issparse(vec2):
        vec2 = vec2.toarray()
    if isbow(vec1) and isbow(vec2):
        # if it's in bow format, we use the following definitions:
        # union = sum of the 'weights' of both the bags
        # intersection = lowest weight for a particular id; basically the number of common words or items
        union = sum(weight for id_, weight in vec1) + sum(weight for id_, weight in vec2)
        vec1, vec2 = dict(vec1), dict(vec2)
        intersection = 0.0
        for feature_id, feature_weight in iteritems(vec1):
            intersection += min(feature_weight, vec2.get(feature_id, 0.0))
        return 1 - float(intersection) / float(union)
    else:
        # if it isn't in bag of words format, we can use sets to calculate intersection and union
        if isinstance(vec1, np.ndarray):
            vec1 = vec1.tolist()
        if isinstance(vec2, np.ndarray):
            vec2 = vec2.tolist()
        vec1 = set(vec1)
        vec2 = set(vec2)
        intersection = vec1 & vec2
        union = vec1 | vec2
        return 1 - float(len(intersection)) / float(len(union))


def jaccard_distance(set1, set2):
    """Calculate Jaccard distance between two sets.

    Parameters
    ----------
    set1 : set
        Input set.
    set2 : set
        Input set.

    Returns
    -------
    float
        Jaccard distance between `set1` and `set2`.
        Value in range `[0, 1]`, where 0 is min distance (max similarity) and 1 is max distance (min similarity).
    """

    union_cardinality = len(set1 | set2)
    if union_cardinality == 0:  # Both sets are empty
        return 1.

    return 1. - float(len(set1 & set2)) / float(union_cardinality)


try:
    # try to load fast, cythonized code if possible
    from gensim._matutils import logsumexp, mean_absolute_difference, dirichlet_expectation

except ImportError:
    def logsumexp(x):
        """Log of sum of exponentials.

        Parameters
        ----------
        x : numpy.ndarray
            Input 2d matrix.

        Returns
        -------
        float
            log of sum of exponentials of elements in `x`.

        Warnings
        --------
        For performance reasons, doesn't support NaNs or 1d, 3d, etc arrays like :func:`scipy.special.logsumexp`.

        """
        x_max = np.max(x)
        x = np.log(np.sum(np.exp(x - x_max)))
        x += x_max

        return x

    def mean_absolute_difference(a, b):
        """Mean absolute difference between two arrays.

        Parameters
        ----------
        a : numpy.ndarray
            Input 1d array.
        b : numpy.ndarray
            Input 1d array.

        Returns
        -------
        float
            mean(abs(a - b)).

        """
        return np.mean(np.abs(a - b))

    def dirichlet_expectation(alpha):
        """Expected value of log(theta) where theta is drawn from a Dirichlet distribution.

        Parameters
        ----------
        alpha : numpy.ndarray
            Dirichlet parameter 2d matrix or 1d vector, if 2d - each row is treated as a separate parameter vector.

        Returns
        -------
        numpy.ndarray
            Log of expected values, dimension same as `alpha.ndim`.

        """
        if len(alpha.shape) == 1:
            result = psi(alpha) - psi(np.sum(alpha))
        else:
            result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
        return result.astype(alpha.dtype, copy=False)  # keep the same precision as input


def qr_destroy(la):
    """Get QR decomposition of `la[0]`.

    Parameters
    ----------
    la : list of numpy.ndarray
        Run QR decomposition on the first elements of `la`. Must not be empty.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Matrices :math:`Q` and :math:`R`.

    Notes
    -----
    Using this function is less memory intense than calling `scipy.linalg.qr(la[0])`,
    because the memory used in `la[0]` is reclaimed earlier. This makes a difference when
    decomposing very large arrays, where every memory copy counts.

    Warnings
    --------
    Content of `la` as well as `la[0]` gets destroyed in the process. Again, for memory-effiency reasons.

    """
    a = np.asfortranarray(la[0])
    del la[0], la  # now `a` is the only reference to the input matrix
    m, n = a.shape
    # perform q, r = QR(a); code hacked out of scipy.linalg.qr
    logger.debug("computing QR of %s dense matrix", str(a.shape))
    geqrf, = get_lapack_funcs(('geqrf',), (a,))
    qr, tau, work, info = geqrf(a, lwork=-1, overwrite_a=True)
    qr, tau, work, info = geqrf(a, lwork=work[0], overwrite_a=True)
    del a  # free up mem
    assert info >= 0
    r = triu(qr[:n, :n])
    if m < n:  # rare case, #features < #topics
        qr = qr[:, :m]  # retains fortran order
    gorgqr, = get_lapack_funcs(('orgqr',), (qr,))
    q, work, info = gorgqr(qr, tau, lwork=-1, overwrite_a=True)
    q, work, info = gorgqr(qr, tau, lwork=work[0], overwrite_a=True)
    assert info >= 0, "qr failed"
    assert q.flags.f_contiguous
    return q, r


class MmWriter(object):
    """Store a corpus in `Matrix Market format <https://math.nist.gov/MatrixMarket/formats.html>`_,
    using :class:`~gensim.corpora.mmcorpus.MmCorpus`.

    Notes
    -----
    The output is written one document at a time, not the whole matrix at once (unlike e.g. `scipy.io.mmread`).
    This allows you to write corpora which are larger than the available RAM.

    The output file is created in a single pass through the input corpus, so that the input can be
    a once-only stream (generator).

    To achieve this, a fake MM header is written first, corpus statistics are collected
    during the pass (shape of the matrix, number of non-zeroes), followed by a seek back to the beginning of the file,
    rewriting the fake header with the final values.

    """
    HEADER_LINE = b'%%MatrixMarket matrix coordinate real general\n'  # the only supported MM format

    def __init__(self, fname):
        """

        Parameters
        ----------
        fname : str
            Path to output file.

        """
        self.fname = fname
        if fname.endswith(".gz") or fname.endswith('.bz2'):
            raise NotImplementedError("compressed output not supported with MmWriter")
        self.fout = utils.smart_open(self.fname, 'wb+')  # open for both reading and writing
        self.headers_written = False

    def write_headers(self, num_docs, num_terms, num_nnz):
        """Write headers to file.

        Parameters
        ----------
        num_docs : int
            Number of documents in corpus.
        num_terms : int
            Number of term in corpus.
        num_nnz : int
            Number of non-zero elements in corpus.

        """
        self.fout.write(MmWriter.HEADER_LINE)

        if num_nnz < 0:
            # we don't know the matrix shape/density yet, so only log a general line
            logger.info("saving sparse matrix to %s", self.fname)
            self.fout.write(utils.to_utf8(' ' * 50 + '\n'))  # 48 digits must be enough for everybody
        else:
            logger.info(
                "saving sparse %sx%s matrix with %i non-zero entries to %s",
                num_docs, num_terms, num_nnz, self.fname
            )
            self.fout.write(utils.to_utf8('%s %s %s\n' % (num_docs, num_terms, num_nnz)))
        self.last_docno = -1
        self.headers_written = True

    def fake_headers(self, num_docs, num_terms, num_nnz):
        """Write "fake" headers to file, to be rewritten once we've scanned the entire corpus.

        Parameters
        ----------
        num_docs : int
            Number of documents in corpus.
        num_terms : int
            Number of term in corpus.
        num_nnz : int
            Number of non-zero elements in corpus.

        """
        stats = '%i %i %i' % (num_docs, num_terms, num_nnz)
        if len(stats) > 50:
            raise ValueError('Invalid stats: matrix too large!')
        self.fout.seek(len(MmWriter.HEADER_LINE))
        self.fout.write(utils.to_utf8(stats))

    def write_vector(self, docno, vector):
        """Write a single sparse vector to the file.

        Parameters
        ----------
        docno : int
            Number of document.
        vector : list of (int, number)
            Document in BoW format.

        Returns
        -------
        (int, int)
            Max word index in vector and len of vector. If vector is empty, return (-1, 0).

        """
        assert self.headers_written, "must write Matrix Market file headers before writing data!"
        assert self.last_docno < docno, "documents %i and %i not in sequential order!" % (self.last_docno, docno)
        vector = sorted((i, w) for i, w in vector if abs(w) > 1e-12)  # ignore near-zero entries
        for termid, weight in vector:  # write term ids in sorted order
            # +1 because MM format starts counting from 1
            self.fout.write(utils.to_utf8("%i %i %s\n" % (docno + 1, termid + 1, weight)))
        self.last_docno = docno
        return (vector[-1][0], len(vector)) if vector else (-1, 0)

    @staticmethod
    def write_corpus(fname, corpus, progress_cnt=1000, index=False, num_terms=None, metadata=False):
        """Save the corpus to disk in `Matrix Market format <https://math.nist.gov/MatrixMarket/formats.html>`_.

        Parameters
        ----------
        fname : str
            Filename of the resulting file.
        corpus : iterable of list of (int, number)
            Corpus in streamed bag-of-words format.
        progress_cnt : int, optional
            Print progress for every `progress_cnt` number of documents.
        index : bool, optional
            Return offsets?
        num_terms : int, optional
            Number of terms in the corpus. If provided, the `corpus.num_terms` attribute (if any) will be ignored.
        metadata : bool, optional
            Generate a metadata file?

        Returns
        -------
        offsets : {list of int, None}
            List of offsets (if index=True) or nothing.

        Notes
        -----
        Documents are processed one at a time, so the whole corpus is allowed to be larger than the available RAM.

        See Also
        --------
        :func:`gensim.corpora.mmcorpus.MmCorpus.save_corpus`
            Save corpus to disk.

        """
        mw = MmWriter(fname)

        # write empty headers to the file (with enough space to be overwritten later)
        mw.write_headers(-1, -1, -1)  # will print 50 spaces followed by newline on the stats line

        # calculate necessary header info (nnz elements, num terms, num docs) while writing out vectors
        _num_terms, num_nnz = 0, 0
        docno, poslast = -1, -1
        offsets = []
        if hasattr(corpus, 'metadata'):
            orig_metadata = corpus.metadata
            corpus.metadata = metadata
            if metadata:
                docno2metadata = {}
        else:
            metadata = False
        for docno, doc in enumerate(corpus):
            if metadata:
                bow, data = doc
                docno2metadata[docno] = data
            else:
                bow = doc
            if docno % progress_cnt == 0:
                logger.info("PROGRESS: saving document #%i", docno)
            if index:
                posnow = mw.fout.tell()
                if posnow == poslast:
                    offsets[-1] = -1
                offsets.append(posnow)
                poslast = posnow
            max_id, veclen = mw.write_vector(docno, bow)
            _num_terms = max(_num_terms, 1 + max_id)
            num_nnz += veclen
        if metadata:
            utils.pickle(docno2metadata, fname + '.metadata.cpickle')
            corpus.metadata = orig_metadata

        num_docs = docno + 1
        num_terms = num_terms or _num_terms

        if num_docs * num_terms != 0:
            logger.info(
                "saved %ix%i matrix, density=%.3f%% (%i/%i)",
                num_docs, num_terms, 100.0 * num_nnz / (num_docs * num_terms), num_nnz, num_docs * num_terms
            )

        # now write proper headers, by seeking and overwriting the spaces written earlier
        mw.fake_headers(num_docs, num_terms, num_nnz)

        mw.close()
        if index:
            return offsets

    def __del__(self):
        """Close `self.fout` file. Alias for :meth:`~gensim.matutils.MmWriter.close`.

        Warnings
        --------
        Closing the file explicitly via the close() method is preferred and safer.

        """
        self.close()  # does nothing if called twice (on an already closed file), so no worries

    def close(self):
        """Close `self.fout` file."""
        logger.debug("closing %s", self.fname)
        if hasattr(self, 'fout'):
            self.fout.close()


try:
    # try to load fast, cythonized code if possible
    from gensim.corpora._mmreader import MmReader
except ImportError:
    FAST_VERSION = -1

    class MmReader(object):
        """Matrix market file reader, used internally in :class:`~gensim.corpora.mmcorpus.MmCorpus`.

        Wrap a term-document matrix on disk (in matrix-market format), and present it
        as an object which supports iteration over the rows (~documents).

        Attributes
        ----------
        num_docs : int
            Number of documents in market matrix file.
        num_terms : int
            Number of terms.
        num_nnz : int
            Number of non-zero terms.

        Notes
        -----
        Note that the file is read into memory one document at a time, not the whole matrix at once
        (unlike e.g. `scipy.io.mmread` and other implementations).
        This allows us to process corpora which are larger than the available RAM.

        """
        def __init__(self, input, transposed=True):
            """

            Parameters
            ----------
            input : {str, file-like object}
                Path to the input file in MM format or a file-like object that supports `seek()`
                (e.g. smart_open objects).
            transposed : bool, optional
                Do lines represent `doc_id, term_id, value`, instead of `term_id, doc_id, value`?

            """
            logger.info("initializing corpus reader from %s", input)
            self.input, self.transposed = input, transposed
            with utils.open_file(self.input) as lines:
                try:
                    header = utils.to_unicode(next(lines)).strip()
                    if not header.lower().startswith('%%matrixmarket matrix coordinate real general'):
                        raise ValueError(
                            "File %s not in Matrix Market format with coordinate real general; instead found: \n%s" %
                            (self.input, header)
                        )
                except StopIteration:
                    pass

                self.num_docs = self.num_terms = self.num_nnz = 0
                for lineno, line in enumerate(lines):
                    line = utils.to_unicode(line)
                    if not line.startswith('%'):
                        self.num_docs, self.num_terms, self.num_nnz = (int(x) for x in line.split())
                        if not self.transposed:
                            self.num_docs, self.num_terms = self.num_terms, self.num_docs
                        break

            logger.info(
                "accepted corpus with %i documents, %i features, %i non-zero entries",
                self.num_docs, self.num_terms, self.num_nnz
            )

        def __len__(self):
            """Get the corpus size: total number of documents."""
            return self.num_docs

        def __str__(self):
            return ("MmCorpus(%i documents, %i features, %i non-zero entries)" %
                    (self.num_docs, self.num_terms, self.num_nnz))

        def skip_headers(self, input_file):
            """Skip file headers that appear before the first document.

            Parameters
            ----------
            input_file : iterable of str
                Iterable taken from file in MM format.

            """
            for line in input_file:
                if line.startswith(b'%'):
                    continue
                break

        def __iter__(self):
            """Iterate through all documents in the corpus.

            Notes
            ------
            Note that the total number of vectors returned is always equal to the number of rows specified
            in the header: empty documents are inserted and yielded where appropriate, even if they are not explicitly
            stored in the Matrix Market file.

            Yields
            ------
            (int, list of (int, number))
                Document id and document in sparse bag-of-words format.

            """
            with utils.file_or_filename(self.input) as lines:
                self.skip_headers(lines)

                previd = -1
                for line in lines:
                    docid, termid, val = utils.to_unicode(line).split()  # needed for python3
                    if not self.transposed:
                        termid, docid = docid, termid
                    # -1 because matrix market indexes are 1-based => convert to 0-based
                    docid, termid, val = int(docid) - 1, int(termid) - 1, float(val)
                    assert previd <= docid, "matrix columns must come in ascending order"
                    if docid != previd:
                        # change of document: return the document read so far (its id is prevId)
                        if previd >= 0:
                            yield previd, document  # noqa:F821

                        # return implicit (empty) documents between previous id and new id
                        # too, to keep consistent document numbering and corpus length
                        for previd in xrange(previd + 1, docid):
                            yield previd, []

                        # from now on start adding fields to a new document, with a new id
                        previd = docid
                        document = []

                    document.append((termid, val,))  # add another field to the current document

            # handle the last document, as a special case
            if previd >= 0:
                yield previd, document

            # return empty documents between the last explicit document and the number
            # of documents as specified in the header
            for previd in xrange(previd + 1, self.num_docs):
                yield previd, []

        def docbyoffset(self, offset):
            """Get the document at file offset `offset` (in bytes).

            Parameters
            ----------
            offset : int
                File offset, in bytes, of the desired document.

            Returns
            ------
            list of (int, str)
                Document in sparse bag-of-words format.

            """
            # empty documents are not stored explicitly in MM format, so the index marks
            # them with a special offset, -1.
            if offset == -1:
                return []
            if isinstance(self.input, string_types):
                fin, close_fin = utils.smart_open(self.input), True
            else:
                fin, close_fin = self.input, False

            fin.seek(offset)  # works for gzip/bz2 input, too
            previd, document = -1, []
            for line in fin:
                docid, termid, val = line.split()
                if not self.transposed:
                    termid, docid = docid, termid
                # -1 because matrix market indexes are 1-based => convert to 0-based
                docid, termid, val = int(docid) - 1, int(termid) - 1, float(val)
                assert previd <= docid, "matrix columns must come in ascending order"
                if docid != previd:
                    if previd >= 0:
                        break
                    previd = docid

                document.append((termid, val,))  # add another field to the current document

            if close_fin:
                fin.close()
            return document
