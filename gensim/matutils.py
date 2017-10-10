#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains math helper functions.
"""

from __future__ import with_statement


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


blas = lambda name, ndarray: scipy.linalg.get_blas_funcs((name,), (ndarray,))[0]

logger = logging.getLogger(__name__)


def argsort(x, topn=None, reverse=False):
    """
    Return indices of the `topn` smallest elements in array `x`, in ascending order.

    If reverse is True, return the greatest elements instead, in descending order.

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
    """
    Convert a streamed corpus into a sparse matrix, in scipy.sparse.csc_matrix format,
    with documents as columns.

    If the number of terms, documents and non-zero elements is known, you can pass
    them here as parameters and a more memory efficient code path will be taken.

    The input corpus may be a non-repeatable stream (generator).

    This is the mirror function to `Sparse2Corpus`.

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
    """
    Add additional rows/columns to a np.matrix `mat`. The new rows/columns
    will be initialized with zeros.
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
    """Like `np.zeros()`, but the array will be aligned at `align` byte boundary."""
    nbytes = np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize
    buffer = np.zeros(nbytes + align, dtype=np.uint8)  # problematic on win64 ("maximum allowed dimension exceeded")
    start_index = -buffer.ctypes.data % align
    return buffer[start_index: start_index + nbytes].view(dtype).reshape(shape, order=order)


def ismatrix(m):
    return isinstance(m, np.ndarray) and m.ndim == 2 or scipy.sparse.issparse(m)


def any2sparse(vec, eps=1e-9):
    """Convert a np/scipy vector into gensim document format (=list of 2-tuples)."""
    if isinstance(vec, np.ndarray):
        return dense2vec(vec, eps)
    if scipy.sparse.issparse(vec):
        return scipy2sparse(vec, eps)
    return [(int(fid), float(fw)) for fid, fw in vec if np.abs(fw) > eps]


def scipy2scipy_clipped(matrix, topn, eps=1e-9):
    """
    Return a scipy.sparse vector/matrix consisting of 'topn' elements of the greatest magnitude (absolute value).
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
        return scipy.sparse.csr.csr_matrix((matrix_data, matrix_indices, matrix_indptr), shape=(matrix.shape[0], np.max(matrix_indices) + 1))


def scipy2sparse(vec, eps=1e-9):
    """Convert a scipy.sparse vector into gensim document format (=list of 2-tuples)."""
    vec = vec.tocsr()
    assert vec.shape[0] == 1
    return [(int(pos), float(val)) for pos, val in zip(vec.indices, vec.data) if np.abs(val) > eps]


class Scipy2Corpus(object):
    """
    Convert a sequence of dense/sparse vectors into a streamed gensim corpus object.

    This is the mirror function to `corpus2csc`.

    """

    def __init__(self, vecs):
        """
        `vecs` is a sequence of dense and/or sparse vectors, such as a 2d np array,
        or a scipy.sparse.csc_matrix, or any sequence containing a mix of 1d np/scipy vectors.

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
    """
    Convert a document in sparse document format (=sequence of 2-tuples) into a dense
    np array (of size `length`).

    This is the mirror function to `full2sparse`.

    """
    result = np.zeros(length, dtype=np.float32)  # fill with zeroes (default value)
    # convert indices to int as numpy 1.12 no longer indexes by floats
    doc = ((int(id_), float(val_)) for (id_, val_) in doc)

    doc = dict(doc)
    # overwrite some of the zeroes with explicit values
    result[list(doc)] = list(itervalues(doc))
    return result


def full2sparse(vec, eps=1e-9):
    """
    Convert a dense np array into the sparse document format (sequence of 2-tuples).

    Values of magnitude < `eps` are treated as zero (ignored).

    This is the mirror function to `sparse2full`.

    """
    vec = np.asarray(vec, dtype=float)
    nnz = np.nonzero(abs(vec) > eps)[0]
    return list(zip(nnz, vec.take(nnz)))


dense2vec = full2sparse


def full2sparse_clipped(vec, topn, eps=1e-9):
    """
    Like `full2sparse`, but only return the `topn` elements of the greatest magnitude (abs).

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
    """
    Convert corpus into a dense np array (documents will be columns). You
    must supply the number of features `num_terms`, because dimensionality
    cannot be deduced from the sparse vectors alone.

    You can optionally supply `num_docs` (=the corpus length) as well, so that
    a more memory-efficient code path is taken.

    This is the mirror function to `Dense2Corpus`.

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
    """
    Treat dense np array as a sparse, streamed gensim corpus.

    No data copy is made (changes to the underlying matrix imply changes in the
    corpus).

    This is the mirror function to `corpus2dense`.

    """

    def __init__(self, dense, documents_columns=True):
        if documents_columns:
            self.dense = dense.T
        else:
            self.dense = dense

    def __iter__(self):
        for doc in self.dense:
            yield full2sparse(doc.flat)

    def __len__(self):
        return len(self.dense)


class Sparse2Corpus(object):
    """
    Convert a matrix in scipy.sparse format into a streaming gensim corpus.

    This is the mirror function to `corpus2csc`.

    """

    def __init__(self, sparse, documents_columns=True):
        if documents_columns:
            self.sparse = sparse.tocsc()
        else:
            self.sparse = sparse.tocsr().T  # make sure shape[1]=number of docs (needed in len())

    def __iter__(self):
        for indprev, indnow in izip(self.sparse.indptr, self.sparse.indptr[1:]):
            yield list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))

    def __len__(self):
        return self.sparse.shape[1]

    def __getitem__(self, item):
        indprev = self.sparse.indptr[item]
        indnow = self.sparse.indptr[item+1]
        return list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))


def veclen(vec):
    if len(vec) == 0:
        return 0.0
    length = 1.0 * math.sqrt(sum(val**2 for _, val in vec))
    assert length > 0.0, "sparse documents must not contain any explicit zero entries"
    return length


def ret_normalized_vec(vec, length):
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


def unitvec(vec, norm='l2'):
    """
    Scale a vector to unit length. The only exception is the zero vector, which
    is returned back unchanged.

    Output will be in the same format as input (i.e., gensim vector=>gensim vector,
    or np array=>np array, scipy.sparse=>scipy.sparse).
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
            return vec / veclen
        else:
            return vec

    if isinstance(vec, np.ndarray):
        vec = np.asarray(vec, dtype=float)
        if norm == 'l1':
            veclen = np.sum(np.abs(vec))
        if norm == 'l2':
            veclen = blas_nrm2(vec)
        if veclen > 0.0:
            return blas_scal(1.0 / veclen, vec)
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
        return ret_normalized_vec(vec, length)
    else:
        raise ValueError("unknown input type")


def cossim(vec1, vec2):
    """
    Return cosine similarity between two sparse vectors.
    The similarity is a number between <-1.0, 1.0>, higher is more similar.
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


def isbow(vec):
    """
    Checks if vector passed is in bag of words representation or not.
    Vec is considered to be in bag of words format if it is 2-tuple format.
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


def convert_vec(vec1, vec2, num_features=None):
    """
    Convert vectors to appropriate forms required by entropy input.
    Checks for sparsity and bag of word format.
    """
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
    """
    A distance metric between two probability distributions.
    Returns a distance value in range <0, +∞> where values closer to 0 mean less distance (and a higher similarity)
    Uses the scipy.stats.entropy method to identify kullback_leibler convergence value.
    If the distribution draws from a certain number of docs, that value must be passed.
    """
    vec1, vec2 = convert_vec(vec1, vec2, num_features=num_features)
    return entropy(vec1, vec2)


def jensen_shannon(vec1, vec2, num_features=None):
    """
    A method of measuring the similarity between two probability distributions.
    It is a symmetrized and finite version of the Kullback–Leibler divergence.
    """
    vec1, vec2 = convert_vec(vec1, vec2, num_features=num_features)
    avg_vec = 0.5 * (vec1 + vec2)
    return 0.5 * (entropy(vec1, avg_vec) + entropy(vec2, avg_vec))


def hellinger(vec1, vec2):
    """
    Hellinger distance is a distance metric to quantify the similarity between two probability distributions.
    Distance between distributions will be a number between <0,1>, where 0 is minimum distance (maximum similarity) and 1 is maximum distance (minimum similarity).
    """
    if scipy.sparse.issparse(vec1):
        vec1 = vec1.toarray()
    if scipy.sparse.issparse(vec2):
        vec2 = vec2.toarray()
    if isbow(vec1) and isbow(vec2):
        # if it is a bag of words format, instead of converting to dense we use dictionaries to calculate appropriate distance
        vec1, vec2 = dict(vec1), dict(vec2)
        if len(vec2) < len(vec1):
            vec1, vec2 = vec2, vec1  # swap references so that we iterate over the shorter vector
        sim = np.sqrt(0.5 * sum((np.sqrt(value) - np.sqrt(vec2.get(index, 0.0)))**2 for index, value in iteritems(vec1)))
        return sim
    else:
        sim = np.sqrt(0.5 * ((np.sqrt(vec1) - np.sqrt(vec2))**2).sum())
        return sim


def jaccard(vec1, vec2):
    """
    A distance metric between bags of words representation.
    Returns 1 minus the intersection divided by union, where union is the sum of the size of the two bags.
    If it is not a bag of words representation, the union and intersection is calculated in the traditional manner.
    Returns a value in range <0,1> where values closer to 0 mean less distance and thus higher similarity.

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
    """
    Calculate a distance between set representation (1 minus the intersection divided by union).
    Return a value in range <0, 1> where values closer to 0 mean smaller distance and thus higher similarity.
    """

    union_cardinality = len(set1 | set2)
    if union_cardinality == 0:  # Both sets are empty
        return 1.

    return 1. - float(len(set1 & set2)) / float(union_cardinality)


def dirichlet_expectation(alpha):
    """
    For a vector `theta~Dir(alpha)`, compute `E[log(theta)]`.

    """
    if len(alpha.shape) == 1:
        result = psi(alpha) - psi(np.sum(alpha))
    else:
        result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
    return result.astype(alpha.dtype)  # keep the same precision as input


def qr_destroy(la):
    """
    Return QR decomposition of `la[0]`. Content of `la` gets destroyed in the process.

    Using this function should be less memory intense than calling `scipy.linalg.qr(la[0])`,
    because the memory used in `la[0]` is reclaimed earlier.
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
    """
    Store a corpus in Matrix Market format.

    Note that the output is written one document at a time, not the whole
    matrix at once (unlike scipy.io.mmread). This allows us to process corpora
    which are larger than the available RAM.

    NOTE: the output file is created in a single pass through the input corpus, so
    that the input can be a once-only stream (iterator).
    To achieve this, a fake MM header is written first, statistics are collected
    during the pass (shape of the matrix, number of non-zeroes), followed by a seek
    back to the beginning of the file, rewriting the fake header with proper values.

    """

    HEADER_LINE = b'%%MatrixMarket matrix coordinate real general\n'  # the only supported MM format

    def __init__(self, fname):
        self.fname = fname
        if fname.endswith(".gz") or fname.endswith('.bz2'):
            raise NotImplementedError("compressed output not supported with MmWriter")
        self.fout = utils.smart_open(self.fname, 'wb+')  # open for both reading and writing
        self.headers_written = False

    def write_headers(self, num_docs, num_terms, num_nnz):
        self.fout.write(MmWriter.HEADER_LINE)

        if num_nnz < 0:
            # we don't know the matrix shape/density yet, so only log a general line
            logger.info("saving sparse matrix to %s", self.fname)
            self.fout.write(utils.to_utf8(' ' * 50 + '\n'))  # 48 digits must be enough for everybody
        else:
            logger.info("saving sparse %sx%s matrix with %i non-zero entries to %s", num_docs, num_terms, num_nnz, self.fname)
            self.fout.write(utils.to_utf8('%s %s %s\n' % (num_docs, num_terms, num_nnz)))
        self.last_docno = -1
        self.headers_written = True

    def fake_headers(self, num_docs, num_terms, num_nnz):
        stats = '%i %i %i' % (num_docs, num_terms, num_nnz)
        if len(stats) > 50:
            raise ValueError('Invalid stats: matrix too large!')
        self.fout.seek(len(MmWriter.HEADER_LINE))
        self.fout.write(utils.to_utf8(stats))

    def write_vector(self, docno, vector):
        """
        Write a single sparse vector to the file.

        Sparse vector is any iterable yielding (field id, field value) pairs.
        """
        assert self.headers_written, "must write Matrix Market file headers before writing data!"
        assert self.last_docno < docno, "documents %i and %i not in sequential order!" % (self.last_docno, docno)
        vector = sorted((i, w) for i, w in vector if abs(w) > 1e-12)  # ignore near-zero entries
        for termid, weight in vector:  # write term ids in sorted order
            self.fout.write(utils.to_utf8("%i %i %s\n" % (docno + 1, termid + 1, weight)))  # +1 because MM format starts counting from 1
        self.last_docno = docno
        return (vector[-1][0], len(vector)) if vector else (-1, 0)

    @staticmethod
    def write_corpus(fname, corpus, progress_cnt=1000, index=False, num_terms=None, metadata=False):
        """
        Save the vector space representation of an entire corpus to disk.

        Note that the documents are processed one at a time, so the whole corpus
        is allowed to be larger than the available RAM.
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
            logger.info("saved %ix%i matrix, density=%.3f%% (%i/%i)", num_docs, num_terms, 100.0 * num_nnz / (num_docs * num_terms), num_nnz, num_docs * num_terms)

        # now write proper headers, by seeking and overwriting the spaces written earlier
        mw.fake_headers(num_docs, num_terms, num_nnz)

        mw.close()
        if index:
            return offsets

    def __del__(self):
        """
        Automatic destructor which closes the underlying file.

        There must be no circular references contained in the object for __del__
        to work! Closing the file explicitly via the close() method is preferred
        and safer.
        """
        self.close()  # does nothing if called twice (on an already closed file), so no worries

    def close(self):
        logger.debug("closing %s", self.fname)
        if hasattr(self, 'fout'):
            self.fout.close()


class MmReader(object):
    """
    Wrap a term-document matrix on disk (in matrix-market format), and present it
    as an object which supports iteration over the rows (~documents).

    Note that the file is read into memory one document at a time, not the whole
    matrix at once (unlike scipy.io.mmread). This allows us to process corpora
    which are larger than the available RAM.
    """

    def __init__(self, input, transposed=True):
        """
        Initialize the matrix reader.

        The `input` refers to a file on local filesystem, which is expected to
        be in the sparse (coordinate) Matrix Market format. Documents are assumed
        to be rows of the matrix (and document features are columns).

        `input` is either a string (file path) or a file-like object that supports
        `seek()` (e.g. gzip.GzipFile, bz2.BZ2File).
        """
        logger.info("initializing corpus reader from %s", input)
        self.input, self.transposed = input, transposed
        with utils.file_or_filename(self.input) as lines:
            try:
                header = utils.to_unicode(next(lines)).strip()
                if not header.lower().startswith('%%matrixmarket matrix coordinate real general'):
                    raise ValueError("File %s not in Matrix Market format with coordinate real general; instead found: \n%s" %
                                    (self.input, header))
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

        logger.info("accepted corpus with %i documents, %i features, %i non-zero entries", self.num_docs, self.num_terms, self.num_nnz)

    def __len__(self):
        return self.num_docs

    def __str__(self):
        return ("MmCorpus(%i documents, %i features, %i non-zero entries)" %
                (self.num_docs, self.num_terms, self.num_nnz))

    def skip_headers(self, input_file):
        """
        Skip file headers that appear before the first document.
        """
        for line in input_file:
            if line.startswith(b'%'):
                continue
            break

    def __iter__(self):
        """
        Iteratively yield vectors from the underlying file, in the format (row_no, vector),
        where vector is a list of (col_no, value) 2-tuples.

        Note that the total number of vectors returned is always equal to the
        number of rows specified in the header; empty documents are inserted and
        yielded where appropriate, even if they are not explicitly stored in the
        Matrix Market file.
        """
        with utils.file_or_filename(self.input) as lines:
            self.skip_headers(lines)

            previd = -1
            for line in lines:
                docid, termid, val = utils.to_unicode(line).split()  # needed for python3
                if not self.transposed:
                    termid, docid = docid, termid
                docid, termid, val = int(docid) - 1, int(termid) - 1, float(val)  # -1 because matrix market indexes are 1-based => convert to 0-based
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
        """Return document at file offset `offset` (in bytes)"""
        # empty documents are not stored explicitly in MM format, so the index marks
        # them with a special offset, -1.
        if offset == -1:
            return []
        if isinstance(self.input, string_types):
            fin = utils.smart_open(self.input)
        else:
            fin = self.input

        fin.seek(offset)  # works for gzip/bz2 input, too
        previd, document = -1, []
        for line in fin:
            docid, termid, val = line.split()
            if not self.transposed:
                termid, docid = docid, termid
            docid, termid, val = int(docid) - 1, int(termid) - 1, float(val)  # -1 because matrix market indexes are 1-based => convert to 0-based
            assert previd <= docid, "matrix columns must come in ascending order"
            if docid != previd:
                if previd >= 0:
                    return document
                previd = docid

            document.append((termid, val,))  # add another field to the current document
        return document
