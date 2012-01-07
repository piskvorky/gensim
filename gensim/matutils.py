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
import os
import itertools

import numpy
import scipy.sparse
import scipy.linalg
from scipy.linalg.lapack import get_lapack_funcs, find_best_lapack_type

# scipy is not a stable package yet, locations change, so try to work
# around differences (currently only concerns location of 'triu' in scipy 0.7 vs. 0.8)
try:
    from scipy.linalg.basic import triu
except ImportError:
    from scipy.linalg.special_matrices import triu

try:
    from numpy import triu_indices
except ImportError:
    # numpy < 1.4
    def triu_indices(n, k=0):
        m = numpy.ones((n,n), int)
        a = triu(m, k)
        return numpy.where(a != 0)

blas = lambda name, ndarray: scipy.linalg.get_blas_funcs((name,), (ndarray,))[0]



logger = logging.getLogger("gensim.matutils")


def corpus2csc(corpus, num_terms=None, dtype=numpy.float64, num_docs=None, num_nnz=None, printprogress=0):
    """
    Convert corpus into a sparse matrix, in scipy.sparse.csc_matrix format,
    with documents as columns.

    If the number of terms, documents and non-zero elements is known, you can pass
    them here as parameters and a more memory efficient code path will be taken.
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
    except AttributeError, e:
        pass # not a MmCorpus...
    if printprogress:
        logger.info("creating sparse matrix from corpus")
    if num_terms is not None and num_docs is not None and num_nnz is not None:
        # faster and much more memory-friendly version of creating the sparse csc
        posnow, indptr = 0, [0]
        indices = numpy.empty((num_nnz,), dtype=numpy.int32) # HACK assume feature ids fit in 32bit integer
        data = numpy.empty((num_nnz,), dtype=dtype)
        for docno, doc in enumerate(corpus):
            if printprogress and docno % printprogress == 0:
                logger.info("PROGRESS: at document #%i/%i" % (docno, num_docs))
            posnext = posnow + len(doc)
            indices[posnow : posnext] = [feature_id for feature_id, _ in doc]
            data[posnow : posnext] = [feature_weight for _, feature_weight in doc]
            indptr.append(posnext)
            posnow = posnext
        assert posnow == num_nnz, "mismatch between supplied and computed number of non-zeros"
        result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, num_docs), dtype=dtype)
    else:
        # slower version; determine the sparse matrix parameters during iteration
        num_nnz, data, indices, indptr = 0, [], [], [0]
        for docno, doc in enumerate(corpus):
            if printprogress and docno % printprogress == 0:
                logger.info("PROGRESS: at document #%i" % (docno))
            indices.extend([feature_id for feature_id, _ in doc])
            data.extend([feature_weight for _, feature_weight in doc])
            num_nnz += len(doc)
            indptr.append(num_nnz)
        if num_terms is None:
            num_terms = max(indices) + 1 if indices else 0
        num_docs = len(indptr) - 1
        # now num_docs, num_terms and num_nnz contain the correct values
        data = numpy.asarray(data, dtype=dtype)
        indices = numpy.asarray(indices)
        result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, num_docs), dtype=dtype)
    return result


def pad(mat, padrow, padcol):
    """
    Add additional rows/columns to a numpy.matrix `mat`. The new rows/columns
    will be initialized with zeros.
    """
    if padrow < 0:
        padrow = 0
    if padcol < 0:
        padcol = 0
    rows, cols = mat.shape
    return numpy.bmat([[mat, numpy.matrix(numpy.zeros((rows, padcol)))],
                      [numpy.matrix(numpy.zeros((padrow, cols + padcol)))]])


def ismatrix(m):
    return isinstance(m, numpy.ndarray) and m.ndim == 2 or scipy.sparse.issparse(m)


def scipy2sparse(vec):
    """Convert a scipy.sparse vector to gensim format (list of 2-tuples)."""
    vec = vec.tocsr()
    assert vec.shape[0] == 1
    return zip(vec.indices, vec.data)


class Scipy2Corpus(object):
    def __init__(self, vecs):
        """Convert a sequence of dense/sparse vector to a gensim corpus object."""
        self.vecs = vecs

    def __iter__(self):
        for vec in self.vecs:
            if isinstance(vec, numpy.ndarray):
                yield full2sparse(vec)
            else:
                yield scipy2sparse(vec)

    def __len__(self):
        return len(self.vecs)


def sparse2full(doc, length):
    """
    Convert a document in sparse corpus format (sequence of 2-tuples) into a dense
    numpy array (of size `length`).
    """
    result = numpy.zeros(length, dtype=numpy.float32) # fill with zeroes (default value)
    doc = dict(doc)
    result[doc.keys()] = doc.values() # overwrite some of the zeroes with explicit values
    return result


def full2sparse(vec, eps=1e-9):
    """
    Convert a dense numpy array into the sparse corpus format (sequence of 2-tuples).

    Values of magnitude < `eps` are treated as zero (ignored).
    """
    return [(pos, val) for pos, val in enumerate(vec) if numpy.abs(val) > eps]
#    # slightly faster but less flexible:
#    nnz = vec.nonzero()[0]
#    return zip(nnz, vec[nnz])

dense2vec = full2sparse


def full2sparse_clipped(vec, topn, eps=1e-9):
    """
    Like `full2sparse`, but only return the `topn` greatest elements (not all).
    """
    # use numpy.argsort and only form tuples that are actually returned.
    # this is about 40x faster than explicitly forming all 2-tuples to run sort() or heapq.nlargest() on.
    if topn <= 0:
        return []
    result = []
    for i in numpy.argsort(vec)[::-1]:
        if abs(vec[i]) > eps: # ignore features with near-zero weight
            result.append((i, vec[i]))
            if len(result) == topn:
                break
    return result


def corpus2dense(corpus, num_terms):
    """
    Convert corpus into a dense numpy array (documents will be columns).
    """
    return numpy.column_stack(sparse2full(doc, num_terms) for doc in corpus)



class Dense2Corpus(object):
    """
    Treat dense numpy array as a sparse gensim corpus.

    No data copy is made (changes to the underlying matrix imply changes in the
    corpus).
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
#endclass DenseCorpus


class Sparse2Corpus(object):
    """
    Convert a matrix in scipy.sparse format into a streaming gensim corpus.
    """
    def __init__(self, sparse, documents_columns=True):
        if documents_columns:
            self.sparse = sparse.tocsc()
        else:
            self.sparse = sparse.tocsr().T # make sure shape[1]=number of docs (needed in len())

    def __iter__(self):
        for indprev, indnow in itertools.izip(self.sparse.indptr, self.sparse.indptr[1:]):
            yield zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow])

    def __len__(self):
        return self.sparse.shape[1]
#endclass Sparse2Corpus


def veclen(vec):
    if len(vec) == 0:
        return 0.0
    length = 1.0 * math.sqrt(sum(val**2 for _, val in vec))
    assert length > 0.0, "sparse documents must not contain any explicit zero entries"
    return length


blas_nrm2 = blas('nrm2', numpy.array([], dtype=float))
blas_scal = blas('scal', numpy.array([], dtype=float))

def unitvec(vec):
    """
    Scale a vector to unit length. The only exception is the zero vector, which
    is returned back unchanged.

    Output will be in the same format as input (i.e., gensim vector=>gensim vector,
    or numpy array=>numpy array, scipy.sparse=>scipy.sparse).
    """
    if scipy.sparse.issparse(vec): # convert scipy.sparse to standard numpy array
        vec = vec.tocsr()
        veclen = numpy.sqrt(numpy.sum(vec.data**2))
        if veclen > 0.0:
            return vec / veclen
        else:
            return vec

    if isinstance(vec, numpy.ndarray):
        vec = numpy.asarray(vec, dtype=float)
        veclen = blas_nrm2(vec)
        if veclen > 0.0:
            return blas_scal(1.0 / veclen, vec)
        else:
            return vec

    try:
        first = iter(vec).next() # is there at least one element?
    except:
        return vec

    if isinstance(first, tuple): # gensim sparse format?
        length = 1.0 * math.sqrt(sum(val**2 for _, val in vec))
        assert length > 0.0, "sparse documents must not contain any explicit zero entries"
        if length != 1.0:
            return [(termid, val / length) for termid, val in vec]
        else:
            return list(vec)
    else:
        raise ValueError("unknown input type")


def cossim(vec1, vec2):
    vec1, vec2 = dict(vec1), dict(vec2)
    if not vec1 or not vec2:
        return 0.0
    vec1len = 1.0 * math.sqrt(sum(val * val for val in vec1.itervalues()))
    vec2len = 1.0 * math.sqrt(sum(val * val for val in vec2.itervalues()))
    assert vec1len > 0.0 and vec2len > 0.0, "sparse documents must not contain any explicit zero entries"
    if len(vec2) < len(vec1):
        vec1, vec2 = vec2, vec1 # swap references so that we iterate over the shorter vector
    result = sum(value * vec2.get(index, 0.0) for index, value in vec1.iteritems())
    result /= vec1len * vec2len # rescale by vector lengths
    return result


def qr_destroy(la):
    """
    Return QR decomposition of `la[0]`. Content of `la` gets destroyed in the process.

    Using this function should be less memory intense than calling `scipy.linalg.qr(la[0])`,
    because the memory used in `la[0]` is reclaimed earlier.
    """
    a = numpy.asfortranarray(la[0])
    del la[0], la # now `a` is the only reference to the input matrix
    m, n = a.shape
    # perform q, r = QR(a); code hacked out of scipy.linalg.qr
    logger.debug("computing QR of %s dense matrix" % str(a.shape))
    geqrf, = get_lapack_funcs(('geqrf',), (a,))
    qr, tau, work, info = geqrf(a, lwork=-1, overwrite_a=True)
    qr, tau, work, info = geqrf(a, lwork=work[0], overwrite_a=True)
    del a # free up mem
    assert info >= 0
    r = triu(qr[:n, :n])
    if m < n: # rare case, #features < #topics
        qr = qr[:, :m] # retains fortran order
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

    HEADER_LINE = '%%MatrixMarket matrix coordinate real general\n' # the only supported MM format

    def __init__(self, fname):
        self.fname = fname
        tmp = open(self.fname, 'w') # reset/create the target file
        tmp.close()
        self.fout = open(self.fname, 'rb+') # open for both reading and writing
        self.headers_written = False


    def write_headers(self, num_docs, num_terms, num_nnz):
        self.fout.write(MmWriter.HEADER_LINE)

        if num_nnz < 0:
            # we don't know the matrix shape/density yet, so only log a general line
            logger.info("saving sparse matrix to %s" % self.fname)
            self.fout.write(' ' * 50 + '\n') # 48 digits must be enough for everybody
        else:
            logger.info("saving sparse %sx%s matrix with %i non-zero entries to %s" %
                         (num_docs, num_terms, num_nnz, self.fname))
            self.fout.write('%s %s %s\n' % (num_docs, num_terms, num_nnz))
        self.last_docno = -1
        self.headers_written = True


    def fake_headers(self, num_docs, num_terms, num_nnz):
        stats = '%i %i %i' % (num_docs, num_terms, num_nnz)
        if len(stats) > 50:
            raise ValueError('Invalid stats: matrix too large!')
        self.fout.seek(len(MmWriter.HEADER_LINE))
        self.fout.write(stats)


    def write_vector(self, docno, vector):
        """
        Write a single sparse vector to the file.

        Sparse vector is any iterable yielding (field id, field value) pairs.
        """
        assert self.headers_written, "must write Matrix Market file headers before writing data!"
        assert self.last_docno < docno, "documents %i and %i not in sequential order!" % (self.last_docno, docno)
        vector = [(i, w) for i, w in vector if abs(w) > 1e-12] # ignore near-zero entries
        vector.sort()
        for termid, weight in vector: # write term ids in sorted order
            self.fout.write("%i %i %s\n" % (docno + 1, termid + 1, weight)) # +1 because MM format starts counting from 1
        self.last_docno = docno
        return (vector[-1][0], len(vector)) if vector else (-1, 0)


    @staticmethod
    def write_corpus(fname, corpus, progress_cnt=1000, index=False):
        """
        Save the vector space representation of an entire corpus to disk.

        Note that the documents are processed one at a time, so the whole corpus
        is allowed to be larger than the available RAM.
        """
        mw = MmWriter(fname)

        # write empty headers to the file (with enough space to be overwritten later)
        mw.write_headers(-1, -1, -1) # will print 50 spaces followed by newline on the stats line

        # calculate necessary header info (nnz elements, num terms, num docs) while writing out vectors
        num_terms, num_nnz = 0, 0
        docno, poslast = -1, -1
        offsets = []
        for docno, bow in enumerate(corpus):
            if docno % progress_cnt == 0:
                logger.info("PROGRESS: saving document #%i" % docno)
            if index:
                posnow = mw.fout.tell()
                if posnow == poslast:
                    offsets[-1] = -1
                offsets.append(posnow)
                poslast = posnow
            max_id, veclen = mw.write_vector(docno, bow)
            num_terms = max(num_terms, 1 + max_id)
            num_nnz += veclen
        num_docs = docno + 1

        if num_docs * num_terms != 0:
            logger.info("saved %ix%i matrix, density=%.3f%% (%i/%i)" %
                         (num_docs, num_terms,
                          100.0 * num_nnz / (num_docs * num_terms),
                          num_nnz,
                          num_docs * num_terms))

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
        self.close() # does nothing if called twice (on an already closed file), so no worries


    def close(self):
        logger.debug("closing %s" % self.fname)
        self.fout.close()
#endclass MmWriter



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
        logger.info("initializing corpus reader from %s" % input)
        self.input, self.transposed = input, transposed
        if isinstance(input, basestring):
            input = open(input)
        header = input.next().strip()
        if not header.lower().startswith('%%matrixmarket matrix coordinate real general'):
            raise ValueError("File %s not in Matrix Market format with coordinate real general; instead found: \n%s" %
                             (self.input, header))
        self.num_docs = self.num_terms = self.num_nnz = 0
        for lineno, line in enumerate(input):
            if not line.startswith('%'):
                self.num_docs, self.num_terms, self.num_nnz = map(int, line.split())
                if not self.transposed:
                    self.num_docs, self.num_terms = self.num_terms, self.num_docs
                break
        logger.info("accepted corpus with %i documents, %i features, %i non-zero entries" %
                     (self.num_docs, self.num_terms, self.num_nnz))

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
            if line.startswith('%'):
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
        if isinstance(self.input, basestring):
            fin = open(self.input)
        else:
            fin = self.input
            fin.seek(0)
        self.skip_headers(fin)

        previd = -1
        for line in fin:
            docid, termid, val = line.split()
            if not self.transposed:
                termid, docid = docid, termid
            docid, termid, val = int(docid) - 1, int(termid) - 1, float(val) # -1 because matrix market indexes are 1-based => convert to 0-based
            assert previd <= docid, "matrix columns must come in ascending order"
            if docid != previd:
                # change of document: return the document read so far (its id is prevId)
                if previd >= 0:
                    yield previd, document

                # return implicit (empty) documents between previous id and new id
                # too, to keep consistent document numbering and corpus length
                for previd in xrange(previd + 1, docid):
                    yield previd, []

                # from now on start adding fields to a new document, with a new id
                previd = docid
                document = []

            document.append((termid, val,)) # add another field to the current document

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
        if isinstance(self.input, basestring):
            fin = open(self.input)
        else:
            fin = self.input

        fin.seek(offset) # works for gzip/bz2 input, too
        previd, document = -1, []
        for line in fin:
            docid, termid, val = line.split()
            if not self.transposed:
                termid, docid = docid, termid
            docid, termid, val = int(docid) - 1, int(termid) - 1, float(val) # -1 because matrix market indexes are 1-based => convert to 0-based
            assert previd <= docid, "matrix columns must come in ascending order"
            if docid != previd:
                if previd >= 0:
                    return document
                previd = docid

            document.append((termid, val,)) # add another field to the current document
        return document
#endclass MmReader
