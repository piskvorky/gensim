#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains math helper functions.
"""

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

blas = lambda name, ndarray: scipy.linalg.get_blas_funcs((name,), (ndarray,))[0]



logger = logging.getLogger("matutils")
logger.setLevel(logging.INFO)


def corpus2csc(corpus, num_terms, dtype=numpy.float64):
    """
    Convert corpus into a sparse matrix, in scipy.sparse.csc_matrix format, 
    with documents as columns.
    """
    logger.info("constructing sparse document matrix")
    docs, data, indices, indptr = 0, [], [], [0]
    for doc in corpus:
        indices.extend([feature_id for feature_id, _ in doc])
        data.extend([feature_weight for _, feature_weight in doc])
        indptr.append(len(doc))
        docs += 1
    indptr = numpy.cumsum(indptr)
    data = numpy.asarray(data, dtype=dtype)
    indices = numpy.asarray(indices)
    return scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, docs), dtype=dtype)


def pad(mat, padRow, padCol):
    """
    Add additional rows/columns to a numpy.matrix `mat`. The new rows/columns 
    will be initialized with zeros.
    """
    if padRow < 0:
        padRow = 0
    if padCol < 0:
        padCol = 0
    rows, cols = mat.shape
    return numpy.bmat([[mat, numpy.matrix(numpy.zeros((rows, padCol)))],
                      [numpy.matrix(numpy.zeros((padRow, cols + padCol)))]])


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

dense2vec = full2sparse

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


def vecLen(vec):
    if len(vec) == 0:
        return 0.0
    vecLen = 1.0 * math.sqrt(sum(val * val for _, val in vec))
    assert vecLen > 0.0, "sparse documents must not contain any explicit zero entries"
    return vecLen


blas_nrm2 = blas('nrm2', numpy.array([], dtype = float))
blas_scal = blas('scal', numpy.array([], dtype = float))

def unitVec(vec):
    """
    Scale a vector to unit length. The only exception is the zero vector, which
    is returned back unchanged.
    
    If the input is sparse (list of 2-tuples), output will also be sparse. Otherwise,
    output will be a numpy array.
    """
    if scipy.sparse.issparse(vec): # convert scipy.sparse to standard numpy array
        vec = vec.toarray().flatten()
    
    try:
        first = iter(vec).next() # is there at least one element?
    except:
        return vec
    
    if isinstance(first, tuple): # sparse format?
        vecLen = 1.0 * math.sqrt(sum(val * val for _, val in vec))
        assert vecLen > 0.0, "sparse documents must not contain any explicit zero entries"
        if vecLen != 1.0:
            return [(termId, val / vecLen) for termId, val in vec]
        else:
            return list(vec)
    else: # dense format
        vec = numpy.asarray(vec, dtype=float)
        veclen = blas_nrm2(vec)
        if veclen > 0.0:
            return blas_scal(1.0 / veclen, vec)
        else:
            return vec


def cossim(vec1, vec2):
    vec1, vec2 = dict(vec1), dict(vec2)
    if not vec1 or not vec2:
        return 0.0
    vec1Len = 1.0 * math.sqrt(sum(val * val for val in vec1.itervalues()))
    vec2Len = 1.0 * math.sqrt(sum(val * val for val in vec2.itervalues()))
    assert vec1Len > 0.0 and vec2Len > 0.0, "sparse documents must not contain any explicit zero entries"
    if len(vec2) < len(vec1):
        vec1, vec2 = vec2, vec1 # swap references so that we iterate over the shorter vector
    result = sum(value * vec2.get(index, 0.0) for index, value in vec1.iteritems())
    result /= vec1Len * vec2Len # rescale by vector lengths
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
    # perform q, r = QR(component); code hacked out of scipy.linalg.qr
    logger.debug("computing QR of %s dense matrix" % str(a.shape))
    geqrf, = get_lapack_funcs(('geqrf',), (a,))
    qr, tau, work, info = geqrf(a, lwork = -1, overwrite_a = True)
    qr, tau, work, info = geqrf(a, lwork = work[0], overwrite_a = True)
    del a # free up mem
    assert info >= 0
    r = triu(qr[:n, :n])
    if m < n: # rare case, #features < #topics
        qr = qr[:, :m] # retains fortran order
    gorgqr, = get_lapack_funcs(('orgqr',), (qr,))
    q, work, info = gorgqr(qr, tau, lwork = -1, overwrite_a = True)
    q, work, info = gorgqr(qr, tau, lwork = work[0], overwrite_a = True)
    assert info >= 0, "qr failed"
    assert q.flags.f_contiguous
    return q, r



class MmWriter(object):
    """
    Store corpus in Matrix Market format.
    """
    
    HEADER_LINE = '%%MatrixMarket matrix coordinate real general\n'
    
    def __init__(self, fname):
        self.fname = fname
        tmp = open(self.fname, 'w') # reset/create the target file
        tmp.close()
        self.fout = open(self.fname, 'rb+') # open for both reading and writing
        self.headersWritten = False
    

    def writeHeaders(self, numDocs, numTerms, numNnz):
        self.fout.write(MmWriter.HEADER_LINE)
        
        if numNnz < 0:
            # we don't know the matrix shape/density yet, so only log a general line
            logger.info("saving sparse matrix to %s" % self.fname)
            self.fout.write(' ' * 50 + '\n') # 48 digits must be enough for everybody
        else:
            logger.info("saving sparse %sx%s matrix with %i non-zero entries to %s" %
                         (numDocs, numTerms, numNnz, self.fname))
            self.fout.write('%s %s %s\n' % (numDocs, numTerms, numNnz))
        self.lastDocNo = -1
        self.headersWritten = True
    
    
    def fakeHeaders(self, numDocs, numTerms, numNnz):
        stats = '%i %i %i' % (numDocs, numTerms, numNnz)
        if len(stats) > 50:
            raise ValueError('Invalid stats: matrix too large!')
        self.fout.seek(len(MmWriter.HEADER_LINE))
        self.fout.write(stats)
    
    
    def writeVector(self, docNo, vector):
        """
        Write a single sparse vector to the file.
        
        Sparse vector is any iterable yielding (field id, field value) pairs.
        """
        assert self.headersWritten, "must write Matrix Market file headers before writing data!"
        assert self.lastDocNo < docNo, "documents %i and %i not in sequential order!" % (self.lastDocNo, docNo)
        for termId, weight in sorted(vector): # write term ids in sorted order
            if weight == 0:
                # to ensure len(doc) does what is expected, there must not be any zero elements in the sparse document
                raise ValueError("zero weights not allowed in sparse documents; check your document generator")
            self.fout.write("%i %i %s\n" % (docNo + 1, termId + 1, weight)) # +1 because MM format starts counting from 1
        self.lastDocNo = docNo


    @staticmethod
    def writeCorpus(fname, corpus, progressCnt = 1000):
        """
        Save the vector space representation of an entire corpus to disk.
        
        Note that the documents are processed one at a time, so the whole corpus 
        is allowed to be larger than the available RAM.
        """
        mw = MmWriter(fname)
        
        # write empty headers to the file (with enough space to be overwritten later)
        mw.writeHeaders(-1, -1, -1) # will print 50 spaces followed by newline on the stats line
        
        # calculate necessary header info (nnz elements, num terms, num docs) while writing out vectors
        numTerms = numNnz = 0
        docNo = -1
        
        for docNo, bow in enumerate(corpus):
            if docNo % progressCnt == 0:
                logger.info("PROGRESS: saving document #%i" % docNo)
            if len(bow) > 0:
                numTerms = max(numTerms, 1 + max(wordId for wordId, val in bow))
                numNnz += len(bow)
            mw.writeVector(docNo, bow)
        numDocs = docNo + 1
            
        if numDocs * numTerms != 0:
            logger.info("saved %ix%i matrix, density=%.3f%% (%i/%i)" % 
                         (numDocs, numTerms,
                          100.0 * numNnz / (numDocs * numTerms),
                          numNnz,
                          numDocs * numTerms))
            
        # now write proper headers, by seeking and overwriting a part of the file
        mw.fakeHeaders(numDocs, numTerms, numNnz)
        
        mw.close()


    def __del__(self):
        """
        Automatic destructor which closes the underlying file. 
        
        There must be no circular references contained in the object for __del__
        to work! Closing the file explicitly via the close() method is preferred
        and safer.
        """
        self.close() # does nothing if called twice (on an already closed file), so no worries
    
    
    def close(self):
        logging.debug("closing %s" % self.fname)
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
        `seek(0)` (e.g. gzip.GzipFile, bz2.BZ2File).
        """
        logger.info("initializing corpus reader from %s" % input)
        self.input, self.transposed = input, transposed
        if isinstance(input, basestring):
            input = open(input)
        header = input.next().strip()
        if not header.lower().startswith('%%matrixmarket matrix coordinate real general'):
            raise ValueError("File %s not in Matrix Market format with coordinate real general; instead found: \n%s" % 
                             (self.input, header))
        self.numDocs = self.numTerms = self.numElements = 0
        for lineNo, line in enumerate(input):
            if not line.startswith('%'):
                self.numDocs, self.numTerms, self.numElements = map(int, line.split())
                if not self.transposed:
                    self.numDocs, self.numTerms = self.numTerms, self.numDocs
                break
        logger.info("accepted corpus with %i documents, %i terms, %i non-zero entries" %
                     (self.numDocs, self.numTerms, self.numElements))
    
    def __len__(self):
        return self.numDocs
    
    def __str__(self):
        return ("MmCorpus(%i documents, %i features, %i non-zero entries)" % 
                (self.numDocs, self.numTerms, self.numElements))
        
    def __iter__(self):
        """
        Iteratively yield vectors from the underlying file, in the format (rowNo, vector),
        where vector is a list of (colId, value) 2-tuples.
        
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
        # skip headers
        for line in fin:
            if line.startswith('%'):
                continue
            break
        
        prevId = -1
        for line in fin:
            docId, termId, val = line.split()
            if not self.transposed:
                termId, docId = docId, termId
            docId, termId, val = int(docId) - 1, int(termId) - 1, float(val) # -1 because matrix market indexes are 1-based => convert to 0-based
            assert prevId <= docId, "matrix columns must come in ascending order"
            if docId != prevId:
                # change of document: return the document read so far (its id is prevId)
                if prevId >= 0:
                    yield prevId, document
                
                # return implicit (empty) documents between previous id and new id 
                # too, to keep consistent document numbering and corpus length
                for prevId in xrange(prevId + 1, docId):
                    yield prevId, []
                
                # from now on start adding fields to a new document, with a new id
                prevId = docId
                document = []
            
            document.append((termId, val,)) # add another field to the current document
        
        # handle the last document, as a special case
        if prevId >= 0:
            yield prevId, document
        
        # return empty documents between the last explicit document and the number 
        # of documents as specified in the header
        for prevId in xrange(prevId + 1, self.numDocs):
            yield prevId, []
#endclass MmReader

