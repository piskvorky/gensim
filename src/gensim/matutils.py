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

import numpy
import scipy.sparse


def pad(mat, padRow, padCol):
    """
    Add additional rows/columns to a numpy.matrix `mat`. The new rows/columns 
    will be initialized with zeros.
    """
    assert padRow >= 0, padCol >= 0
    rows, cols = mat.shape
    return numpy.bmat([[mat, numpy.matrix(numpy.zeros((rows, padCol)))],
                      [numpy.matrix(numpy.zeros((padRow, cols + padCol)))]])


def sparse2full(doc, length):
    """
    Convert document in sparse format (sequence of 2-tuples) into a full numpy
    array (of size `length`).
    """
    result = numpy.zeros(length, dtype = numpy.float32)
    doc = dict(doc)
    result[doc.keys()] = doc.values()
    return result


def vecLen(vec):
    if len(vec) == 0:
        return 0.0
    vecLen = 1.0 * math.sqrt(sum(val * val for _, val in vec))
    assert vecLen > 0.0, "sparse documents must not contain any explicit zero entries"
    return vecLen


def unitVec(vec):
    """
    Scale a vector to unit length. The only exception is zero vector, which
    is returned back unchanged.
    
    If the input is sparse (list of 2-tuples), output will also be sparse. Otherwise,
    output will be a numpy array.
    """
    if scipy.sparse.issparse(vec):
        vec = vec.toarray().flatten()
    try:
        first = iter(vec).next()
    except:
        return vec
    
    if isinstance(first, tuple):
        vecLen = 1.0 * math.sqrt(sum(val * val for _, val in vec))
        assert vecLen > 0.0, "sparse documents must not contain any explicit zero entries"
        if vecLen != 1.0:
            result = [(termId, val / vecLen) for termId, val in vec]
        else:
            result = list(vec)
    else:
        vec = numpy.asarray(vec, dtype = float)
        vecLen = numpy.sqrt(numpy.sum(vec * vec))
        if vecLen > 0.0:
            result = vec / vecLen
        else:
            result = vec
            
    return result


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


class MmWriter(object):
    """
    Store corpus in Matrix Market format.
    """
    
    HEADER_LINE = '%%matrixmarket matrix coordinate real general\n'
    
    def __init__(self, fname):
        self.fname = fname
        tmp = open(self.fname, 'w')
        tmp.close()
        self.fout = open(self.fname, 'r+')
        self.headersWritten = False
    

    def writeHeaders(self, numDocs, numTerms, numNnz):
        self.fout.write(MmWriter.HEADER_LINE)
        
        if numNnz < 0:
            # we don't know the matrix shape/density yet, so only log a general line
            logging.info("saving sparse matrix to %s" % self.fname)
            self.fout.write(' ' * 50 + '\n') # 48 digits must be enough for everybody
        else:
            logging.info("saving sparse %sx%s matrix with %i non-zero entries to %s" %
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
            self.fout.write("%i %i %f\n" % (docNo + 1, termId + 1, weight)) # +1 because MM format starts counting from 1
        self.lastDocNo = docNo


    @staticmethod
    def writeCorpus(fname, corpus):
        """
        Save the vector space representation of an entire corpus to disk.
        
        Note that the documents are processed one at a time, so the whole corpus 
        is allowed to be larger than the available RAM.
        """
        mw = MmWriter(fname)
        
        # write empty headers to the file (with enough space to be overwritten later)
        mw.writeHeaders(-1, -1, -1) # will print 50 spaces followed by newline on the stats line
        
        # calculate necessary header info (nnz elements, num terms, num docs) while writing out vectors
        numDocs = len(corpus)
        numTerms = numNnz = 0
        
        for docNo, bow in enumerate(corpus):
            if docNo % 1000 == 0:
                logging.info("PROGRESS: saving document %i/%i" % 
                             (docNo, len(corpus)))
            if len(bow) > 0:
                numTerms = max(numTerms, 1 + max(wordId for wordId, val in bow))
                numNnz += len(bow)
            mw.writeVector(docNo, bow)
            
        if numDocs * numTerms != 0:
            logging.info("saved %ix%i matrix, density=%.3f%% (%i/%i)" % 
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
    matrix at once (unlike scipy.io.mmread). This allows for representing corpora 
    which are larger than the available RAM.
    """
    def __init__(self, fname):
        """
        Initialize the matrix reader. 
        
        The `fname` is a path to a file on local filesystem, which is expected to 
        be in sparse (coordinate) Matrix Market format. Documents are assumed to 
        be rows of the matrix (and document features are columns).
        """
        logging.info("initializing corpus reader from %s" % fname)
        self.fname = fname
        fin = open(fname)
        header = fin.next().strip()
        if not header.lower().startswith('%%matrixmarket matrix coordinate real general'):
            raise ValueError("File %s not in Matrix Market format with coordinate real general; instead found: \n%s" % 
                             (fname, header))
        self.numDocs = self.numTerms = self.numElements = 0
        for lineNo, line in enumerate(fin):
            if not line.startswith('%'):
                self.numDocs, self.numTerms, self.numElements = map(int, line.split())
                break
        logging.info("accepted corpus with %i documents, %i terms, %i non-zero entries" %
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
        fin = open(self.fname)
        
        # skip headers
        for line in fin:
            if line.startswith('%'):
                continue
            break
        
        prevId = -1
        for line in fin:
            docId, termId, val = line.split()
            docId, termId, val = int(docId) - 1, int(termId) - 1, float(val) # -1 because matrix market indexes are 1-based => convert to 0-based
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
        for prevId in xrange(prevId + 1, self.numDocs):
            yield prevId, []
#endclass MmReader

