#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains functions and classes for computing similarities across
a collection of vectors=documents in the Vector Space Model.

The main classes are :

1. `Similarity` -- answers similarity queries by linearly scanning over the corpus.
   This is slow but memory independent.
2. `MatrixSimilarity` -- stores the whole corpus **in memory**, computes similarity 
   by in-memory matrix-vector multiplication. This is much faster than the general 
   `Similarity`, so use this when dealing with smaller corpora (must fit in RAM).
3. `SparseMatrixSimilarity` -- same as `MatrixSimilarity`, but uses less memory 
   if the vectors are sparse.

Once the similarity object has been initialized, you can query for document
similarity simply by

>>> similarities = similarity_object[query_vector]

or iterate over within-corpus similarities with

>>> for similarities in similarity_object:
>>>     ...

-------
"""


import logging

import numpy
import scipy.sparse

from gensim import interfaces, utils, matutils



class Similarity(interfaces.SimilarityABC):
    """
    Compute cosine similarity against a corpus of documents. This is done by a full 
    sequential scan of the corpus. 
    
    If your corpus is reasonably small (fits in RAM), consider using `MatrixSimilarity`
    or `SparseMatrixSimilarity` instead, for (much) faster similarity searches.
    """
    def __init__(self, corpus, numBest = None):
        """
        If `numBest` is left unspecified, similarity queries return a full list (one 
        float for every document in the corpus, including the query document):
        
        If `numBest` is set, queries return `numBest` most similar documents, as a 
        sorted list:
        
        >>> sms = MatrixSimilarity(corpus, numBest = 3)
        >>> sms[vec12]
        [(12, 1.0), (30, 0.95), (5, 0.45)]
        
        """
        self.corpus = corpus
        self.numBest = numBest
        self.normalize = True
    
    
    def getSimilarities(self, doc):
        return [matutils.cossim(doc, other) for other in self.corpus]
#endclass Similarity    


class MatrixSimilarity(interfaces.SimilarityABC):
    """
    Compute similarity against a corpus of documents by storing its 
    term-document (or concept-document) matrix in memory. The similarity measure 
    used is cosine between two vectors.
    
    This allows fast similarity searches (simple sparse matrix-vector multiplication),
    but loses the memory-independence of an iterative corpus.

    The matrix is internally stored as a numpy array.
    """
    def __init__(self, corpus, numBest = None, dtype = numpy.float32, numFeatures = None):
        """
        If `numBest` is left unspecified, similarity queries return a full list (one 
        float for every document in the corpus, including the query document):
        
        If `numBest` is set, queries return `numBest` most similar documents, as a 
        sorted list:
        
        >>> sms = MatrixSimilarity(corpus, numBest = 3)
        >>> sms[vec12]
        [(12, 1.0), (30, 0.95), (5, 0.45)]
        
        """
        if numFeatures is None:
            logging.info("scanning corpus of %i documents to determine the number of features" %
                         len(corpus))
            numFeatures = 1 + utils.getMaxId(corpus)
            
        logging.info("creating matrix for %i documents and %i features" % 
                     (len(corpus), numFeatures))
        self.numFeatures = numFeatures
        self.numBest = numBest
        self.corpus = numpy.empty(shape = (len(corpus), numFeatures), dtype = dtype)
        self.normalize = True
        
        # iterate over corpus, populating the numpy matrix
        for docNo, vector in enumerate(corpus):
            if docNo % 10000 == 0:
                logging.info("PROGRESS: at document #%i/%i" % (docNo, len(corpus)))
            vector = matutils.unitVec(matutils.sparse2full(vector, numFeatures))
            self.corpus[docNo] = vector
        
        self.corpus = numpy.asmatrix(self.corpus)
    
    
    def getSimilarities(self, doc):
        """
        Return similarity of sparse vector `doc` to all documents in the corpus.
        
        `doc` may be either a bag-of-words iterable (standard corpus document), 
        or a numpy array, or a `scipy.sparse` matrix.
        """
        if scipy.sparse.issparse(doc):
            vec = doc.toarray().flatten()
        elif isinstance(doc, numpy.ndarray):
            vec = doc
        else:
            vec = matutils.sparse2full(doc, self.numFeatures)
        
        vec.shape = (vec.size, 1)
        if vec.shape != (self.corpus.shape[1], 1):
            raise ValueError("vector shape mismatch; expected %s, got %s" % 
                             ((self.corpus.shape[1], 1,), vec.shape))
        
        # compute cosine similarity against every other document in the collection
        allSims = self.corpus * vec # N x T * T x 1 = N x 1
        allSims = list(allSims.flat) # convert to plain python list
        assert len(allSims) == self.corpus.shape[0] # make sure no document got lost!
        return allSims
#endclass MatrixSimilarity



class SparseMatrixSimilarity(interfaces.SimilarityABC):
    """
    Compute similarity against a corpus of documents by storing its sparse 
    term-document (or concept-document) matrix in memory. The similarity measure 
    used is cosine between two vectors.
    
    This allows fast similarity searches (simple sparse matrix-vector multiplication),
    but loses the memory-independence of an iterative corpus.

    The matrix is internally stored as a `scipy.sparse.csr` matrix.
    """
    def __init__(self, corpus, numBest = None, dtype = numpy.float32):
        """
        If `numBest` is left unspecified, similarity queries return a full list (one 
        float for every document in the corpus, including the query document):
        
        If `numBest` is set, queries return `numBest` most similar documents, as a 
        sorted list:
        
        >>> sms = SparseMatrixSimilarity(corpus, numBest = 3)
        >>> sms[vec12]
        [(12, 1.0), (30, 0.95), (5, 0.45)]
        
        """
        logging.info("creating sparse matrix for %i documents" % len(corpus))
        self.numBest = numBest
        self.corpus = scipy.sparse.lil_matrix((len(corpus), 1), dtype = dtype) # set no of columns to 1 for now, as the number of terms is unknown yet
        self.normalize = True
        
        # iterate over corpus, populating the sparse matrix
        for docNo, vector in enumerate(corpus):
            if docNo % 10000 == 0:
                logging.info("PROGRESS: at document #%i/%i" % (docNo, len(corpus)))
            vector = matutils.unitVec(vector) # make all vectors unit length, so that cosine similarity = simple dot product
            self.corpus.rows[docNo] = [termId for termId, _ in vector]
            self.corpus.data[docNo] = [dtype(val) for _, val in vector]
        
        # now set the shape properly, using no. columns = highest term index in the corpus + 1
        numTerms = 1 + max(max(row + [-1]) for row in self.corpus.rows) # + [-1] to avoid exceptions from max(empty)
        self.corpus._shape = (len(corpus), numTerms)
        
        # convert to Compressed Sparse Row for efficient row slicing and multiplications
        self.corpus = self.corpus.tocsr()
        logging.info("created %s" % repr(self.corpus))
    
    
    def getSimilarities(self, doc):
        """
        Return similarity of sparse vector `doc` to all documents in the corpus.
        
        `doc` may be either a bag-of-words iterable (standard corpus document), 
        or a numpy array, or a `scipy.sparse` matrix.
        """
        if scipy.sparse.issparse(doc):
            vec = doc.T
        elif isinstance(doc, numpy.ndarray):
            vec = scipy.sparse.csr_matrix(doc).T # Tx1 array
        else:
            vec = scipy.sparse.dok_matrix((self.corpus.shape[1], 1), dtype = self.corpus.dtype)
            for fieldId, fieldValue in doc:
                vec[fieldId, 0] = fieldValue
        if vec.shape != (self.corpus.shape[1], 1):
            raise ValueError("vector shape mismatch; expected %s, got %s" % 
                             ((self.corpus.shape[1], 1,), vec.shape))
        
        # compute cosine similarity against every other document in the collection
        allSims = self.corpus * vec.tocsc() # N x T * T x 1 = N x 1
        allSims = list(allSims.toarray().flat) # convert to plain python list
        assert len(allSims) == self.corpus.shape[0] # make sure no document got lost!
        return allSims
#endclass SparseMatrixSimilarity

