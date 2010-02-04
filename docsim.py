#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# author Radim Rehurek, radimrehurek@seznam.cz

"""
This module contains functions and classes for computing similarity in Vector
Space Model.
"""


import logging

import numpy
import scipy.sparse

import utils
import matutils



class SimilarityABC(utils.SaveLoad):
    def __init__(self, corpus, numBest = None):
        """
        Initialize the similarity search.
        
        If numBest is left unspecified, iter(self) will yield similarities as a list 
        (one float for every document in the corpus, including the query document).
        
        If numBest is set, iter(self) will yield indices and similarities of numBest most 
        similar documents, as a sorted list, eg. [(docIndex1, 1.0), (docIndex2, 0.95), 
        ..., (docIndexnumBest, 0.45)].
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")


    def __getitem__(self, doc):
        """
        Return similarity of doc to all documents in the corpus.
        
        doc may be either a bag-of-words iterable (corpus document), or a numpy 
        array, or a scipy.sparse matrix.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")


    def __iter__(self):
        """
        For each document, compute cosine similarity against all other documents 
        and yield the result.
        """
        for docNo, doc in enumerate(self.corpus):
            # compute cosine similarity against every other document in the collection
            allSims = self[doc]
            
            # return either all similarities as a list, or only self.numBest most similar, depending on settings from the constructor
            if self.numBest is None:
                yield allSims
            else:
                tops = [(docNo, sim) for docNo, sim in enumerate(allSims) if sim > 0]
                tops = sorted(tops, key = lambda item: -item[1]) # sort by -sim => highest cossim first
                yield tops[ : self.numBest] # return at most numBest top 2-tuples (docId, docSim)
#endclass SimilarityABC
        

class Similarity(SimilarityABC):
    """
    Compute cosine similary against a corpus of documents. This is done by a full 
    sequential scan of the corpus. If your corpus is reasonably small (fits in RAM), 
    consider using SparseMatrixSimilarity for (much) faster similarity searches.
    """
    def __init__(self, corpus, numBest = None, normalize = True):
        """
        If numBest is left unspecified, similarities will be returned as a full
        list (one float for every document in the corpus, including itself).
        
        If numBest is set, return indices and similarities of numBest most 
        similar documents, as a sorted list of eg. [(docIndex1, 1.0), (docIndex2, 0.95), 
        ..., (docIndexnumBest, 0.45)].
        
        Set normalize to False of the vectors in corpus are all either unit length 
        or zero (faster search).
        """
        self.corpus = corpus
        self.numBest = numBest
        self.normalize = normalize
    
    
    def __getitem__(self, doc):
        """
        Return similarity of doc to all documents in the corpus.
        
        doc may be either a bag-of-words iterable (corpus document), or a numpy 
        array, or a scipy.sparse matrix.
        """
        doc = matutils.unitVec(doc)
        result = [matutils.cossim(doc, other, norm1 = False, norm2 = self.normalize)
                  for other in self.corpus]
        return result
#endclass Similarity    



class SparseMatrixSimilarity(SimilarityABC):
    """
    Compute similarity against a corpus of documents by storing its sparse 
    term-document (or concept-document) matrix in memory. The similarity measure 
    used is cosine between two vectors.
    
    This allows for faster similarity searches (simple sparse matrix-vector multiplication),
    but loses the memory-independence of an iterative corpus.

    The matrix is internally stored as a scipy.sparse array.
    """
    def __init__(self, corpus, numBest = None, dtype = numpy.float32):
        """
        If numBest is left unspecified, similarities will be returned as a full
        list (one float for every document in the corpus, including itself).
        
        If numBest is set, return indices and similarities of numBest most 
        similar documents, as a sorted list of eg. [(docIndex1, 1.0), (docIndex2, 0.95), 
        ..., (docIndexnumBest, 0.45)].
        """
        logging.info("creating sparse matrix for %i documents" % len(corpus))
        self.numBest = numBest
        self.corpus = scipy.sparse.lil_matrix((len(corpus), 1), dtype = dtype) # set no of columns to 1 for now, as the number of terms is unknown yet
        
        # iterate over the corpus, filling the sparse matrix
        for docNo, vector in enumerate(corpus):
            if docNo % 10000 == 0:
                logging.info("PROGRESS: at document #%i/%i" % (docNo, len(corpus)))
            vector = matutils.unitVec(vector) # make all vectors unit length, so that cosine similarity = simple dot product
            self.corpus.rows[docNo] = [termId for termId, _ in vector]
            self.corpus.data[docNo] = [dtype(val) for _, val in vector]
        
        # now set the shape properly, using no. columns = highest term index in the corpus + 1
        numTerms = 1 + max(max(row + [-1]) for row in self.corpus.rows) # + [0] to avoid exceptions from max([]) 
        self.corpus._shape = (len(corpus), numTerms)
        
        # convert to Compressed Sparse Row for efficient row slicing and multiplications
        self.corpus = self.corpus.tocsr()
        logging.info("created %s" % repr(self.corpus))
    
    
    def __getitem__(self, doc):
        """
        Return similarity of doc to all documents in the corpus.
        
        doc may be either a bag-of-words iterable (corpus document), or a numpy 
        array, or a scipy.sparse matrix.
        
        The document is assumed to be unit length -- ie. it is up to you to ensure
        that it is either empty or sum(val**2) == 1.0.
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
        allSims = list(allSims.toarray().flatten()) # convert to plain python list
        assert len(allSims) == self.corpus.shape[0] # make sure no document got lost!
        return allSims
#endclass SparseMatrixSimilarity


