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
import scipy

import utils
import matutils

import lsi # needed for Latent Semantic Indexing
import ldamodel # needed for Latent Dirichlet Allocation
import randomprojections # needed for Random Projections

import sources
import corpora



class SimilarityABC(utils.SaveLoad):
    def __init__(self, corpus):
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def __iter__(self):
        """
        Proceed one document at a time, returning similarities to other documents
        in the corpus.
        
        The order of iteration must be the same as the order of iteration over the
        input corpus.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")


class SparseMatrixSimilarity(SimilarityABC):
    """
    Compute similarity of a corpus of documents by storing the sparse 
    term-document (or concept-document) matrix in memory. The similarity measure 
    used is cosine between two vectors.
    
    This allows for faster similarity searches (simple sparse matrix-vector multiplication),
    but loses the memory-independence of an iterative corpus.

    The matrix is internally stored as a scipy.sparse array.
    """
    def __init__(self, corpus, numTerms, normalize = True, dtype = numpy.float32):
        assert numTerms > 0, "called MatrixSimilarity() with zero terms; something went wrong"
        logging.info("creating sparse matrix of shape %ix%i" % (len(corpus), numTerms))
        self.mat = scipy.sparse.lil_matrix((len(corpus), numTerms), dtype = dtype)
        for docNo, vector in enumerate(corpus):
            if normalize:
                vector = matutils.unitVec(vector)
            self.mat.rows[docNo] = [termId for termId, _ in vector]
            self.mat.data[docNo] = [dtype(val) for _, val in vector]
        self.mat = self.mat.tocsr() # convert to Compressed Sparse Row for efficient row slicing and multiplications
        logging.info("created %s" % repr(self.mat))
    
    
    def __iter__(self):
        for docNo in xrange(self.mat.shape[0]):
            yield self.getSimilar(docNo, numBest = None)
    
    
    def getSimilar(self, docNo, numBest = None):
        """
        Compute cosine similarity for document docNo against all other documents.
        
        If numBest is left unspecified, return array of similarities against all 
        documents in the corpus (including self). This is an array of floats, its
        length equals the length of the corpus.
        
        If numBest is set, return indices and similarities of numBest most 
        similar documents, as a sorted list of eg. [(docIndex1, 1.0), (docIndex2, 0.95), 
        ..., (docIndexnumBest, 0.45)].
        
        Note that the input document, docNo, always has (trivially) cosine 
        similarity of 1.0 to itself, which is the theoretical maximum. It therefore 
        always ranks among the most similar documents.
        """
        # get the sparse document vector, of shape T x 1
        vec = self.mat[docNo, :].T
        
        # compute cosine similarity against every other document in the collection
        allSims = self.mat * vec # N x T * T x 1 = N x 1
        allSims = list(allSims.toarray().flatten()) # convert to plain python list
        assert len(allSims) == self.mat.shape[0] # make sure no document got lost!
        
        if not numBest:
            return allSims
        else:
            tops = [(docNo, sim) for docNo, sim in enumerate(allSims) if sim > 0]
            tops = sorted(tops, key = lambda item: -item[1]) # sort by -sim => highest cossim first
            return tops[ : numBest] # return at most numBest top 2-tuples (docId, docSim)
#endclass SparseMatrixSimilarity



def buildDmlCorpus(language):
    numdam = sources.DmlSource('numdam', sourcePath['numdam'])
    dmlcz = sources.DmlSource('dmlcz', sourcePath['dmlcz'])
    arxmliv = sources.ArxmlivSource('arxmliv', sourcePath['arxmliv'])
    
    config = corpora.DmlConfig('gensim', resultDir = sourcePath['results'], acceptLangs = [language])
    
    dml = corpora.DmlCorpus()
    dml.processConfig(config)
    dml.buildDictionary()
    dml.dictionary.filterExtremes(noBelow = 5, noAbove = 0.3)
    
    dml.save() # save the whole object as binary data
    dml.saveDebug() # save docNo->docId mapping, and termId->term mapping in text format
    dml.saveAsMatrix() # save word count matrix and tfidf matrix
    
                                                               
def generateSimilar(method, docSim):
    pass

    
