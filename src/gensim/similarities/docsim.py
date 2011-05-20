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
import scipy
import scipy.sparse
from scipy.sparse import sparsetools
from scipy.linalg import get_blas_funcs

from gensim import interfaces, utils, matutils


logger = logging.getLogger('gensim.similarity.docsim')


class Similarity(interfaces.SimilarityABC):
    """
    Compute cosine similarity against a corpus of documents. This is done by a full
    sequential scan of the corpus.

    If your corpus is reasonably small (fits in RAM), consider using `MatrixSimilarity`
    or `SparseMatrixSimilarity` instead, for (much) faster similarity searches.
    """
    def __init__(self, corpus, numBest=None):
        """
        If `numBest` is left unspecified, similarity queries return a full list (one
        float for every document in the corpus, including the query document):

        If `numBest` is set, queries return `numBest` most similar documents, as a
        sorted list:

        >>> sms = Similarity(corpus, numBest=3)
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
    def __init__(self, corpus, numBest=None, dtype=numpy.float32, numFeatures=None, chunks=256):
        """
        If `numBest` is left unspecified, similarity queries return a full list (one
        float for every document in the corpus):

        >>> sms = MatrixSimilarity(corpus)
        >>> sms[vec12]
        [0.0, 0.0, 0.2, 0.13, 0.8, 0.0, 0.0]

        If `numBest` is set, queries return `numBest` most similar documents, as a
        sorted list:

        >>> sms = MatrixSimilarity(corpus, numBest=3)
        >>> sms[vec12]
        [(2, 0.2), (3, 0.13), (4, 0.8)]

        """
        if numFeatures is None:
            logger.info("scanning corpus to determine the number of features")
            numFeatures = 1 + utils.getMaxId(corpus)

        self.numFeatures = numFeatures
        self.numBest = numBest
        self.normalize = True
        self.chunks = chunks

        if corpus is not None:
            logger.info("creating matrix for %s documents and %i features" %
                         (len(corpus), numFeatures))
            self.corpus = numpy.empty(shape=(len(corpus), numFeatures), dtype=dtype)
            # iterate over corpus, populating the numpy index matrix with (normalized)
            # document vectors
            for docNo, vector in enumerate(corpus):
                if docNo % 1000 == 0:
                    logger.info("PROGRESS: at document #%i/%i" % (docNo, len(corpus)))
                vector = matutils.unitVec(matutils.sparse2full(vector, numFeatures))
                self.corpus[docNo] = vector


    def getSimilarities(self, query):
        """
        Return similarity of sparse vector `query` to all documents in the corpus,
        as a numpy array.

        If `query` is a collection of documents, return a 2D array of similarities
        of each document in `query` to all documents in the corpus (=batch query,
        faster than processing each document in turn).
        """
        is_corpus, query = utils.isCorpus(query)
        if is_corpus:
            query = numpy.asarray([matutils.sparse2full(vec, self.numFeatures) for vec in query],
                                  dtype=self.corpus.dtype)
        else:
            if scipy.sparse.issparse(query):
                query = query.toarray() # convert sparse to dense
            elif isinstance(query, numpy.ndarray):
                pass
            else:
                # default case: query is a single vector in sparse gensim format
                query = matutils.sparse2full(query, self.numFeatures)
            query = numpy.asarray(query, dtype=self.corpus.dtype)

        # do a little transposition dance to stop numpy from making a copy of
        # self.corpus internally in numpy.dot (very slow).
        result = numpy.dot(self.corpus, query.T).T # return #queries x #index
        return result # XXX: removed casting the result to list; does anyone care?
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
    def __init__(self, corpus, numBest=None, chunks=500, dtype=numpy.float32):
        """
        If `numBest` is left unspecified, similarity queries return a full list (one
        float for every document in the corpus, including the query document):

        If `numBest` is set, queries return `numBest` most similar documents, as a
        sorted list:

        >>> sms = SparseMatrixSimilarity(corpus, numBest=3)
        >>> sms[vec12]
        [(12, 1.0), (30, 0.95), (5, 0.45)]

        """
        self.numBest = numBest
        self.normalize = True
        self.chunks = chunks

        if corpus is not None:
            logger.info("creating sparse index")

            # iterate over input corpus, populating the sparse index matrix
            try:
                # use the more efficient corpus generation version, if the input
                # `corpus` is MmCorpus-like.
                num_terms, num_docs, num_nnz = corpus.numTerms, corpus.numDocs, corpus.numElements
            except AttributeError:
                # no MmCorpus, use the slower version :(
                num_terms, num_docs, num_nnz = None, None, None
            self.corpus = matutils.corpus2csc((matutils.unitVec(vector) for vector in corpus),
                                              num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,
                                              dtype=numpy.float32).T

            # convert to Compressed Sparse Row for efficient row slicing and multiplications
            self.corpus = self.corpus.tocsr() # currently does nothing, CSC.T is already CSR
            logger.info("created %s" % repr(self.corpus))


    def getSimilarities(self, query):
        """
        Return similarity of sparse vector `query` to all documents in the corpus,
        as a numpy array.

        If `query` is a collection of documents, return a 2D array of similarities
        of each document in `query` to all documents in the corpus (=batch query,
        faster than processing each document in turn).
        """
        is_corpus, query = utils.isCorpus(query)
        if is_corpus:
            query = matutils.corpus2csc(query, self.corpus.shape[1])
        else:
            if scipy.sparse.issparse(query):
                query = query.T # convert documents=rows to documents=columns
            elif isinstance(query, numpy.ndarray):
                if query.ndim == 1:
                    query.shape = (len(query), 1)
                query = scipy.sparse.csc_matrix(query)
            else:
                # default case: query is a single vector, in sparse gensim format
                query = matutils.corpus2csc([query], self.corpus.shape[1])

        # compute cosine similarity against every other document in the collection
        result = self.corpus * query.tocsc() # N x T * T x C = N x C
        if result.shape[1] == 1:
            # for queries of one document, return a 1d array
            result = result.toarray().flatten()
        else:
            # otherwise, return a 2d matrix (#queries x #index)
            result = result.toarray().T
        return result
#endclass SparseMatrixSimilarity

