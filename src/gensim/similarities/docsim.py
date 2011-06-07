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
    def __init__(self, corpus, num_best=None):
        """
        If `num_best` is left unspecified, similarity queries return a full list (one
        float for every document in the corpus, including the query document):

        If `num_best` is set, queries return `num_best` most similar documents, as a
        sorted list:

        >>> sms = Similarity(corpus, num_best=3)
        >>> sms[vec12]
        [(12, 1.0), (30, 0.95), (5, 0.45)]

        """
        self.corpus = corpus
        self.num_best = num_best
        self.normalize = True


    def get_similarities(self, doc):
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
    def __init__(self, corpus, num_best=None, dtype=numpy.float32, num_features=None, chunks=256):
        """
        If `num_best` is left unspecified, similarity queries return a full list (one
        float for every document in the corpus):

        >>> sms = MatrixSimilarity(corpus)
        >>> sms[vec12]
        [0.0, 0.0, 0.2, 0.13, 0.8, 0.0, 0.0]

        If `num_best` is set, queries return `num_best` most similar documents, as a
        sorted list:

        >>> sms = MatrixSimilarity(corpus, num_best=3)
        >>> sms[vec12]
        [(2, 0.2), (3, 0.13), (4, 0.8)]

        """
        if num_features is None:
            logger.info("scanning corpus to determine the number of features")
            num_features = 1 + utils.get_max_id(corpus)

        self.num_features = num_features
        self.num_best = num_best
        self.normalize = True
        self.chunks = chunks

        if corpus is not None:
            logger.info("creating matrix for %s documents and %i features" %
                         (len(corpus), num_features))
            self.corpus = numpy.empty(shape=(len(corpus), num_features), dtype=dtype)
            # iterate over corpus, populating the numpy index matrix with (normalized)
            # document vectors
            for docno, vector in enumerate(corpus):
                if docno % 1000 == 0:
                    logger.info("PROGRESS: at document #%i/%i" % (docno, len(corpus)))
                vector = matutils.unitvec(matutils.sparse2full(vector, num_features))
                self.corpus[docno] = vector


    def get_similarities(self, query):
        """
        Return similarity of sparse vector `query` to all documents in the corpus,
        as a numpy array.

        If `query` is a collection of documents, return a 2D array of similarities
        of each document in `query` to all documents in the corpus (=batch query,
        faster than processing each document in turn).
        """
        is_corpus, query = utils.is_corpus(query)
        if is_corpus:
            query = numpy.asarray([matutils.sparse2full(vec, self.num_features) for vec in query],
                                  dtype=self.corpus.dtype)
        else:
            if scipy.sparse.issparse(query):
                query = query.toarray() # convert sparse to dense
            elif isinstance(query, numpy.ndarray):
                pass
            else:
                # default case: query is a single vector in sparse gensim format
                query = matutils.sparse2full(query, self.num_features)
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
    def __init__(self, corpus, num_best=None, chunks=500, dtype=numpy.float32):
        """
        If `num_best` is left unspecified, similarity queries return a full list (one
        float for every document in the corpus, including the query document):

        If `num_best` is set, queries return `num_best` most similar documents, as a
        sorted list:

        >>> sms = SparseMatrixSimilarity(corpus, num_best=3)
        >>> sms[vec12]
        [(12, 1.0), (30, 0.95), (5, 0.45)]

        """
        self.num_best = num_best
        self.normalize = True
        self.chunks = chunks

        if corpus is not None:
            logger.info("creating sparse index")

            # iterate over input corpus, populating the sparse index matrix
            try:
                # use the more efficient corpus generation version, if the input
                # `corpus` is MmCorpus-like (knows its shape and number of non-zeroes).
                num_terms, num_docs, num_nnz = corpus.num_terms, corpus.num_docs, corpus.num_nnz
                logger.debug("using efficient sparse index creation")
            except AttributeError:
                # no MmCorpus, use the slower version :(
                num_terms, num_docs, num_nnz = None, None, None
            self.corpus = matutils.corpus2csc((matutils.unitvec(vector) for vector in corpus),
                                              num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,
                                              dtype=numpy.float32, printprogress=10000).T

            # convert to Compressed Sparse Row for efficient row slicing and multiplications
            self.corpus = self.corpus.tocsr() # currently no-op, CSC.T is already CSR
            logger.info("created %s" % repr(self.corpus))


    def get_similarities(self, query):
        """
        Return similarity of sparse vector `query` to all documents in the corpus,
        as a numpy array.

        If `query` is a collection of documents, return a 2D array of similarities
        of each document in `query` to all documents in the corpus (=batch query,
        faster than processing each document in turn).
        """
        is_corpus, query = utils.is_corpus(query)
        if is_corpus:
            query = matutils.corpus2csc(query, self.corpus.shape[1], dtype=self.corpus.dtype)
        else:
            if scipy.sparse.issparse(query):
                query = query.T # convert documents=rows to documents=columns
            elif isinstance(query, numpy.ndarray):
                if query.ndim == 1:
                    query.shape = (len(query), 1)
                query = scipy.sparse.csc_matrix(query, dtype=self.corpus.dtype)
            else:
                # default case: query is a single vector, in sparse gensim format
                query = matutils.corpus2csc([query], self.corpus.shape[1], dtype=self.corpus.dtype)

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

