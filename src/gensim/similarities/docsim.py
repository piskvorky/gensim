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
from scipy.linalg import get_blas_funcs

from gensim import interfaces, utils, matutils


logger = logging.getLogger('gensim.similarity.docsim')


class Shard(utils.SaveLoad):
    def __init__(self, fname, index):
        self.fname = fname
        self.length = len(index)
        self.cls = index.__class__
        logger.info("saving index shard to %s" % fname)
        index.save(fname)
        # TODO: load back with mmap immediately?
        # TODO: support for remote shards

    def __len__(self):
        return self.length


    def get_index(self):
        logger.debug("loading index from %s" % self.fname)
        index = self.cls.load(self.fname) # FIXME yuck, optimize


    def __getitem__(self, query):
        index = self.get_index()
        try:
            index.num_best = self.num_best
        except:
            pass
        return index[query]


class Similarity(interfaces.SimilarityABC):
    """
    Compute cosine similarity of a dynamic query against a static corpus of documents
    ("the index").

    Scalability is achieved by sharding the index into smaller pieces, each of which
    fits into core memory (see the `(Sparse)MatrixSimilarity` classes in this module).
    The shard themselves are simply stored as files to disk

    """
    def __init__(self, output_prefix, corpus, num_features, num_best=None, shardsize=20000):
        """
        Construct the index from `corpus`. The index can be later extended by calling
        the `add_documents` method. Documents are split into shards of `shardsize`
        each, converted to a matrix and stored to disk under `output_prefix.shard_number`.

        `num_features` is the number of features in the `corpus` (e.g. size of the
        dictionary, or the number of latent topics).

        If `num_best` is left unspecified, similarity queries will return a full
        vector with one float for every document in the index:

        >>> index = Similarity('/tmp/index', corpus, num_features=400)
        >>> index[vec]
        [0.0, 0.0, 0.2, 0.13, 0.8, 0.0, 0.0]

        If `num_best` is set, queries return only the `num_best` most similar documents.

        >>> index = Similarity(corpus, num_best=3)
        >>> indexsms[vec]
        [(12, 1.0), (30, 0.95), (5, 0.45)]

        """
#       TODO: make the index more robust against disk/power failures; use some db/pytables?
        self.output_prefix = output_prefix
        self.num_features = num_features
        self.num_best = num_best
        self.normalize = True
        self.shardsize = shardsize
        self.shards = []
        self.fresh_docs, self.fresh_nnz = [], 0

        if corpus is not None:
            self.add_documents(corpus)


    def add_documents(self, corpus):
        """
        Extend the index with new documents.

        Internally, documents are buffered and then spilled to disk when there's
        at least `self.shardsize` of them (or when a query is issued).
        """
        if self.shards and len(self.shards[-1]) < self.shardsize:
            # last shard was incomplete; load it back and add the documents there,
            # don't start a new shard.
            # TODO mention in docs that the pattern of "add 1 doc/query/add 1 doc/query/.."
            # is very inefficient! add many docs (ideally, all of them), then query.
            self.reopen_shard()
        for doc in corpus:
            self.fresh_docs.append(doc)
            self.fresh_nnz += len(doc)
            if len(self.fresh_docs) >= self.shardsize:
                self.close_shard()


    def shardid2filename(self, shardid):
        if self.output_prefix.endswith('.'):
            return "%s%s" % (self.output_prefix, shardid)
        else:
            return "%s.%s" % (self.output_prefix, shardid)


    def close_shard(self):
        """
        This forces the latest shard to close (be converted to matrix and stored
        to disk).

        NOTE: the shard is closed even if it is not full yet (its size is smaller
        than `self.shardsize`). If documents are added later, this incomplete
        shard will be loaded again and completed.
        """
        if not self.fresh_docs:
            return
        shardid = len(self.shards)
        # consider the shard sparse if its density is < 30%
        issparse = 0.3 > 1.0 * self.fresh_nnz / (len(self.fresh_docs) * self.num_features)
        if issparse:
            index = SparseMatrixSimilarity(self.fresh_docs, num_terms=self.num_features,
                                           num_docs=len(self.fresh_docs), num_nnz=self.fresh_nnz)
        else:
            index = MatrixSimilarity(self.fresh_docs, num_features=self.num_features)
        logger.info("creating %s shard #%s" % ('sparse' if issparse else 'dense', shardid))
        shard = Shard(self.shardid2filename(shardid), index)
        shard.num_best = self.num_best
        shard.num_nnz = self.fresh_nnz
        self.shards.append(shard)
        self.fresh_docs, self.fresh_nnz = [], 0


    def reopen_shard(self):
        assert self.shards
        if self.fresh_docs:
            raise ValueError("cannot reopen shard with fresh documents in index")
        last_shard = self.shards[-1]
        last_index = last_shard.get_index()
        if scipy.issparse(last_index.corpus):
            self.fresh_docs = [doc for doc in matutils.Sparse2Corpus(last_index.corpus)]
        else:
            self.fresh_docs = [doc for doc in matutils.Dense2Corpus(last_index.corpus)]
        self.fresh_nnz = last_shard.num_nnz
        del self.shards[-1] # remove the shard from index NOTE: but its file is not deleted


    def get_similarities(self, query):
        self.close_shard() # no-op if no documents added to index since last query
        results = []
        for shard in self.shards:
            shard.num_best = self.num_best
            results.append(shard[query])

        if self.num_best:
            # FIXME when query a chunk
            result = sorted(sum(results, []), key=lambda item: -item[1])[ : self.num_best]
        else:
            result = numpy.hstack(results)
        return result
#endclass Similarity


class MatrixSimilarity(interfaces.SimilarityABC):
    """
    Compute similarity against a corpus of documents by storing its
    term-document (or concept-document) matrix in memory. The similarity measure
    used is cosine between two vectors.

    The matrix is internally stored as a *dense* numpy array. Unless the entire matrix
    fits into main memory, use `Similarity` instead.

    See also `Similarity` and `SparseMatrixSimilarity` in this module.

    """
    def __init__(self, corpus, num_best=None, dtype=numpy.float32, num_features=None, chunks=256):
        """
        `num_features` is the number of features in the corpus (will be determined
        automatically by scanning the corpus if not specified). See `Similarity`
        class for description of the other parameters.

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
            self.index = numpy.empty(shape=(len(corpus), num_features), dtype=dtype)
            # iterate over corpus, populating the numpy index matrix with (normalized)
            # document vectors
            for docno, vector in enumerate(corpus):
                if docno % 1000 == 0:
                    logger.info("PROGRESS: at document #%i/%i" % (docno, len(corpus)))
                vector = matutils.unitvec(matutils.sparse2full(vector, num_features))
                self.index[docno] = vector


    def __len__(self):
        return self.corpus.shape[1]


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


    The matrix is internally stored as a `scipy.sparse.csr` matrix. Unless the entire
    matrix fits into main memory, use `Similarity` instead.

    See also `Similarity` and `MatrixSimilarity` in this module.
    """
    def __init__(self, corpus, num_best=None, chunks=500, dtype=numpy.float32,
                 num_terms=None, num_docs=None, num_nnz=None):
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
                # no MmCorpus, use the slower version (or maybe user supplied the
                # num_* params in constructor)
                pass
            self.corpus = matutils.corpus2csc((matutils.unitvec(vector) for vector in corpus),
                                              num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,
                                              dtype=numpy.float32, printprogress=10000).T

            # convert to Compressed Sparse Row for efficient row slicing and multiplications
            self.corpus = self.corpus.tocsr() # currently no-op, CSC.T is already CSR
            logger.info("created %s" % repr(self.corpus))


    def __len__(self):
        return self.corpus.shape[1]


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

