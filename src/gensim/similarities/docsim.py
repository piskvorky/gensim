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
        # TODO: support for remote shards (Pyro? multiprocessing?)


    def __len__(self):
        return self.length


    def __str__(self):
        return ("Shard(%i documents in %s)" % (len(self), self.fname))


    def get_index(self):
        logger.debug("loading index from %s" % self.fname)
        return self.cls.load(self.fname)


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
    def __init__(self, output_prefix, corpus, num_features, num_best=None, shardsize=5000):
        """
        Construct the index from `corpus`. The index can be later extended by calling
        the `add_documents` method. Documents are split into shards of `shardsize`
        each, converted to a matrix and stored to disk under `output_prefix.shard_number`.
        `shardsize` should be chosen so that a shardsize x shardsize matrix fits
        comfortably into main memory.

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
        self.output_prefix = output_prefix
        self.num_features = num_features
        self.num_best = num_best
        self.normalize = True
        self.shardsize = shardsize
        self.shards = []
        self.fresh_docs, self.fresh_nnz = [], 0

        if corpus is not None:
            self.add_documents(corpus)
#       TODO: make the index more robust against disk/power failures; use some db/pytables?


    def __len__(self):
        return len(self.fresh_docs) + sum([len(shard) for shard in self.shards])


    def __str__(self):
        return ("Similarity index with %i documents in %i shards (stored under %s)" %
                (len(self), len(self.shards), self.output_prefix))


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
        to disk). Does nothing if no new documents added since last call.

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
        if scipy.sparse.issparse(last_index.index):
            self.fresh_docs = list(matutils.Sparse2Corpus(last_index.index))
        else:
            self.fresh_docs = list(matutils.Dense2Corpus(last_index.index))
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


    def __iter__(self):
        """
        For each index document, compute cosine similarity against all other
        documents in the index and yield the result.
        """
        # turn off query normalization (vectors in the index are assumed to be already normalized)
        norm = self.normalize
        self.normalize = False

        for shard in self.shards:
            shard_index = shard.get_index()
            # use the entire shard index as a gigantic query!
            # FIXME or maybe chunk it into smaller pieces?
            for sims in self[shard_index.index]:
                yield sims
        self.normalize = norm
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
                self.index[docno] = matutils.unitvec(matutils.sparse2full(vector, num_features))


    def __len__(self):
        return self.index.shape[1]


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
                                  dtype=self.index.dtype)
        else:
            if scipy.sparse.issparse(query):
                query = query.toarray() # convert sparse to dense
            elif isinstance(query, numpy.ndarray):
                pass
            else:
                # default case: query is a single vector in sparse gensim format
                query = matutils.sparse2full(query, self.num_features)
            query = numpy.asarray(query, dtype=self.index.dtype)

        # do a little transposition dance to stop numpy from making a copy of
        # self.corpus internally in numpy.dot (very slow).
        result = numpy.dot(self.index, query.T).T # return #queries x #index
        return result # XXX: removed casting the result from array to list; does anyone care?


    def save(self, fname):
        """
        Override the default `save` (which uses cPickle), because that's
        too inefficient. Instead, single out the large index matrix and store that
        separately in binary format (that can be directly mmap'ed), under `fname.npy`.
        """
        logger.info("storing %s object to %s and %s" % (self.__class__.__name__, fname, fname + '.npy'))
        # first, remove the index from self.__dict__, so it doesn't get pickled
        index = self.index
        del self.index
        try:
            utils.pickle(self, fname) # store index-less object
            numpy.save(fname + '.npy', index) # store index
        finally:
            self.index = index


    @classmethod
    def load(cls, fname):
        """
        Load a previously saved object from file (also see `save`).
        """
        logger.info("loading %s object from %s and %s" % (cls.__name__, fname, fname + '.npy'))
        result = utils.unpickle(fname)
        result.index = numpy.load(fname + '.npy', mmap_mode='r') # load back as read-only
        return result
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
            self.index = matutils.corpus2csc((matutils.unitvec(vector) for vector in corpus),
                                              num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,
                                              dtype=numpy.float32, printprogress=10000).T

            # convert to Compressed Sparse Row for efficient row slicing and multiplications
            self.index = self.index.tocsr() # currently no-op, CSC.T is already CSR
            logger.info("created %r" % self.index)


    def __len__(self):
        return self.index.shape[1]


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
            query = matutils.corpus2csc(query, self.index.shape[1], dtype=self.index.dtype)
        else:
            if scipy.sparse.issparse(query):
                query = query.T # convert documents=rows to documents=columns
            elif isinstance(query, numpy.ndarray):
                if query.ndim == 1:
                    query.shape = (len(query), 1)
                query = scipy.sparse.csc_matrix(query, dtype=self.index.dtype)
            else:
                # default case: query is a single vector, in sparse gensim format
                query = matutils.corpus2csc([query], self.index.shape[1], dtype=self.index.dtype)

        # compute cosine similarity against every other document in the collection
        result = self.index * query.tocsc() # N x T * T x C = N x C
        if result.shape[1] == 1:
            # for queries of one document, return a 1d array
            result = result.toarray().flatten()
        else:
            # otherwise, return a 2d matrix (#queries x #index)
            result = result.toarray().T
        return result


    def save(self, fname):
        """
        Override the default `save` (which uses cPickle), because that's
        too inefficient. Instead, single out the large internal arrays and store
        them separately in binary format (that can be directly mmap'ed), under
        `fname.array_name.npy`.
        """
        logger.info("storing %s object to %s and %s.npy" % (self.__class__.__name__, fname, fname))
        assert isinstance(self.index, scipy.sparse.csr_matrix)
        # first, remove the arrays from self.__dict__, so they don't get pickled
        data, indptr, indices = self.index.data, self.index.indptr, self.index.indices
        del self.index.data, self.index.indptr, self.index.indices
        try:
            utils.pickle(self, fname) # store array-less object
            # store arrays (.npy suffix is appended by numpy automatically)
            numpy.save(fname + '.data.npy', data)
            numpy.save(fname + '.indptr.npy', indptr)
            numpy.save(fname + '.indices.npy', indices)
        finally:
            self.index.data, self.index.indptr, self.index.indices = data, indptr, indices


    @classmethod
    def load(cls, fname):
        """
        Load a previously saved object from file (also see `save`).
        """
        logger.info("loading %s object from %s and %s.*.npy" % (cls.__name__, fname, fname))
        result = utils.unpickle(fname)
        data = numpy.load(fname + '.data.npy', mmap_mode='r') # load back as read-only
        indptr = numpy.load(fname + '.indptr.npy', mmap_mode='r')
        indices = numpy.load(fname + '.indices.npy', mmap_mode='r')
        result.index.data, result.index.indptr, result.index.indices = data, indptr, indices
        return result
#endclass SparseMatrixSimilarity

