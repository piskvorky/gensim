#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains functions and classes for computing similarities across
a collection of documents in the Vector Space Model.

The main class is `Similarity`, which builds an index for a given set of documents.
Once the index is built, you can perform efficient queries like "Tell me how similar
is this query document to each document in the index?". The result is a vector
of numbers as large as the size of the initial set of documents, that is, one float
for each index document. Alternatively, you can also request only the top-N most
similar index documents to the query.

You can later add new documents to the index via `Similarity.add_documents()`.

How It Works
------------

The `Similarity` class splits the index into several smaller sub-indexes ("shards"),
which are disk-based. If your entire index fits in memory (~hundreds of thousands
documents for 1GB of RAM), you can also use the `MatrixSimilarity` or `SparseMatrixSimilarity`
classes directly. These are more simple but do not scale well (the entire index is
kept in RAM).

Once the index has been initialized, you can query for document similarity simply by:

>>> index = Similarity('/tmp/tst', corpus, num_features=12) # build the index
>>> similarities = index[query] # get similarities between the query and all index documents

If you have more query documents, you can submit them all at once, in a batch:

>>> for similarities in index[batch_of_documents]: # the batch is simply an iterable of documents (=gensim corpus)
>>>     ...

The benefit of this batch (aka "chunked") querying is much better performance.
To see the speed-up on your machine, run ``python -m gensim.test.simspeed``
(compare to my results `here <http://groups.google.com/group/gensim/msg/4f6f171a869e4fca?>`_).

There is also a special syntax for when you need similarity of documents in the index
to the index itself (i.e. query=index documents themselves). This special syntax
already uses the faster, batch queries internally:

>>> for similarities in index: # return all similarities of the 1st index document, then 2nd...
>>>     ...


"""


import logging
import itertools
import os
import random
import heapq

import numpy
import scipy
import scipy.sparse

from gensim import interfaces, utils, matutils


logger = logging.getLogger('gensim.similarities.docsim')


class Shard(utils.SaveLoad):
    """A proxy class that represents a single shard instance within a Similarity
    index.

    Basically just wraps (Sparse)MatrixSimilarity so that it mmaps from disk on
    request (query).

    """
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
        return ("%s Shard(%i documents in %s)" % (self.cls.__name__, len(self), self.fname))


    def get_index(self):
        logger.debug("loading index from %s" % self.fname)
        return self.cls.load(self.fname)


    def get_document_id(self, pos):
        """Return index vector at position `pos`.

        The vector is of the same type as the underlying index (ie., dense for
        MatrixSimilarity and scipy.sparse for SparseMatrixSimilarity.
        """
        assert 0 <= pos < len(self), "requested position out of range"
        index = self.get_index()
        return index.index[pos]


    def __getitem__(self, query):
        index = self.get_index()
        try:
            index.num_best = self.num_best
            index.normalize = self.normalize
        except:
            raise ValueError("num_best and normalize have to be set before querying a proxy Shard object")
        return index[query]



class Similarity(interfaces.SimilarityABC):
    """
    Compute cosine similarity of a dynamic query against a static corpus of documents
    ("the index").

    Scalability is achieved by sharding the index into smaller pieces, each of which
    fits into core memory (see the `(Sparse)MatrixSimilarity` classes in this module).
    The shards themselves are simply stored as files to disk and mmap'ed back as needed.

    """
    def __init__(self, output_prefix, corpus, num_features, num_best=None, chunksize=1024, shardsize=32768):
        """
        Construct the index from `corpus`. The index can be later extended by calling
        the `add_documents` method. Documents are split into shards of `shardsize`
        documents each, converted to a matrix (for fast BLAS calls) and stored to disk
        under `output_prefix.shard_number` (=you need write access to that location).
        If you don't specify an output prefix, a random filename in temp will be used.

        `shardsize` should be chosen so that a `shardsize x chunksize` matrix of floats
        fits comfortably into main memory.

        `num_features` is the number of features in the `corpus` (e.g. size of the
        dictionary, or the number of latent topics for latent semantic models).

        If `num_best` is left unspecified, similarity queries will return a full
        vector with one float for every document in the index:

        >>> index = Similarity('/path/to/index', corpus, num_features=400) # if corpus has 7 documents...
        >>> index[query] # ... then result will have 7 floats
        [0.0, 0.0, 0.2, 0.13, 0.8, 0.0, 0.1]

        If `num_best` is set, queries return only the `num_best` most similar documents,
        always leaving out documents for which the similarity is 0
        (i.e. vectors with all dimensions zero and vectors with non-zero dimensions, but none in common).
        If the input vector itself has only dimensions with value zero (the sparse representation is empty),
        the returned list will always be empty.

        >>> index.num_best = 3
        >>> index[query] # return at most "num_best" of `(index_of_document, similarity)` tuples
        [(4, 0.8), (2, 0.13), (3, 0.13)]

        You can also override `num_best` dynamically, simply by setting e.g.
        `self.num_best = 10` before doing a query.

        """
        if output_prefix is None:
            # undocumented feature: set output_prefix=None to create the server in temp
            self.output_prefix = utils.randfname(prefix='simserver')
        else:
            self.output_prefix = output_prefix
        self.num_features = num_features
        self.num_best = num_best
        self.normalize = True
        self.chunksize = int(chunksize)
        self.shardsize = shardsize
        self.shards = []
        self.fresh_docs, self.fresh_nnz = [], 0

        if corpus is not None:
            self.add_documents(corpus)


    def __len__(self):
        return len(self.fresh_docs) + sum([len(shard) for shard in self.shards])


    def __str__(self):
        return ("Similarity index with %i documents in %i shards (stored under %s)" %
                (len(self), len(self.shards), self.output_prefix))


    def add_documents(self, corpus):
        """
        Extend the index with new documents.

        Internally, documents are buffered and then spilled to disk when there's
        `self.shardsize` of them (or when a query is issued).
        """
        min_ratio = 1.0 # 0.5 to only reopen shards that are <50% complete
        if self.shards and len(self.shards[-1]) < min_ratio * self.shardsize:
            # The last shard was incomplete (<; load it back and add the documents there, don't start a new shard
            self.reopen_shard()
        for doc in corpus:
            if isinstance(doc, numpy.ndarray):
                doclen = len(doc)
            elif scipy.sparse.issparse(doc):
                doclen = doc.nnz
            else:
                doclen = len(doc)
                if doclen < 0.3 * self.num_features:
                    doc = matutils.unitvec(matutils.corpus2csc([doc], self.num_features).T)
                else:
                    doc = matutils.unitvec(matutils.sparse2full(doc, self.num_features))
            self.fresh_docs.append(doc)
            self.fresh_nnz += doclen
            if len(self.fresh_docs) >= self.shardsize:
                self.close_shard()
            if len(self.fresh_docs) % 10000 == 0:
                logger.info("PROGRESS: fresh_shard size=%i" % len(self.fresh_docs))


    def shardid2filename(self, shardid):
        if self.output_prefix.endswith('.'):
            return "%s%s" % (self.output_prefix, shardid)
        else:
            return "%s.%s" % (self.output_prefix, shardid)


    def close_shard(self):
        """
        Force the latest shard to close (be converted to a matrix and stored
        to disk). Do nothing if no new documents added since last call.

        **NOTE**: the shard is closed even if it is not full yet (its size is smaller
        than `self.shardsize`). If documents are added later via `add_documents()`,
        this incomplete shard will be loaded again and completed.
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
            raise ValueError("cannot reopen a shard with fresh documents in index")
        last_shard = self.shards[-1]
        last_index = last_shard.get_index()
        logger.info("reopening an incomplete shard of %i documents" % len(last_shard))

        self.fresh_docs = list(last_index.index)
        self.fresh_nnz = last_shard.num_nnz
        del self.shards[-1] # remove the shard from index, *but its file on disk is not deleted*
        logger.debug("reopen complete")


    def __getitem__(self, query):
        """Get similarities of document `query` to all documents in the corpus.

        **or**

        If `query` is a corpus (iterable of documents), return a matrix of similarities
        of all query documents vs. all corpus document. This batch query is more
        efficient than computing the similarities one document after another.
        """
        self.close_shard() # no-op if no documents added to index since last query

        # reset num_best and normalize parameters, in case they were changed dynamically
        for shard in self.shards:
            shard.num_best = self.num_best
            shard.normalize = self.normalize

        # there are 4 distinct code paths, depending on whether input `query` is
        # a corpus (or numpy/scipy matrix) or a single document, and whether the
        # similarity result is a full array or only num_best most similar documents.

        if self.num_best is None:
            # user asked for all documents => just stack the sub-results into a single matrix
            # (works for both corpus / single doc query)
            return numpy.hstack(shard[query] for shard in self.shards)

        offsets = numpy.cumsum([0] + [len(shard) for shard in self.shards])
        convert = lambda doc, shard_no: [(doc_index + offsets[shard_no], sim)
                                         for doc_index, sim in doc]
        is_corpus, query = utils.is_corpus(query)
        is_corpus = is_corpus or hasattr(query, 'ndim') and query.ndim > 1 and query.shape[0] > 1
        if not is_corpus:
            # user asked for num_best most similar and query is a single doc
            results = (convert(shard[query], shard_no)
                       for shard_no, shard in enumerate(self.shards))
            return heapq.nlargest(self.num_best, itertools.chain(*results), key=lambda item: item[1])

        # the trickiest combination: returning num_best results when query was a corpus
        shard_results = []
        for shard_no, shard in enumerate(self.shards):
            shard_result = [convert(doc, shard_no) for doc in shard[query]]
            shard_results.append(shard_result)
        result = []
        for parts in itertools.izip(*shard_results):
            merged = heapq.nlargest(self.num_best, itertools.chain(*parts), key=lambda item: item[1])
            result.append(merged)
        return result


    def similarity_by_id(self, docpos):
        """
        Return similarity of the given document only. `pos` is the position
        of the query document within index.
        """
        self.close_shard() # no-op if no documents added to index since last query
        pos = 0
        for shard in self.shards:
            pos += len(shard)
            if docpos < pos:
                break
        if not self.shards or docpos < 0 or docpos >= pos:
            raise ValueError("invalid document position: %s (must be 0 <= x < %s)" %
                             (docpos, len(self)))
        norm, self.normalize = self.normalize, False
        query = shard.get_document_id(docpos - pos + len(shard))
        result = self[query]
        self.normalize = norm
        return result


    def __iter__(self):
        """
        For each index document, compute cosine similarity against all other
        documents in the index and yield the result.
        """
        # turn off query normalization (vectors in the index are already normalized, save some CPU)
        norm, self.normalize = self.normalize, False

        for chunk in self.iter_chunks():
            if chunk.shape[0] > 1:
                for sim in self[chunk]:
                    yield sim
            else:
                yield self[chunk]

        self.normalize = norm # restore normalization


    def iter_chunks(self, chunksize=None):
        """
        Iteratively yield the index as chunks of documents, each of size <= chunksize.

        The chunk is returned in its raw form (matrix or sparse matrix slice).
        The size of the chunk may be smaller than requested; it is up to the caller
        to check the result for real length, using `chunk.shape[0]`.
        """
        self.close_shard()

        if chunksize is None:
            # if not explicitly specified, use the chunksize from the constructor
            chunksize = self.chunksize

        for shard in self.shards:
            query = shard.get_index().index
            for chunk_start in xrange(0, query.shape[0], chunksize):
                # scipy.sparse doesn't allow slicing beyond real size of the matrix
                # (unlike numpy). so, clip the end of the chunk explicitly to make
                # scipy.sparse happy
                chunk_end = min(query.shape[0], chunk_start + chunksize)
                chunk = query[chunk_start : chunk_end] # create a view
                yield chunk


    def save(self, fname=None):
        """
        Save the object via pickling (also see load) under filename specified in
        the constructor.

        Calls `close_shard` internally to spill any unfinished shards to disk first.

        """
        self.close_shard()
        if fname is None:
            fname = self.output_prefix
        super(Similarity, self).save(fname)
#endclass Similarity



class MatrixSimilarity(interfaces.SimilarityABC):
    """
    Compute similarity against a corpus of documents by storing the index matrix
    in memory. The similarity measure used is cosine between two vectors.

    Use this if your input corpus contains dense vectors (such as documents in LSI
    space) and fits into RAM.

    The matrix is internally stored as a *dense* numpy array. Unless the entire matrix
    fits into main memory, use `Similarity` instead.

    See also `Similarity` and `SparseMatrixSimilarity` in this module.

    """
    def __init__(self, corpus, num_best=None, dtype=numpy.float32, num_features=None, chunksize=256):
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
        self.chunksize = chunksize

        if corpus is not None:
            logger.info("creating matrix for %s documents and %i features" %
                         (len(corpus), num_features))
            self.index = numpy.empty(shape=(len(corpus), num_features), dtype=dtype)
            # iterate over corpus, populating the numpy index matrix with (normalized)
            # document vectors
            for docno, vector in enumerate(corpus):
                if docno % 1000 == 0:
                    logger.debug("PROGRESS: at document #%i/%i" % (docno, len(corpus)))
                # individual documents in fact may be in numpy.scipy.sparse format as well.
                # it's not documented because other it's not fully supported throughout.
                # the user better know what he's doing (no normalization, must
                # explicitly supply num_features etc).
                if isinstance(vector, numpy.ndarray):
                    pass
                elif scipy.sparse.issparse(vector):
                    vector = vector.toarray().flatten()
                else:
                    vector = matutils.unitvec(matutils.sparse2full(vector, num_features))
                self.index[docno] = vector


    def __len__(self):
        return self.index.shape[0]


    def get_similarities(self, query):
        """
        Return similarity of sparse vector `query` to all documents in the corpus,
        as a numpy array.

        If `query` is a collection of documents, return a 2D array of similarities
        of each document in `query` to all documents in the corpus (=batch query,
        faster than processing each document in turn).

        **Do not use this function directly; use the self[query] syntax instead.**

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
        # self.index internally in numpy.dot (very slow).
        result = numpy.dot(self.index, query.T).T # return #queries x #index
        return result # XXX: removed casting the result from array to list; does anyone care?


    def save(self, fname):
        """
        Override the default `save` (which uses cPickle), because that's
        too inefficient and cPickle has bugs. Instead, single out the large index
        matrix and store that separately in binary format (that can be directly
        mmap'ed), under `fname.npy`. The rest of the object is pickled to `fname`.
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
        logger.debug("loading %s object from %s" % (cls.__name__, fname))
        result = utils.unpickle(fname)
        result.index = numpy.load(fname + '.npy', mmap_mode='r') # load back as read-only
        return result
#endclass MatrixSimilarity



class SparseMatrixSimilarity(interfaces.SimilarityABC):
    """
    Compute similarity against a corpus of documents by storing the sparse index
    matrix in memory. The similarity measure used is cosine between two vectors.

    Use this if your input corpus contains sparse vectors (such as documents in
    bag-of-words format) and fits into RAM.

    The matrix is internally stored as a `scipy.sparse.csr` matrix. Unless the entire
    matrix fits into main memory, use `Similarity` instead.

    See also `Similarity` and `MatrixSimilarity` in this module.
    """
    def __init__(self, corpus, num_best=None, chunksize=500, dtype=numpy.float32,
                 num_terms=None, num_docs=None, num_nnz=None):
        self.num_best = num_best
        self.normalize = True
        self.chunksize = chunksize

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
            corpus = (matutils.scipy2sparse(v) if scipy.sparse.issparse(v) else
                      (matutils.full2sparse(v) if isinstance(v, numpy.ndarray) else
                       matutils.unitvec(v)) for v in corpus)
            self.index = matutils.corpus2csc(corpus, num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,
                                              dtype=dtype, printprogress=10000).T

            # convert to Compressed Sparse Row for efficient row slicing and multiplications
            self.index = self.index.tocsr() # currently no-op, CSC.T is already CSR
            logger.info("created %r" % self.index)


    def __len__(self):
        return self.index.shape[0]


    def get_similarities(self, query):
        """
        Return similarity of sparse vector `query` to all documents in the corpus,
        as a numpy array.

        If `query` is a collection of documents, return a 2D array of similarities
        of each document in `query` to all documents in the corpus (=batch query,
        faster than processing each document in turn).

        **Do not use this function directly; use the self[query] syntax instead.**

        """
        is_corpus, query = utils.is_corpus(query)
        if is_corpus:
            query = matutils.corpus2csc(query, self.index.shape[1], dtype=self.index.dtype)
        else:
            if scipy.sparse.issparse(query):
                query = query.T # convert documents=rows to documents=columns
            elif isinstance(query, numpy.ndarray):
                if query.ndim == 1:
                    query.shape = (1, len(query))
                query = scipy.sparse.csr_matrix(query, dtype=self.index.dtype).T
            else:
                # default case: query is a single vector, in sparse gensim format
                query = matutils.corpus2csc([query], self.index.shape[1], dtype=self.index.dtype)

        # compute cosine similarity against every other document in the collection
        result = self.index * query.tocsc() # N x T * T x C = N x C
        if result.shape[1] == 1 and not is_corpus:
            # for queries of one document, return a 1d array
            result = result.toarray().flatten()
        else:
            # otherwise, return a 2d matrix (#queries x #index)
            result = result.toarray().T
        return result


    def save(self, fname):
        """
        Override the default `save` (which uses cPickle), because that's
        too inefficient and cPickle has bugs. Instead, single out the large internal
        arrays and store them separately in binary format (that can be directly
        mmap'ed), under `fname.array_name.npy`.
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
        logger.debug("loading %s object from %s and %s.*.npy" % (cls.__name__, fname, fname))
        result = utils.unpickle(fname)
        data = numpy.load(fname + '.data.npy', mmap_mode='r') # load back as read-only
        indptr = numpy.load(fname + '.indptr.npy', mmap_mode='r')
        indices = numpy.load(fname + '.indices.npy', mmap_mode='r')
        result.index.data, result.index.indptr, result.index.indices = data, indptr, indices
        return result
#endclass SparseMatrixSimilarity
