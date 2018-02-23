#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Computing similarities across a collection of documents in the Vector Space Model.

The main class is :class:`~gensim.similarity.docsim.Similarity`, which builds an index for a given set of documents.
Once the index is built, you can perform efficient queries like "Tell me how similar is this query document to each
document in the index?". The result is a vector of numbers as large as the size of the initial set of documents,
that is, one float for each index document. Alternatively, you can also request only the top-N most
similar index documents to the query.


How It Works
------------
The :class:`~gensim.similarity.docsim.Similarity` class splits the index into several smaller sub-indexes ("shards"),
which are disk-based. If your entire index fits in memory (~hundreds of thousands documents for 1GB of RAM),
you can also use the :class:`~gensim.similarity.docsim.MatrixSimilarity`
or :class:`~gensim.similarity.docsim.SparseMatrixSimilarity` classes directly.
These are more simple but do not scale as well (they keep the entire index in RAM, no sharding).

Once the index has been initialized, you can query for document similarity simply by:
>>> from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
>>>
>>> index_tmpfile = get_tmpfile("index")
>>> query = [(1, 2), (6, 1), (7, 2)]
>>>
>>> index = Similarity(index_tmpfile, common_corpus, num_features=len(common_dictionary)) # build the index
>>> similarities = index[query] # get similarities between the query and all index documents

If you have more query documents, you can submit them all at once, in a batch:

>>> from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
>>>
>>> index_tmpfile = get_tmpfile("index")
>>> batch_of_documents = common_corpus[:]  # only as example
>>> index = Similarity(index_tmpfile, common_corpus, num_features=len(common_dictionary)) # build the index
>>>
>>> for similarities in index[batch_of_documents]: # the batch is simply an iterable of documents, aka gensim corpus.
...     pass

The benefit of this batch (aka "chunked") querying is much better performance.
To see the speed-up on your machine, run ``python -m gensim.test.simspeed``
(compare to my results `here <http://groups.google.com/group/gensim/msg/4f6f171a869e4fca?>`_).

There is also a special syntax for when you need similarity of documents in the index
to the index itself (i.e. queries=indexed documents themselves). This special syntax
uses the faster, batch queries internally and **is ideal for all-vs-all pairwise similarities**:

>>> from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
>>>
>>> index_tmpfile = get_tmpfile("index")
>>> index = Similarity(index_tmpfile, common_corpus, num_features=len(common_dictionary)) # build the index
>>>
>>> for similarities in index: # yield similarities of the 1st indexed document, then 2nd...
...     pass

"""
import logging
import itertools
import os
import heapq

import numpy
import scipy.sparse

from gensim import interfaces, utils, matutils
from six.moves import map as imap, xrange, zip as izip


logger = logging.getLogger(__name__)

PARALLEL_SHARDS = False
try:
    import multiprocessing
    # by default, don't parallelize queries. uncomment the following line if you want that.
#    PARALLEL_SHARDS = multiprocessing.cpu_count() # use #parallel processes = #CPus
except ImportError:
    pass


class Shard(utils.SaveLoad):
    """A proxy that represents a single shard instance within :class:`~gensim.similarity.docsim.Similarity` index.

    Basically just wraps :class:`~gensim.similarities.docsim.MatrixSimilarity`,
    :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`, etc, so that it mmaps from disk on request (query).

    """

    def __init__(self, fname, index):
        """
        Parameters
        ----------
        fname : str
            Path to top-level directory (file) to traverse for corpus documents.
        index : :class:`~gensim.interfaces.SimilarityABC`
            Index object.

        """
        self.dirname, self.fname = os.path.split(fname)
        self.length = len(index)
        self.cls = index.__class__
        logger.info("saving index shard to %s", self.fullname())
        index.save(self.fullname())
        self.index = self.get_index()

    def fullname(self):
        """Get full path to shard file.

        Return
        ------
        str
            Path to shard instance.

        """
        return os.path.join(self.dirname, self.fname)

    def __len__(self):
        """Get length."""
        return self.length

    def __getstate__(self):
        """Special handler for pickle.

        Returns
        -------
        dict
            Object that contains state of current instance without `index`.

        """
        result = self.__dict__.copy()
        # (S)MS objects must be loaded via load() because of mmap (simple pickle.load won't do)
        if 'index' in result:
            del result['index']
        return result

    def __str__(self):
        return "%s Shard(%i documents in %s)" % (self.cls.__name__, len(self), self.fullname())

    def get_index(self):
        """Load & get index.

        Returns
        -------
        :class:`~gensim.interfaces.SimilarityABC`
            Index instance.

        """
        if not hasattr(self, 'index'):
            logger.debug("mmaping index from %s", self.fullname())
            self.index = self.cls.load(self.fullname(), mmap='r')
        return self.index

    def get_document_id(self, pos):
        """Get index vector at position `pos`.

        Parameters
        ----------
        pos : int
            Vector position.

        Return
        ------
        {:class:`scipy.sparse.csr_matrix`, :class:`numpy.ndarray`}
            Index vector. Type depends on underlying index.

        Notes
        -----
        The vector is of the same type as the underlying index (ie., dense for
        :class:`~gensim.similarities.docsim.MatrixSimilarity`
        and scipy.sparse for :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`.
        TODO: Can dense be scipy.sparse?

        """
        assert 0 <= pos < len(self), "requested position out of range"
        return self.get_index().index[pos]

    def __getitem__(self, query):
        """Get similarities of document (or corpus) `query` to all documents in the corpus.

        Parameters
        ----------
        query : {iterable of list of (int, number) , list of (int, number))}
            Document or corpus.

        Returns
        -------
        :class:`numpy.ndarray`
            Similarities of document/corpus if index is :class:`~gensim.similarities.docsim.MatrixSimilarity` **or**
        :class:`scipy.sparse.csr_matrix`
            for case if index is :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`.

        """
        index = self.get_index()
        try:
            index.num_best = self.num_best
            index.normalize = self.normalize
        except Exception:
            raise ValueError("num_best and normalize have to be set before querying a proxy Shard object")
        return index[query]


def query_shard(args):
    """Helper for request query from shard, same as shard[query].

    Parameters
    ---------
    args : (list of (int, number), :class:`~gensim.interfaces.SimilarityABC`)
        Query and Shard instances

    Returns
    -------
    :class:`numpy.ndarray`
        Similarities of document/corpus if index is :class:`~gensim.similarities.docsim.MatrixSimilarity` **or**
    :class:`scipy.sparse.csr_matrix`
        for case if index is :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`.

    """
    query, shard = args  # simulate starmap (not part of multiprocessing in older Pythons)
    logger.debug("querying shard %s num_best=%s in process %s", shard, shard.num_best, os.getpid())
    result = shard[query]
    logger.debug("finished querying shard %s in process %s", shard, os.getpid())
    return result


class Similarity(interfaces.SimilarityABC):
    """Compute cosine similarity of a dynamic query against a static corpus of documents ('the index').

    Notes
    -----
    Scalability is achieved by sharding the index into smaller pieces, each of which fits into core memory
    The shards themselves are simply stored as files to disk and mmap'ed back as needed.



    Examples
    --------
    >>> from gensim.corpora.textcorpus import TextCorpus
    >>> from gensim.test.utils import datapath, get_tmpfile
    >>> from gensim.similarities import Similarity
    >>>
    >>> corpus = TextCorpus(datapath('testcorpus.mm'))
    >>> index_temp = get_tmpfile("index")
    >>> index = Similarity(index_temp, corpus, num_features=400)  # create index
    >>>
    >>> query = next(iter(corpus))
    >>> result = index[query]  # search similar to `query` in index
    >>>
    >>> for sims in index[corpus]: # if you have more query documents, you can submit them all at once, in a batch
    ...     pass
    >>>
    >>> # There is also a special syntax for when you need similarity of documents in the index
    >>> # to the index itself (i.e. queries=indexed documents themselves). This special syntax
    >>> # uses the faster, batch queries internally and **is ideal for all-vs-all pairwise similarities**:
    >>> for similarities in index: # yield similarities of the 1st indexed document, then 2nd...
    ...     pass

    See Also
    --------
    :class:`~gensim.similarities.docsim.MatrixSimilarity`
        Index similarity (dense with cosine distance).
    :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`
        Index similarity (sparse with cosine distance).
    :class:`~gensim.similarities.docsim.SoftCosineSimilarity`
        Index similarity (with soft-cosine distance).
    :class:`~gensim.similarities.docsim.WmdSimilarity`
        Index similarity (with word-mover distance).

    """

    def __init__(self, output_prefix, corpus, num_features, num_best=None, chunksize=256, shardsize=32768, norm='l2'):
        """

        Parameters
        ----------
        output_prefix : str
            Prefix for shard filename. If None - random filename in temp will be used.
        corpus : iterable of list of (int, number)
            Corpus in BoW format.
        num_features : int
            Size of the dictionary (number of features).
        num_best : int, optional
            If set, return only the `num_best` most similar documents, always leaving out documents with similarity = 0.
            Otherwise, return a full vector with one float for every document in the index.
        chunksize : int, optional
            Size of block.
        shardsize : int, optional
            Size of shards should be chosen so that a `shardsize x chunksize` matrix of floats fits comfortably
            into memory.
        norm : {'l1', 'l2'}, optional
            Normalization to use.

        Notes
        ------------
        Documents are split (internally, transparently) into shards of `shardsize` documents each, converted to matrix,
        for faster BLAS calls. Each shard is stored to disk under `output_prefix.shard_number`.
        If you don't specify an output prefix, a random filename in temp will be used.
        If your entire index fits in memory (~hundreds of thousands
        documents for 1GB of RAM), you can also use the :class:`~gensim.similarities.docsim.MatrixSimilarity`
        or :class:`~gensim.similarities.docsim.SparseMatrixSimilarity` classes directly. These are more simple
        but do not scale as well (they keep the entire index in RAM, no sharding).

        """
        if output_prefix is None:
            # undocumented feature: set output_prefix=None to create the server in temp
            self.output_prefix = utils.randfname(prefix='simserver')
        else:
            self.output_prefix = output_prefix
        logger.info("starting similarity index under %s", self.output_prefix)
        self.num_features = num_features
        self.num_best = num_best
        self.norm = norm
        self.chunksize = int(chunksize)
        self.shardsize = shardsize
        self.shards = []
        self.fresh_docs, self.fresh_nnz = [], 0

        if corpus is not None:
            self.add_documents(corpus)

    def __len__(self):
        """Get length of index."""
        return len(self.fresh_docs) + sum([len(shard) for shard in self.shards])

    def __str__(self):
        return "Similarity index with %i documents in %i shards (stored under %s)" % (
            len(self), len(self.shards), self.output_prefix
        )

    def add_documents(self, corpus):
        """Extend the index with new documents.

        Parameters
        ----------
        corpus : iterable of list of (int, number)
            Corpus in BoW format.

        Notes
        -----
        Internally, documents are buffered and then spilled to disk when there's `self.shardsize` of them
        (or when a query is issued).

        Examples
        --------
        >>> from gensim.corpora.textcorpus import TextCorpus
        >>> from gensim.test.utils import datapath, get_tmpfile
        >>> from gensim.similarities import Similarity
        >>>
        >>> corpus = TextCorpus(datapath('testcorpus.mm'))
        >>> index_temp = get_tmpfile("index")
        >>> index = Similarity(index_temp, corpus, num_features=400)  # create index
        >>>
        >>> one_more_corpus = TextCorpus(datapath('testcorpus.txt'))
        >>> index.add_documents(one_more_corpus)  # add more documents in corpus

        """
        min_ratio = 1.0  # 0.5 to only reopen shards that are <50% complete
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
                    doc = matutils.unitvec(matutils.corpus2csc([doc], self.num_features).T, self.norm)
                else:
                    doc = matutils.unitvec(matutils.sparse2full(doc, self.num_features), self.norm)
            self.fresh_docs.append(doc)
            self.fresh_nnz += doclen
            if len(self.fresh_docs) >= self.shardsize:
                self.close_shard()
            if len(self.fresh_docs) % 10000 == 0:
                logger.info("PROGRESS: fresh_shard size=%i", len(self.fresh_docs))

    def shardid2filename(self, shardid):
        """Get shard file by `shardid`.

        Parameters
        ----------
        shardid : int
            Shard index.

        Return
        ------
        str
            Path to shard file.

        """
        if self.output_prefix.endswith('.'):
            return "%s%s" % (self.output_prefix, shardid)
        else:
            return "%s.%s" % (self.output_prefix, shardid)

    def close_shard(self):
        """Force the latest shard to close (be converted to a matrix and stored to disk).
         Do nothing if no new documents added since last call.

        Notes
        -----
        The shard is closed even if it is not full yet (its size is smaller than `self.shardsize`).
        If documents are added later via :meth:`~gensim.similarities.docsim.MatrixSimilarity.add_documents`
        this incomplete shard will be loaded again and completed.

        """
        if not self.fresh_docs:
            return
        shardid = len(self.shards)
        # consider the shard sparse if its density is < 30%
        issparse = 0.3 > 1.0 * self.fresh_nnz / (len(self.fresh_docs) * self.num_features)
        if issparse:
            index = SparseMatrixSimilarity(
                self.fresh_docs, num_terms=self.num_features, num_docs=len(self.fresh_docs), num_nnz=self.fresh_nnz
            )
        else:
            index = MatrixSimilarity(self.fresh_docs, num_features=self.num_features)
        logger.info("creating %s shard #%s", 'sparse' if issparse else 'dense', shardid)
        shard = Shard(self.shardid2filename(shardid), index)
        shard.num_best = self.num_best
        shard.num_nnz = self.fresh_nnz
        self.shards.append(shard)
        self.fresh_docs, self.fresh_nnz = [], 0

    def reopen_shard(self):
        """Reopen incomplete shard."""
        assert self.shards
        if self.fresh_docs:
            raise ValueError("cannot reopen a shard with fresh documents in index")
        last_shard = self.shards[-1]
        last_index = last_shard.get_index()
        logger.info("reopening an incomplete shard of %i documents", len(last_shard))

        self.fresh_docs = list(last_index.index)
        self.fresh_nnz = last_shard.num_nnz
        del self.shards[-1]  # remove the shard from index, *but its file on disk is not deleted*
        logger.debug("reopen complete")

    def query_shards(self, query):
        """Applying shard[query] for each shard in `self.shards`, as a sequence.

        Parameters
        ----------
        query : {iterable of list of (int, number) , list of (int, number))}
            Document in BoW format or corpus of documents.


        Returns
        -------
        (None, list of ...)
            Result of search.

        """
        args = zip([query] * len(self.shards), self.shards)
        if PARALLEL_SHARDS and PARALLEL_SHARDS > 1:
            logger.debug("spawning %i query processes", PARALLEL_SHARDS)
            pool = multiprocessing.Pool(PARALLEL_SHARDS)
            result = pool.imap(query_shard, args, chunksize=1 + len(args) / PARALLEL_SHARDS)
        else:
            # serial processing, one shard after another
            pool = None
            result = imap(query_shard, args)
        return pool, result

    def __getitem__(self, query):
        """Get similarities of document (or corpus) `query` to all documents in the corpus.

        Parameters
        ----------
        query : {iterable of list of (int, number) , list of (int, number))}
            Corpus or document of corpus.

        Return
        ------
        :class:`numpy.ndarray`
            Similarities of document/corpus if index is :class:`~gensim.similarities.docsim.MatrixSimilarity` **or**
        :class:`scipy.sparse.csr_matrix`
            for case if index is :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`.

        Notes
        -----
        If `query` is a corpus (iterable of documents), return a matrix of similarities of
        all query documents vs. all corpus document. This batch query is more efficient than computing the similarities
        one document after another.

        Examples
        --------
        >>> from gensim.corpora.textcorpus import TextCorpus
        >>> from gensim.test.utils import datapath
        >>> from gensim.similarities import Similarity
        >>> import gensim.downloader as api
        >>>
        >>> corpus = TextCorpus(datapath('testcorpus.txt'))
        >>> index = Similarity('temp', corpus, num_features=400)
        >>> result = index[corpus]  # similarities matrix

        """
        self.close_shard()  # no-op if no documents added to index since last query

        # reset num_best and normalize parameters, in case they were changed dynamically
        for shard in self.shards:
            shard.num_best = self.num_best
            shard.normalize = self.norm

        # there are 4 distinct code paths, depending on whether input `query` is
        # a corpus (or numpy/scipy matrix) or a single document, and whether the
        # similarity result should be a full array or only num_best most similar
        # documents.
        pool, shard_results = self.query_shards(query)
        if self.num_best is None:
            # user asked for all documents => just stack the sub-results into a single matrix
            # (works for both corpus / single doc query)
            result = numpy.hstack(shard_results)
        else:
            # the following uses a lot of lazy evaluation and (optionally) parallel
            # processing, to improve query latency and minimize memory footprint.
            offsets = numpy.cumsum([0] + [len(shard) for shard in self.shards])

            def convert(shard_no, doc):
                return [(doc_index + offsets[shard_no], sim) for doc_index, sim in doc]

            is_corpus, query = utils.is_corpus(query)
            is_corpus = is_corpus or hasattr(query, 'ndim') and query.ndim > 1 and query.shape[0] > 1
            if not is_corpus:
                # user asked for num_best most similar and query is a single doc
                results = (convert(shard_no, result) for shard_no, result in enumerate(shard_results))
                result = heapq.nlargest(self.num_best, itertools.chain(*results), key=lambda item: item[1])
            else:
                # the trickiest combination: returning num_best results when query was a corpus
                results = []
                for shard_no, result in enumerate(shard_results):
                    shard_result = [convert(shard_no, doc) for doc in result]
                    results.append(shard_result)
                result = []
                for parts in izip(*results):
                    merged = heapq.nlargest(self.num_best, itertools.chain(*parts), key=lambda item: item[1])
                    result.append(merged)
        if pool:
            # gc doesn't seem to collect the Pools, eventually leading to
            # "IOError 24: too many open files". so let's terminate it manually.
            pool.terminate()

        return result

    def vector_by_id(self, docpos):
        """Get indexed vector corresponding to the document at position `docpos`.

        Parameters
        ----------
        docpos : int
            Document position

        Return
        ------
        :class:`scipy.sparse.csr_matrix`
            Indexed vector, internal type depends on underlying index.

        Examples
        --------
        >>> from gensim.corpora.textcorpus import TextCorpus
        >>> from gensim.test.utils import datapath
        >>> from gensim.similarities import Similarity
        >>> import gensim.downloader as api
        >>>
        >>> # Create index:
        >>> corpus = TextCorpus(datapath('testcorpus.txt'))
        >>> index = Similarity('temp', corpus, num_features=400)
        >>> vector = index.vector_by_id(1)

        """
        self.close_shard()  # no-op if no documents added to index since last query
        pos = 0
        for shard in self.shards:
            pos += len(shard)
            if docpos < pos:
                break
        if not self.shards or docpos < 0 or docpos >= pos:
            raise ValueError("invalid document position: %s (must be 0 <= x < %s)" % (docpos, len(self)))
        result = shard.get_document_id(docpos - pos + len(shard))
        return result

    def similarity_by_id(self, docpos):
        """Get similarity of the given document only by `docpos`.

        Parameters
        ----------
        docpos : int
            Document position in index

        Return
        ------
        :class:`numpy.ndarray`
            Similarities of document/corpus if index is :class:`~gensim.similarities.docsim.MatrixSimilarity` **or**
        :class:`scipy.sparse.csr_matrix`
            for case if index is :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`.

        Examples
        --------
        >>> from gensim.corpora.textcorpus import TextCorpus
        >>> from gensim.test.utils import datapath
        >>> from gensim.similarities import Similarity
        >>>
        >>> corpus = TextCorpus(datapath('testcorpus.txt'))
        >>> index = Similarity('temp', corpus, num_features=400)
        >>> similarities = index.similarity_by_id(1)

        """
        query = self.vector_by_id(docpos)
        norm, self.norm = self.norm, False
        result = self[query]
        self.norm = norm
        return result

    def __iter__(self):
        """For each index document in index, compute cosine similarity against all other documents in the index.
        Using :meth:`~gensim.similarities.docsim.Similarity.iter_chunks`.

        Yields
        ------
        :class:`numpy.ndarray`
            Similarities of document if index is :class:`~gensim.similarities.docsim.MatrixSimilarity` **or**
        :class:`scipy.sparse.csr_matrix`
            for case if index is :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`.

        """
        # turn off query normalization (vectors in the index are already normalized, save some CPU)
        norm, self.norm = self.norm, False

        for chunk in self.iter_chunks():
            if chunk.shape[0] > 1:
                for sim in self[chunk]:
                    yield sim
            else:
                yield self[chunk]

        self.norm = norm  # restore normalization

    def iter_chunks(self, chunksize=None):
        """Iteratively yield the index as chunks of documents, each of size <= chunksize.

        Parameters
        ----------
        chunksize : int, optional
            Size of chunk,, if None - `self.chunksize` will be used.

        Notes
        -----
        The chunk is returned in its raw form.
        The size of the chunk may be smaller than requested; it is up to the caller to check the result for real length.

        Yields
        ------
        :class:`numpy.ndarray`
            Similarities of document if index is :class:`~gensim.similarities.docsim.MatrixSimilarity` **or**
        :class:`scipy.sparse.csr_matrix`
            for case if index is :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`.

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
                chunk = query[chunk_start: chunk_end]  # create a view
                yield chunk

    def check_moved(self):
        """Update shard locations (for case if the server directory has moved on filesystem)."""
        dirname = os.path.dirname(self.output_prefix)
        for shard in self.shards:
            shard.dirname = dirname

    def save(self, fname=None, *args, **kwargs):
        """Save the object via pickling (also see load) under filename specified in the constructor.

        Parameters
        ----------
        fname : str, optional
            Path for save index, if not provided - will be saved to `self.output_prefix`.
        *args : object
            Arguments, look at :meth:`gensim.interfaces.SimilarityABC.save`.
        **kwargs : object
            Keyword arguments, look at :meth:`gensim.interfaces.SimilarityABC.save`.

        Notes
        -----
        Call :meth:`~gensim.similarities.Similarity.close_shard` internally to spill unfinished shards to disk first.

        Examples
        --------
        >>> from gensim.corpora.textcorpus import TextCorpus
        >>> from gensim.test.utils import datapath, get_tmpfile
        >>> from gensim.similarities import Similarity
        >>>
        >>> temp_fname = get_tmpfile("index")
        >>> output_fname = get_tmpfile("saved_index")
        >>>
        >>> corpus = TextCorpus(datapath('testcorpus.txt'))
        >>> index = Similarity(temp_fname, corpus, num_features=400)
        >>>
        >>> index.save(output_fname)
        >>> loaded_index = index.load(output_fname)

        """
        self.close_shard()
        if fname is None:
            fname = self.output_prefix
        super(Similarity, self).save(fname, *args, **kwargs)

    def destroy(self):
        """Delete all files under self.output_prefix, object is not usable after calling this method anymore."""
        import glob
        for fname in glob.glob(self.output_prefix + '*'):
            logger.info("deleting %s", fname)
            os.remove(fname)


class MatrixSimilarity(interfaces.SimilarityABC):
    """Compute cosine similarity against a corpus of documents by storing the index matrix in memory.

    Unless the entire matrix fits into main memory, use :class:`~gensim.similarities.docsim.Similarity` instead.

    Examples
    --------
    >>> from gensim.test.utils import common_corpus, common_dictionary
    >>> from gensim.similarities import MatrixSimilarity
    >>>
    >>> query = [(1, 2), (5, 4)]
    >>> index = MatrixSimilarity(common_corpus, num_features=len(common_dictionary))
    >>> sims = index[query]

    """
    def __init__(self, corpus, num_best=None, dtype=numpy.float32, num_features=None, chunksize=256, corpus_len=None):
        """

        Parameters
        ----------
        corpus : iterable of list of (int, number)
            Corpus in BoW format.
        num_best : int, optional
            If set, return only the `num_best` most similar documents, always leaving out documents with similarity = 0.
            Otherwise, return a full vector with one float for every document in the index.
        dtype : numpy.dtype
            Datatype of internal matrix
        num_features : int, optional
            Size of the dictionary.
        chunksize : int, optional
            Size of chunk.
        corpus_len : int, optional
            Size of `corpus`, if not specified - will scan corpus to determine size.

        """
        if num_features is None:
            logger.warning(
                "scanning corpus to determine the number of features (consider setting `num_features` explicitly)"
            )
            num_features = 1 + utils.get_max_id(corpus)

        self.num_features = num_features
        self.num_best = num_best
        self.normalize = True
        self.chunksize = chunksize
        if corpus_len is None:
            corpus_len = len(corpus)

        if corpus is not None:
            if self.num_features <= 0:
                raise ValueError(
                    "cannot index a corpus with zero features (you must specify either `num_features` "
                    "or a non-empty corpus in the constructor)"
                )
            logger.info("creating matrix with %i documents and %i features", corpus_len, num_features)
            self.index = numpy.empty(shape=(corpus_len, num_features), dtype=dtype)
            # iterate over corpus, populating the numpy index matrix with (normalized)
            # document vectors
            for docno, vector in enumerate(corpus):
                if docno % 1000 == 0:
                    logger.debug("PROGRESS: at document #%i/%i", docno, corpus_len)
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
        """Get similarity between `query` and current index instance.

        Warnings
        --------
        Do not use this function directly, use the :class:`~gensim.similarities.docsim.MatrixSimilarity.__getitem__`
        instead.

        Parameters
        ----------
        query : {list of (int, number), iterable of list of (int, number), :class:`scipy.sparse.csr_matrix`
            Document or collection of documents.

        Return
        ------
        :class:`numpy.ndarray`
            Similarity matrix.

        """
        is_corpus, query = utils.is_corpus(query)
        if is_corpus:
            query = numpy.asarray(
                [matutils.sparse2full(vec, self.num_features) for vec in query],
                dtype=self.index.dtype
            )
        else:
            if scipy.sparse.issparse(query):
                query = query.toarray()  # convert sparse to dense
            elif isinstance(query, numpy.ndarray):
                pass
            else:
                # default case: query is a single vector in sparse gensim format
                query = matutils.sparse2full(query, self.num_features)
            query = numpy.asarray(query, dtype=self.index.dtype)

        # do a little transposition dance to stop numpy from making a copy of
        # self.index internally in numpy.dot (very slow).
        result = numpy.dot(self.index, query.T).T  # return #queries x #index
        return result  # XXX: removed casting the result from array to list; does anyone care?

    def __str__(self):
        return "%s<%i docs, %i features>" % (self.__class__.__name__, len(self), self.index.shape[1])


class SoftCosineSimilarity(interfaces.SimilarityABC):
    """Compute soft cosine similarity against a corpus of documents by storing the index matrix in memory.

    Examples
    --------
    >>> from gensim.test.utils import common_texts
    >>> from gensim.corpora import Dictionary
    >>> from gensim.models import Word2Vec
    >>> from gensim.similarities import SoftCosineSimilarity
    >>>
    >>> model = Word2Vec(common_texts, size=20, min_count=1)  # train word-vectors
    >>> dictionary = Dictionary(common_texts)
    >>> bow_corpus = [dictionary.doc2bow(document) for document in common_texts]
    >>>
    >>> similarity_matrix = model.wv.similarity_matrix(dictionary)  # construct similarity matrix
    >>> index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=10)
    >>>
    >>> # Make a query.
    >>> query = 'graph trees computer'.split()
    >>> # calculate similarity between query and each doc from bow_corpus
    >>> sims = index[dictionary.doc2bow(query)]

    Check out `Tutorial Notebook
    <https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/soft_cosine_tutorial.ipynb>`_
    for more examples.

    """
    def __init__(self, corpus, similarity_matrix, num_best=None, chunksize=256):
        """

        Parameters
        ----------
        corpus: iterable of list of (int, float)
            A list of documents in the BoW format.
        similarity_matrix : :class:`scipy.sparse.csc_matrix`
            A term similarity matrix, typically produced by
            :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similarity_matrix`.
        num_best : int, optional
            The number of results to retrieve for a query, if None - return similarities with all elements from corpus.
        chunksize: int, optional
            Size of one corpus chunk.

        See Also
        --------
        :meth:`gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similarity_matrix`
            A term similarity matrix produced from term embeddings.
        :func:`gensim.matutils.softcossim`
            The Soft Cosine Measure.

        """
        self.corpus = corpus
        self.similarity_matrix = similarity_matrix
        self.num_best = num_best
        self.chunksize = chunksize

        # Normalization of features is undesirable, since soft cosine similarity requires special
        # normalization using the similarity matrix. Therefore, we would just be normalizing twice,
        # increasing the numerical error.
        self.normalize = False

        # index is simply an array from 0 to size of corpus.
        self.index = numpy.arange(len(corpus))

    def __len__(self):
        return len(self.corpus)

    def get_similarities(self, query):
        """Get similarity between `query` and current index instance.

        Warnings
        --------
        Do not use this function directly; use the self[query] syntax instead.

        Parameters
        ----------
        query : {list of (int, number), iterable of list of (int, number), :class:`scipy.sparse.csr_matrix`
            Document or collection of documents.

        Return
        ------
        :class:`numpy.ndarray`
            Similarity matrix.

        """
        if isinstance(query, numpy.ndarray):
            # Convert document indexes to actual documents.
            query = [self.corpus[i] for i in query]

        if not query or not isinstance(query[0], list):
            query = [query]

        n_queries = len(query)
        result = []
        for qidx in range(n_queries):
            # Compute similarity for each query.
            qresult = [matutils.softcossim(document, query[qidx], self.similarity_matrix)
                       for document in self.corpus]
            qresult = numpy.array(qresult)

            # Append single query result to list of all results.
            result.append(qresult)

        if len(result) == 1:
            # Only one query.
            result = result[0]
        else:
            result = numpy.array(result)

        return result

    def __str__(self):
        return "%s<%i docs, %i features>" % (self.__class__.__name__, len(self), self.similarity_matrix.shape[0])


class WmdSimilarity(interfaces.SimilarityABC):
    """Compute negative WMD similarity against a corpus of documents by storing the index matrix in memory.


    See :class:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors` for more information.
    Also, tutorial `notebook
    <https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/WMD_tutorial.ipynb>`_ for more examples.

    When using this code, please consider citing the following papers:
    `Ofir Pele and Michael Werman, "A linear time histogram metric for improved SIFT matching"
    <http://www.cs.huji.ac.il/~werman/Papers/ECCV2008.pdf>`_, `Ofir Pele and Michael Werman, "Fast and robust earth
    mover's distances" <http://www.cs.huji.ac.il/~werman/Papers/ICCV2009.pdf>`_, `"Matt Kusner et al. "From Word
    Embeddings To Document Distances" <http://proceedings.mlr.press/v37/kusnerb15.pdf>`_.

    Example
    -------
    >>> from gensim.test.utils import common_texts
    >>> from gensim.corpora import Dictionary
    >>> from gensim.models import Word2Vec
    >>> from gensim.similarities import WmdSimilarity
    >>>
    >>> model = Word2Vec(common_texts, size=20, min_count=1)  # train word-vectors
    >>> dictionary = Dictionary(common_texts)
    >>> bow_corpus = [dictionary.doc2bow(document) for document in common_texts]
    >>>
    >>> index = WmdSimilarity(bow_corpus, model)
    >>> # Make query.
    >>> query = 'trees'
    >>> sims = index[query]

    """

    def __init__(self, corpus, w2v_model, num_best=None, normalize_w2v_and_replace=True, chunksize=256):
        """

        Parameters
        ----------
        corpus: iterable of list of (int, float)
            A list of documents in the BoW format.
        w2v_model: :class:`~gensim.models.word2vec.Word2VecTrainables`
            A trained word2vec model.
        num_best: int, optional
            Number of results to retrieve.
        normalize_w2v_and_replace: bool, optional
            Whether or not to normalize the word2vec vectors to length 1.
        chunksize : int, optional
            Size of chunk.

        """
        self.corpus = corpus
        self.w2v_model = w2v_model
        self.num_best = num_best
        self.chunksize = chunksize

        # Normalization of features is not possible, as corpus is a list (of lists) of strings.
        self.normalize = False

        # index is simply an array from 0 to size of corpus.
        self.index = numpy.arange(len(corpus))

        if normalize_w2v_and_replace:
            # Normalize vectors in word2vec class to length 1.
            w2v_model.init_sims(replace=True)

    def __len__(self):
        """Get size of corpus."""
        return len(self.corpus)

    def get_similarities(self, query):
        """Get similarity between `query` and current index instance.

        Warnings
        --------
        Do not use this function directly; use the self[query] syntax instead.

        Parameters
        ----------
        query : {list of (int, number), iterable of list of (int, number), :class:`scipy.sparse.csr_matrix`
            Document or collection of documents.

        Return
        ------
        :class:`numpy.ndarray`
            Similarity matrix.

        """
        if isinstance(query, numpy.ndarray):
            # Convert document indexes to actual documents.
            query = [self.corpus[i] for i in query]

        if not query or not isinstance(query[0], list):
            query = [query]

        n_queries = len(query)
        result = []
        for qidx in range(n_queries):
            # Compute similarity for each query.
            qresult = [self.w2v_model.wmdistance(document, query[qidx]) for document in self.corpus]
            qresult = numpy.array(qresult)
            qresult = 1. / (1. + qresult)  # Similarity is the negative of the distance.

            # Append single query result to list of all results.
            result.append(qresult)

        if len(result) == 1:
            # Only one query.
            result = result[0]
        else:
            result = numpy.array(result)

        return result

    def __str__(self):
        return "%s<%i docs, %i features>" % (self.__class__.__name__, len(self), self.w2v_model.wv.syn0.shape[1])


class SparseMatrixSimilarity(interfaces.SimilarityABC):
    """Compute cosine similarity against a corpus of documents by storing the index matrix in memory.

    Notes
    -----
    Use this if your input corpus contains sparse vectors (such as documents in bag-of-words format) and fits into RAM.

    The matrix is internally stored as a :class:`scipy.sparse.csr_matrix` matrix. Unless the entire
    matrix fits into main memory, use :class:`~gensim.similarities.docsim.Similarity` instead.

    Takes an optional `maintain_sparsity` argument, setting this to True
    causes `get_similarities` to return a sparse matrix instead of a
    dense representation if possible.

    See also
    --------
    :class:`~gensim.similarities.docsim.Similarity`
        Index similarity (wrapper for other inheritors of :class:`~gensim.interfaces.SimilarityABC`).
    :class:`~gensim.similarities.docsim.MatrixSimilarity`
        Index similarity (dense with cosine distance).

    """

    def __init__(self, corpus, num_features=None, num_terms=None, num_docs=None, num_nnz=None,
                 num_best=None, chunksize=500, dtype=numpy.float32, maintain_sparsity=False):
        """
        Parameters
        ----------
        corpus: iterable of list of (int, float)
            A list of documents in the BoW format.
        num_features : int, optional
            Size of the dictionary.
        num_terms : int, optional
            Number of terms, **must be specified**.
        num_docs : int, optional
            Number of documents in `corpus`.
        num_nnz : int, optional
            Number of non-zero terms.
        num_best : int, optional
            If set, return only the `num_best` most similar documents, always leaving out documents with similarity = 0.
            Otherwise, return a full vector with one float for every document in the index.
        chunksize : int, optional
            Size of chunk.
        dtype : numpy.dtype, optional
            Data type of internal matrix.
        maintain_sparsity : bool, optional
            if True - will return sparse arr from
            :meth:`~gensim.similarities.docsim.SparseMatrixSimilarity.get_similarities`.

        """

        self.num_best = num_best
        self.normalize = True
        self.chunksize = chunksize
        self.maintain_sparsity = maintain_sparsity

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
            if num_features is not None:
                # num_terms is just an alias for num_features, for compatibility with MatrixSimilarity
                num_terms = num_features
            if num_terms is None:
                raise ValueError("refusing to guess the number of sparse features: specify num_features explicitly")
            corpus = (matutils.scipy2sparse(v) if scipy.sparse.issparse(v) else
                      (matutils.full2sparse(v) if isinstance(v, numpy.ndarray) else
                       matutils.unitvec(v)) for v in corpus)
            self.index = matutils.corpus2csc(
                corpus, num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,
                dtype=dtype, printprogress=10000
            ).T

            # convert to Compressed Sparse Row for efficient row slicing and multiplications
            self.index = self.index.tocsr()  # currently no-op, CSC.T is already CSR
            logger.info("created %r", self.index)

    def __len__(self):
        """Get size of index."""
        return self.index.shape[0]

    def get_similarities(self, query):
        """Get similarity between `query` and current index instance.

        Warnings
        --------
        Do not use this function directly; use the self[query] syntax instead.

        Parameters
        ----------
        query : {list of (int, number), iterable of list of (int, number), :class:`scipy.sparse.csr_matrix`
            Document or collection of documents.

        Return
        ------
        :class:`numpy.ndarray`
            Similarity matrix (if maintain_sparsity=False) **OR**
        :class:`scipy.sparse.csc`
            otherwise

        """
        is_corpus, query = utils.is_corpus(query)
        if is_corpus:
            query = matutils.corpus2csc(query, self.index.shape[1], dtype=self.index.dtype)
        else:
            if scipy.sparse.issparse(query):
                query = query.T  # convert documents=rows to documents=columns
            elif isinstance(query, numpy.ndarray):
                if query.ndim == 1:
                    query.shape = (1, len(query))
                query = scipy.sparse.csr_matrix(query, dtype=self.index.dtype).T
            else:
                # default case: query is a single vector, in sparse gensim format
                query = matutils.corpus2csc([query], self.index.shape[1], dtype=self.index.dtype)

        # compute cosine similarity against every other document in the collection
        result = self.index * query.tocsc()  # N x T * T x C = N x C
        if result.shape[1] == 1 and not is_corpus:
            # for queries of one document, return a 1d array
            result = result.toarray().flatten()
        elif self.maintain_sparsity:
            # avoid converting to dense array if maintaining sparsity
            result = result.T
        else:
            # otherwise, return a 2d matrix (#queries x #index)
            result = result.toarray().T
        return result
