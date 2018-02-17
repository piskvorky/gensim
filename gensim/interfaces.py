#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains implementations of basic interfaces used across the
whole gensim package. These interfaces usable for building corpus,
transformation and similarity classes.

All interfaces are realized as abstract base classes (i.e. some optional
functionality is provided in the interface itself, so that the interfaces should
be inherited).

"""

from __future__ import with_statement

import logging

from gensim import utils, matutils
from six.moves import xrange


logger = logging.getLogger('gensim.interfaces')


class CorpusABC(utils.SaveLoad):
    """This class implements interface (abstract base class) for corpus
    classes. A corpus is simply an iterable object, where each iteration step
    yields one document:

    >>> for doc in corpus:
    >>>     # do something with the doc...

    A document is a sequence of `(attr_id, attr_value)` 2-tuples:

    >>> for attr_id, attr_value in doc:
    >>>     # do something with the attribute

    See :mod:`gensim.corpora.svmlightcorpus` module for an example of a corpus
    class.

    Saving the corpus with the `save` method (inherited from
    :mod:`gensim.utils.SaveLoad`) will only store current in-memory
    representation of object (its stream state) but not documents themselves.
    Use :meth:`gensim.interfacese.save_corpus()` static method for serializing
    actual stream content.

    Note
    ----
    Although default :func:`len` method is provided, it is very inefficient
    because it based on linear scan through the corpus to determine its length.
    Wherever the corpus size is needed and known in advance (or at least doesn't
    change so that it can be cached), the :func:`len` method should be
    overridden.

    """

    def __iter__(self):
        """Iterator protocol for the corpus.

        Raises
        ------
        NotImplementedError
            Since it's abstract class this iterator protocol should be
            overwritten in the inherited class.

        """
        raise NotImplementedError('cannot instantiate abstract base class')

    def save(self, *args, **kwargs):
        """Saves corpus in-memory state (but not documents).

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        """
        import warnings
        warnings.warn(
            "corpus.save() stores only the (tiny) iteration object; "
            "to serialize the actual corpus content, use e.g. MmCorpus.serialize(corpus)"
        )
        super(CorpusABC, self).save(*args, **kwargs)

    def __len__(self):
        """Returns size of the corpus (number of documents).

        Result obtained by iterating over whole corpus thus it's ineffective.
        This method is just the least common denominator and should be
        overridden when possible.

        Raises
        ------
        NotImplementedError
            Since it's abstract class this method should be reimplemented later.

        """
        raise NotImplementedError("must override __len__() before calling len(corpus)")
#        logger.warning("performing full corpus scan to determine its length; was this intended?")
#        return sum(1 for doc in self) # sum(empty generator) == 0, so this works even for an empty corpus

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        """Saves given `corpus` to disk.

        Some formats support saving the dictionary (`feature_id -> word`
        mapping), which can be provided by the optional `id2word` parameter.

        >>> MmCorpus.save_corpus('file.mm', corpus)

        Some corpus also support an index of where each document begins, so
        that the documents on disk can be accessed in O(1) time (see the
        :mod:`gensim.corpora.IndexedCorpus` base class). In this case,
        :func:`save_corpus` is automatically called internally by
        :func:`serialize`, which does :func:`save_corpus` plus saves the index
        at the same time, so you may want to store the corpus with:

        >>> MmCorpus.serialize('file.mm', corpus) # stores index as well, allowing random access to individual documents

        Calling :func:`serialize()` is preferred to calling
        :func:`save_corpus()`.

        Parameters
        ----------
        fname : str
            Path to corpus output file.
        corpus : :class:`~gensim.interfaces.CorpusABC`
            Corpus to be saved.
        id2word : {dict of (int, str), :class:`~gensim.corpora.Dictionary`}, optional
            Dictionary of corpus.
        metadata : bool, optional
            If True :func:`serialize` will write out article titles to a pickle
            file.

        """
        raise NotImplementedError('cannot instantiate abstract base class')

        # example code:
        logger.info("converting corpus to ??? format: %s", fname)
        with utils.smart_open(fname, 'wb') as fout:
            for doc in corpus:  # iterate over the document stream
                fmt = str(doc)  # format the document appropriately...
                fout.write(utils.to_utf8("%s\n" % fmt))  # serialize the formatted document to disk


class TransformedCorpus(CorpusABC):
    """Interface (abstract class) for corpus supports transformations."""
    def __init__(self, obj, corpus, chunksize=None, **kwargs):
        """Initialization of corpus. Documents storage, corpus should be
        provided.

        Parameters
        ----------
        obj : object
            Object where documents are stored.
        corpus : :class:`~gensim.interfaces.CorpusABC`
            Given corpus.
        chunksize : int, optional
            If provided more effective processing (by group of documents) will
            performed.
        kwargs
            Arbitrary keyword arguments.

        """
        self.obj, self.corpus, self.chunksize = obj, corpus, chunksize
        # add the new parameters like per_word_topics to base class object of LdaModel
        for key, value in kwargs.items():
            setattr(self.obj, key, value)
        self.metadata = False

    def __len__(self):
        """Returns size of the corpus."""
        return len(self.corpus)

    def __iter__(self):
        """Iterator protocol for corpus. If `chunksize` is set more effective
        processing will performed. Yields document.

        Yields
        ------
        iterable of (int, int)
            Current document.

        """
        if self.chunksize:
            for chunk in utils.grouper(self.corpus, self.chunksize):
                for transformed in self.obj.__getitem__(chunk, chunksize=None):
                    yield transformed
        else:
            for doc in self.corpus:
                yield self.obj[doc]

    def __getitem__(self, docno):
        """Provides access to corpus elements by index `docno`.

        Parameters
        ----------
        docno : int
            Index of document in corpus.

        Returns
        -------
        iterable of (int, int)
            Selected document of corpus.

        Raises
        ------
        RuntimeError
            If corpus doesn't support slicing.

        """
        if hasattr(self.corpus, '__getitem__'):
            return self.obj[self.corpus[docno]]
        else:
            raise RuntimeError('Type {} does not support slicing.'.format(type(self.corpus)))


class TransformationABC(utils.SaveLoad):
    """Interface for transformations. A 'transformation' is any object which
    accepts a sparse document via the dictionary notation `[]` and returns
    another sparse document in its stead:

    >>> transformed_doc = transformation[doc]

    or also:

    >>> transformed_corpus = transformation[corpus]

    See the :mod:`gensim.models.tfidfmodel` module for an example of a
    transformation.

    """

    def __getitem__(self, vec):
        """Provide access to element of `transformations`.

        Transforms vector from one vector space into another

        **or**

        Transforms a whole corpus into another.

        Parameters
        ----------
        vec : iterable
            Given vector.

        Raises
        ------
        NotImplementedError
            Since it's abstract class this method should be reimplemented later.

        """
        raise NotImplementedError('cannot instantiate abstract base class')

    def _apply(self, corpus, chunksize=None, **kwargs):
        """Applies the transformation to a whole corpus (as opposed to a
        single document) and returns the result as another corpus.

        Parameters
        ----------
        corpus : :class:`~gensim.interfaces.CorpusABC`
            given corpus.
        chunksize : int, optional
            If provided more effective processing (by group of documents) will
            performed.
        kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        :class:`~gensim.interfaces.TransformedCorpus`
            Transformed corpus.

        """
        return TransformedCorpus(self, corpus, chunksize, **kwargs)


class SimilarityABC(utils.SaveLoad):
    """Abstract class for similarity search over a corpus.

    In all instances, there is a corpus against which we want to perform the
    similarity search.

    For each similarity search, the input is a document and the output are its
    similarities to individual corpus documents.

    Similarity queries are realized by calling ``self[query_document]``.

    There is also a convenience wrapper, where iterating over `self` yields
    similarities of each document in the corpus against the whole corpus (i.e.
    the query is each corpus document in turn).

    """

    def __init__(self, corpus):
        """Initialization of object. Since it is an abstract class this
        initialization need to be overwritten in the inherited class.

        Parameters
        ----------
        corpus : :class:`~gensim.interfaces.CorpusABC`
            Given corpus.

        Raises
        ------
        NotImplementedError
            Since it's abstract class this method should be reimplemented later.

        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def get_similarities(self, doc):
        """Returns similarity measures of documents of corpus to given `doc`.

        Parameters
        ----------
        doc : iterable of (int, int)
            Given document.

        Raises
        ------
        NotImplementedError
            Since it's abstract class this method should be reimplemented later.

        """
        # (Sparse)MatrixSimilarity override this method so that they both use the
        # same  __getitem__ method, defined below
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def __getitem__(self, query):
        """Provides access to similarities of document `query` to all documents
        in the corpus.

        **or**

        If `query` is a corpus (iterable of documents), returns a matrix of
        similarities of all query documents vs. all corpus document. Using this
        type of batch query is more efficient than computing the similarities
        one document after another.

        Parameters
        ----------
        query : {iterable of (int, int), :class:`~gensim.interfaces.CorpusABC`}
            Given document or corpus.

        Returns
        -------
        {`scipy.sparse.csr.csr_matrix`, list of (int, float)}
            Simiarities of given document or corpus and objects corpus.

        """
        is_corpus, query = utils.is_corpus(query)
        if self.normalize:
            # self.normalize only works if the input is a plain gensim vector/corpus (as
            # advertised in the doc). in fact, input can be a numpy or scipy.sparse matrix
            # as well, but in that case assume tricks are happening and don't normalize
            # anything (self.normalize has no effect).
            if matutils.ismatrix(query):
                import warnings  # noqa:F401
                # warnings.warn("non-gensim input must already come normalized")
            else:
                if is_corpus:
                    query = [matutils.unitvec(v) for v in query]
                else:
                    query = matutils.unitvec(query)
        result = self.get_similarities(query)

        if self.num_best is None:
            return result

        # if maintain_sparsity is True, result is scipy sparse. Sort, clip the
        # topn and return as a scipy sparse matrix.
        if getattr(self, 'maintain_sparsity', False):
            return matutils.scipy2scipy_clipped(result, self.num_best)

        # if the input query was a corpus (=more documents), compute the top-n
        # most similar for each document in turn
        if matutils.ismatrix(result):
            return [matutils.full2sparse_clipped(v, self.num_best) for v in result]
        else:
            # otherwise, return top-n of the single input document
            return matutils.full2sparse_clipped(result, self.num_best)

    def __iter__(self):
        """Implements iterator protocol. For each index document, computes
        cosine similarity against all other documents in the index and yields
        result.

        Yields
        ------
        list of (int, float)
            Cosine similarity of current document and all documents of corpus.

        """
        # turn off query normalization (vectors in the index are assumed to be already normalized)
        norm = self.normalize
        self.normalize = False

        # Try to compute similarities in bigger chunks of documents (not
        # one query = a single document after another). The point is, a
        # bigger query of N documents is faster than N small queries of one
        # document.
        #
        # After computing similarities of the bigger query in `self[chunk]`,
        # yield the resulting similarities one after another, so that it looks
        # exactly the same as if they had been computed with many small queries.
        try:
            chunking = self.chunksize > 1
        except AttributeError:
            # chunking not supported; fall back to the (slower) mode of 1 query=1 document
            chunking = False
        if chunking:
            # assumes `self.corpus` holds the index as a 2-d numpy array.
            # this is true for MatrixSimilarity and SparseMatrixSimilarity, but
            # may not be true for other (future) classes..?
            for chunk_start in xrange(0, self.index.shape[0], self.chunksize):
                # scipy.sparse doesn't allow slicing beyond real size of the matrix
                # (unlike numpy). so, clip the end of the chunk explicitly to make
                # scipy.sparse happy
                chunk_end = min(self.index.shape[0], chunk_start + self.chunksize)
                chunk = self.index[chunk_start: chunk_end]
                for sim in self[chunk]:
                    yield sim
        else:
            for doc in self.index:
                yield self[doc]

        # restore old normalization value
        self.normalize = norm
