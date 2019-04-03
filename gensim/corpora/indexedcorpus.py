#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""Base Indexed Corpus class."""

import logging
import six

import numpy

from gensim import interfaces, utils

logger = logging.getLogger(__name__)


class IndexedCorpus(interfaces.CorpusABC):
    """Indexed corpus is a mechanism for random-accessing corpora.

    While the standard corpus interface in gensim allows iterating over corpus,
    we'll show it with :class:`~gensim.corpora.mmcorpus.MmCorpus`.

    .. sourcecode:: pycon

        >>> from gensim.corpora import MmCorpus
        >>> from gensim.test.utils import datapath
        >>>
        >>> corpus = MmCorpus(datapath('testcorpus.mm'))
        >>> for doc in corpus:
        ...     pass

    :class:`~gensim.corpora.indexedcorpus.IndexedCorpus` allows accessing the documents with index
    in :math:`{O}(1)` look-up time.

    .. sourcecode:: pycon

        >>> document_index = 3
        >>> doc = corpus[document_index]

    Notes
    -----
    This functionality is achieved by storing an extra file (by default named the same as the `fname.index`)
    that stores the byte offset of the beginning of each document.

    """

    def __init__(self, fname, index_fname=None):
        """

        Parameters
        ----------
        fname : str
            Path to corpus.
        index_fname : str, optional
            Path to index, if not provided - used `fname.index`.

        """
        try:
            if index_fname is None:
                index_fname = utils.smart_extension(fname, '.index')
            self.index = utils.unpickle(index_fname)
            # change self.index into a numpy.ndarray to support fancy indexing
            self.index = numpy.asarray(self.index)
            logger.info("loaded corpus index from %s", index_fname)
        except Exception:
            self.index = None
        self.length = None

    @classmethod
    def serialize(serializer, fname, corpus, id2word=None, index_fname=None,
                  progress_cnt=None, labels=None, metadata=False):
        """Serialize corpus with offset metadata, allows to use direct indexes after loading.

        Parameters
        ----------
        fname : str
            Path to output file.
        corpus : iterable of iterable of (int, float)
            Corpus in BoW format.
        id2word : dict of (str, str), optional
            Mapping id -> word.
        index_fname : str, optional
             Where to save resulting index, if None - store index to `fname`.index.
        progress_cnt : int, optional
            Number of documents after which progress info is printed.
        labels : bool, optional
             If True - ignore first column (class labels).
        metadata : bool, optional
            If True - ensure that serialize will write out article titles to a pickle file.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.corpora import MmCorpus
            >>> from gensim.test.utils import get_tmpfile
            >>>
            >>> corpus = [[(1, 0.3), (2, 0.1)], [(1, 0.1)], [(2, 0.3)]]
            >>> output_fname = get_tmpfile("test.mm")
            >>>
            >>> MmCorpus.serialize(output_fname, corpus)
            >>> mm = MmCorpus(output_fname)  # `mm` document stream now has random access
            >>> print(mm[1])  # retrieve document no. 42, etc.
            [(1, 0.1)]

        """
        if getattr(corpus, 'fname', None) == fname:
            raise ValueError("identical input vs. output corpus filename, refusing to serialize: %s" % fname)

        if index_fname is None:
            index_fname = utils.smart_extension(fname, '.index')

        kwargs = {'metadata': metadata}
        if progress_cnt is not None:
            kwargs['progress_cnt'] = progress_cnt

        if labels is not None:
            kwargs['labels'] = labels

        offsets = serializer.save_corpus(fname, corpus, id2word, **kwargs)

        if offsets is None:
            raise NotImplementedError(
                "Called serialize on class %s which doesn't support indexing!" % serializer.__name__
            )

        # store offsets persistently, using pickle
        # we shouldn't have to worry about self.index being a numpy.ndarray as the serializer will return
        # the offsets that are actually stored on disk - we're not storing self.index in any case, the
        # load just needs to turn whatever is loaded from disk back into a ndarray - this should also ensure
        # backwards compatibility
        logger.info("saving %s index to %s", serializer.__name__, index_fname)
        utils.pickle(offsets, index_fname)

    def __len__(self):
        """Get the index length.

        Notes
        -----
        If the corpus is not indexed, also count corpus length and cache this value.

        Returns
        -------
        int
            Length of index.

        """
        if self.index is not None:
            return len(self.index)
        if self.length is None:
            logger.info("caching corpus length")
            self.length = sum(1 for _ in self)
        return self.length

    def __getitem__(self, docno):
        """Get document by `docno` index.

        Parameters
        ----------
        docno : {int, iterable of int}
            Document number or iterable of numbers (like a list of str).

        Returns
        -------
        list of (int, float)
            If `docno` is int - return document in BoW format.

        :class:`~gensim.utils.SlicedCorpus`
            If `docno` is iterable of int - return several documents in BoW format
            wrapped to :class:`~gensim.utils.SlicedCorpus`.

        Raises
        ------
        RuntimeError
            If index isn't exist.

        """
        if self.index is None:
            raise RuntimeError("Cannot call corpus[docid] without an index")
        if isinstance(docno, (slice, list, numpy.ndarray)):
            return utils.SlicedCorpus(self, docno)
        elif isinstance(docno, six.integer_types + (numpy.integer,)):
            return self.docbyoffset(self.index[docno])
            # TODO: no `docbyoffset` method, should be defined in this class
        else:
            raise ValueError('Unrecognised value for docno, use either a single integer, a slice or a numpy.ndarray')
