#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Corpus in the `Matrix Market format <https://math.nist.gov/MatrixMarket/formats.html>`_."""

import logging

from gensim import matutils
from gensim.corpora import IndexedCorpus


logger = logging.getLogger(__name__)


class MmCorpus(matutils.MmReader, IndexedCorpus):
    """Corpus serialized using the `sparse coordinate Matrix Market format
    <https://math.nist.gov/MatrixMarket/formats.html>`_.

    Wrap a term-document matrix on disk (in matrix-market format), and present it
    as an object which supports iteration over the matrix rows (~documents).

    Notable instance attributes:

    Attributes
    ------------------
    num_docs : int
        Number of documents in the market matrix file.
    num_terms : int
        Number of features (terms, topics).
    num_nnz : int
        Number of non-zero elements in the sparse MM matrix.

    Notes
    -----
    The file is read into memory one document at a time, not the whole matrix at once,
    unlike e.g. `scipy.io.mmread` and other implementations. This allows you to **process corpora which are larger
    than the available RAM**, in a streamed manner.

    Example
    --------
    .. sourcecode:: pycon

        >>> from gensim.corpora.mmcorpus import MmCorpus
        >>> from gensim.test.utils import datapath
        >>>
        >>> corpus = MmCorpus(datapath('test_mmcorpus_with_index.mm'))
        >>> for document in corpus:
        ...     pass

    """
    def __init__(self, fname):
        """

        Parameters
        ----------
        fname : {str, file-like object}
            Path to file in MM format or a file-like object that supports `seek()`
            (e.g. a compressed file opened by `smart_open <https://github.com/RaRe-Technologies/smart_open>`_).

        """
        # avoid calling super(), too confusing
        IndexedCorpus.__init__(self, fname)
        matutils.MmReader.__init__(self, fname)

    def __iter__(self):
        """Iterate through all documents.

        Yields
        ------
        list of (int, numeric)
            Document in the `sparse Gensim bag-of-words format <intro.rst#core-concepts>`__.

        Notes
        ------
        The total number of vectors returned is always equal to the number of rows specified in the header.
        Empty documents are inserted and yielded where appropriate, even if they are not explicitly stored in the
        (sparse) Matrix Market file.

        """
        for doc_id, doc in super(MmCorpus, self).__iter__():
            yield doc  # get rid of doc id, return the sparse vector only

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, progress_cnt=1000, metadata=False):
        """Save a corpus to disk in the sparse coordinate Matrix Market format.

        Parameters
        ----------
        fname : str
            Path to file.
        corpus : iterable of list of (int, number)
            Corpus in Bow format.
        id2word : dict of (int, str), optional
            Mapping between word_id -> word. Used to retrieve the total vocabulary size if provided.
            Otherwise, the total vocabulary size is estimated based on the highest feature id encountered in `corpus`.
        progress_cnt : int, optional
            How often to report (log) progress.
        metadata : bool, optional
            Writes out additional metadata?

        Warnings
        --------
        This function is automatically called by :class:`~gensim.corpora.mmcorpus.MmCorpus.serialize`, don't
        call it directly, call :class:`~gensim.corpora.mmcorpus.MmCorpus.serialize` instead.

        Example
        -------
        .. sourcecode:: pycon

            >>> from gensim.corpora.mmcorpus import MmCorpus
            >>> from gensim.test.utils import datapath
            >>>
            >>> corpus = MmCorpus(datapath('test_mmcorpus_with_index.mm'))
            >>>
            >>> MmCorpus.save_corpus("random", corpus)  # Do not do it, use `serialize` instead.
            [97, 121, 169, 201, 225, 249, 258, 276, 303]

        """
        logger.info("storing corpus in Matrix Market format to %s", fname)
        num_terms = len(id2word) if id2word is not None else None
        return matutils.MmWriter.write_corpus(
            fname, corpus, num_terms=num_terms, index=True, progress_cnt=progress_cnt, metadata=metadata
        )
