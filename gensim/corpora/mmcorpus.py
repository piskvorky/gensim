#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""Corpus in the Matrix Market format."""

import logging

from gensim import matutils
from gensim.corpora import IndexedCorpus


logger = logging.getLogger(__name__)


class MmCorpus(matutils.MmReader, IndexedCorpus):
    """Corpus in matrix market format.
    Wrap a term-document matrix on disk (in matrix-market format), and present it
    as an object which supports iteration over the rows (~documents).

    Attributes
    ----------
    num_docs : int
        Number of documents in market matrix file.
    num_terms : int
        Number of terms.
    num_nnz : int
        Number of non-zero terms.

    Notes
    ----------
    Note that the file is read into memory one document at a time, not the whole
    matrix at once (unlike :meth:`~scipy.io.mmread`). This allows us to process corpora
    which are larger than the available RAM.

    """

    def __init__(self, fname):
        """Read corpus in matrix market format.

        Parameters
        ----------
        fname : {str, file-like object}
            String (file path) or a file-like object that supports
            `seek()` (e.g. :class:`gzip.GzipFile`, :class:`bz2.BZ2File`).

        Notes
        -----
        File-like objects are not closed automatically.

        Example
        --------
        >>> from gensim.corpora.mmcorpus import MmCorpus
        >>> from gensim.test.utils import datapath
        >>> import gensim.downloader as api
        >>> from gensim.utils import simple_preprocess
        >>> corpus = MmCorpus(datapath('test_mmcorpus_with_index.mm'))
        >>> print corpus
        MmCorpus(9 documents, 12 features, 28 non-zero entries)

        """

        # avoid calling super(), too confusing
        IndexedCorpus.__init__(self, fname)
        matutils.MmReader.__init__(self, fname)

    def __iter__(self):
        """Iterate through vectors from underlying matrix.

        Yields
        ------
        list of (int, str)
            "Vector" of terms for next document in matrix. Vector of terms is represented as a
            list of (termid, val) tuples

        Notes
        ------
        Note that the total number of vectors returned is always equal to the
        number of rows specified in the header; empty documents are inserted and
        yielded where appropriate, even if they are not explicitly stored in the
        Matrix Market file.

        """
        for doc_id, doc in super(MmCorpus, self).__iter__():
            yield doc  # get rid of doc id, return the sparse vector only

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, progress_cnt=1000, metadata=False):
        """Save a corpus in the Matrix Market format to disk.

        Parameters
        ----------
        fname : str
            Path to file.
        corpus : iterable of list of (int, number)
            Corpus in Bow format.
        id2word : dict of (int, str), optional
            WordId -> Word.
        progress_cnt : int, optional
            Progress counter.
        metadata : bool, optional
            If true, writes out additional metadata.

        Notes
        -----
        This function is automatically called by `MmCorpus.serialize`; don't
        call it directly, call `serialize` instead.

        Example
        -------
        >>> from gensim.corpora.mmcorpus import MmCorpus
        >>> from gensim.test.utils import datapath
        >>> import gensim.downloader as api
        >>> from gensim.utils import simple_preprocess
        >>> corpus = MmCorpus(datapath('test_mmcorpus_with_index.mm'))
        >>> # Do not do it, use `serialize` instead.
        >>> MmCorpus.save_corpus("random", corpus)
        [97, 121, 169, 201, 225, 249, 258, 276, 303]

        """
        logger.info("storing corpus in Matrix Market format to %s", fname)
        num_terms = len(id2word) if id2word is not None else None
        return matutils.MmWriter.write_corpus(
            fname, corpus, num_terms=num_terms, index=True, progress_cnt=progress_cnt, metadata=metadata
        )
