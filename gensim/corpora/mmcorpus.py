#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Corpus in the Matrix Market format.
"""


import logging

from gensim import matutils
from gensim.corpora import IndexedCorpus


logger = logging.getLogger('gensim.corpora.mmcorpus')


try:
    # try to load fast, cythonized code if possible
    from gensim.corpora._mmreader import MmReader
    FAST_VERSION = 1
    logger.info('Fast version of MmReader is being used')
except ImportError:
    # else fall back to python/numpy
    from gensim.corpora.mmreader import MmReader
    FAST_VERSION = -1
    logger.warning('Slow version of MmReader is being used')


class MmCorpus(MmReader, IndexedCorpus):
    """
    Corpus in matrix market format

    Wrap a term-document matrix on disk (in matrix-market format), and present it
    as an object which supports iteration over the rows (~documents).

    Attributes
    ----------
    num_docs : int
        number of documents in market matrix file
    num_terms : int
        number of terms
    num_nnz : int
        number of non-zero terms

    Notes
    ----------
    Note that the file is read into memory one document at a time, not the whole
    matrix at once (unlike scipy.io.mmread). This allows us to process corpora
    which are larger than the available RAM.

    """

    def __init__(self, fname):
        """
        Read corpus in matrix market format

        Parameters
        ----------
        fname : string or file-like
            string (file path) or a file-like object that supports
            `seek()` (e.g. gzip.GzipFile, bz2.BZ2File). File-like objects are
            not closed automatically.

        """

        # avoid calling super(), too confusing
        IndexedCorpus.__init__(self, fname)
        MmReader.__init__(self, fname)

    def __iter__(self):
        """
        Iterate through vectors from underlying matrix

        Yields
        ------
        list of (termid, val)
            "vector" of terms for next document in matrix
            vector of terms is represented as a list of (termid, val) tuples

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
        """
        Save a corpus in the Matrix Market format to disk.

        This function is automatically called by `MmCorpus.serialize`; don't
        call it directly, call `serialize` instead.
        """
        logger.info("storing corpus in Matrix Market format to %s", fname)
        num_terms = len(id2word) if id2word is not None else None
        return matutils.MmWriter.write_corpus(
            fname, corpus, num_terms=num_terms, index=True, progress_cnt=progress_cnt, metadata=metadata
        )
