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


class MmCorpus(matutils.MmReader, IndexedCorpus):
    """
    Corpus in the Matrix Market format.
    """

    def __init__(self, fname):
        # avoid calling super(), too confusing
        IndexedCorpus.__init__(self, fname)
        matutils.MmReader.__init__(self, fname)

    def __iter__(self):
        """
        Interpret a matrix in Matrix Market format as a streamed gensim corpus
        (yielding one document at a time).
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
