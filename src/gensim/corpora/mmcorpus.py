#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Corpus in the Matrix Market format.
"""


import logging

from gensim import interfaces, matutils
from gensim.corpora import IndexedCorpus



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
        for docId, doc in super(MmCorpus, self).__iter__():
            yield doc # get rid of docId, return the sparse vector only

    @staticmethod
    def saveCorpus(fname, corpus, id2word=None, progressCnt=1000):
        """
        Save a corpus in the Matrix Market format to disk.
        
        This function is automatically called by `MmCorpus.serialize`; don't
        call it directly, call `serialize` instead.
        """
        logging.info("storing corpus in Matrix Market format to %s" % fname)
        return matutils.MmWriter.writeCorpus(fname, corpus, index=True, progressCnt=progressCnt)
#endclass MmCorpus


