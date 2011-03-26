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


logger = logging.getLogger('gensim.corpora.mmcorpus')
logger.setLevel(logging.INFO)


class MmCorpus(matutils.MmReader, interfaces.CorpusABC):
    """
    Corpus in the Matrix Market format.
    """
    def __iter__(self):
        """
        Interpret a matrix in Matrix Market format as a streaming corpus.
        
        This simply wraps the I/O reader of MM format, to comply with the corpus 
        interface.
        """
        for docId, doc in super(MmCorpus, self).__iter__():
            yield doc # get rid of docId, return the sparse vector only
    
    @staticmethod
    def saveCorpus(fname, corpus, id2word=None, progressCnt=1000):
        """
        Save a corpus in the Matrix Market format to disk.
        """
        logger.info("storing corpus in Matrix Market format to %s" % fname)
        matutils.MmWriter.writeCorpus(fname, corpus, progressCnt=progressCnt)
#endclass MmCorpus


