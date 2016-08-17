#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Corpus in the Matrix Market format.
"""


import logging

from gensim.matutils import MmReader, MmWriter
from gensim.corpora import IndexedCorpus


logger = logging.getLogger('gensim.corpora.mmcorpus')


class MmCorpus(MmReader, IndexedCorpus):
    """
    Corpus in the Matrix Market format.
    """
    def __init__(self, fname, index_fname=None, transposed=True):
        # avoid calling super(), too confusing
        print('Loading the reader for the main docs file')
        MmReader.__init__(self, fname, transposed=transposed)
        print('Now trying to init the index for {} or {}'.format(fname, index_fname))
        IndexedCorpus.__init__(self, fname, index_fname=index_fname)

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
        num_terms = len(id2word) if id2word is not None else None
        logger.info("storing corpus with {} terms in dict, and {} docs in Matrix Market format to file named {}".format(
                    num_terms, len(corpus), fname))
        return MmWriter.write_corpus(fname, corpus, num_terms=num_terms, index=True, progress_cnt=progress_cnt, metadata=metadata)

# endclass MmCorpus
