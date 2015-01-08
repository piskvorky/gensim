#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated tests for DTM/DIM model
"""


import logging
import gensim
import os
import unittest
from gensim import corpora
from gensim import utils


# needed because sample data files are located in the same folder
module_path = os.path.dirname(__file__)
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_dtm.tst')


class DTMcorpus(corpora.textcorpus.TextCorpus):
    def get_texts(self):
        return self.input

    def __len__(self):
        return len(self.input)

class TestDtmModel(unittest.TestCase):
    def setUp(self):
        self.corpus, self.time_seq = utils.unpickle(datapath('dtm_test'))
        # first you need to setup the environment variable for the dtm binary
        dtm_home = os.environ.get('DTM_HOME', "C:/Users/Artyom/SkyDrive/TopicModels/dtm-master/")
        self.dtm_path = os.path.join(dtm_home, 'bin', 'dtm') if dtm_home else None

    def testDtm(self):
        model = gensim.models.DtmModel(self.dtm_path, self.corpus, self.time_seq, num_topics=2, id2word=self.corpus.dictionary, model='dtm', initialize_lda=True)
        topics = model.show_topics(topics=2, times=2, topn=10)

    def testDim(self):
        model = gensim.models.DtmModel(self.dtm_path, self.corpus, self.time_seq, num_topics=2, id2word=self.corpus.dictionary, model='fixed', initialize_lda=True)
        topics = model.show_topics(topics=2, times=2, topn=10)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
