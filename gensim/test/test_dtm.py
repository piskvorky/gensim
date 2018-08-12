#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated tests for DTM/DIM model
"""


import logging
from subprocess import CalledProcessError
import gensim
import os
import unittest
from gensim import corpora
from gensim.test.utils import datapath


class TestDtmModel(unittest.TestCase):

    def setUp(self):
        self.time_slices = [3, 7]
        self.corpus = corpora.mmcorpus.MmCorpus(datapath('dtm_test.mm'))
        self.id2word = corpora.Dictionary.load(datapath('dtm_test.dict'))
        # first you need to setup the environment variable $DTM_PATH for the dtm executable file
        self.dtm_path = os.environ.get('DTM_PATH', None)
        if not self.dtm_path:
            self.skipTest("$DTM_PATH is not properly set up.")

    def testDtm(self):
        if self.dtm_path is not None:
            model = gensim.models.wrappers.DtmModel(
                self.dtm_path, self.corpus, self.time_slices, num_topics=2,
                id2word=self.id2word, model='dtm', initialize_lda=True,
                rng_seed=1
            )
            topics = model.show_topics(num_topics=2, times=2, num_words=10)
            self.assertEqual(len(topics), 4)

            one_topic = model.show_topic(topicid=1, time=1, num_words=10)
            self.assertEqual(len(one_topic), 10)
            self.assertEqual(one_topic[0][1], u'idexx')

    def testDim(self):
        if self.dtm_path is not None:
            model = gensim.models.wrappers.DtmModel(
                self.dtm_path, self.corpus, self.time_slices, num_topics=2,
                id2word=self.id2word, model='fixed', initialize_lda=True,
                rng_seed=1
            )
            topics = model.show_topics(num_topics=2, times=2, num_words=10)
            self.assertEqual(len(topics), 4)

            one_topic = model.show_topic(topicid=1, time=1, num_words=10)
            self.assertEqual(len(one_topic), 10)
            self.assertEqual(one_topic[0][1], u'skills')

    # In stderr expect "Error opening file /tmp/a65419_train_out/initial-lda-ss.dat. Failing."
    def testCalledProcessError(self):
        if self.dtm_path is not None:
            with self.assertRaises(CalledProcessError):
                gensim.models.wrappers.DtmModel(
                    self.dtm_path, self.corpus, self.time_slices, num_topics=2,
                    id2word=self.id2word, model='dtm', initialize_lda=False,
                    rng_seed=1
                )


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
