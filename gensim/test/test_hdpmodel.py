#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""


import logging
import unittest

from gensim.corpora import mmcorpus, Dictionary
from gensim.models import hdpmodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, common_texts

import numpy as np

dictionary = Dictionary(common_texts)
corpus = [dictionary.doc2bow(text) for text in common_texts]


class TestHdpModel(unittest.TestCase, basetmtests.TestBaseTopicModel):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = hdpmodel.HdpModel
        self.model = self.class_(corpus, id2word=dictionary, random_state=np.random.seed(0))

    def testTopicValues(self):
        """
        Check show topics method
        """
        results = self.model.show_topics()[0]
        expected_prob, expected_word = '0.264', 'trees '
        prob, word = results[1].split('+')[0].split('*')
        self.assertEqual(results[0], 0)
        self.assertEqual(prob, expected_prob)
        self.assertEqual(word, expected_word)

        return

    def testLDAmodel(self):
        """
        Create ldamodel object, and check if the corresponding alphas are equal.
        """
        ldam = self.model.suggested_lda_model()
        self.assertEqual(ldam.alpha[0], self.model.lda_alpha[0])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
