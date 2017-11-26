#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import logging
import unittest
import numbers

import six
import numpy as np
from numpy.testing import assert_allclose

from gensim.corpora import mmcorpus, Dictionary
from gensim.models import sldamodel
from gensim import matutils, utils
from gensim.test import basetmtests

dictionary = Dictionary(common_texts)
corpus = [dictionary.doc2bow(text) for text in common_texts]

def testRandomState():
    testcases = [np.random.seed(0), None, np.random.RandomState(0), 0]
    for testcase in testcases:
        assert(isinstance(utils.get_random_state(testcase), np.random.RandomState))


class TestSLdaModel(unittest.TestCase, basetmtests.TestBaseTopicModel):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = sldamodel.SLdaModel
        self.model = self.class_(corpus, id2word=dictionary, num_topics=2, passes=100)

        def testEta(self):
        kwargs = dict(
            id2word=dictionary,
            num_topics=2,
            eta=None
        )
        num_terms = len(dictionary)
        expected_shape = (num_terms,)

        # should not raise anything
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        assert_allclose(model.eta, np.array([0.5] * num_terms))

        kwargs['eta'] = 0.3
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        assert_allclose(model.eta, np.array([0.3] * num_terms))

        kwargs['eta'] = 3
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        assert_allclose(model.eta, np.array([3] * num_terms))

        kwargs['eta'] = [0.3] * num_terms
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        assert_allclose(model.eta, np.array([0.3] * num_terms))

        kwargs['eta'] = np.array([0.3] * num_terms)
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        assert_allclose(model.eta, np.array([0.3] * num_terms))

        # should be ok with num_topics x num_terms
        testeta = np.array([[0.5] * len(dictionary)] * 2)
        kwargs['eta'] = testeta
        self.class_(**kwargs)

        kwargs['eta'] = [0.3] * (num_terms + 1)
        self.assertRaises(AssertionError, self.class_, **kwargs)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
