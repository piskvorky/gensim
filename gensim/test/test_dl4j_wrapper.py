#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""


import logging
import unittest
import os
import tempfile
import numpy
from gensim.models.wrappers.dl4j import dl4jwrapper

module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)

def testfile():
    # temporary model will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_dl4j.tst')

class TestDl4j(unittest.TestCase):
    def setUp(self):
        if not os.path.isfile("dl4j_jar.jar"):
            raise unittest.SkipTest("dlfj libraries not found. Skipping dl4j wrapper tests")
        self.out_path = 'testmodel'
        self.test_model_file = datapath('raw_dl4j.txt')
        self.test_model = dl4jwrapper.dl4jWrapper.load_dl4j_w2v_format(self.test_model_file)

    def testTraining(self):
        """Test self.test_model successfully trained, parameters and weights correctly loaded"""
        vocab_size, model_size = 1750, 1
        trained_model = dl4jwrapper.dl4jWrapper.train("dl4j_jar.jar", minWordFrequency=5, iterations=1, layerSize=100, seed=42, windowSize=5, output_file="raw_dl4j.txt")

        self.assertEqual(trained_model.syn0.shape, (vocab_size, model_size))
        self.assertEqual(len(trained_model.vocab), vocab_size)

    def testLoadDl4jFormat(self):
        """Test model successfully loaded from Wordrank format file"""
        vocab_size, dim = 1750, 100
        self.assertEqual(self.test_model.syn0.shape, (vocab_size, dim))
        self.assertEqual(len(self.test_model.vocab), vocab_size)
        os.remove(self.test_model_file+'.w2vformat')

    def testPersistence(self):
        """Test storing/loading the entire model"""
        self.test_model.save(testfile())
        loaded = dl4jwrapper.dl4jWrapper.load(testfile())
        self.models_equal(self.test_model, loaded)

    def testSimilarity(self):
        """Test n_similarity for vocab words"""
        self.assertTrue(numpy.allclose(self.test_model.n_similarity(['the', 'and'], ['and', 'the']), 1.0))
        self.assertEqual(self.test_model.similarity('the', 'and'), self.test_model.similarity('the', 'and'))

    def testLookup(self):
        self.assertTrue(numpy.allclose(self.test_model['night'], self.test_model[['night']]))

    def models_equal(self, model, model2):
        self.assertEqual(len(model.vocab), len(model2.vocab))
        self.assertEqual(set(model.vocab.keys()), set(model2.vocab.keys()))
        self.assertTrue(numpy.allclose(model.syn0, model2.syn0))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
