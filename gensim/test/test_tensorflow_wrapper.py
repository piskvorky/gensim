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
import os
import tempfile

import numpy

from gensim.models.tfword2vec import TfWord2Vec

module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)

def testfile():
    # temporary model will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_tensorflow.test')

class TestTensorFlow(unittest.TestCase):
    def setUp(self):
        self.corpus_file = datapath('lee.cor')
        self.out_path = 'testmodel'
        self.tf_file = datapath('test_glove.txt')
        vocab_path = '.'
        self.test_model = TfWord2Vec(self.corpus_file, epochs_to_train=1, embedding_size=100, batch_size=100000, save_path=vocab_path)

    #TODO fix this after saving works
    def LoadTensorFlowFormat(self):
        """Test model successfully loaded from Wordrank format file"""
        model = TfWord2Vec.load_tf_model(self.tf_file)
        vocab_size, dim = 76, 50
        self.assertEqual(model.syn0.shape, (vocab_size, dim))
        self.assertEqual(len(model.vocab), vocab_size)
        os.remove(self.tf_file+'.w2vformat')

    #TODO fix this after saving works
    def Persistence(self):
        """Test storing/loading the entire model"""
        if not self.wr_path:
            return
        self.test_model.save(testfile())
        loaded = TfWord2Vec.load_tf_model(testfile())
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


