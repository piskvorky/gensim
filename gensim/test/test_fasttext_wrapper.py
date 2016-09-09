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

from gensim.models.wrappers import fasttext

module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)

def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_fasttext.tst')

class TestFastText(unittest.TestCase):
    def setUp(self):
        ft_home = os.environ.get('FT_HOME', None)
        self.ft_path = os.path.join(ft_home, 'fasttext') if ft_home else None
        self.corpus_file = datapath('lee.cor')
        if self.ft_path:
            self.test_model = fasttext.FastText.train(self.ft_path, self.corpus_file)

    def testTraining(self):
        """Test model successfully trained"""
        if not self.ft_path:
            return
        model = fasttext.FastText.train(self.ft_path, self.corpus_file)
        self.assertEqual(model.kv.syn0.shape, (len(model.vocab), model.size))
        self.assertEqual(model.syn0_all.shape, (model.num_vectors, model.size))

    def testPersistence(self):
        """Test storing/loading the entire model."""
        if not self.ft_path:
            return
        model = fasttext.FastText.train(self.ft_path, self.corpus_file)
        model.save(testfile())
        loaded = fasttext.FastText.load(testfile())
        self.models_equal(model, loaded)

    def testLoadFastTextFormat(self):
        """Test model successfully loaded from fastText .vec and .bin files"""
        if not self.ft_path:
            return
        fasttext.FastText.train(self.ft_path, self.corpus_file, output_file=testfile())
        model = fasttext.FastText.load_fasttext_format(testfile())
        self.assertEqual(model.kv.syn0.shape, (len(model.vocab), model.size))
        self.assertEqual(model.syn0_all.shape, (model.num_vectors, model.size))

    def testSimilarity(self):
        """Test n_similarity for in-vocab and out-of-vocab words"""
        if not self.ft_path:
            return
        # In vocab, sanity check
        self.assertTrue(numpy.allclose(self.test_model.n_similarity(['the', 'and'], ['and', 'the']), 1.0))
        self.assertEqual(self.test_model.similarity('the', 'and'), self.test_model.similarity('the', 'and'))
        # Out of vocab check
        self.assertTrue(numpy.allclose(self.test_model.n_similarity(['night', 'nights'], ['nights', 'night']), 1.0))
        self.assertEqual(self.test_model.similarity('night', 'nights'), self.test_model.similarity('nights', 'night'))

    def testLookup(self):
        if not self.ft_path:
            return
        # In vocab, sanity check
        self.assertTrue(numpy.allclose(self.test_model['night'], self.test_model[['night']]))
        # Out of vocab check
        self.assertTrue(numpy.allclose(self.test_model['nights'], self.test_model[['nights']]))

    def models_equal(self, model, model2):
        self.assertEqual(len(model.vocab), len(model2.vocab))
        self.assertEqual(set(model.vocab.keys()), set(model2.vocab.keys()))
        self.assertTrue(numpy.allclose(model.kv.syn0, model2.kv.syn0))
        self.assertTrue(numpy.allclose(model.syn0_all, model2.syn0_all))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()