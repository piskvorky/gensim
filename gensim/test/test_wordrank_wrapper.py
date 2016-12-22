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

from gensim.models.wrappers import wordrank

module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)

def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_wordrank.test')

class TestWordrank(unittest.TestCase):
    def setUp(self):
        wr_home = os.environ.get('WR_HOME', None)
        self.wr_path = wr_home if wr_home else None
        self.corpus_file = datapath('lee.cor')
        if self.wr_path:
            self.test_model = wordrank.Wordrank.train(self.wr_path, self.corpus_file)

    def testTraining(self):
        """Test model successfully trained"""
        if not self.wr_path:
            return
        model = wordrank.Wordrank.train(self.wr_path, self.corpus_file, size=10)
        self.assertEqual(model.wv.syn0.shape, (len(model.wv.vocab), 10))

    def testPersistence(self):
        """Test storing/loading the entire model."""
        if not self.wr_path:
            return
        model = wordrank.Wordrank.train(self.wr_path, self.corpus_file)
        model.save(testfile())
        loaded = wordrank.Wordrank.load(testfile())
        self.models_equal(model, loaded)

    def testLoadWordrank(self):
        """Test model successfully loaded from wordrank .test files"""
        if not self.wr_path:
            return
        model = wordrank.Wordrank.load_wordrank_model(testfile())
        self.assertTrue(model.wv.syn0.shape == (len(model.wv.vocab), 10))

    def testSimilarity(self):
        """Test n_similarity for vocab words"""
        if not self.wr_path:
            return
        # In vocab, sanity check
        self.assertTrue(numpy.allclose(self.test_model.n_similarity(['the', 'and'], ['and', 'the']), 1.0))
        self.assertEqual(self.test_model.similarity('the', 'and'), self.test_model.similarity('the', 'and'))

    def testLookup(self):
        if not self.wr_path:
            return
        # In vocab, sanity check
        self.assertTrue(numpy.allclose(self.test_model['night'], self.test_model[['night']]))

    def models_equal(self, model, model2):
        self.assertEqual(len(model.vocab), len(model2.vocab))
        self.assertEqual(set(model.vocab.keys()), set(model2.vocab.keys()))
        self.assertTrue(numpy.allclose(model.wv.syn0, model2.wv.syn0))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()