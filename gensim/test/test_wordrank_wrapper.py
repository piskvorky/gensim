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

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


def testfile():
    # temporary model will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_wordrank.test')


class TestWordrank(unittest.TestCase):
    def setUp(self):
        wr_home = os.environ.get('WR_HOME', None)
        self.wr_path = wr_home if wr_home else None
        self.corpus_file = datapath('lee.cor')
        self.out_name = 'testmodel'
        self.wr_file = datapath('test_glove.txt')
        if not self.wr_path:
            return
        self.test_model = wordrank.Wordrank.train(
            self.wr_path, self.corpus_file, self.out_name, iter=6,
            dump_period=5, period=5, np=2, cleanup_files=True
        )

    def testLoadWordrankFormat(self):
        """Test model successfully loaded from Wordrank format file"""
        model = wordrank.Wordrank.load_wordrank_model(self.wr_file)
        vocab_size, dim = 76, 50
        self.assertEqual(model.syn0.shape, (vocab_size, dim))
        self.assertEqual(len(model.vocab), vocab_size)
        os.remove(self.wr_file + '.w2vformat')

    def testEnsemble(self):
        """Test ensemble of two embeddings"""
        if not self.wr_path:
            return
        new_emb = self.test_model.ensemble_embedding(self.wr_file, self.wr_file)
        self.assertEqual(new_emb.shape, (76, 50))
        os.remove(self.wr_file + '.w2vformat')

    def testPersistence(self):
        """Test storing/loading the entire model"""
        if not self.wr_path:
            return
        self.test_model.save(testfile())
        loaded = wordrank.Wordrank.load(testfile())
        self.models_equal(self.test_model, loaded)

    def testSimilarity(self):
        """Test n_similarity for vocab words"""
        if not self.wr_path:
            return
        self.assertTrue(numpy.allclose(self.test_model.n_similarity(['the', 'and'], ['and', 'the']), 1.0))
        self.assertEqual(self.test_model.similarity('the', 'and'), self.test_model.similarity('the', 'and'))

    def testLookup(self):
        if not self.wr_path:
            return
        self.assertTrue(numpy.allclose(self.test_model['night'], self.test_model[['night']]))

    def models_equal(self, model, model2):
        self.assertEqual(len(model.vocab), len(model2.vocab))
        self.assertEqual(set(model.vocab.keys()), set(model2.vocab.keys()))
        self.assertTrue(numpy.allclose(model.syn0, model2.syn0))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
