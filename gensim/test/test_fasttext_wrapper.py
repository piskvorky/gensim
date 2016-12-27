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
from gensim.models import keyedvectors

module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)

def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_fasttext.tst')

class TestFastText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ft_home = os.environ.get('FT_HOME', None)
        cls.ft_path = os.path.join(ft_home, 'fasttext') if ft_home else None
        cls.corpus_file = datapath('lee_background.cor')
        cls.test_model_file = os.path.join(tempfile.gettempdir(), 'ft_model')
        if cls.ft_path:
            cls.test_model = fasttext.FastText.train(
                cls.ft_path, cls.corpus_file, output_file=cls.test_model_file, size=10)
        else:
            cls.skipTest(cls, "FT_HOME env variable not set, skipping test")

    def model_sanity(self, model):
        """Even tiny models trained on LeeCorpus should pass these sanity checks"""
        self.assertEqual(model.wv.syn0.shape, (len(model.vocab), model.size))
        self.assertEqual(model.wv.syn0_all.shape, (model.num_ngram_vectors, model.size))
        sims = model.most_similar('war', topn=len(model.index2word))

    def models_equal(self, model1, model2):
        self.assertEqual(len(model1.vocab), len(model2.vocab))
        self.assertEqual(set(model1.vocab.keys()), set(model2.vocab.keys()))
        self.assertTrue(numpy.allclose(model1.wv.syn0, model2.wv.syn0))
        self.assertTrue(numpy.allclose(model1.wv.syn0_all, model2.wv.syn0_all))

    def testTraining(self):
        """Test self.test_model successfully trained"""
        vocab_size, model_size = 1762, 10
        self.assertEqual(self.test_model.wv.syn0.shape, (vocab_size, model_size))
        self.assertEqual(len(self.test_model.wv.vocab), vocab_size)
        self.assertEqual(self.test_model.wv.syn0_all.shape[1], model_size)
        self.model_sanity(self.test_model)

    def testMinCount(self):
        self.assertTrue('forests' not in self.test_model)
        test_model_min_count_1 = fasttext.FastText.train(
                self.ft_path, self.corpus_file, output_file=self.test_model_file, size=10, min_count=1)
        self.assertTrue('forests' in test_model_min_count_1)

    def testModelSize(self):
        test_model_size_20 = fasttext.FastText.train(
                self.ft_path, self.corpus_file, output_file=self.test_model_file, size=20)
        self.assertEqual(test_model_size_20.size, 20)
        self.assertEqual(test_model_size_20.syn0.shape[1], 20)
        self.assertEqual(test_model_size_20.wv.syn0_all.shape[1], 20)

    def testPersistence(self):
        """Test storing/loading the entire model."""
        self.test_model.save(testfile())
        loaded = fasttext.FastText.load(testfile())
        self.models_equal(self.test_model, loaded)

        self.test_model.save(testfile(), sep_limit=0)
        self.models_equal(self.test_model, fasttext.FastText.load(testfile()))

    def testNormalizedVectorsNotSaved(self):
        """Test syn0norm isn't saved in model file"""
        self.test_model.init_sims()
        self.test_model.save(testfile())
        loaded = fasttext.FastText.load(testfile())
        self.assertTrue(loaded.wv.syn0norm is None)
        self.assertTrue(loaded.wv.syn0_all_norm is None)

        wv = self.test_model.wv
        wv.save(testfile())
        loaded_kv = keyedvectors.KeyedVectors.load(testfile())
        self.assertTrue(loaded_kv.syn0norm is None)
        self.assertTrue(loaded_kv.syn0_all_norm is None)

    def testLoadFastTextFormat(self):
        """Test model successfully loaded from fastText .vec and .bin files"""
        model = fasttext.FastText.load_fasttext_format(self.test_model_file)
        vocab_size, model_size = 1762, 10
        self.assertEqual(self.test_model.wv.syn0.shape, (vocab_size, model_size))
        self.assertEqual(len(self.test_model.wv.vocab), vocab_size, model_size)
        self.assertEqual(self.test_model.wv.syn0_all.shape, (self.test_model.num_ngram_vectors, model_size))
        self.model_sanity(model)

    def testNSimilarity(self):
        """Test n_similarity for in-vocab and out-of-vocab words"""
        # In vocab, sanity check
        self.assertTrue(numpy.allclose(self.test_model.n_similarity(['the', 'and'], ['and', 'the']), 1.0))
        self.assertEqual(self.test_model.n_similarity(['the'], ['and']), self.test_model.n_similarity(['and'], ['the']))
        # Out of vocab check
        self.assertTrue(numpy.allclose(self.test_model.n_similarity(['night', 'nights'], ['nights', 'night']), 1.0))
        self.assertEqual(self.test_model.n_similarity(['night'], ['nights']), self.test_model.n_similarity(['nights'], ['night']))

    def testSimilarity(self):
        """Test n_similarity for in-vocab and out-of-vocab words"""
        # In vocab, sanity check
        self.assertTrue(numpy.allclose(self.test_model.similarity('the', 'the'), 1.0))
        self.assertEqual(self.test_model.similarity('the', 'and'), self.test_model.similarity('and', 'the'))
        # Out of vocab check
        self.assertTrue(numpy.allclose(self.test_model.similarity('nights', 'nights'), 1.0))
        self.assertEqual(self.test_model.similarity('night', 'nights'), self.test_model.similarity('nights', 'night'))

    def testMostSimilar(self):
        """Test n_similarity for in-vocab and out-of-vocab words"""
        # In vocab, sanity check
        self.assertEqual(len(self.test_model.most_similar(positive=['the', 'and'], topn=5)), 5)
        self.assertEqual(self.test_model.most_similar('the'), self.test_model.most_similar(positive=['the']))
        # Out of vocab check
        self.assertEqual(len(self.test_model.most_similar(['night', 'nights'], topn=5)), 5)
        self.assertEqual(self.test_model.most_similar('nights'), self.test_model.most_similar(positive=['nights']))

    def testMostSimilarCosmul(self):
        """Test n_similarity for in-vocab and out-of-vocab words"""
        # In vocab, sanity check
        self.assertEqual(len(self.test_model.most_similar(positive=['the', 'and'], topn=5)), 5)
        self.assertEqual(self.test_model.most_similar('the'), self.test_model.most_similar(positive=['the']))
        # Out of vocab check
        self.assertEqual(len(self.test_model.most_similar(['night', 'nights'], topn=5)), 5)
        self.assertEqual(self.test_model.most_similar('nights'), self.test_model.most_similar(positive=['nights']))

    def testLookup(self):
        # In vocab, sanity check
        self.assertTrue('night' in self.test_model)
        self.assertTrue(numpy.allclose(self.test_model['night'], self.test_model[['night']]))
        # Out of vocab check
        self.assertFalse('nights' in self.test_model)
        self.assertTrue(numpy.allclose(self.test_model['nights'], self.test_model[['nights']]))

    def testHash(self):
        # Tests FastText.ft_hash method return values to those obtained from original C implementation
        ft_hash = fasttext.FastText.ft_hash('test')
        self.assertEqual(ft_hash, 2949673445)
        ft_hash = fasttext.FastText.ft_hash('word')
        self.assertEqual(ft_hash, 1788406269)

    def testWordVectorEqualsFastTextCLIOutput(self):
        pass

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()