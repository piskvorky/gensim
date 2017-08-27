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

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)
logger = logging.getLogger(__name__)


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_fasttext.tst')


class TestFastText(unittest.TestCase):
    def setUp(self):
        ft_home = os.environ.get('FT_HOME', None)
        self.ft_path = os.path.join(ft_home, 'fasttext') if ft_home else None
        self.corpus_file = datapath('lee_background.cor')
        self.test_model_file = datapath('lee_fasttext')
        self.test_new_model_file = datapath('lee_fasttext_new')
        # Load pre-trained model to perform tests in case FastText binary isn't available in test environment
        self.test_model = fasttext.FastText.load(self.test_model_file)

    def model_sanity(self, model):
        """Even tiny models trained on any corpus should pass these sanity checks"""
        self.assertEqual(model.wv.syn0.shape, (len(model.wv.vocab), model.vector_size))
        self.assertEqual(model.wv.syn0_all.shape, (model.num_ngram_vectors, model.vector_size))

    def models_equal(self, model1, model2):
        self.assertEqual(len(model1.wv.vocab), len(model2.wv.vocab))
        self.assertEqual(set(model1.wv.vocab.keys()), set(model2.wv.vocab.keys()))
        self.assertTrue(numpy.allclose(model1.wv.syn0, model2.wv.syn0))
        self.assertTrue(numpy.allclose(model1.wv.syn0_all, model2.wv.syn0_all))

    def testTraining(self):
        """Test self.test_model successfully trained, parameters and weights correctly loaded"""
        if self.ft_path is None:
            logger.info("FT_HOME env variable not set, skipping test")
            return  # Use self.skipTest once python < 2.7 is no longer supported
        vocab_size, model_size = 1763, 10
        trained_model = fasttext.FastText.train(
            self.ft_path, self.corpus_file, size=model_size, output_file=testfile())

        self.assertEqual(trained_model.wv.syn0.shape, (vocab_size, model_size))
        self.assertEqual(len(trained_model.wv.vocab), vocab_size)
        self.assertEqual(trained_model.wv.syn0_all.shape[1], model_size)
        self.model_sanity(trained_model)

        # Tests temporary training files deleted
        self.assertFalse(os.path.exists('%s.bin' % testfile()))

    def testMinCount(self):
        """Tests words with frequency less than `min_count` absent from vocab"""
        if self.ft_path is None:
            logger.info("FT_HOME env variable not set, skipping test")
            return  # Use self.skipTest once python < 2.7 is no longer supported
        test_model_min_count_5 = fasttext.FastText.train(
            self.ft_path, self.corpus_file, output_file=testfile(), size=10, min_count=5)
        self.assertTrue('forests' not in test_model_min_count_5.wv.vocab)

        test_model_min_count_1 = fasttext.FastText.train(
            self.ft_path, self.corpus_file, output_file=testfile(), size=10, min_count=1)
        self.assertTrue('forests' in test_model_min_count_1.wv.vocab)

    def testModelSize(self):
        """Tests output vector dimensions are the same as the value for `size` param"""
        if self.ft_path is None:
            logger.info("FT_HOME env variable not set, skipping test")
            return  # Use self.skipTest once python < 2.7 is no longer supported
        test_model_size_20 = fasttext.FastText.train(
            self.ft_path, self.corpus_file, output_file=testfile(), size=20)
        self.assertEqual(test_model_size_20.vector_size, 20)
        self.assertEqual(test_model_size_20.wv.syn0.shape[1], 20)
        self.assertEqual(test_model_size_20.wv.syn0_all.shape[1], 20)

    def testPersistence(self):
        """Test storing/loading the entire model."""
        self.test_model.save(testfile())
        loaded = fasttext.FastText.load(testfile())
        self.models_equal(self.test_model, loaded)

        self.test_model.save(testfile(), sep_limit=0)
        self.models_equal(self.test_model, fasttext.FastText.load(testfile()))

    def testNormalizedVectorsNotSaved(self):
        """Test syn0norm/syn0_all_norm aren't saved in model file"""
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
        """Test model successfully loaded from fastText .bin file"""
        try:
            model = fasttext.FastText.load_fasttext_format(self.test_model_file)
        except Exception as exc:
            self.fail('Unable to load FastText model from file %s: %s' % (self.test_model_file, exc))
        vocab_size, model_size = 1762, 10
        self.assertEqual(model.wv.syn0.shape, (vocab_size, model_size))
        self.assertEqual(len(model.wv.vocab), vocab_size, model_size)
        self.assertEqual(model.wv.syn0_all.shape, (model.num_ngram_vectors, model_size))

        expected_vec = [
            -0.57144,
            -0.0085561,
            0.15748,
            -0.67855,
            -0.25459,
            -0.58077,
            -0.09913,
            1.1447,
            0.23418,
            0.060007
        ]  # obtained using ./fasttext print-word-vectors lee_fasttext_new.bin
        self.assertTrue(numpy.allclose(model["hundred"], expected_vec, atol=1e-4))

        # vector for oov words are slightly different from original FastText due to discarding unused ngrams
        # obtained using a modified version of ./fasttext print-word-vectors lee_fasttext_new.bin
        expected_vec_oov = [
            -0.23825,
            -0.58482,
            -0.22276,
            -0.41215,
            0.91015,
            -1.6786,
            -0.26724,
            0.58818,
            0.57828,
            0.75801
        ]
        self.assertTrue(numpy.allclose(model["rejection"], expected_vec_oov, atol=1e-4))

        self.assertEquals(model.min_count, 5)
        self.assertEquals(model.window, 5)
        self.assertEquals(model.iter, 5)
        self.assertEquals(model.negative, 5)
        self.assertEquals(model.sample, 0.0001)
        self.assertEquals(model.bucket, 1000)
        self.assertEquals(model.wv.max_n, 6)
        self.assertEquals(model.wv.min_n, 3)
        self.model_sanity(model)

    def testLoadFastTextNewFormat(self):
        """ Test model successfully loaded from fastText (new format) .bin file """
        try:
            new_model = fasttext.FastText.load_fasttext_format(self.test_new_model_file)
        except Exception as exc:
            self.fail('Unable to load FastText model from file %s: %s' % (self.test_new_model_file, exc))
        vocab_size, model_size = 1763, 10
        self.assertEqual(new_model.wv.syn0.shape, (vocab_size, model_size))
        self.assertEqual(len(new_model.wv.vocab), vocab_size, model_size)
        self.assertEqual(new_model.wv.syn0_all.shape, (new_model.num_ngram_vectors, model_size))

        expected_vec = [
            -0.025627,
            -0.11448,
            0.18116,
            -0.96779,
            0.2532,
            -0.93224,
            0.3929,
            0.12679,
            -0.19685,
            -0.13179
        ]  # obtained using ./fasttext print-word-vectors lee_fasttext_new.bin
        self.assertTrue(numpy.allclose(new_model["hundred"], expected_vec, atol=1e-4))

        # vector for oov words are slightly different from original FastText due to discarding unused ngrams
        # obtained using a modified version of ./fasttext print-word-vectors lee_fasttext_new.bin
        expected_vec_oov = [
            -0.53378,
            -0.19,
            0.013482,
            -0.86767,
            -0.21684,
            -0.89928,
            0.45124,
            0.18025,
            -0.14128,
            0.22508
        ]
        self.assertTrue(numpy.allclose(new_model["rejection"], expected_vec_oov, atol=1e-4))

        self.assertEquals(new_model.min_count, 5)
        self.assertEquals(new_model.window, 5)
        self.assertEquals(new_model.iter, 5)
        self.assertEquals(new_model.negative, 5)
        self.assertEquals(new_model.sample, 0.0001)
        self.assertEquals(new_model.bucket, 1000)
        self.assertEquals(new_model.wv.max_n, 6)
        self.assertEquals(new_model.wv.min_n, 3)
        self.model_sanity(new_model)

    def testLoadFileName(self):
        """ Test model accepts input as both `/path/to/model` or `/path/to/model.bin` """
        self.assertTrue(fasttext.FastText.load_fasttext_format(datapath('lee_fasttext_new')))
        self.assertTrue(fasttext.FastText.load_fasttext_format(datapath('lee_fasttext_new.bin')))

    def testLoadModelWithNonAsciiVocab(self):
        """Test loading model with non-ascii words in vocab"""
        model = fasttext.FastText.load_fasttext_format(datapath('non_ascii_fasttext'))
        self.assertTrue(u'který' in model)
        try:
            vector = model[u'který']  # noqa:F841
        except UnicodeDecodeError:
            self.fail('Unable to access vector for utf8 encoded non-ascii word')

    def testLoadModelNonUtf8Encoding(self):
        """Test loading model with words in user-specified encoding"""
        model = fasttext.FastText.load_fasttext_format(datapath('cp852_fasttext'), encoding='cp852')
        self.assertTrue(u'který' in model)
        try:
            vector = model[u'který']  # noqa:F841
        except KeyError:
            self.fail('Unable to access vector for cp-852 word')

    def testNSimilarity(self):
        """Test n_similarity for in-vocab and out-of-vocab words"""
        # In vocab, sanity check
        self.assertTrue(numpy.allclose(self.test_model.n_similarity(['the', 'and'], ['and', 'the']), 1.0))
        self.assertEqual(self.test_model.n_similarity(['the'], ['and']), self.test_model.n_similarity(['and'], ['the']))
        # Out of vocab check
        self.assertTrue(numpy.allclose(self.test_model.n_similarity(['night', 'nights'], ['nights', 'night']), 1.0))
        self.assertEqual(self.test_model.n_similarity(['night'], ['nights']),
                         self.test_model.n_similarity(['nights'], ['night']))

    def testSimilarity(self):
        """Test similarity for in-vocab and out-of-vocab words"""
        # In vocab, sanity check
        self.assertTrue(numpy.allclose(self.test_model.similarity('the', 'the'), 1.0))
        self.assertEqual(self.test_model.similarity('the', 'and'), self.test_model.similarity('and', 'the'))
        # Out of vocab check
        self.assertTrue(numpy.allclose(self.test_model.similarity('nights', 'nights'), 1.0))
        self.assertEqual(self.test_model.similarity('night', 'nights'), self.test_model.similarity('nights', 'night'))

    def testMostSimilar(self):
        """Test most_similar for in-vocab and out-of-vocab words"""
        # In vocab, sanity check
        self.assertEqual(len(self.test_model.most_similar(positive=['the', 'and'], topn=5)), 5)
        self.assertEqual(self.test_model.most_similar('the'), self.test_model.most_similar(positive=['the']))
        # Out of vocab check
        self.assertEqual(len(self.test_model.most_similar(['night', 'nights'], topn=5)), 5)
        self.assertEqual(self.test_model.most_similar('nights'), self.test_model.most_similar(positive=['nights']))

    def testMostSimilarCosmul(self):
        """Test most_similar_cosmul for in-vocab and out-of-vocab words"""
        # In vocab, sanity check
        self.assertEqual(len(self.test_model.most_similar_cosmul(positive=['the', 'and'], topn=5)), 5)
        self.assertEqual(
            self.test_model.most_similar_cosmul('the'),
            self.test_model.most_similar_cosmul(positive=['the']))
        # Out of vocab check
        self.assertEqual(len(self.test_model.most_similar_cosmul(['night', 'nights'], topn=5)), 5)
        self.assertEqual(
            self.test_model.most_similar_cosmul('nights'),
            self.test_model.most_similar_cosmul(positive=['nights']))

    def testLookup(self):
        """Tests word vector lookup for in-vocab and out-of-vocab words"""
        # In vocab, sanity check
        self.assertTrue('night' in self.test_model.wv.vocab)
        self.assertTrue(numpy.allclose(self.test_model['night'], self.test_model[['night']]))
        # Out of vocab check
        self.assertFalse('nights' in self.test_model.wv.vocab)
        self.assertTrue(numpy.allclose(self.test_model['nights'], self.test_model[['nights']]))
        # Word with no ngrams in model
        self.assertRaises(KeyError, lambda: self.test_model['a!@'])

    def testContains(self):
        """Tests __contains__ for in-vocab and out-of-vocab words"""
        # In vocab, sanity check
        self.assertTrue('night' in self.test_model.wv.vocab)
        self.assertTrue('night' in self.test_model)
        # Out of vocab check
        self.assertFalse('nights' in self.test_model.wv.vocab)
        self.assertTrue('nights' in self.test_model)
        # Word with no ngrams in model
        self.assertFalse('a!@' in self.test_model.wv.vocab)
        self.assertFalse('a!@' in self.test_model)

    def testWmdistance(self):
        """Tests wmdistance for docs with in-vocab and out-of-vocab words"""
        doc = ['night', 'payment']
        oov_doc = ['nights', 'forests', 'payments']
        ngrams_absent_doc = ['a!@', 'b#$']

        dist = self.test_model.wmdistance(doc, oov_doc)
        self.assertNotEqual(float('inf'), dist)
        dist = self.test_model.wmdistance(doc, ngrams_absent_doc)
        self.assertEqual(float('inf'), dist)

    def testDoesntMatch(self):
        """Tests doesnt_match for list of out-of-vocab words"""
        oov_words = ['nights', 'forests', 'payments']
        # Out of vocab check
        for word in oov_words:
            self.assertFalse(word in self.test_model.wv.vocab)
        try:
            self.test_model.doesnt_match(oov_words)
        except Exception:
            self.fail('model.doesnt_match raises exception for oov words')

    def testHash(self):
        # Tests FastText.ft_hash method return values to those obtained from original C implementation
        ft_hash = fasttext.FastText.ft_hash('test')
        self.assertEqual(ft_hash, 2949673445)
        ft_hash = fasttext.FastText.ft_hash('word')
        self.assertEqual(ft_hash, 1788406269)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
