#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayantjain1992@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking the poincare module from the models package.
"""

import logging
import unittest

import numpy as np

from gensim.corpora import Dictionary
from gensim.models import KeyedVectors as EuclideanKeyedVectors
from gensim.test.utils import datapath


logger = logging.getLogger(__name__)


class TestEuclideanKeyedVectors(unittest.TestCase):
    def setUp(self):
        self.vectors = EuclideanKeyedVectors.load_word2vec_format(
            datapath('euclidean_vectors.bin'), binary=True, datatype=np.float64)

    def similarity_matrix(self):
        """Test similarity_matrix returns expected results."""

        corpus = [["government", "denied", "holiday"], ["holiday", "slowing", "hollingworth"]]
        dictionary = Dictionary(corpus)
        corpus = [dictionary.doc2bow(document) for document in corpus]

        # checking symmetry and the existence of ones on the diagonal
        similarity_matrix = self.similarity_matrix(corpus, dictionary).todense()
        self.assertTrue((similarity_matrix.T == similarity_matrix).all())
        self.assertTrue((np.diag(similarity_matrix) == similarity_matrix).all())

        # checking that thresholding works as expected
        similarity_matrix = self.similarity_matrix(corpus, dictionary, threshold=0.45).todense()
        self.assertEquals(18, np.sum(similarity_matrix == 0))

        # checking that exponent works as expected
        similarity_matrix = self.similarity_matrix(corpus, dictionary, exponent=1.0).todense()
        self.assertAlmostEqual(9.5788956, np.sum(similarity_matrix))

        # checking that nonzero_limit works as expected
        similarity_matrix = self.similarity_matrix(corpus, dictionary, nonzero_limit=4).todense()
        self.assertEquals(4, np.sum(similarity_matrix == 0))

        similarity_matrix = self.similarity_matrix(corpus, dictionary, nonzero_limit=3).todense()
        self.assertEquals(20, np.sum(similarity_matrix == 0))

    def test_most_similar(self):
        """Test most_similar returns expected results."""
        expected = [
            'conflict',
            'administration',
            'terrorism',
            'call',
            'israel'
        ]
        predicted = [result[0] for result in self.vectors.most_similar('war', topn=5)]
        self.assertEqual(expected, predicted)

    def test_most_similar_topn(self):
        """Test most_similar returns correct results when `topn` is specified."""
        self.assertEqual(len(self.vectors.most_similar('war', topn=5)), 5)
        self.assertEqual(len(self.vectors.most_similar('war', topn=10)), 10)

        predicted = self.vectors.most_similar('war', topn=None)
        self.assertEqual(len(predicted), len(self.vectors.vocab))

    def test_most_similar_raises_keyerror(self):
        """Test most_similar raises KeyError when input is out of vocab."""
        with self.assertRaises(KeyError):
            self.vectors.most_similar('not_in_vocab')

    def test_most_similar_restrict_vocab(self):
        """Test most_similar returns handles restrict_vocab correctly."""
        expected = set(self.vectors.index2word[:5])
        predicted = set(result[0] for result in self.vectors.most_similar('war', topn=5, restrict_vocab=5))
        self.assertEqual(expected, predicted)

    def test_most_similar_with_vector_input(self):
        """Test most_similar returns expected results with an input vector instead of an input word."""
        expected = [
            'war',
            'conflict',
            'administration',
            'terrorism',
            'call',
        ]
        input_vector = self.vectors['war']
        predicted = [result[0] for result in self.vectors.most_similar([input_vector], topn=5)]
        self.assertEqual(expected, predicted)

    def test_most_similar_to_given(self):
        """Test most_similar_to_given returns correct results."""
        predicted = self.vectors.most_similar_to_given('war', ['terrorism', 'call', 'waging'])
        self.assertEqual(predicted, 'terrorism')

    def test_similar_by_word(self):
        """Test similar_by_word returns expected results."""
        expected = [
            'conflict',
            'administration',
            'terrorism',
            'call',
            'israel'
        ]
        predicted = [result[0] for result in self.vectors.similar_by_word('war', topn=5)]
        self.assertEqual(expected, predicted)

    def test_similar_by_vector(self):
        """Test similar_by_word returns expected results."""
        expected = [
            'war',
            'conflict',
            'administration',
            'terrorism',
            'call',
        ]
        input_vector = self.vectors['war']
        predicted = [result[0] for result in self.vectors.similar_by_vector(input_vector, topn=5)]
        self.assertEqual(expected, predicted)

    def test_distance(self):
        """Test that distance returns expected values."""
        self.assertTrue(np.allclose(self.vectors.distance('war', 'conflict'), 0.06694602))
        self.assertEqual(self.vectors.distance('war', 'war'), 0)

    def test_similarity(self):
        """Test similarity returns expected value for two words, and for identical words."""
        self.assertTrue(np.allclose(self.vectors.similarity('war', 'war'), 1))
        self.assertTrue(np.allclose(self.vectors.similarity('war', 'conflict'), 0.93305397))

    def test_words_closer_than(self):
        """Test words_closer_than returns expected value for distinct and identical nodes."""
        self.assertEqual(self.vectors.words_closer_than('war', 'war'), [])
        expected = set(['conflict', 'administration'])
        self.assertEqual(set(self.vectors.words_closer_than('war', 'terrorism')), expected)

    def test_rank(self):
        """Test rank returns expected value for distinct and identical nodes."""
        self.assertEqual(self.vectors.rank('war', 'war'), 1)
        self.assertEqual(self.vectors.rank('war', 'terrorism'), 3)

    def test_wv_property(self):
        """Test that the deprecated `wv` property returns `self`. To be removed in v4.0.0."""
        self.assertTrue(self.vectors is self.vectors.wv)

    def test_add_word(self):
        """Test that adding word in a manual way works correctly."""
        words = ['___some_word{}_not_present_in_keyed_vectors___'.format(i) for i in range(5)]
        word_vectors = [np.random.randn(self.vectors.vector_size) for _ in range(5)]

        # Test `add_entity` on already filled kv.
        for word, vector in zip(words, word_vectors):
            self.vectors.add_entity(word, vector)

        for word, vector in zip(words, word_vectors):
            self.assertTrue(np.allclose(self.vectors[word], vector))

        # Test `add_entity` on empty kv.
        kv = EuclideanKeyedVectors(self.vectors.vector_size)
        for word, vector in zip(words, word_vectors):
            kv.add_entity(word, vector)

        for word, vector in zip(words, word_vectors):
            self.assertTrue(np.allclose(kv[word], vector))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
