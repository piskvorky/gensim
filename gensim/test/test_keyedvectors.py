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
from gensim.models import KeyedVectors as EuclideanKeyedVectors, TfidfModel
from gensim.test.utils import datapath


logger = logging.getLogger(__name__)


class TestEuclideanKeyedVectors(unittest.TestCase):
    def setUp(self):
        self.vectors = EuclideanKeyedVectors.load_word2vec_format(
            datapath('euclidean_vectors.bin'), binary=True, datatype=np.float64)

    def test_similarity_matrix(self):
        """Test similarity_matrix returns expected results."""

        documents = [["government", "denied", "holiday"],
                  ["holiday", "slowing", "hollingworth"]]
        dictionary = Dictionary(documents)

        # checking symmetry
        similarity_matrix = self.vectors.similarity_matrix(dictionary).todense()
        self.assertTrue((similarity_matrix.T == similarity_matrix).all())

        # checking the existence of ones on the main diagonal
        self.assertTrue(
            (np.diag(similarity_matrix) ==
             np.ones(similarity_matrix.shape[0])).all())

        # checking that thresholding works as expected
        similarity_matrix = self.vectors.similarity_matrix(dictionary, threshold=0.45).todense()
        self.assertEquals(18, np.sum(similarity_matrix == 0))

        # checking that exponent works as expected
        similarity_matrix = self.vectors.similarity_matrix(dictionary, exponent=1.0).todense()
        self.assertAlmostEqual(9.5788956, np.sum(similarity_matrix), places=5)

        # checking that nonzero_limit works as expected
        similarity_matrix = self.vectors.similarity_matrix(dictionary, nonzero_limit=4).todense()
        self.assertEquals(4, np.sum(similarity_matrix == 0))

        similarity_matrix = self.vectors.similarity_matrix(dictionary, nonzero_limit=3).todense()
        self.assertEquals(20, np.sum(similarity_matrix == 0))

        # check that processing rows in the order given by IDF has desired effect

        # The complete similarity matrix we would obtain with nonzero_limit would look as follows:
        documents = [["honour", "understanding"], ["understanding", "mean", "knop"]]
        dictionary = Dictionary(documents)
        tfidf = TfidfModel(dictionary=dictionary)

        # All terms except for "understanding" have IDF of log2(2 / 1) = log2(2) = 1.
        # The term "understanding" has IDF of log2(2 / 2) = log2(1) = 0.
        #
        # If we do not pass the tfidf parameter to the similarity_matrix
        # method, then we process rows in the order from 1 to 4. If we do pass
        # the tfidf parameter to the similarity_matrix method, then we first
        # process the rows 1, 3, 4 that correspond to terms with IDF of 1.0 and
        # then the row 2 that corresponds to the term "understanding" with IDF
        # of 0. Since the method is greedy, we will end up with two different
        # similarity matrices.

        similarity_matrix = self.vectors.similarity_matrix(
            dictionary, nonzero_limit=2).todense()
        self.assertTrue(np.all(np.isclose(similarity_matrix, np.array([
            [1, 0.9348248, 0, 0], [0.9348248, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))))

        similarity_matrix = self.vectors.similarity_matrix(
            dictionary, tfidf, nonzero_limit=2).todense()
        self.assertTrue(np.all(np.isclose(similarity_matrix, np.array([
            [1, 0.9348248, 0, 0.9112908], [0.9348248, 1, 0.90007025, 0], [0, 0.90007025, 1, 0],
            [0.9112908, 0, 0, 1]]))))

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

    def test_add_single(self):
        """Test that adding entity in a manual way works correctly."""
        entities = ['___some_entity{}_not_present_in_keyed_vectors___'.format(i) for i in range(5)]
        vectors = [np.random.randn(self.vectors.vector_size) for _ in range(5)]

        # Test `add` on already filled kv.
        for ent, vector in zip(entities, vectors):
            self.vectors.add(ent, vector)

        for ent, vector in zip(entities, vectors):
            self.assertTrue(np.allclose(self.vectors[ent], vector))

        # Test `add` on empty kv.
        kv = EuclideanKeyedVectors(self.vectors.vector_size)
        for ent, vector in zip(entities, vectors):
            kv.add(ent, vector)

        for ent, vector in zip(entities, vectors):
            self.assertTrue(np.allclose(kv[ent], vector))

    def test_add_multiple(self):
        """Test that adding a bulk of entities in a manual way works correctly."""
        entities = ['___some_entity{}_not_present_in_keyed_vectors___'.format(i) for i in range(5)]
        vectors = [np.random.randn(self.vectors.vector_size) for _ in range(5)]

        # Test `add` on already filled kv.
        vocab_size = len(self.vectors.vocab)
        self.vectors.add(entities, vectors, replace=False)
        self.assertEqual(vocab_size + len(entities), len(self.vectors.vocab))

        for ent, vector in zip(entities, vectors):
            self.assertTrue(np.allclose(self.vectors[ent], vector))

        # Test `add` on empty kv.
        kv = EuclideanKeyedVectors(self.vectors.vector_size)
        kv[entities] = vectors
        self.assertEqual(len(kv.vocab), len(entities))

        for ent, vector in zip(entities, vectors):
            self.assertTrue(np.allclose(kv[ent], vector))

    def test_set_item(self):
        """Test that __setitem__ works correctly."""
        vocab_size = len(self.vectors.vocab)

        # Add new entity.
        entity = '___some_new_entity___'
        vector = np.random.randn(self.vectors.vector_size)
        self.vectors[entity] = vector

        self.assertEqual(len(self.vectors.vocab), vocab_size + 1)
        self.assertTrue(np.allclose(self.vectors[entity], vector))

        # Replace vector for entity in vocab.
        vocab_size = len(self.vectors.vocab)
        vector = np.random.randn(self.vectors.vector_size)
        self.vectors['war'] = vector

        self.assertEqual(len(self.vectors.vocab), vocab_size)
        self.assertTrue(np.allclose(self.vectors['war'], vector))

        # __setitem__ on several entities.
        vocab_size = len(self.vectors.vocab)
        entities = ['war', '___some_new_entity1___', '___some_new_entity2___', 'terrorism', 'conflict']
        vectors = [np.random.randn(self.vectors.vector_size) for _ in range(len(entities))]

        self.vectors[entities] = vectors

        self.assertEqual(len(self.vectors.vocab), vocab_size + 2)
        for ent, vector in zip(entities, vectors):
            self.assertTrue(np.allclose(self.vectors[ent], vector))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
