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
from gensim.models.keyedvectors import KeyedVectors as EuclideanKeyedVectors, WordEmbeddingSimilarityIndex, \
    FastTextKeyedVectors
from gensim.test.utils import datapath

import gensim.models.keyedvectors


logger = logging.getLogger(__name__)


class TestWordEmbeddingSimilarityIndex(unittest.TestCase):
    def setUp(self):
        self.vectors = EuclideanKeyedVectors.load_word2vec_format(
            datapath('euclidean_vectors.bin'), binary=True, datatype=np.float64)

    def test_most_similar(self):
        """Test most_similar returns expected results."""

        # check the handling of out-of-dictionary terms
        index = WordEmbeddingSimilarityIndex(self.vectors)
        self.assertLess(0, len(list(index.most_similar(u"holiday", topn=10))))
        self.assertEqual(0, len(list(index.most_similar(u"out-of-dictionary term", topn=10))))

        # check that the topn works as expected
        index = WordEmbeddingSimilarityIndex(self.vectors)
        results = list(index.most_similar(u"holiday", topn=10))
        self.assertLess(0, len(results))
        self.assertGreaterEqual(10, len(results))
        results = list(index.most_similar(u"holiday", topn=20))
        self.assertLess(10, len(results))
        self.assertGreaterEqual(20, len(results))

        # check that the term itself is not returned
        index = WordEmbeddingSimilarityIndex(self.vectors)
        terms = [term for term, similarity in index.most_similar(u"holiday", topn=len(self.vectors.vocab))]
        self.assertFalse(u"holiday" in terms)

        # check that the threshold works as expected
        index = WordEmbeddingSimilarityIndex(self.vectors, threshold=0.0)
        results = list(index.most_similar(u"holiday", topn=10))
        self.assertLess(0, len(results))
        self.assertGreaterEqual(10, len(results))

        index = WordEmbeddingSimilarityIndex(self.vectors, threshold=1.0)
        results = list(index.most_similar(u"holiday", topn=10))
        self.assertEqual(0, len(results))

        # check that the exponent works as expected
        index = WordEmbeddingSimilarityIndex(self.vectors, exponent=1.0)
        first_similarities = np.array([similarity for term, similarity in index.most_similar(u"holiday", topn=10)])
        index = WordEmbeddingSimilarityIndex(self.vectors, exponent=2.0)
        second_similarities = np.array([similarity for term, similarity in index.most_similar(u"holiday", topn=10)])
        self.assertTrue(np.allclose(first_similarities**2.0, second_similarities))


class TestEuclideanKeyedVectors(unittest.TestCase):
    def setUp(self):
        self.vectors = EuclideanKeyedVectors.load_word2vec_format(
            datapath('euclidean_vectors.bin'), binary=True, datatype=np.float64)

    def test_similarity_matrix(self):
        """Test similarity_matrix returns expected results."""

        documents = [[u"government", u"denied", u"holiday"], [u"holiday", u"slowing", u"hollingworth"]]
        dictionary = Dictionary(documents)
        similarity_matrix = self.vectors.similarity_matrix(dictionary).todense()

        # checking the existence of ones on the main diagonal
        self.assertTrue(
            (np.diag(similarity_matrix) == np.ones(similarity_matrix.shape[0])).all())

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

        predicted = self.vectors.most_similar('war', topn=0)
        self.assertEqual(len(predicted), 0)

        predicted = self.vectors.most_similar('war', topn=np.uint8(0))
        self.assertEqual(len(predicted), 0)

    def test_relative_cosine_similarity(self):
        """Test relative_cosine_similarity returns expected results with an input of a word pair and topn"""
        wordnet_syn = [
            'good', 'goodness', 'commodity', 'trade_good', 'full', 'estimable', 'honorable',
            'respectable', 'beneficial', 'just', 'upright', 'adept', 'expert', 'practiced', 'proficient',
            'skillful', 'skilful', 'dear', 'near', 'dependable', 'safe', 'secure', 'right', 'ripe', 'well',
            'effective', 'in_effect', 'in_force', 'serious', 'sound', 'salutary', 'honest', 'undecomposed',
            'unspoiled', 'unspoilt', 'thoroughly', 'soundly'
        ]   # synonyms for "good" as per wordnet
        cos_sim = []
        for i in range(len(wordnet_syn)):
            if wordnet_syn[i] in self.vectors.vocab:
                cos_sim.append(self.vectors.similarity("good", wordnet_syn[i]))
        cos_sim = sorted(cos_sim, reverse=True)  # cosine_similarity of "good" with wordnet_syn in decreasing order
        # computing relative_cosine_similarity of two similar words
        rcs_wordnet = self.vectors.similarity("good", "nice") / sum(cos_sim[i] for i in range(10))
        rcs = self.vectors.relative_cosine_similarity("good", "nice", 10)
        self.assertTrue(rcs_wordnet >= rcs)
        self.assertTrue(np.allclose(rcs_wordnet, rcs, 0, 0.125))
        # computing relative_cosine_similarity for two non-similar words
        rcs = self.vectors.relative_cosine_similarity("good", "worst", 10)
        self.assertTrue(rcs < 0.10)

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
        self.assertTrue(self.vectors is self.vectors)

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

    def test_ft_kv_backward_compat_w_360(self):
        kv = EuclideanKeyedVectors.load(datapath("ft_kv_3.6.0.model.gz"))
        ft_kv = FastTextKeyedVectors.load(datapath("ft_kv_3.6.0.model.gz"))

        expected = ['trees', 'survey', 'system', 'graph', 'interface']
        actual = [word for (word, similarity) in kv.most_similar("human", topn=5)]

        self.assertEqual(actual, expected)

        actual = [word for (word, similarity) in ft_kv.most_similar("human", topn=5)]

        self.assertEqual(actual, expected)


class L2NormTest(unittest.TestCase):
    def test(self):
        m = np.array(range(1, 10), dtype=np.float32)
        m.shape = (3, 3)

        norm = gensim.models.keyedvectors._l2_norm(m)
        self.assertFalse(np.allclose(m, norm))

        gensim.models.keyedvectors._l2_norm(m, replace=True)
        self.assertTrue(np.allclose(m, norm))


class UnpackTest(unittest.TestCase):
    def test_copy_sanity(self):
        m = np.array(range(9))
        m.shape = (3, 3)
        hash2index = {10: 0, 11: 1, 12: 2}

        n = gensim.models.keyedvectors._unpack_copy(m, 25, hash2index)
        self.assertTrue(np.all(m[0] == n[10]))
        self.assertTrue(np.all(m[1] == n[11]))
        self.assertTrue(np.all(m[2] == n[12]))

    def test_sanity(self):
        m = np.array(range(9))
        m.shape = (3, 3)
        hash2index = {10: 0, 11: 1, 12: 2}

        n = gensim.models.keyedvectors._unpack(m, 25, hash2index)
        self.assertTrue(np.all(np.array([0, 1, 2]) == n[10]))
        self.assertTrue(np.all(np.array([3, 4, 5]) == n[11]))
        self.assertTrue(np.all(np.array([6, 7, 8]) == n[12]))

    def test_tricky(self):
        m = np.array(range(9))
        m.shape = (3, 3)
        hash2index = {1: 0, 0: 1, 12: 2}

        n = gensim.models.keyedvectors._unpack(m, 25, hash2index)
        self.assertTrue(np.all(np.array([3, 4, 5]) == n[0]))
        self.assertTrue(np.all(np.array([0, 1, 2]) == n[1]))
        self.assertTrue(np.all(np.array([6, 7, 8]) == n[12]))

    def test_identity(self):
        m = np.array(range(9))
        m.shape = (3, 3)
        hash2index = {0: 0, 1: 1, 2: 2}

        n = gensim.models.keyedvectors._unpack(m, 25, hash2index)
        self.assertTrue(np.all(np.array([0, 1, 2]) == n[0]))
        self.assertTrue(np.all(np.array([3, 4, 5]) == n[1]))
        self.assertTrue(np.all(np.array([6, 7, 8]) == n[2]))


class Gensim320Test(unittest.TestCase):
    def test(self):
        path = datapath('old_keyedvectors_320.dat')
        vectors = gensim.models.keyedvectors.KeyedVectors.load(path)
        self.assertTrue(vectors.word_vec('computer') is not None)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
