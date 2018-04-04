#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Vit Novotny <witiko@mail.muni.cz>
# Copyright (C) 2018 Vit Novotny <witiko@mail.muni.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking the gensim.models.term_similarity module.
"""

import logging
from math import sqrt
import unittest

from gensim.corpora import Dictionary
from gensim.models import UniformTermSimilarityIndex, SparseTermSimilarityMatrix, TfidfModel

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

logger = logging.getLogger(__name__)


class TestUniformTermSimilarityIndex(unittest.TestCase):
    def setUp(self):
        self.documents = [["government", "denied", "holiday"], ["holiday", "slowing", "hollingworth"]]
        self.dictionary = Dictionary(self.documents)

    def test_most_similar(self):
        """Test most_similar returns expected results."""

        # check that the topn works as expected
        index = UniformTermSimilarityIndex(self.dictionary)
        results = list(index.most_similar("holiday", topn=1))
        self.assertLess(0, len(results))
        self.assertGreaterEqual(1, len(results))
        results = list(index.most_similar("holiday", topn=4))
        self.assertLess(1, len(results))
        self.assertGreaterEqual(4, len(results))

        # check that the term itself is not returned
        index = UniformTermSimilarityIndex(self.dictionary)
        terms = [term for term, similarity in index.most_similar("holiday", topn=len(self.dictionary))]
        self.assertFalse("holiday" in terms)

        # check that the term_similarity works as expected
        index = UniformTermSimilarityIndex(self.dictionary, term_similarity=0.2)
        similarities = np.array([
            similarity for term, similarity in index.most_similar("holiday", topn=len(self.dictionary))])
        self.assertTrue(np.all(similarities == 0.2))


class TestSparseTermSimilarityMatrix(unittest.TestCase):
    def setUp(self):
        self.documents = [
            ["government", "denied", "holiday"],
            ["government", "denied", "holiday", "slowing", "hollingworth"]]
        self.dictionary = Dictionary(self.documents)
        self.tfidf = TfidfModel(dictionary=self.dictionary)
        self.index = UniformTermSimilarityIndex(self.dictionary, term_similarity=0.5)

    def test_building(self):
        """Test the matrix building algorithm."""

        # check matrix type
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary).matrix
        self.assertTrue(isinstance(matrix, csc_matrix))

        # check symmetry
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary).matrix.todense()
        self.assertTrue(np.all(matrix == matrix.T))

        # check the existence of ones on the main diagonal
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary).matrix.todense()
        self.assertTrue(np.all(np.diag(matrix) == np.ones(matrix.shape[0])))

        # check the matrix order
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary).matrix.todense()
        self.assertEqual(matrix.shape[0], len(self.dictionary))
        self.assertEqual(matrix.shape[1], len(self.dictionary))

        # check that the dtype works as expected
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, dtype=np.float32).matrix.todense()
        self.assertEqual(np.float32, matrix.dtype)

        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, dtype=np.float64).matrix.todense()
        self.assertEqual(np.float64, matrix.dtype)

        # check that the nonzero_limit works as expected
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=100).matrix.todense()
        self.assertGreaterEqual(101, np.max(np.sum(matrix != 0, axis=0)))

        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=4).matrix.todense()
        self.assertGreaterEqual(5, np.max(np.sum(matrix != 0, axis=0)))

        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=1).matrix.todense()
        self.assertGreaterEqual(2, np.max(np.sum(matrix != 0, axis=0)))

        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=0).matrix.todense()
        self.assertEqual(1, np.max(np.sum(matrix != 0, axis=0)))
        self.assertTrue(np.all(matrix == np.eye(matrix.shape[0])))

        # check that tfidf works as expected
        matrix = SparseTermSimilarityMatrix(
            self.index, self.dictionary, nonzero_limit=1).matrix.todense()
        expected_matrix = np.array([
            [1.0, 0.5, 0.0, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(np.all(expected_matrix == matrix))

        matrix = SparseTermSimilarityMatrix(
            self.index, self.dictionary, nonzero_limit=1, tfidf=self.tfidf).matrix.todense()
        expected_matrix = np.array([
            [1.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(np.all(expected_matrix == matrix))

    def test_encapsulation(self):
        """Test the matrix encapsulation."""

        # check that a sparse matrix will be converted to a CSC format
        expected_matrix = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [0.0, 0.0, 1.0]])

        matrix = SparseTermSimilarityMatrix(csc_matrix(expected_matrix)).matrix
        self.assertTrue(isinstance(matrix, csc_matrix))
        self.assertTrue(np.all(matrix.todense() == expected_matrix))

        matrix = SparseTermSimilarityMatrix(csr_matrix(expected_matrix)).matrix
        self.assertTrue(isinstance(matrix, csc_matrix))
        self.assertTrue(np.all(matrix.todense() == expected_matrix))

    def test_inner_product(self):
        """Test the inner product."""

        matrix = SparseTermSimilarityMatrix(
            UniformTermSimilarityIndex(self.dictionary, term_similarity=0.5), self.dictionary)

        # check zero vectors work as expected
        vec1 = self.dictionary.doc2bow(["government", "government", "denied"])
        vec2 = self.dictionary.doc2bow(["government", "holiday"])

        self.assertEqual(0.0, matrix.inner_product([], vec2))
        self.assertEqual(0.0, matrix.inner_product(vec1, []))
        self.assertEqual(0.0, matrix.inner_product([], []))

        self.assertEqual(0.0, matrix.inner_product([], vec2, normalized=True))
        self.assertEqual(0.0, matrix.inner_product(vec1, [], normalized=True))
        self.assertEqual(0.0, matrix.inner_product([], [], normalized=True))

        # check that real-world vectors work as expected
        vec1 = self.dictionary.doc2bow(["government", "government", "denied"])
        vec2 = self.dictionary.doc2bow(["government", "holiday"])
        expected_result = 0.0
        expected_result += 2 * 1.0 * 1  # government * s_{ij} * government
        expected_result += 2 * 0.5 * 1  # government * s_{ij} * holiday
        expected_result += 1 * 0.5 * 1  # denied * s_{ij} * government
        expected_result += 1 * 0.5 * 1  # denied * s_{ij} * holiday
        self.assertEqual(expected_result, matrix.inner_product(vec1, vec2))

        vec1 = self.dictionary.doc2bow(["government", "government", "denied"])
        vec2 = self.dictionary.doc2bow(["government", "holiday"])
        expected_result = matrix.inner_product(vec1, vec2)
        expected_result /= sqrt(matrix.inner_product(vec1, vec1))
        expected_result /= sqrt(matrix.inner_product(vec2, vec2))
        self.assertEqual(expected_result, matrix.inner_product(vec1, vec2, normalized=True))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
