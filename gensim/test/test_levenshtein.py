#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Vit Novotny <witiko@mail.muni.cz>
# Copyright (C) 2018 Vit Novotny <witiko@mail.muni.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking the poincare module from the models package.
"""

import logging
import unittest

import numpy as np

from gensim.corpora import Dictionary
from gensim.models import levenshtein, TfidfModel


logger = logging.getLogger(__name__)


class TestLevenshtein(unittest.TestCase):
    def test_similarity_matrix(self):
        """Test similarity_matrix returns expected results."""

        documents = [["government", "denied", "holiday"],
                  ["holiday", "slowing", "hollingworth"]]
        dictionary = Dictionary(documents)

        # checking symmetry
        similarity_matrix = levenshtein.similarity_matrix(dictionary).todense()
        self.assertTrue((similarity_matrix.T == similarity_matrix).all())

        # checking the existence of ones on the main diagonal
        self.assertTrue(
            (np.diag(similarity_matrix) ==
             np.ones(similarity_matrix.shape[0])).all())

        # checking that thresholding works as expected
        similarity_matrix = levenshtein.similarity_matrix(dictionary).todense()
        self.assertEquals(0, np.sum(similarity_matrix == 0))

        similarity_matrix = levenshtein.similarity_matrix(dictionary, threshold=0.1).todense()
        self.assertEquals(20, np.sum(similarity_matrix == 0))

        # checking that alpha and beta work as expected
        distances = np.array([
            [ 1,  7,  6, 11,  6],
            [ 7,  1,  9,  9,  9],
            [ 6,  9,  1,  8,  6],
            [11,  9,  8,  1,  9],
            [ 6,  9,  6,  9,  1]])
        lengths = np.array([
            [ 6, 10,  7, 12,  7],
            [10, 10, 10, 12, 10],
            [ 7, 10,  7, 12,  7],
            [12, 12, 12, 12, 12],
            [ 7, 10,  7, 12,  7]])
        alpha = 1.2
        beta = 3.4
        expected_similarity_matrix = alpha * (1.0 - distances * 1.0 / lengths)**beta
        np.fill_diagonal(expected_similarity_matrix, 1)
        similarity_matrix = levenshtein.similarity_matrix(dictionary, alpha=alpha, beta=beta).todense()
        self.assertTrue(np.allclose(expected_similarity_matrix, similarity_matrix))

        # checking that nonzero_limit works as expected
        similarity_matrix = levenshtein.similarity_matrix(dictionary).todense()
        self.assertEquals(0, np.sum(similarity_matrix == 0))

        zeros = np.array([
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0]])
        similarity_matrix = levenshtein.similarity_matrix(dictionary, nonzero_limit=2).todense()
        self.assertTrue(np.all(zeros == (similarity_matrix == 0)))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
