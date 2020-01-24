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

from gensim.summarization.bm25 import get_bm25_weights, iter_bm25_bow, BM25
from gensim.test.utils import common_texts


class TestBM25(unittest.TestCase):
    def test_max_match_with_itself(self):
        """ Document should show maximum matching with itself """
        weights = get_bm25_weights(common_texts)
        for index, doc_weights in enumerate(weights):
            expected = max(doc_weights)
            predicted = doc_weights[index]
            self.assertAlmostEqual(expected, predicted)

    def test_with_generator(self):
        """ Check above function with input as generator """
        text_gen = (i for i in common_texts)
        weights = get_bm25_weights(text_gen)
        for index, doc_weights in enumerate(weights):
            expected = max(doc_weights)
            predicted = doc_weights[index]
            self.assertAlmostEqual(expected, predicted)

    def test_nonnegative_weights(self):
        """ All the weights for a partiular document should be non negative """
        weights = get_bm25_weights(common_texts)
        for doc_weights in weights:
            for weight in doc_weights:
                self.assertTrue(weight >= 0.)

    def test_same_match_with_same_document(self):
        """ A document should always get the same weight when matched with a particular document """
        corpus = [['cat', 'dog', 'mouse'], ['cat', 'lion'], ['cat', 'lion']]
        weights = get_bm25_weights(corpus)
        self.assertAlmostEqual(weights[0][1], weights[0][2])

    def test_disjoint_docs_if_weight_zero(self):
        """ Two disjoint documents should have zero matching"""
        corpus = [['cat', 'dog', 'lion'], ['goat', 'fish', 'tiger']]
        weights = get_bm25_weights(corpus)
        self.assertAlmostEqual(weights[0][1], 0)
        self.assertAlmostEqual(weights[1][0], 0)

    def test_multiprocessing(self):
        """ Result should be the same using different processes """
        weights1 = get_bm25_weights(common_texts)
        weights2 = get_bm25_weights(common_texts, n_jobs=2)
        weights3 = get_bm25_weights(common_texts, n_jobs=-1)
        self.assertAlmostEqual(weights1, weights2)
        self.assertAlmostEqual(weights1, weights3)
        self.assertAlmostEqual(weights2, weights3)

    def test_k1(self):
        """ Changing the k1 parameter should give consistent results """
        corpus = common_texts
        index = 0
        doc = corpus[index]
        first_k1 = 1.0
        second_k1 = 2.0

        first_bm25 = BM25(corpus, k1=first_k1)
        second_bm25 = BM25(corpus, k1=second_k1)
        first_score = first_bm25.get_score(doc, index)
        second_score = second_bm25.get_score(doc, index)
        self.assertLess(first_score, second_score)

        first_iter = iter_bm25_bow(corpus, k1=first_k1)
        second_iter = iter_bm25_bow(corpus, k1=second_k1)
        first_score = dict(next(iter(first_iter)))[index]
        second_score = dict(next(iter(second_iter)))[index]
        self.assertLess(first_score, second_score)

        first_weights = get_bm25_weights(corpus, k1=first_k1)
        second_weights = get_bm25_weights(corpus, k1=second_k1)
        first_score = first_weights[index]
        second_score = second_weights[index]
        self.assertLess(first_score, second_score)

    def test_b(self):
        """ Changing the b parameter should give consistent results """
        corpus = common_texts
        index = 0
        doc = corpus[index]
        first_b = 1.0
        second_b = 2.0

        first_bm25 = BM25(corpus, b=first_b)
        second_bm25 = BM25(corpus, b=second_b)
        first_score = first_bm25.get_score(doc, index)
        second_score = second_bm25.get_score(doc, index)
        self.assertLess(first_score, second_score)

        first_iter = iter_bm25_bow(corpus, b=first_b)
        second_iter = iter_bm25_bow(corpus, b=second_b)
        first_score = dict(next(iter(first_iter)))[index]
        second_score = dict(next(iter(second_iter)))[index]
        self.assertLess(first_score, second_score)

        first_weights = get_bm25_weights(corpus, b=first_b)
        second_weights = get_bm25_weights(corpus, b=second_b)
        first_score = first_weights[index]
        second_score = second_weights[index]
        self.assertLess(first_score, second_score)

    def test_epsilon(self):
        """ Changing the b parameter should give consistent results """
        corpus = [['cat', 'dog', 'mouse'], ['cat', 'lion'], ['cat', 'lion']]
        first_epsilon = 1.0
        second_epsilon = 2.0
        bm25 = BM25(corpus)
        words_with_negative_idfs = set([
            word
            for word, idf in bm25.idf.items()
            if idf < 0
        ])
        index, doc = [
            (index, document)
            for index, document
            in enumerate(corpus)
            if words_with_negative_idfs & set(document)
        ][0]

        first_bm25 = BM25(corpus, epsilon=first_epsilon)
        second_bm25 = BM25(corpus, epsilon=second_epsilon)
        first_score = first_bm25.get_score(doc, index)
        second_score = second_bm25.get_score(doc, index)
        self.assertGreater(first_score, second_score)

        first_iter = iter_bm25_bow(corpus, epsilon=first_epsilon)
        second_iter = iter_bm25_bow(corpus, epsilon=second_epsilon)
        first_score = dict(next(iter(first_iter)))[index]
        second_score = dict(next(iter(second_iter)))[index]
        self.assertGreater(first_score, second_score)

        first_weights = get_bm25_weights(corpus, epsilon=first_epsilon)
        second_weights = get_bm25_weights(corpus, epsilon=second_epsilon)
        first_score = first_weights[index]
        second_score = second_weights[index]
        self.assertGreater(first_score, second_score)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
