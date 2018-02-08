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

from gensim.summarization.bm25 import get_bm25_weights
from gensim.test.utils import common_texts


class TestBM25(unittest.TestCase):
    def test_max_match_with_itself(self):
        """ Document should show maximum matching with itself """
        weights = get_bm25_weights(common_texts)
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
