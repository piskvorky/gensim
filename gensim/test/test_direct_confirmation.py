#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for direct confirmation measures in the direct_confirmation_measure module.
"""

import logging
import unittest
from collections import namedtuple

from gensim.topic_coherence import direct_confirmation_measure
from gensim.topic_coherence import text_analysis


class TestDirectConfirmationMeasure(unittest.TestCase):
    def setUp(self):
        # Set up toy example for better understanding and testing
        # of this module. See the modules for the mathematical formulas
        self.segmentation = [[(1, 2)]]
        self.posting_list = {1: {2, 3, 4}, 2: {3, 5}}
        self.num_docs = 5

        id2token = {1: 'test', 2: 'doc'}
        token2id = {v: k for k, v in id2token.items()}
        dictionary = namedtuple('Dictionary', 'token2id, id2token')(token2id, id2token)
        self.accumulator = text_analysis.InvertedIndexAccumulator({1, 2}, dictionary)
        self.accumulator._inverted_index = {0: {2, 3, 4}, 1: {3, 5}}
        self.accumulator._num_docs = self.num_docs

    def testLogConditionalProbability(self):
        """Test log_conditional_probability()"""
        obtained = direct_confirmation_measure.log_conditional_probability(
            self.segmentation, self.accumulator)[0]
        # Answer should be ~ ln(1 / 2) = -0.693147181
        expected = -0.693147181
        self.assertAlmostEqual(obtained, expected)

    def testLogRatioMeasure(self):
        """Test log_ratio_measure()"""
        obtained = direct_confirmation_measure.log_ratio_measure(
            self.segmentation, self.accumulator)[0]
        # Answer should be ~ ln{(1 / 5) / [(3 / 5) * (2 / 5)]} = -0.182321557
        expected = -0.182321557
        self.assertAlmostEqual(obtained, expected)

    def testNormalizedLogRatioMeasure(self):
        """Test normalized_log_ratio_measure()"""
        obtained = direct_confirmation_measure.log_ratio_measure(
            self.segmentation, self.accumulator, normalize=True)[0]
        # Answer should be ~ -0.182321557 / -ln(1 / 5) = -0.113282753
        expected = -0.113282753
        self.assertAlmostEqual(obtained, expected)


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
