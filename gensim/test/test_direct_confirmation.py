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

from gensim.topic_coherence import direct_confirmation_measure

class TestDirectConfirmationMeasure(unittest.TestCase):
    def setUp(self):
        # Set up toy example for better understanding and testing
        # of this module. See the modules for the mathematical formulas
        self.segmentation = [[(1, 2)]]
        self.posting_list = {1: set([2, 3, 4]), 2: set([3, 5])}
        self.num_docs = 5

    def testLogConditionalProbability(self):
        """Test log_conditional_probability()"""
        obtained = direct_confirmation_measure.log_conditional_probability(self.segmentation, self.posting_list, self.num_docs)[0]
        # Answer should be ~ ln(1 / 2) = -0.693147181
        expected = -0.693147181
        self.assertAlmostEqual(obtained, expected)

    def testLogRatioMeasure(self):
        """Test log_ratio_measure()"""
        obtained = direct_confirmation_measure.log_ratio_measure(self.segmentation, self.posting_list, self.num_docs)[0]
        # Answer should be ~ ln{(1 / 5) / [(3 / 5) * (2 / 5)]} = -0.182321557
        expected = -0.182321557
        self.assertAlmostEqual(obtained, expected)

    def testNormalizedLogRatioMeasure(self):
        """Test normalized_log_ratio_measure()"""
        obtained = direct_confirmation_measure.log_ratio_measure(self.segmentation, self.posting_list, self.num_docs, normalize=True)[0]
        # Answer should be ~ -0.182321557 / -ln(1 / 5) = -0.113282753
        expected = -0.113282753
        self.assertAlmostEqual(obtained, expected)

if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
