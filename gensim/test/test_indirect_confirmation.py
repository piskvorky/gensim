#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for indirect confirmation measures in the indirect_confirmation_measure module.
"""

import logging
import unittest

from gensim.topic_coherence import indirect_confirmation_measure

import numpy as np
from numpy import array

class TestIndirectConfirmation(unittest.TestCase):
    def setUp(self):
        # Set up toy example for better understanding and testing
        # of this module. See the modules for the mathematical formulas
        self.topics = [np.array([1, 2])]
        # Result from s_one_set segmentation:
        self.segmentation = [[(1, array([1, 2])), (2, array([1, 2]))]]
        self.posting_list = {1: set([2, 3, 4]), 2: set([3, 5])}
        self.gamma = 1
        self.measure = 'nlr'
        self.num_docs = 5

    def testCosineSimilarity(self):
        """Test cosine_similarity()"""
        obtained = indirect_confirmation_measure.cosine_similarity(self.topics, self.segmentation,
                                                                   self.posting_list, self.measure,
                                                                   self.gamma, self.num_docs)
        # The steps involved in this calculation are as follows:
        # 1. Take (1, array([1, 2]). Take w' which is 1.
        # 2. Calculate nlr(1, 1), nlr(1, 2). This is our first vector.
        # 3. Take w* which is array([1, 2]).
        # 4. Calculate nlr(1, 1) + nlr(2, 1). Calculate nlr(1, 2), nlr(2, 2). This is our second vector.
        # 5. Find out cosine similarity between these two vectors.
        # 6. Similarly for the second segmentation.
        expected = [0.6230, 0.6230]  # To account for EPSILON approximation
        self.assertAlmostEqual(obtained[0], expected[0], 4)
        self.assertAlmostEqual(obtained[1], expected[1], 4)

if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
