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
from gensim.topic_coherence import text_analysis
from gensim.corpora.dictionary import Dictionary

import numpy as np


class TestIndirectConfirmation(unittest.TestCase):
    def setUp(self):
        # Set up toy example for better understanding and testing
        # of this module. See the modules for the mathematical formulas
        self.topics = [np.array([1, 2])]
        # Result from s_one_set segmentation:
        self.segmentation = [[(1, np.array([1, 2])), (2, np.array([1, 2]))]]
        self.gamma = 1
        self.measure = 'nlr'

        dictionary = Dictionary()
        dictionary.id2token = {1: 'fake', 2: 'tokens'}
        self.accumulator = text_analysis.InvertedIndexAccumulator({1, 2}, dictionary)
        self.accumulator._inverted_index = {0: {2, 3, 4}, 1: {3, 5}}
        self.accumulator._num_docs = 5

    def testCosineSimilarity(self):
        """Test cosine_similarity()"""
        obtained = indirect_confirmation_measure.cosine_similarity(
            self.segmentation, self.accumulator, self.topics, self.measure, self.gamma)

        # The steps involved in this calculation are as follows:
        # 1. Take (1, array([1, 2]). Take w' which is 1.
        # 2. Calculate nlr(1, 1), nlr(1, 2). This is our first vector.
        # 3. Take w* which is array([1, 2]).
        # 4. Calculate nlr(1, 1) + nlr(2, 1). Calculate nlr(1, 2), nlr(2, 2). This is our second vector.
        # 5. Find out cosine similarity between these two vectors.
        # 6. Similarly for the second segmentation.
        expected = [0.6230, 0.6230]  # To account for EPSILON approximation
        for i in range(len(expected)):
            self.assertAlmostEqual(obtained[i], expected[i], 4)


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
