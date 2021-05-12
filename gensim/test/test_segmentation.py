#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for segmentation algorithms in the segmentation module.
"""


import logging
import unittest

import numpy as np

from gensim.topic_coherence import segmentation
from numpy import array


class TestSegmentation(unittest.TestCase):
    def setUp(self):
        self.topics = [
            array([9, 4, 6]),
            array([9, 10, 7]),
            array([5, 2, 7])
        ]

    def test_s_one_pre(self):
        """Test s_one_pre segmentation."""
        actual = segmentation.s_one_pre(self.topics)
        expected = [
            [(4, 9), (6, 9), (6, 4)],
            [(10, 9), (7, 9), (7, 10)],
            [(2, 5), (7, 5), (7, 2)]
        ]
        self.assertTrue(np.allclose(actual, expected))

    def test_s_one_one(self):
        """Test s_one_one segmentation."""
        actual = segmentation.s_one_one(self.topics)
        expected = [
            [(9, 4), (9, 6), (4, 9), (4, 6), (6, 9), (6, 4)],
            [(9, 10), (9, 7), (10, 9), (10, 7), (7, 9), (7, 10)],
            [(5, 2), (5, 7), (2, 5), (2, 7), (7, 5), (7, 2)]
        ]
        self.assertTrue(np.allclose(actual, expected))

    def test_s_one_set(self):
        """Test s_one_set segmentation."""
        actual = segmentation.s_one_set(self.topics)
        expected = [
            [(9, array([9, 4, 6])), (4, array([9, 4, 6])), (6, array([9, 4, 6]))],
            [(9, array([9, 10, 7])), (10, array([9, 10, 7])), (7, array([9, 10, 7]))],
            [(5, array([5, 2, 7])), (2, array([5, 2, 7])), (7, array([5, 2, 7]))]
        ]
        for s_i in range(len(actual)):
            for j in range(len(actual[s_i])):
                self.assertEqual(actual[s_i][j][0], expected[s_i][j][0])
                self.assertTrue(np.allclose(actual[s_i][j][1], expected[s_i][j][1]))


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
