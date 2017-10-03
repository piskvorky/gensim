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
import gensim.models.coherence_utils


class TestAggregation(unittest.TestCase):
    def setUp(self):
        self.confirmed_measures = [1.1, 2.2, 3.3, 4.4]

    def testArithmeticMean(self):
        """Test arithmetic_mean()"""
        obtained = gensim.models.coherence_utils.arithmetic_mean(self.confirmed_measures)
        expected = 2.75
        self.assertEqual(obtained, expected)


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
