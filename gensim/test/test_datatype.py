#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking various matutils functions.
"""

import logging
import unittest

import numpy as np

from gensim.test.utils import datapath
from gensim.models.keyedvectors import KeyedVectors


class TestDataType(unittest.TestCase):
    def load_model(self, datatype):
        path = datapath('test.kv.txt')
        kv = KeyedVectors.load_word2vec_format(path, binary=False,
                                               datatype=datatype)
        return kv

    def test_high_precision(self):
        kv = self.load_model(np.float64)
        self.assertAlmostEqual(kv['horse.n.01'][0], -0.0008546282343595379)
        self.assertEqual(kv['horse.n.01'][0].dtype, np.float64)

    def test_medium_precision(self):
        kv = self.load_model(np.float32)
        self.assertAlmostEqual(kv['horse.n.01'][0], -0.00085462822)
        self.assertEqual(kv['horse.n.01'][0].dtype, np.float32)

    def test_low_precision(self):
        kv = self.load_model(np.float16)
        self.assertAlmostEqual(kv['horse.n.01'][0], -0.00085449)
        self.assertEqual(kv['horse.n.01'][0].dtype, np.float16)


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
