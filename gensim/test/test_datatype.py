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
    def test_text(self):
        path = datapath('test.kv.txt')
        kv = KeyedVectors.load_word2vec_format(path, binary=False,
                                               datatype=np.float64)
        self.assertAlmostEqual(kv['horse.n.01'][0], -0.0008546282343595379)
        self.assertEqual(kv['horse.n.01'][0].dtype, np.float64)


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
