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

from gensim import matutils


class TestMatutils(unittest.TestCase):
    def test_unitvec(self):
        input_vector = np.random.uniform(size=(100,)).astype(np.float32)
        unit_vector = matutils.unitvec(input_vector)
        self.assertEqual(input_vector.dtype, unit_vector.dtype)


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
