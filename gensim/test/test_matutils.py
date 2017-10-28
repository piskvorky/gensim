#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking various math helper functions.
"""


import logging
import unittest

import numpy as np

from gensim import matutils


class TestMatUtils(unittest.TestCase):
    def test_dirichlet_expectation_keeps_precision(self):
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            alpha_1d = np.array([0.5, 0.5], dtype=dtype)
            result = matutils.dirichlet_expectation(alpha_1d)
            self.assertEqual(dtype, result.dtype)

            alpha_2d = np.array([[0.5, 0.5], [1.0, 2.0]], dtype=dtype)
            result = matutils.dirichlet_expectation(alpha_2d)
            self.assertEqual(dtype, result.dtype)


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
