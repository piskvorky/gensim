#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
from gensim import csparse
from gensim.csparse import gaxpy

from numpy.testing import assert_array_equal
from numpy import asfortranarray
from numpy.random import normal
from functools import partial
from scipy.sparse import rand
from os import cpu_count
from timeit import Timer
import unittest
import logging
import scipy

class TestCSparse(unittest.TestCase):
    def testPmultiply(self):

        if csparse.openmp:

            def matmul(X, Y): return X * Y
            def pmatmul(X, Y): return gaxpy(X, Y)

            X_rows, X_cols, factors = 1, 100000, 1000

            X = rand(X_rows, X_cols, format="csc")
            Y = asfortranarray(normal(0.0, 1.0, (X.shape[1], factors)))

            if cpu_count() > 1:
                t_matmul = Timer(partial(matmul, X, Y)).timeit(1)
                t_pmatmul = Timer(partial(pmatmul, X, Y)).timeit(1)

                assert t_pmatmul < t_matmul

            assert_array_equal(matmul(X, Y), pmatmul(X, Y))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
