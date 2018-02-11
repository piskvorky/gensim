#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import logging
import unittest

import numpy, scipy
from gensim.csparse.psparse import pmultiply
from numpy.testing import assert_array_equal

class TestCSparse(unittest.TestCase):
    def testPmultiply(self):
        X = scipy.sparse.rand(3000, 8000, format="csc")
        Y = numpy.asfortranarray(numpy.random.normal(0.0, 1.0, (X.shape[1], 1000)))
        Z_np = X @ Y
        Z = pmultiply(X, Y)

        assert_array_equal(Z_np, Z)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
