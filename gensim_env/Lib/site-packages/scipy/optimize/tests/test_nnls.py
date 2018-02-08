""" Unit tests for nonnegative least squares
Author: Uwe Schmitt
Sep 2008
"""
from __future__ import division, print_function, absolute_import

from numpy.testing import assert_

from scipy.optimize import nnls
from numpy import arange, dot
from numpy.linalg import norm


class TestNNLS(object):

    def test_nnls(self):
        a = arange(25.0).reshape(-1,5)
        x = arange(5.0)
        y = dot(a,x)
        x, res = nnls(a,y)
        assert_(res < 1e-7)
        assert_(norm(dot(a,x)-y) < 1e-7)
