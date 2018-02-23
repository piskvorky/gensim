#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
import logging
import unittest
import numpy as np
from scipy.special import psi  # gamma function utils

import gensim.matutils as matutils


# we'll define known, good (slow) version of functions here
# and compare results from these functions vs. cython ones
def logsumexp(x):
    """
    Log of sum of exponentials

    Parameters
    ----------
    x : array_like
        Input data

    Returns
    -------
    float
        log of sum of exponentials of elements in `x`

    Notes
    -----
        for performance, does not support NaNs or > 1d arrays like
        scipy.special.logsumexp()

    """

    x_max = np.max(x)
    x = np.log(np.sum(np.exp(x - x_max)))
    x += x_max

    return x


def mean_absolute_difference(a, b):
    """
    Mean absolute difference between two arrays

    Parameters
    ----------
    a : (M,) array_like of float32
    b : (M,) array_like of float32

    Returns
    -------
    float
        mean(abs(a - b))

    """
    return np.mean(np.abs(a - b))


def dirichlet_expectation(alpha):
    """For a vector :math:`\\theta \sim Dir(\\alpha)`, compute :math:`E[log \\theta]`.

    Parameters
    ----------
    alpha : numpy.ndarray
        Input vector or matrix.

    Returns
    -------
    numpy.ndarray:
        :math:`E[log \\theta]`

    """
    if len(alpha.shape) == 1:
        result = psi(alpha) - psi(np.sum(alpha))
    else:
        result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
    return result.astype(alpha.dtype, copy=False)  # keep the same precision as input


dirichlet_expectation_1d = dirichlet_expectation
dirichlet_expectation_2d = dirichlet_expectation


class TestLdaModelInner(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState()
        self.num_runs = 100  # test functions with *num_runs* random inputs
        self.num_topics = 100

    def testLogSumExp(self):
        # test logsumexp
        rs = self.random_state

        for dtype in [np.float16, np.float32, np.float64]:
            for i in range(self.num_runs):
                input = rs.uniform(-1000, 1000, size=(self.num_topics, 1))

                known_good = logsumexp(input)
                test_values = matutils.logsumexp(input)

                msg = "logsumexp failed for dtype={}".format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)

    def testMeanAbsoluteDifference(self):
        # test mean_absolute_difference
        rs = self.random_state

        for dtype in [np.float16, np.float32, np.float64]:
            for i in range(self.num_runs):
                input1 = rs.uniform(-10000, 10000, size=(self.num_topics,))
                input2 = rs.uniform(-10000, 10000, size=(self.num_topics,))

                known_good = mean_absolute_difference(input1, input2)
                test_values = matutils.mean_absolute_difference(input1, input2)

                msg = "mean_absolute_difference failed for dtype={}".format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)

    def testDirichletExpectation(self):
        # test dirichlet_expectation
        rs = self.random_state

        for dtype in [np.float16, np.float32, np.float64]:
            for i in range(self.num_runs):
                # 1 dimensional case
                input_1d = rs.uniform(.01, 10000, size=(self.num_topics,))
                known_good = dirichlet_expectation(input_1d)
                test_values = matutils.dirichlet_expectation_1d(input_1d)

                msg = "dirichlet_expectation_1d failed for dtype={}".format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)

                # 2 dimensional case
                input_2d = rs.uniform(.01, 10000, size=(1, self.num_topics,))
                known_good = dirichlet_expectation(input_2d)
                test_values = matutils.dirichlet_expectation_2d(input_2d)

                msg = "dirichlet_expectation_2d failed for dtype={}".format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
