from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_allclose

from scipy.special._testutils import FuncData
from scipy.special import gamma, gammaln, loggamma


def test_identities1():
    # test the identity exp(loggamma(z)) = gamma(z)
    x = np.array([-99.5, -9.5, -0.5, 0.5, 9.5, 99.5])
    y = x.copy()
    x, y = np.meshgrid(x, y)
    z = (x + 1J*y).flatten()
    dataset = np.vstack((z, gamma(z))).T

    def f(z):
        return np.exp(loggamma(z))

    FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()


def test_identities2():
    # test the identity loggamma(z + 1) = log(z) + loggamma(z)
    x = np.array([-99.5, -9.5, -0.5, 0.5, 9.5, 99.5])
    y = x.copy()
    x, y = np.meshgrid(x, y)
    z = (x + 1J*y).flatten()
    dataset = np.vstack((z, np.log(z) + loggamma(z))).T

    def f(z):
        return loggamma(z + 1)

    FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()


def test_realpart():
    # Test that the real parts of loggamma and gammaln agree on the
    # real axis.
    x = np.r_[-np.logspace(10, -10), np.logspace(-10, 10)] + 0.5
    dataset = np.vstack((x, gammaln(x))).T

    def f(z):
        return loggamma(z).real
    
    FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()


def test_gh_6536():
    z = loggamma(complex(-3.4, +0.0))
    zbar = loggamma(complex(-3.4, -0.0))
    assert_allclose(z, zbar.conjugate(), rtol=1e-15, atol=0)
    

def test_branch_cut():
    # Make sure negative zero is treated correctly
    x = -np.logspace(300, -30, 100)
    z = np.asarray([complex(x0, 0.0) for x0 in x])
    zbar = np.asarray([complex(x0, -0.0) for x0 in x])
    assert_allclose(z, zbar.conjugate(), rtol=1e-15, atol=0)
