#!/usr/bin/env cython
# coding: utf-8
# cython: embedsignature=True

from __future__ import division
cimport cython
import numpy as np
cimport numpy as np
ctypedef cython.floating DTYPE_t
from libc.math cimport log, exp, fabs
from cython.parallel import prange


def mean_absolute_difference(a, b):
    """Mean absolute difference between two arrays, using :func:`~gensim._matutils._mean_absolute_difference`.

    Parameters
    ----------
    a : numpy.ndarray
        Input 1d array, supports float16, float32 and float64.
    b : numpy.ndarray
        Input 1d array, supports float16, float32 and float64.

    Returns
    -------
    float
        mean(abs(a - b)).

    """
    if a.shape != b.shape:
        raise ValueError("a and b must have same shape")

    if a.dtype == np.float64:
        return _mean_absolute_difference[double](a, b)
    elif a.dtype == np.float32:
        return _mean_absolute_difference[float](a, b)
    elif a.dtype == np.float16:
        return _mean_absolute_difference[float](a.astype(np.float32), b.astype(np.float32))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_t _mean_absolute_difference(DTYPE_t[:] a, DTYPE_t[:] b) nogil:
    """Mean absolute difference between two arrays.

    Parameters
    ----------
    a : numpy.ndarray
        Input 1d array.
    b : numpy.ndarray
        Input 1d array.

    Returns
    -------
    DTYPE_t
        mean(abs(a - b))

    """

    cdef DTYPE_t result = 0.0
    cdef size_t i
    cdef size_t j

    cdef size_t I = a.shape[0]
    cdef size_t N = I

    for i in range(I):
        result += fabs(a[i] - b[i])
    result /= N

    return result


def _logsumexp(x):
    """Log of sum of exponentials, using :func:`~gensim._matutils._logsumexp_2d`.

    Parameters
    ----------
    x : numpy.ndarray
        Input 2d matrix, supports float16, float32 and float64.

    Returns
    -------
    float
        log of sum of exponentials of elements in `x`.

    Warnings
    --------
    By performance reasons, doesn't support NaNs or 1d, 3d, etc arrays like :func:`scipy.special.logsumexp`.

    """

    if x.dtype == np.float64:
        return _logsumexp_2d[double](x)
    elif x.dtype == np.float32:
        return _logsumexp_2d[float](x)
    elif x.dtype == np.float16:
        return _logsumexp_2d[float](x.astype(np.float32))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_t _logsumexp_2d(DTYPE_t[:, :] data) nogil:
    """Log of sum of exponentials.

    Parameters
    ----------
    x : numpy.ndarray
        Input 2d matrix.

    Returns
    -------
    DTYPE_t
        log of sum of exponentials of elements in `data`.

    """

    cdef DTYPE_t max_val = data[0, 0]
    cdef DTYPE_t result = 0.0
    cdef size_t i
    cdef size_t j

    cdef size_t I = data.shape[0]
    cdef size_t J = data.shape[1]

    for i in range(I):
        for j in range(J):
            if data[i, j] > max_val:
                max_val = data[i, j]

    for i in range(I):
        for j in range(J):
            result += exp(data[i, j] - max_val)

    result = log(result) + max_val

    return result


def _dirichlet_expectation(alpha):
    """Expected value of log(theta) where theta is drawn from a Dirichlet distribution.
    Using :func:`~gensim._matutils.dirichlet_expectation_1d` or :func:`~gensim._matutils.dirichlet_expectation_2d`.

    Parameters
    ----------
    alpha : numpy.ndarray
        Dirichlet parameter 2d matrix or 1d vector, if 2d - each row is treated as a separate parameter vector,
        supports float16, float32 and float64.

    Returns
    -------
    numpy.ndarray
        Log of expected values, dimension same as `alpha.ndim`.

    """
    if alpha.ndim == 2:
        return dirichlet_expectation_2d(alpha)
    else:
        return dirichlet_expectation_1d(alpha)


def dirichlet_expectation_2d(alpha):
    """Expected value of log(theta) where theta is drawn from a Dirichlet distribution.
    Using :func:`~gensim._matutils._dirichlet_expectation_2d`.

    Parameters
    ----------
    alpha : numpy.ndarray
        Dirichlet parameter 2d matrix, each row is treated as a separate parameter vector,
        supports float16, float32 and float64.

    Returns
    -------
    numpy.ndarray
        Log of expected values, 2d matrix.

    """
    if alpha.dtype == np.float64:
        out = np.zeros(alpha.shape, dtype=alpha.dtype)
        _dirichlet_expectation_2d[double](alpha, out)
    elif alpha.dtype == np.float32:
        out = np.zeros(alpha.shape, dtype=alpha.dtype)
        _dirichlet_expectation_2d[float](alpha, out)
    elif alpha.dtype == np.float16:
        out = np.zeros(alpha.shape, dtype=np.float32)
        _dirichlet_expectation_2d[float](alpha.astype(np.float32), out)
        out = out.astype(np.float16)

    return out


def dirichlet_expectation_1d(alpha):
    """Expected value of log(theta) where theta is drawn from a Dirichlet distribution.
    Using :func:`~gensim._matutils._dirichlet_expectation_1d`.

    Parameters
    ----------
    alpha : numpy.ndarray
        Dirichlet parameter 1d vector, supports float16, float32 and float64.

    Returns
    -------
    numpy.ndarray
        Log of expected values, 1d vector.

    """
    if alpha.dtype == np.float64:
        out = np.zeros(alpha.shape, dtype=alpha.dtype)
        _dirichlet_expectation_1d[double](alpha, out)
    elif alpha.dtype == np.float32:
        out = np.zeros(alpha.shape, dtype=alpha.dtype)
        _dirichlet_expectation_1d[float](alpha, out)
    elif alpha.dtype == np.float16:
        out = np.zeros(alpha.shape, dtype=np.float32)
        _dirichlet_expectation_1d[float](alpha.astype(np.float32), out)
        out = out.astype(np.float16)

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _dirichlet_expectation_1d(DTYPE_t[:] alpha, DTYPE_t[:] out) nogil:
    """Expected value of log(theta) where theta is drawn from a Dirichlet distribution.

    Parameters
    ----------
    alpha : numpy.ndarray
        Dirichlet parameter 1d vector.

    out : numpy.ndarray
        Output array, contains log of expected values.

    """
    cdef DTYPE_t sum_alpha = 0.0
    cdef DTYPE_t psi_sum_alpha = 0.0
    cdef size_t i
    cdef size_t I = alpha.shape[0]

    for i in range(I):
        sum_alpha += alpha[i]

    psi_sum_alpha = _digamma(sum_alpha)

    for i in range(I):
        out[i] = _digamma(alpha[i]) - psi_sum_alpha


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _dirichlet_expectation_2d(DTYPE_t[:, :] alpha, DTYPE_t[:, :] out) nogil:
    """Expected value of log(theta) where theta is drawn from a Dirichlet distribution.

    Parameters
    ----------
    alpha : numpy.ndarray
        Dirichlet parameter matrix, each row is treated as a parameter vector for its own Dirichlet.

    out : numpy.ndarray
        Log of expected values, 2d matrix.

    """
    cdef DTYPE_t sum_alpha = 0.0
    cdef DTYPE_t psi_sum_alpha = 0.0
    cdef size_t i, j
    cdef size_t I = alpha.shape[0]
    cdef size_t J = alpha.shape[1]

    for i in range(I):
        sum_alpha = 0.0
        for j in range(J):
            sum_alpha += alpha[i, j]

        psi_sum_alpha = _digamma(sum_alpha)

        for j in range(J):
            out[i, j] = _digamma(alpha[i, j]) - psi_sum_alpha


def digamma(DTYPE_t x):
    """Digamma function for positive floats, using :func:`~gensim._matutils._digamma`.

    Parameters
    ----------
    x : float
        Positive value.

    Returns
    -------
    float
        Digamma(x).

    """
    return _digamma(x)


@cython.cdivision(True)
cdef inline DTYPE_t _digamma(DTYPE_t x,) nogil:
    """Digamma function for positive floats.

    Parameters
    ----------
    x : float
        Positive value.

    Notes
    -----

    Adapted from:

    * Authors:
        * Original FORTRAN77 version by Jose Bernardo.
        * C version by John Burkardt.

    * Reference: Jose Bernardo, Algorithm AS 103: Psi (Digamma) Function,
      Applied Statistics, Volume 25, Number 3, 1976, pages 315-317.

    * Licensing: This code is distributed under the GNU LGPL license.


    Returns
    -------
    float
        Digamma(x).

    """
    cdef DTYPE_t c = 8.5;
    cdef DTYPE_t euler_mascheroni = 0.57721566490153286060;
    cdef DTYPE_t r;
    cdef DTYPE_t value;
    cdef DTYPE_t x2;

    if ( x <= 0.000001 ):
        value = - euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x;
        return value;

    # Reduce to DIGAMA(X + N).
    value = 0.0;
    x2 = x;
    while ( x2 < c ):
        value = value - 1.0 / x2;
        x2 = x2 + 1.0;

    # Use Stirling's (actually de Moivre's) expansion.
    r = 1.0 / x2;
    value = value + log ( x2 ) - 0.5 * r;

    r = r * r;

    value = value \
        - r * ( 1.0 / 12.0  \
        - r * ( 1.0 / 120.0 \
        - r * ( 1.0 / 252.0 \
        - r * ( 1.0 / 240.0 \
        - r * ( 1.0 / 132.0 ) ) ) ) )

    return value;
