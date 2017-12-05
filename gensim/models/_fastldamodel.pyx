from __future__ import division
import numpy as np
cimport numpy as np
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
from libc.math cimport log, exp, fabs
cimport cython
from cython.parallel import prange


def digamma(DTYPE_t x):
    """
    Digamma function for positive floats

    Parameters
    ----------
    x : float32

    Returns
    -------
    digamma : float32
    
    """

    return _digamma(x)


@cython.cdivision(True)
cdef inline DTYPE_t _digamma (DTYPE_t x,) nogil:
    """ 
    Digamma over positive floats only

    Adapted from:

    Author:
        Original FORTRAN77 version by Jose Bernardo.
        C version by John Burkardt.

    Reference:
        Jose Bernardo,
        Algorithm AS 103:
        Psi ( Digamma ) Function,
        Applied Statistics,
        Volume 25, Number 3, 1976, pages 315-317.
    
    Licensing:
        This code is distributed under the GNU LGPL license. 

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef mean_absolute_difference(DTYPE_t[:] a, DTYPE_t[:] b):
    """
    Mean absolute difference between two arrays

    Parameters
    ----------
    a : array_like of float32
    b : array_like of float32

    Returns
    -------
    float32
        mean(abs(a - b))
    
    """

    cdef DTYPE_t result = 0.0
    cdef size_t i
    cdef size_t j

    cdef size_t I = a.shape[0]
    cdef size_t N = I

    if (a.shape[0] != b.shape[0]):
        raise ValueError("a and b must have same shape")

    for i in range(I):
        result += fabs(a[i] - b[i])
    result /= N
        
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef logsumexp_2d(DTYPE_t[:, :] data):
    """
    Log of sum of exponentials for 2d array
    
    Parameters
    ----------
    x : array_like of float32_t
    
    Returns
    -------
    float
        log of sum of exponentials of elements in `x`
 
    """
    
    cdef DTYPE_t max_val = data[0, 0]
    cdef DTYPE_t result = 0.0
    cdef size_t i
    cdef size_t j

    cdef size_t I = data.shape[0]
    cdef size_t J = data.shape[1]
    
    with nogil:
        for i in range(I):
            for j in range(J):
                if data[i, j] > max_val:
                    max_val = data[i, j]
        
        for i in range(I):
            for j in range(J):
                result += exp(data[i, j] - max_val)

        result = log(result) + max_val

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dirichlet_expectation_1d(DTYPE_t[:] alpha):
    """
    Expected value of log(theta) where theta is drawn from a Dirichlet distribution
    
    Parameters
    ----------
    alpha : 1d array_like
        Dirichlet parameter vector
    
    Returns
    -------
    1d array_like
        log of expected values

    """

    cdef DTYPE_t sum_alpha = 0.0
    cdef DTYPE_t psi_sum_alpha = 0.0
    cdef size_t i
    cdef size_t I = alpha.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros([I], dtype=DTYPE)    

    for i in range(I):
        sum_alpha += alpha[i]

    psi_sum_alpha = _digamma(sum_alpha)

    for i in range(I):
        result[i] = _digamma(alpha[i]) - psi_sum_alpha

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dirichlet_expectation_2d(DTYPE_t[:, :] alpha):
    """
    Expected value of log(theta) where theta is drawn from a Dirichlet distribution
    
    Parameters
    ----------
    alpha : 2d array_like
        Dirichlet parameter vector.  
        Each row is treated as a parameter vector for its own Dirichlet
    
    Returns
    -------
    2d array_like
        log of expected values

    """

    cdef DTYPE_t sum_alpha = 0.0
    cdef DTYPE_t psi_sum_alpha = 0.0
    cdef size_t i, j
    cdef size_t I = alpha.shape[0]
    cdef size_t J = alpha.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros([I, J], dtype=DTYPE)    

    for i in range(I):
        sum_alpha = 0.0
        for j in range(J):
            sum_alpha += alpha[i, j]

        psi_sum_alpha = _digamma(sum_alpha)

        for j in range(J):
            result[i, j] = _digamma(alpha[i, j]) - psi_sum_alpha

    return result
