from __future__ import division
import numpy as np
cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t
from libc.math cimport log, exp
cimport cython


def digamma(DTYPE_t x):
    return _digamma(x)

@cython.cdivision(True)
cdef DTYPE_t _digamma (DTYPE_t x,) nogil:
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
cpdef logsumexp(DTYPE_t[:] data):
    cdef DTYPE_t max_val = data[0]
    cdef DTYPE_t result = 0.0
    cdef size_t i

    cdef size_t imax = data.shape[0]

    for i in range(1, imax):
        if data[i] > max_val:
            max_val = data[i]
    
    for i in range(imax):
        result += exp(data[i] - max_val)

    result = log(result) + max_val
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dirichlet_expectation_1d(DTYPE_t[:] alpha):
    cdef DTYPE_t sum_alpha
    cdef DTYPE_t psi_sum_alpha
    cdef size_t i
    cdef size_t I = alpha.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros([I], dtype=DTYPE)    

    for i in range(I):
        sum_alpha += alpha[i]

    psi_sum_alpha = _digamma(sum_alpha)

    for i in range(I):
        result[i] = _digamma(alpha[i]) - psi_sum_alpha

    return result
