# Author: Timofey Yefimov

# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=True

from libc.math cimport sqrt
from cython.parallel import prange

cdef double fmin(double x, double y) nogil:
    return x if x < y else y

cdef double fmax(double x, double y) nogil:
    return x if x > y else y

def solve_h(double[:, ::1] h, double[:, :] Wtv, double[:, ::1] WtW, int[::1] permutation, double kappa):
    """Find optimal dense vector representation for current W and r matrices.

    Parameters
    ----------
    h : matrix
        Dense representation of documents in current batch.
    Wtv : matrix
    WtW : matrix

    Returns
    -------
    float
        Cumulative difference between previous and current h vectors.

    """

    cdef Py_ssize_t n_components = h.shape[0]
    cdef Py_ssize_t n_samples = h.shape[1]
    cdef double violation = 0
    cdef double grad, projected_grad, hessian
    cdef Py_ssize_t sample_idx = 0
    cdef Py_ssize_t component_idx_1 = 0
    cdef Py_ssize_t component_idx_2 = 0

    for sample_idx in prange(n_samples, nogil=True):
        for component_idx_1 in range(n_components):
            component_idx_1 = permutation[component_idx_1]

            grad = -Wtv[component_idx_1, sample_idx]

            for component_idx_2 in range(n_components):
                grad += WtW[component_idx_1, component_idx_2] * h[component_idx_2, sample_idx]

            hessian = WtW[component_idx_1, component_idx_1]

            grad = grad * kappa / hessian

            projected_grad = fmin(0, grad) if h[component_idx_1, sample_idx] == 0 else grad

            violation += projected_grad * projected_grad

            h[component_idx_1, sample_idx] = fmax(h[component_idx_1, sample_idx] - grad, 0.)

    return sqrt(violation)
