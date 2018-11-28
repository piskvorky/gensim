# Author: Timofey Yefimov

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

from libc.math cimport sqrt, fabs, fmin, fmax, copysign
from cython.parallel import prange

def solve_h(double[:, ::1] h, double[:, :] Wt_v_minus_r, double[:, ::1] WtW, double kappa):
    cdef Py_ssize_t n_components = h.shape[0]
    cdef Py_ssize_t n_samples = h.shape[1]
    cdef double violation = 0
    cdef double grad, projected_grad, hessian
    cdef Py_ssize_t sample_idx = 0
    cdef Py_ssize_t component_idx_1 = 0
    cdef Py_ssize_t component_idx_2 = 0

    for sample_idx in prange(n_samples, nogil=True):
        for component_idx_1 in range(n_components):

            grad = -Wt_v_minus_r[component_idx_1, sample_idx]

            for component_idx_2 in range(n_components):
                grad += WtW[component_idx_1, component_idx_2] * h[component_idx_2, sample_idx]

            hessian = WtW[component_idx_1, component_idx_1]

            grad = grad * kappa / hessian

            projected_grad = fmin(0, grad) if h[component_idx_1, sample_idx] == 0 else grad

            violation += projected_grad * projected_grad

            h[component_idx_1, sample_idx] = fmax(h[component_idx_1, sample_idx] - grad, 0.)

    return sqrt(violation)
