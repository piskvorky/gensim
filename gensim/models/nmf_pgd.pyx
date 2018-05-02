# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: Timofey Yefimov

cimport cython
from libc.math cimport sqrt

def solve_h(double[:, :] h, double[:, :] Wt, double[:, :] r_diff, double eta):
    cdef Py_ssize_t n_components = Wt.shape[0]
    cdef Py_ssize_t n_features = Wt.shape[1]
    cdef Py_ssize_t n_samples = h.shape[1]
    cdef double violation = 0
    cdef double grad, projected_grad
    cdef Py_ssize_t sample_idx, component_idx, feature_idx

    with nogil:
        for sample_idx in range(n_samples):
            for component_idx in range(n_components):

                grad = 0

                for feature_idx in range(n_features):
                    grad += Wt[component_idx, feature_idx] * r_diff[feature_idx, sample_idx]

                grad *= eta

                projected_grad = min(0, grad) if h[component_idx, sample_idx] == 0 else grad

                violation += projected_grad ** 2

                h[component_idx, sample_idx] = max(h[component_idx, sample_idx] - grad, 0)

    return sqrt(violation)

def solve_r():
    pass