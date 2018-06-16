# Author: Timofey Yefimov

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: linetrace=False

cimport cython
from libc.math cimport sqrt, fabs, fmin, fmax, copysign

def solve_h(double[:, ::1] h, double[:, :] Wt_v_minus_r, double[:, ::1] WtW, double kappa):
    cdef Py_ssize_t n_components = h.shape[0]
    cdef Py_ssize_t n_samples = h.shape[1]
    cdef double violation = 0
    cdef double grad, projected_grad, hessian
    cdef Py_ssize_t sample_idx, component_idx_1, component_idx_2

    with nogil:
        for component_idx_1 in range(n_components):
            for sample_idx in range(n_samples):

                grad = -Wt_v_minus_r[component_idx_1, sample_idx]

                for component_idx_2 in range(n_components):
                    grad += WtW[component_idx_1, component_idx_2] * h[component_idx_2, sample_idx]

                hessian = WtW[component_idx_1, component_idx_1]

                h[component_idx_1, sample_idx] = fmax(h[component_idx_1, sample_idx] - grad * kappa / hessian, 0.)

def solve_r(double[:, ::1] r, double[:, ::1] r_actual, double lambda_, double v_max):
    cdef Py_ssize_t n_features = r.shape[0]
    cdef Py_ssize_t n_samples = r.shape[1]
    cdef double violation = 0
    cdef double r_new_element

    with nogil:
        for sample_idx in range(n_samples):
            for feature_idx in range(n_features):
                r_new_element = fabs(r_actual[feature_idx, sample_idx]) - lambda_
                r_new_element = fmax(r_new_element, 0)
                r_new_element = copysign(r_new_element, r_actual[feature_idx, sample_idx])
                r_new_element = fmax(r_new_element, -v_max)
                r_new_element = fmin(r_new_element, v_max)

                r[feature_idx, sample_idx] = r_new_element
