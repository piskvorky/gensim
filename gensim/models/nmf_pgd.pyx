# Author: Timofey Yefimov

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

cimport cython
from libc.math cimport sqrt, fabs, fmin, fmax, copysign
from cython.parallel import prange
import numpy as np

# def solve_h_sparse(h, Wt_v_minus_r, WtW, double kappa):
#     cdef Py_ssize_t [:] h_indptr = h.indptr
#     cdef Py_ssize_t [:] h_indices = h.indices
#     cdef double [:] h_data = h.data
#
#     cdef Py_ssize_t [:] Wt_v_minus_r_indptr = Wt_v_minus_r.indptr
#     cdef Py_ssize_t [:] Wt_v_minus_r_indices = Wt_v_minus_r.indices
#     cdef double [:] Wt_v_minus_r_data = Wt_v_minus_r.data
#
#     cdef Py_ssize_t [:] WtW_indptr = WtW.indptr
#     cdef Py_ssize_t [:] WtW_indices = WtW.indices
#     cdef double [:] WtW_data = WtW.data
#
#     cdef Py_ssize_t n_components = h.shape[0]
#     cdef Py_ssize_t n_samples = h.shape[1]
#     cdef double violation = 0
#     cdef double grad, projected_grad, hessian
#     cdef Py_ssize_t sample_idx = 0
#     cdef Py_ssize_t component_idx_1 = 0
#     cdef Py_ssize_t component_idx_2 = 0
#
#
#     for sample_idx in range(n_samples):
#         for component_idx_1 in prange(n_components, nogil=True):
#
#             grad = -Wt_v_minus_r_data[component_idx_1, sample_idx]
#
#             for component_idx_2 in range(n_components):
#                 grad += WtW[component_idx_1, component_idx_2] * h[component_idx_2, sample_idx]
#
#             hessian = WtW[component_idx_1, component_idx_1]
#
#             grad = grad * kappa / hessian
#
#             projected_grad = fmin(0, grad) if h[component_idx_1, sample_idx] == 0 else grad
#
#             violation += projected_grad * projected_grad
#
#             h[component_idx_1, sample_idx] = fmax(h[component_idx_1, sample_idx] - grad, 0.)
#
#     h.eliminate_zeros()
#
#     return sqrt(violation)

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

def solve_r(
        int[::1] r_indptr,
        int[::1] r_indices,
        double[::1] r_data,
        int[::1] r_actual_indptr,
        int[::1] r_actual_indices,
        double[::1] r_actual_data,
        double lambda_,
        double v_max
    ):
    cdef Py_ssize_t n_samples = r_actual_indptr.shape[0] - 1
    cdef double violation = 0
    cdef double r_actual_sign = 1.0
    cdef Py_ssize_t sample_idx

    cdef Py_ssize_t r_col_size = 0
    cdef Py_ssize_t r_actual_col_size = 0
    cdef Py_ssize_t r_col_idx_idx = 0
    cdef Py_ssize_t r_actual_col_idx_idx

    cdef Py_ssize_t [:] r_col_indices = np.zeros(n_samples, dtype=np.intp)
    cdef Py_ssize_t [:] r_actual_col_indices = np.zeros(n_samples, dtype=np.intp)

    for sample_idx in prange(n_samples, nogil=True):
        r_col_size = r_indptr[sample_idx + 1] - r_indptr[sample_idx]
        r_actual_col_size = r_actual_indptr[sample_idx + 1] - r_actual_indptr[sample_idx]

        r_col_indices[sample_idx] = 0
        r_actual_col_indices[sample_idx] = 0

        while r_col_indices[sample_idx] < r_col_size or r_actual_col_indices[sample_idx] < r_actual_col_size:
            r_col_idx_idx = r_indices[
                r_indptr[sample_idx]
                + r_col_indices[sample_idx]
            ]
            r_actual_col_idx_idx = r_actual_indices[
                r_actual_indptr[sample_idx]
                + r_actual_col_indices[sample_idx]
            ]

            if r_col_idx_idx >= r_actual_col_idx_idx:
                r_actual_sign = copysign(
                    r_actual_sign,
                    r_actual_data[
                        r_actual_indptr[sample_idx]
                        + r_actual_col_indices[sample_idx]
                    ]
                )

                r_actual_data[
                    r_actual_indptr[sample_idx]
                    + r_actual_col_indices[sample_idx]
                ] = fabs(
                    r_actual_data[
                        r_actual_indptr[sample_idx]
                        + r_actual_col_indices[sample_idx]
                    ]
                ) - lambda_

                r_actual_data[
                    r_actual_indptr[sample_idx]
                    + r_actual_col_indices[sample_idx]
                ] = fmax(
                    r_actual_data[
                        r_actual_indptr[sample_idx]
                        + r_actual_col_indices[sample_idx]
                    ],
                    0
                )

                if (
                        r_actual_data[
                            r_actual_indptr[sample_idx]
                            + r_actual_col_indices[sample_idx]
                        ] != 0
                ):
                    r_actual_data[
                        r_actual_indptr[sample_idx]
                        + r_actual_col_indices[sample_idx]
                    ] = copysign(
                        r_actual_data[
                            r_actual_indptr[sample_idx]
                            + r_actual_col_indices[sample_idx]
                        ],
                        r_actual_sign
                    )

                    r_actual_data[
                        r_actual_indptr[sample_idx]
                        + r_actual_col_indices[sample_idx]
                    ] = fmax(
                        r_actual_data[
                            r_actual_indptr[sample_idx]
                            + r_actual_col_indices[sample_idx]
                        ],
                        -v_max
                    )

                    r_actual_data[
                        r_actual_indptr[sample_idx]
                        + r_actual_col_indices[sample_idx]
                    ] = fmin(
                        r_actual_data[
                            r_actual_indptr[sample_idx]
                            + r_actual_col_indices[sample_idx]
                        ],
                        v_max
                    )

                if r_col_idx_idx == r_actual_col_idx_idx:
                    violation += (
                        r_data[
                            r_indptr[sample_idx]
                            + r_col_indices[sample_idx]
                        ]
                        - r_actual_data[
                            r_actual_indptr[sample_idx]
                            + r_actual_col_indices[sample_idx]
                        ]
                    ) ** 2
                else:
                    violation += r_actual_data[
                        r_actual_indptr[sample_idx]
                        + r_actual_col_indices[sample_idx]
                    ] ** 2

                if r_actual_col_indices[sample_idx] < r_actual_col_size:
                    r_actual_col_indices[sample_idx] += 1
                else:
                    r_col_indices[sample_idx] += 1
            else:
                violation += r_data[
                    r_indptr[sample_idx]
                    + r_col_indices[sample_idx]
                ] ** 2

                if r_col_indices[sample_idx] < r_col_size:
                    r_col_indices[sample_idx] += 1
                else:
                    r_actual_col_indices[sample_idx] += 1

    return sqrt(violation)
