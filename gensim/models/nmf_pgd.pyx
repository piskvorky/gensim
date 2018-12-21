# Author: Timofey Yefimov

# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=True

from libc.math cimport sqrt, fabs, copysign
from cython.parallel import prange

cdef double fmin(double x, double y) nogil:
    return x if x < y else y

cdef double fmax(double x, double y) nogil:
    return x if x > y else y

cdef double clip(double a, double a_min, double a_max) nogil:
    a = fmin(a, a_max)
    a = fmax(a, a_min)
    return a

def solve_h(double[:, ::1] h, double[:, :] Wt_v_minus_r, double[:, ::1] WtW, double kappa):
    """Find optimal dense vector representation for current W and r matrices.

    Parameters
    ----------
    h : matrix
        Dense representation of documents in current batch.
    Wt_v_minus_r : matrix
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
        r,
        r_actual,
        double lambda_,
        double v_max
    ):
    """Bound new residuals.

    Parameters
    ----------
    r: sparse matrix
    r_actual: sparse matrix
    lambda_ : double
    v_max : double

    Returns
    -------
    float
        Cumulative difference between previous and current residuals vectors.

    """

    cdef int[::1] r_indptr = r.indptr
    cdef int[::1] r_indices = r.indices
    cdef double[::1] r_data = r.data
    cdef int[::1] r_actual_indptr = r_actual.indptr
    cdef int[::1] r_actual_indices = r_actual.indices
    cdef double[::1] r_actual_data = r_actual.data

    cdef Py_ssize_t r_col_size = 0
    cdef Py_ssize_t r_actual_col_size = 0
    cdef Py_ssize_t r_col_indptr
    cdef Py_ssize_t r_actual_col_indptr
    cdef Py_ssize_t r_col_idx
    cdef Py_ssize_t r_actual_col_idx
    cdef double* r_element
    cdef double* r_actual_element

    cdef double r_actual_sign = 1.0

    cdef Py_ssize_t n_samples = r_actual_indptr.shape[0] - 1
    cdef Py_ssize_t sample_idx

    cdef double violation = 0

    for sample_idx in prange(n_samples, nogil=True):
        r_col_size = r_indptr[sample_idx + 1] - r_indptr[sample_idx]
        r_actual_col_size = r_actual_indptr[sample_idx + 1] - r_actual_indptr[sample_idx]

        r_col_idx = 0
        r_actual_col_idx = 0

        while r_col_idx < r_col_size or r_actual_col_idx < r_actual_col_size:
            r_col_indptr = r_indices[
                r_indptr[sample_idx]
                + r_col_idx
            ]
            r_actual_col_indptr = r_actual_indices[
                r_actual_indptr[sample_idx]
                + r_actual_col_idx
            ]

            r_element = &r_data[
                r_indptr[sample_idx]
                + r_col_idx
            ]
            r_actual_element = &r_actual_data[
                r_actual_indptr[sample_idx]
                + r_actual_col_idx
            ]

            if r_col_indptr >= r_actual_col_indptr:
                r_actual_sign = copysign(r_actual_sign, r_actual_element[0])

                r_actual_element[0] = fabs(r_actual_element[0]) - lambda_
                r_actual_element[0] = fmax(r_actual_element[0], 0)

                if r_actual_element[0] != 0:
                    r_actual_element[0] = copysign(r_actual_element[0], r_actual_sign)
                    r_actual_element[0] = clip(r_actual_element[0], -v_max, v_max)

                if r_col_indptr == r_actual_col_indptr:
                    violation += (r_element[0] - r_actual_element[0]) ** 2
                else:
                    violation += r_actual_element[0] ** 2

                if r_actual_col_idx < r_actual_col_size:
                    r_actual_col_idx = r_actual_col_idx + 1
                else:
                    r_col_idx = r_col_idx + 1
            else:
                violation += r_element[0] ** 2

                if r_col_idx < r_col_size:
                    r_col_idx = r_col_idx + 1
                else:
                    r_actual_col_idx = r_actual_col_idx + 1

    return sqrt(violation)
