#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# distutils: language = c++

from libcpp.vector cimport vector
import cython

# Import both NumPy and the Cython declarations for NumPy
import numpy as np
cimport numpy as np


# Declare the interface to the C++ EMD library
# ============================================

cdef extern from "lib/emd_hat.hpp":
    cdef double emd_hat_gd_metric_double(vector[double],
                                         vector[double],
                                         vector[vector[double]],
                                         double) except +


# Define the API
# ==============

def emd(np.ndarray[np.float64_t, ndim=1, mode="c"] first_signature,
        np.ndarray[np.float64_t, ndim=1, mode="c"] second_signature,
        np.ndarray[np.float64_t, ndim=2, mode="c"] distance_matrix,
        extra_mass_penalty=-1):
    """
    Compute the Earth Mover's Distance between the given signatures with the
    given distance matrix.

    :param first_signature: A 1D numpy array of ``np.double``, of length
        :math:`N`.
    :param second_signature: A 1D numpy array of ``np.double``, also of length
        :math:`N`..
    :param distance_matrix: A 2D numpy array of ``np.double``, of size
        :math:`N \cross N`.
    :param extra_mass_penalty: The penalty for extra mass. If you want the
        resulting distance to be a metric, it should be at least half the
        diameter of the space (maximum possible distance between any two
        points). If you want partial matching you can set it to zero (but then
        the resulting distance is not guaranteed to be a metric). The default
        value is -1 which means the maximum value in the distance matrix is
        used.
    """
    return emd_hat_gd_metric_double(first_signature,
                                    second_signature,
                                    distance_matrix,
                                    extra_mass_penalty)
