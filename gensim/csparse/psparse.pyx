#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
cimport cython
cimport numpy as np
import numpy as np
import scipy.sparse
from libc.stddef cimport ptrdiff_t
from cython.parallel import parallel, prange

#-----------------------------------------------------------------------------
# Headers
#-----------------------------------------------------------------------------

ctypedef int csi

ctypedef struct cs:
    # matrix in compressed-column or triplet form
    csi nzmax       # maximum number of entries
    csi m           # number of rows
    csi n           # number of columns
    csi *p          # column pointers (size n+1) or col indices (size nzmax)
    csi *i          # row indices, size nzmax
    double *x       # numerical values, size nzmax
    csi nz          # # of entries in triplet matrix, -1 for compressed-col
    
cdef extern csi cs_gaxpy (cs *A, double *x, double *y) nogil
cdef extern csi cs_print (cs *A, csi brief) nogil

assert sizeof(csi) == 4

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

@cython.boundscheck(False)
def pmultiply(X not None, np.ndarray[ndim=2, mode='fortran', dtype=np.float64_t] W not None):
    """Multiply a sparse CSC matrix by a dense matrix
    
    Parameters
    ----------
    X : scipy.sparse.csc_matrix
        A sparse matrix, of size N x M
    W : np.ndarray[dtype=float564, ndim=2, mode='fortran']
        A dense matrix, of size M x P. Note, W must be contiguous and in
        fortran (column-major) order. You can ensure this using
        numpy's `asfortranarray` function.
        
    Returns
    -------
    A : np.ndarray[dtype=float64, ndim=2, mode='fortran']
        A dense matrix, of size N x P, the result of multiplying X by W.
     
   Notes
    -----
    This function is parallelized over the columns of W using OpenMP. You
    can control the number of threads at runtime using the OMP_NUM_THREADS
    environment variable. The internal sparse matrix code is from CSPARSE, 
    a Concise Sparse matrix package. Copyright (c) 2006, Timothy A. Davis.
    http://www.cise.ufl.edu/research/sparse/CSparse, licensed under the
    GNU LGPL v2.1+.

    References
    ----------
    .. [1] Davis, Timothy A., "Direct Methods for Sparse Linear Systems
    (Fundamentals of Algorithms 2)," SIAM Press, 2006. ISBN: 0898716136
    """
    if X.shape[1] != W.shape[0]:
        raise ValueError('matrices are not aligned')
    
    cdef int i
    cdef cs csX
    cdef np.ndarray[double, ndim=2, mode='fortran'] result
    cdef np.ndarray[csi, ndim=1, mode = 'c'] indptr  = X.indptr
    cdef np.ndarray[csi, ndim=1, mode = 'c'] indices = X.indices
    cdef np.ndarray[double, ndim=1, mode = 'c']    data = X.data

    # Pack the scipy data into the CSparse struct. This is just copying some
    # pointers.
    csX.nzmax = X.data.shape[0]
    csX.m = X.shape[0]
    csX.n = X.shape[1]
    csX.p = &indptr[0]
    csX.i = &indices[0]
    csX.x = &data[0]
    csX.nz = -1  # to indicate CSC format
    
    result = np.zeros((X.shape[0], W.shape[1]), order='F', dtype=np.double)
    for i in prange(W.shape[1], nogil=True):
        # X is in fortran format, so we can get quick access to each of its
        # columns
        cs_gaxpy(&csX, &W[0, i], &result[0, i])
    
    return result
