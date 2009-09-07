#!/usr/bin/env python

import os
import os.path
import random
import string
import logging
import tempfile
from itertools import izip

import numpy
import numpy.random
import scipy
import scipy.io.mmio

from csc import divisi

import matutils


UID_LENGTH = 8

def toTensor(mat):
    logging.info("creating divisi sparse tensor of shape %s" % (mat.shape,))
    sparseTensor = divisi.DictTensor(ndim = 2)
    sparseTensor.update(iterateCsc(mat.tocsc()))
    return sparseTensor

def iterateCsc(mat):
    """
    Iterate over CSC matrix, returning (key, value) pairs, where key = (row, column) 2-tuple.
    Ie. simulate mat.iteritems() as it should have been in scipy.sparse...
    
    Depends on scipy.sparse.csc_matrix implementation details!
    """
    if not isinstance(mat, scipy.sparse.csc_matrix):
        raise TypeError("iterateCsc expects an CSC matrix on input!")
    for col in xrange(mat.shape[1]):
        if col % 1000 == 0:
            logging.debug("iterating over column %i/%i" % (col, mat.shape[1]))
        for i in xrange(mat.indptr[col], mat.indptr[col + 1]):
            row, value = mat.indices[i], mat.data[i]
            yield (row, col), value


def doSVD(sparseTensor, num, val_only = False):
    """
    Do Singular Value Decomposition on mat using an external program (LIBSVDC via divisi). 
    mat is a scipy.sparse matrix in CSC format.
    Returns:
    - array of 'num' largest singular values if val_only is set
    - otherwise return the triple (U, S, VT) = (left eigenvectors, singular values, right eigenvectors)
    """
    logging.info("computing sparse svd of %s matrix" % (sparseTensor.shape,))
    svdResult = divisi.svd.svd_sparse(sparseTensor, k = num)
    
    # convert sparse tensors (result of sparse_svd in divisi) back to standard numpy arrays
    u = svdResult.u.unwrap()
    v = svdResult.v.unwrap()
    s = svdResult.svals.unwrap()
    logging.info("result of svd: u=%s, s=%s, v=%s" % (u.shape, s.shape, v.shape))
    assert len(s) <= num # make sure we didn't get more than we asked for
    
    if not val_only:
        assert len(s) == u.shape[1] == v.shape[1] # make sure the dimensions fit
        return u, s, v.T
    else:
        return s


if __name__ == '__main__':
    
    logging.basicConfig(level = logging.DEBUG)

    #mat = '/home/radim/mats/tstp.mm'
    #mat = '/home/radim/svd/mats/20n.mm'    
    #mat = scipy.io.mmio.mmread(mat)

    mat = numpy.random.random((200, 100))
#
#    if isinstance(mat, scipy.sparse.spmatrix):
#        mat = mat.todense()
        
    print 'input mat:', repr(mat)
    NUMDIM = 500
    result = doSVD(mat, NUMDIM)
    if result == None:
        logging.critical('SVD failed')
        raise ":("
    S, U, VT = result
    print 'S:', S
    print 'U:', U.shape#, U
    print 'V^T:', VT.shape#, VT
#    full = numpy.dot(U, numpy.dot(numpy.diag(S), VT))
#    print 'MSE(U*S*V^T, A) =', matutils.mse(full, mat)
#    print 'diff =', numpy.sum(full-mat)
    
#    import lsi
#    lsie = lsi.LSIEngine(mat, NUMDIM)
#    print 'S:', numpy.diag(lsie.S)
#    print 'U:', lsie.U.shape#, lsie.U
#    print 'V^T:', lsie.VT.shape#, lsie.VT
#    full = numpy.dot(lsie.U, numpy.dot(lsie.S, lsie.VT))
#    print 'MSE(U*S*V^T, A) =', matutils.mse(full, mat)
#    print 'diff =', numpy.sum(full-mat)
