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


def tensorFromMtx(fname, transposed = False):
    """
    Read sparse matrix from fname (coordinate matrix market format) and return it 
    as sparse divisi tensor.
    
    If transposed is set, return the 2d tensor transposed.
    """
    result = divisi.DictTensor(ndim = 2)
    first = True
    i, j, li, lj = 0, 0, -1, -1
    for linenum, line in enumerate(open(fname)):
        if linenum == 0:
            if not 'coordinate' in line:
                raise ValueError('invalid matrix type (expected coordinate format) in %s' %
                                 fname)
        if line.startswith('%'):
            continue
        if first:
            info = line.split()
            docs, terms, nnz = int(info[0]), int(info[1]), int(info[2])
            logging.info("reading %i non-zero entries of %ix%i matrix from %s" %
                         (nnz, docs, terms, fname))
            first = False
            continue
        parts = line.split()
        if li != i and i % 1000 == 0:
            logging.info("PROGRESS: at item %i/%i" % (i, docs))
        li, lj = i, j
        i, j = int(parts[0]), int(parts[1])
        if transposed:
            i1, i2 = j, i
        else:
            i1, i2 = i, j
        value = float(parts[2])
        result[(i1 - 1, i2 - 1)] = value # -1 because matrix market format starts numbering from 1
    return result


def iterateCsc(mat):
    """
    Iterate over CSC matrix, returning (key, value) pairs, where key = (row, column) 2-tuple.
    Ie. simulate mat.iteritems() as it should have been in scipy.sparse...
    
    Depends on scipy.sparse.csc_matrix implementation details!
    """
    if not isinstance(mat, scipy.sparse.csc_matrix):
        raise TypeError("iterateCsc expects a CSC matrix on input!")
    for col in xrange(mat.shape[1]):
        if col % 1000 == 0:
            logging.debug("iterating over column %i/%i" % (col, mat.shape[1]))
        for i in xrange(mat.indptr[col], mat.indptr[col + 1]):
            row, value = mat.indices[i], mat.data[i]
            yield (row, col), value


def toTensor(sparseMat):
    logging.info("creating divisi sparse tensor of shape %s" % (sparseMat.shape,))
    sparseTensor = divisi.DictTensor(ndim = 2)
    sparseTensor.update(iterateCsc(sparseMat.tocsc()))
    return sparseTensor


def doSVD(sparseTensor, num):
    """
    Do Singular Value Decomposition on mat using an external program (SVDLIBC via divisi).
     
    mat is a sparse matrix in divisi tensor format.
    
    Returns return the triple (U, S, VT) = (left eigenvectors, singular values, right eigenvectors)
    of num most greatest factors.
    """
    logging.info("computing sparse svd of %s matrix" % (sparseTensor.shape,))
    svdResult = divisi.svd.svd_sparse(sparseTensor, k = num)
    
    # convert sparse tensors (result of sparse_svd in divisi) back to standard numpy arrays
    u = svdResult.u.unwrap()
    v = svdResult.v.unwrap()
    s = svdResult.svals.unwrap()
    logging.info("result of svd: u=%s, s=%s, v=%s" % (u.shape, s.shape, v.shape))
    assert len(s) <= num # make sure we didn't get more than we asked for
    assert len(s) == u.shape[1] == v.shape[1] # make sure the dimensions fit
    return u, s, v.T


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
