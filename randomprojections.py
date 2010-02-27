#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import logging
from itertools import izip
import math

import scipy
import scipy.sparse
import numpy
import numpy.random


def getRPVector(dim):
    result = numpy.empty((dim,), dtype = numpy.int8)
    for i in xrange(dim):
        f = numpy.random.random()
        if f < 1.0 / 6:
            result[i] = +1 # with probability 1/6 
        elif f < 2.0 / 6:
            result[i] = -1 # with probability 1/6
        else:
            result[i] = 0 # rest = probability 2/3
#    logging.debug("created random projection vector with sum=%i" % numpy.sum(result))
    return result

def getRPMatrix(oldDim, newDim):
    """
    Return random projection matrix in compressed sparse row format
    """
    logging.info("building random projection matrix of size %ix%i" % (oldDim, newDim))
    result = scipy.sparse.dok_matrix((newDim, oldDim), dtype = numpy.float32)
    for i in xrange(oldDim):
        if i % 10000 == 0:
            logging.info('at term #%i/%i' % (i, oldDim))
        for j, f in izip(xrange(newDim), numpy.random.random(newDim)):
            if f > 0.33333333333333333:
                continue # with prob = 2/3, insert 0.0 = noop in sparse matrices
            if f > 0.16666666666666666:
                result[j, i] = -1 # with probability 1/6
            else:
                result[j, i] = +1 # with probability 1/6 
    logging.debug('converting sparse projection to csr format')
    return result.tocsr()
