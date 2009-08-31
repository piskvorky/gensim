#!/usr/bin/env python2.5

import logging

logging.root.level = 10

import common
import pylab
import numpy
import matutils

mat = matutils.loadMatrix(common.matrixFile("gensim_engTFIDFsim.mm"))
#mat = numpy.zeros((10, 20), float)
#mat[2] = 0.2
#mat[8] = 0.9
logging.info("%ix%i matrix loaded" % mat.shape)
pylab.figure()
pylab.imshow(mat, cmap = pylab.cm.gray, interpolation = "nearest")
pylab.savefig("tfidf_sim")
