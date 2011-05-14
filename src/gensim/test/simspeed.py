#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>

"""
USAGE: %(program)s CORPUS_DENSE.mm CORPUS_SPARSE.mm [NUMDOCS]
    Run speed test of similarity queries. Only use the first NUMDOCS documents of \
each corpus for testing (or use all if no NUMDOCS is given).

Example: ./simspeed.py wikismall.dense.mm wikismall.sparse.mm 5000
"""

import logging
import sys
import itertools
import os
import math
from time import time

import numpy
import scipy.sparse

import gensim


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=10)
    logging.info("running %s" % " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    if len(sys.argv) > 3:
        NUMDOCS = int(sys.argv[3])
        corpus_dense = list(itertools.islice(gensim.corpora.MmCorpus(sys.argv[1]), NUMDOCS))
        corpus_sparse = list(itertools.islice(gensim.corpora.MmCorpus(sys.argv[2]), NUMDOCS))
    else:
        corpus_dense = gensim.corpora.MmCorpus(sys.argv[1])
        corpus_sparse = gensim.corpora.MmCorpus(sys.argv[2])

    # create the query index to be tested (one for dense input, one for sparse)
    index_dense = gensim.similarities.MatrixSimilarity(corpus_dense)
    index_sparse = gensim.similarities.SparseMatrixSimilarity(corpus_sparse)

    logging.info("test 1: similarity of all vs. all, %i documents, %i features" %
                 (len(corpus_dense), index_dense.numFeatures))
    for chunks in [0, 1, 5, 10, 100, 200, 500, 1000]:
        index_dense.chunks = chunks
        start = time()
        # `sims` stores the entire N x N sim matrix in memory!
        # this is not necessary, but i added it to test the accuracy of the result
        # (=report mean diff below)
        sims = [sim for sim in index_dense]
        taken = time() - start
        sims = numpy.asarray(sims)
        if chunks == 0:
            logging.info("chunks=%i, time=%.4fs (%.2f docs/s)" % (chunks, taken, len(corpus_dense) / taken))
            unchunked = sims
        else:
            queries = math.ceil(1.0 * len(corpus_dense) / chunks)
            diff = numpy.mean(numpy.abs(unchunked - sims))
            logging.info("chunks=%i, time=%.4fs (%.2f docs/s, %.2f queries/s), meandiff=%.3e" %
                         (chunks, taken, len(corpus_dense) / taken, queries / taken, diff))
        del sims

    index_dense.numBest = 10
    logging.info("test 2: as above, but only ask for top-10 most similar for each document")
    for chunks in [0, 1, 5, 10, 100, 200, 500, 1000]:
        index_dense.chunks = chunks
        start = time()
        sims = [sim for sim in index_dense]
        taken = time() - start
        if chunks == 0:
            queries = len(corpus_dense)
        else:
            queries = math.ceil(1.0 * len(corpus_dense) / chunks)
        logging.info("chunks=%i, time=%.4fs (%.2f docs/s, %.2f queries/s)" %
                     (chunks, taken, len(corpus_dense) / taken, queries / taken))

    density = 100.0 * index_sparse.corpus.nnz / (index_sparse.corpus.shape[0] * index_sparse.corpus.shape[1])
    logging.info("test 3: sparse all vs. all, %i documents, %i features, %.2f%% density" %
                 (len(corpus_sparse), index_sparse.corpus.shape[1], density))
    for chunks in [0, 5, 10, 100, 500, 1000, 5000]:
        index_sparse.chunks = chunks
        start = time()
        sims = [sim for sim in index_sparse]
        taken = time() - start
        sims = numpy.asarray(sims)
        if chunks == 0:
            logging.info("chunks=%i, time=%.4fs (%.2f docs/s)" % (chunks, taken, len(corpus_sparse) / taken))
            unchunked = sims
        else:
            queries = math.ceil(1.0 * len(corpus_sparse) / chunks)
            diff = numpy.mean(numpy.abs(unchunked - sims))
            logging.info("chunks=%i, time=%.4fs (%.2f docs/s, %.2f queries/s), meadiff=%.3e" %
                         (chunks, taken, len(corpus_sparse) / taken, queries / taken, diff))
        del sims

    logging.info("finished running %s" % program)
