#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s CORPUS_DENSE.mm CORPUS_SPARSE.mm [NUMDOCS]
    Run speed test of similarity queries. Only use the first NUMDOCS documents of \
each corpus for testing (or use all if no NUMDOCS is given).
    The two sample corpora can be downloaded from http://nlp.fi.muni.cz/projekty/gensim/wikismall.tgz

Example: ./simspeed.py wikismall.dense.mm wikismall.sparse.mm 5000
"""

import logging
import sys
import itertools
import os
import math
from time import time

import numpy as np

import gensim


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    corpus_dense = gensim.corpora.MmCorpus(sys.argv[1])
    corpus_sparse = gensim.corpora.MmCorpus(sys.argv[2])
    NUMTERMS = corpus_sparse.num_terms
    if len(sys.argv) > 3:
        NUMDOCS = int(sys.argv[3])
        corpus_dense = list(itertools.islice(corpus_dense, NUMDOCS))
        corpus_sparse = list(itertools.islice(corpus_sparse, NUMDOCS))

    # create the query index to be tested (one for dense input, one for sparse)
    index_dense = gensim.similarities.MatrixSimilarity(corpus_dense)
    index_sparse = gensim.similarities.SparseMatrixSimilarity(corpus_sparse, num_terms=NUMTERMS)

    density = 100.0 * index_sparse.index.nnz / (index_sparse.index.shape[0] * index_sparse.index.shape[1])

    # Difference between test #1 and test #3 is that the query in #1 is a gensim iterable
    # corpus, while in #3, the index is used directly (np arrays). So #1 is slower,
    # because it needs to convert sparse vecs to np arrays and normalize them to
    # unit length=extra work, which #3 avoids.
    query = list(itertools.islice(corpus_dense, 1000))
    logging.info("test 1 (dense): dense corpus of %i docs vs. index (%i documents, %i dense features)" %
                 (len(query), len(index_dense), index_dense.num_features))
    for chunksize in [1, 4, 8, 16, 64, 128, 256, 512, 1024]:
        start = time()
        if chunksize > 1:
            sims = []
            for chunk in gensim.utils.chunkize_serial(query, chunksize):
                sim = index_dense[chunk]
                sims.extend(sim)
        else:
            sims = [index_dense[vec] for vec in query]
        assert len(sims) == len(query)  # make sure we have one result for each query document
        taken = time() - start
        queries = math.ceil(1.0 * len(query) / chunksize)
        logging.info("chunksize=%i, time=%.4fs (%.2f docs/s, %.2f queries/s)" %
                     (chunksize, taken, len(query) / taken, queries / taken))

    # Same comment as for test #1 but vs. test #4.
    query = list(itertools.islice(corpus_sparse, 1000))
    logging.info("test 2 (sparse): sparse corpus of %i docs vs. sparse index (%i documents, %i features, %.2f%% density)" %
                 (len(query), len(corpus_sparse), index_sparse.index.shape[1], density))
    for chunksize in [1, 5, 10, 100, 500, 1000]:
        start = time()
        if chunksize > 1:
            sims = []
            for chunk in gensim.utils.chunkize_serial(query, chunksize):
                sim = index_sparse[chunk]
                sims.extend(sim)
        else:
            sims = [index_sparse[vec] for vec in query]
        assert len(sims) == len(query)  # make sure we have one result for each query document
        taken = time() - start
        queries = math.ceil(1.0 * len(query) / chunksize)
        logging.info("chunksize=%i, time=%.4fs (%.2f docs/s, %.2f queries/s)" %
                     (chunksize, taken, len(query) / taken, queries / taken))

    logging.info("test 3 (dense): similarity of all vs. all (%i documents, %i dense features)" %
                 (len(corpus_dense), index_dense.num_features))
    for chunksize in [0, 1, 4, 8, 16, 64, 128, 256, 512, 1024]:
        index_dense.chunksize = chunksize
        start = time()
        # `sims` stores the entire N x N sim matrix in memory!
        # this is not necessary, but i added it to test the accuracy of the result
        # (=report mean diff below)
        sims = [sim for sim in index_dense]
        taken = time() - start
        sims = np.asarray(sims)
        if chunksize == 0:
            logging.info("chunksize=%i, time=%.4fs (%.2f docs/s)" % (chunksize, taken, len(corpus_dense) / taken))
            unchunksizeed = sims
        else:
            queries = math.ceil(1.0 * len(corpus_dense) / chunksize)
            diff = np.mean(np.abs(unchunksizeed - sims))
            logging.info("chunksize=%i, time=%.4fs (%.2f docs/s, %.2f queries/s), meandiff=%.3e" %
                         (chunksize, taken, len(corpus_dense) / taken, queries / taken, diff))
        del sims

    index_dense.num_best = 10
    logging.info("test 4 (dense): as above, but only ask for the top-10 most similar for each document")
    for chunksize in [0, 1, 4, 8, 16, 64, 128, 256, 512, 1024]:
        index_dense.chunksize = chunksize
        start = time()
        sims = [sim for sim in index_dense]
        taken = time() - start
        if chunksize == 0:
            queries = len(corpus_dense)
        else:
            queries = math.ceil(1.0 * len(corpus_dense) / chunksize)
        logging.info("chunksize=%i, time=%.4fs (%.2f docs/s, %.2f queries/s)" %
                     (chunksize, taken, len(corpus_dense) / taken, queries / taken))
    index_dense.num_best = None

    logging.info("test 5 (sparse): similarity of all vs. all (%i documents, %i features, %.2f%% density)" %
                 (len(corpus_sparse), index_sparse.index.shape[1], density))
    for chunksize in [0, 5, 10, 100, 500, 1000, 5000]:
        index_sparse.chunksize = chunksize
        start = time()
        sims = [sim for sim in index_sparse]
        taken = time() - start
        sims = np.asarray(sims)
        if chunksize == 0:
            logging.info("chunksize=%i, time=%.4fs (%.2f docs/s)" % (chunksize, taken, len(corpus_sparse) / taken))
            unchunksizeed = sims
        else:
            queries = math.ceil(1.0 * len(corpus_sparse) / chunksize)
            diff = np.mean(np.abs(unchunksizeed - sims))
            logging.info("chunksize=%i, time=%.4fs (%.2f docs/s, %.2f queries/s), meandiff=%.3e" %
                         (chunksize, taken, len(corpus_sparse) / taken, queries / taken, diff))
        del sims

    index_sparse.num_best = 10
    logging.info("test 6 (sparse): as above, but only ask for the top-10 most similar for each document")
    for chunksize in [0, 5, 10, 100, 500, 1000, 5000]:
        index_sparse.chunksize = chunksize
        start = time()
        sims = [sim for sim in index_sparse]
        taken = time() - start
        if chunksize == 0:
            queries = len(corpus_sparse)
        else:
            queries = math.ceil(1.0 * len(corpus_sparse) / chunksize)
        logging.info("chunksize=%i, time=%.4fs (%.2f docs/s, %.2f queries/s)" %
                     (chunksize, taken, len(corpus_sparse) / taken, queries / taken))
    index_sparse.num_best = None

    logging.info("finished running %s" % program)
