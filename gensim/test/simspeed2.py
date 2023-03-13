#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""
USAGE: %(program)s CORPUS_DENSE.mm CORPUS_SPARSE.mm [NUMDOCS]
    Run speed test of similarity queries. Only use the first NUMDOCS documents of \
each corpus for testing (or use all if no NUMDOCS is given).
    The two sample corpora can be downloaded from http://nlp.fi.muni.cz/projekty/gensim/wikismall.tgz

Example: ./simspeed2.py wikismall.dense.mm wikismall.sparse.mm
"""

import logging
import sys
import itertools
import os
import math
from time import time

import gensim


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s", " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    corpus_dense = gensim.corpora.MmCorpus(sys.argv[1])
    corpus_sparse = gensim.corpora.MmCorpus(sys.argv[2])
    dense_features, sparse_features = corpus_dense.num_terms, corpus_sparse.num_terms
    if len(sys.argv) > 3:
        NUMDOCS = int(sys.argv[3])
        corpus_dense = list(itertools.islice(corpus_dense, NUMDOCS))
        corpus_sparse = list(itertools.islice(corpus_sparse, NUMDOCS))

    # create the query index to be tested (one for dense input, one for sparse)
    index_dense = gensim.similarities.Similarity('/tmp/tstdense', corpus_dense, dense_features)
    index_sparse = gensim.similarities.Similarity('/tmp/tstsparse', corpus_sparse, sparse_features)

    density = 100.0 * sum(shard.num_nnz for shard in index_sparse.shards) / (len(index_sparse) * sparse_features)

    logging.info(
        "test 1 (dense): similarity of all vs. all (%i documents, %i dense features)",
        len(corpus_dense), index_dense.num_features
    )
    for chunksize in [1, 8, 32, 64, 128, 256, 512, 1024, index_dense.shardsize]:
        index_dense.chunksize = chunksize
        start = time()
        for sim in index_dense:
            pass
        taken = time() - start
        queries = math.ceil(1.0 * len(corpus_dense) / chunksize)
        logging.info(
            "chunksize=%i, time=%.4fs (%.2f docs/s, %.2f queries/s)",
            chunksize, taken, len(corpus_dense) / taken, queries / taken
        )

    index_dense.num_best = 10
    logging.info("test 2 (dense): as above, but only ask for the top-10 most similar for each document")
    for chunksize in [1, 8, 32, 64, 128, 256, 512, 1024, index_dense.shardsize]:
        index_dense.chunksize = chunksize
        start = time()
        sims = [sim for sim in index_dense]
        taken = time() - start
        queries = math.ceil(1.0 * len(corpus_dense) / chunksize)
        logging.info(
            "chunksize=%i, time=%.4fs (%.2f docs/s, %.2f queries/s)",
            chunksize, taken, len(corpus_dense) / taken, queries / taken
        )
    index_dense.num_best = None

    logging.info(
        "test 3 (sparse): similarity of all vs. all (%i documents, %i features, %.2f%% density)",
        len(corpus_sparse), index_sparse.num_features, density
    )

    for chunksize in [1, 5, 10, 100, 256, 500, 1000, index_sparse.shardsize]:
        index_sparse.chunksize = chunksize
        start = time()
        for sim in index_sparse:
            pass
        taken = time() - start
        queries = math.ceil(1.0 * len(corpus_sparse) / chunksize)
        logging.info(
            "chunksize=%i, time=%.4fs (%.2f docs/s, %.2f queries/s)",
            chunksize, taken, len(corpus_sparse) / taken, queries / taken
        )

    index_sparse.num_best = 10
    logging.info("test 4 (sparse): as above, but only ask for the top-10 most similar for each document")
    for chunksize in [1, 5, 10, 100, 256, 500, 1000, index_sparse.shardsize]:
        index_sparse.chunksize = chunksize
        start = time()
        for sim in index_sparse:
            pass
        taken = time() - start
        queries = math.ceil(1.0 * len(corpus_sparse) / chunksize)
        logging.info(
            "chunksize=%i, time=%.4fs (%.2f docs/s, %.2f queries/s)",
            chunksize, taken, len(corpus_sparse) / taken, queries / taken
        )
    index_sparse.num_best = None

    # Difference between test #5 and test #1 is that the query in #5 is a gensim iterable
    # corpus, while in #1, the index is used directly (numpy arrays). So #5 is slower,
    # because it needs to convert sparse vecs to numpy arrays and normalize them to
    # unit length=extra work, which #1 avoids.
    query = list(itertools.islice(corpus_dense, 1000))
    logging.info(
        "test 5 (dense): dense corpus of %i docs vs. index (%i documents, %i dense features)",
        len(query), len(index_dense), index_dense.num_features
    )
    for chunksize in [1, 8, 32, 64, 128, 256, 512, 1024]:
        start = time()
        if chunksize > 1:
            sims = []
            for chunk in gensim.utils.chunkize_serial(query, chunksize):
                _ = index_dense[chunk]
        else:
            for vec in query:
                _ = index_dense[vec]
        taken = time() - start
        queries = math.ceil(1.0 * len(query) / chunksize)
        logging.info(
            "chunksize=%i, time=%.4fs (%.2f docs/s, %.2f queries/s)",
            chunksize, taken, len(query) / taken, queries / taken
        )

    # Same comment as for test #5.
    query = list(itertools.islice(corpus_dense, 1000))
    logging.info(
        "test 6 (sparse): sparse corpus of %i docs vs. sparse index (%i documents, %i features, %.2f%% density)",
        len(query), len(corpus_sparse), index_sparse.num_features, density
    )
    for chunksize in [1, 5, 10, 100, 500, 1000]:
        start = time()
        if chunksize > 1:
            sims = []
            for chunk in gensim.utils.chunkize_serial(query, chunksize):
                _ = index_sparse[chunk]
        else:
            for vec in query:
                _ = index_sparse[vec]
        taken = time() - start
        queries = math.ceil(1.0 * len(query) / chunksize)
        logging.info(
            "chunksize=%i, time=%.4fs (%.2f docs/s, %.2f queries/s)",
            chunksize, taken, len(query) / taken, queries / taken
        )

    logging.info("finished running %s", program)
