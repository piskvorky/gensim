#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>


"""USAGE: %(program)s MATRIX.mm [CLIP_DOCS] [CLIP_TERMS]

Check truncated SVD error for the algo in gensim, using a given corpus. This script
runs the decomposition with several internal parameters (number of requested factors,
iterative chunk size) and reports error for each parameter combination.

The number of input documents is clipped to the first CLIP_DOCS. Similarly,
only the first CLIP_TERMS are considered (features with id >= CLIP_TERMS are
ignored, effectively restricting the vocabulary size). If you don't specify them,
the entire matrix will be used.

Example: ./svd_error.py ~/gensim/results/wiki_en_v10k.mm.bz2 100000 10000
"""

from __future__ import print_function, with_statement

import logging
import os
import sys
import time
import bz2
import itertools

import numpy as np
import scipy.linalg

import gensim

try:
    from sparsesvd import sparsesvd
except ImportError:
    # no SVDLIBC: install with `easy_install sparsesvd` if you want SVDLIBC results as well
    sparsesvd = None

sparsesvd = None  # don't use SVDLIBC

FACTORS = [300]  # which num_topics to try
CHUNKSIZE = [10000, 1000]  # which chunksize to try
POWER_ITERS = [0, 1, 2, 4, 6]  # extra power iterations for the randomized algo

# when reporting reconstruction error, also report spectral norm error? (very slow)
COMPUTE_NORM2 = False


def norm2(a):
    """Spectral norm ("norm 2") of a symmetric matrix `a`."""
    if COMPUTE_NORM2:
        logging.info("computing spectral norm of a %s matrix", str(a.shape))
        return scipy.linalg.eigvalsh(a).max()  # much faster than np.linalg.norm(2)
    else:
        return np.nan


def rmse(diff):
    return np.sqrt(1.0 * np.multiply(diff, diff).sum() / diff.size)


def print_error(name, aat, u, s, ideal_nf, ideal_n2):
    err = -np.dot(u, np.dot(np.diag(s), u.T))
    err += aat
    nf, n2 = np.linalg.norm(err), norm2(err)
    print(
        '%s error: norm_frobenius=%f (/ideal=%g), norm2=%f (/ideal=%g), RMSE=%g' %
        (name, nf, nf / ideal_nf, n2, n2 / ideal_n2, rmse(err))
    )
    sys.stdout.flush()


class ClippedCorpus(object):
    def __init__(self, corpus, max_docs, max_terms):
        self.corpus = corpus
        self.max_docs, self.max_terms = max_docs, max_terms

    def __iter__(self):
        for doc in itertools.islice(self.corpus, self.max_docs):
            yield [(f, w) for f, w in doc if f < self.max_terms]


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s", " ".join(sys.argv))

    program = os.path.basename(sys.argv[0])
    # do we have enough cmd line arguments?
    if len(sys.argv) < 2:
        print(globals()["__doc__"] % locals())
        sys.exit(1)

    fname = sys.argv[1]
    if fname.endswith('bz2'):
        mm = gensim.corpora.MmCorpus(bz2.BZ2File(fname))
    else:
        mm = gensim.corpora.MmCorpus(fname)

    # extra cmd parameters = use a subcorpus (fewer docs, smaller vocab)
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    else:
        n = mm.num_docs
    if len(sys.argv) > 3:
        m = int(sys.argv[3])
    else:
        m = mm.num_terms
    logging.info("using %i documents and %i features", n, m)
    corpus = ClippedCorpus(mm, n, m)
    id2word = gensim.utils.FakeDict(m)

    logging.info("computing corpus * corpus^T")  # eigenvalues of this matrix are singular values of `corpus`, squared
    aat = np.zeros((m, m), dtype=np.float64)
    for chunk in gensim.utils.grouper(corpus, chunksize=5000):
        num_nnz = sum(len(doc) for doc in chunk)
        chunk = gensim.matutils.corpus2csc(chunk, num_nnz=num_nnz, num_terms=m, num_docs=len(chunk), dtype=np.float32)
        chunk = chunk * chunk.T
        chunk = chunk.toarray()
        aat += chunk
        del chunk

    logging.info("computing full decomposition of corpus * corpus^t")
    aat = aat.astype(np.float32)
    spectrum_s, spectrum_u = scipy.linalg.eigh(aat)
    spectrum_s = spectrum_s[::-1]  # re-order to descending eigenvalue order
    spectrum_u = spectrum_u.T[::-1].T
    np.save(fname + '.spectrum.npy', spectrum_s)

    for factors in FACTORS:
        err = -np.dot(spectrum_u[:, :factors], np.dot(np.diag(spectrum_s[:factors]), spectrum_u[:, :factors].T))
        err += aat
        ideal_fro = np.linalg.norm(err)
        del err
        ideal_n2 = spectrum_s[factors + 1]
        print('*' * 40, "%i factors, ideal error norm_frobenius=%f, norm_2=%f" % (factors, ideal_fro, ideal_n2))
        print("*" * 30, end="")
        print_error("baseline", aat,
                    np.zeros((m, factors)), np.zeros((factors)), ideal_fro, ideal_n2)
        if sparsesvd:
            logging.info("computing SVDLIBC SVD for %i factors", factors)
            taken = time.time()
            corpus_ram = gensim.matutils.corpus2csc(corpus, num_terms=m)
            ut, s, vt = sparsesvd(corpus_ram, factors)
            taken = time.time() - taken
            del corpus_ram
            del vt
            u, s = ut.T.astype(np.float32), s.astype(np.float32)**2  # convert singular values to eigenvalues
            del ut
            print("SVDLIBC SVD for %i factors took %s s (spectrum %f .. %f)"
                  % (factors, taken, s[0], s[-1]))
            print_error("SVDLIBC", aat, u, s, ideal_fro, ideal_n2)
            del u
        for power_iters in POWER_ITERS:
            for chunksize in CHUNKSIZE:
                logging.info(
                    "computing incremental SVD for %i factors, %i power iterations, chunksize %i",
                    factors, power_iters, chunksize
                )
                taken = time.time()
                gensim.models.lsimodel.P2_EXTRA_ITERS = power_iters
                model = gensim.models.LsiModel(
                    corpus, id2word=id2word, num_topics=factors,
                    chunksize=chunksize, power_iters=power_iters
                )
                taken = time.time() - taken
                u, s = model.projection.u.astype(np.float32), model.projection.s.astype(np.float32)**2
                del model
                print(
                    "incremental SVD for %i factors, %i power iterations, "
                    "chunksize %i took %s s (spectrum %f .. %f)" %
                    (factors, power_iters, chunksize, taken, s[0], s[-1])
                )
                print_error('incremental SVD', aat, u, s, ideal_fro, ideal_n2)
                del u
            logging.info("computing multipass SVD for %i factors, %i power iterations", factors, power_iters)
            taken = time.time()
            model = gensim.models.LsiModel(
                corpus, id2word=id2word, num_topics=factors, chunksize=2000,
                onepass=False, power_iters=power_iters
            )
            taken = time.time() - taken
            u, s = model.projection.u.astype(np.float32), model.projection.s.astype(np.float32)**2
            del model
            print(
                "multipass SVD for %i factors, "
                "%i power iterations took %s s (spectrum %f .. %f)" %
                (factors, power_iters, taken, s[0], s[-1])
            )
            print_error('multipass SVD', aat, u, s, ideal_fro, ideal_n2)
            del u

    logging.info("finished running %s", program)
