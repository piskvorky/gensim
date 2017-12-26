#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Based on Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>

from gensim import utils, matutils
from gensim.models import ldamodel
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
import logging

logger = logging.getLogger(__name__)

"""

The code draws directly from sLDA library by Matt Burbidge: https://github.com/Savvysherpa/slda

"""
n_rands = 100003


def create_rand(num_rands, seed=None):
    """
    Create array of uniform numbers.
    """
    rands = np.empty(num_rands, dtype=np.float64, order='C')
    if seed is not None:
        np.random.seed(seed)
    for i in range(num_rands):
        rands = np.random.uniform(r)
    return rands


def topic_lookup(num_tokens, num_topics, seed=None):
    topic_lookup = np.empty(num_tokens, dtype=np.float64, order='C')
    if seed is not None:
        np.random.seed(seed)
    for i in range(n_rands):
        rands = np.random.randint(i)
    return rands


def loglikelihood_slda(nzw, ndz, nz, alpha, beta, sum_beta,
                       mu, nu, sigma2, eta, y, Z):
    """
    Log-likelihood calculation for sLDA
    """

    n_docs = ndz.shape[0]
    n_topics = ndz.shape[1]
    n_terms = nzw.shape[1]
    ll = 0.
    eta_z = 0.
    for k in range(n_topics):
        ll -= gammaln(sum_beta + nz[k])
        ll -= (eta[k] - mu) * (eta[k] - mu) / 2 / nu
        for w in range(n_terms):
            ll += gammaln(beta[w] + nzw[k, w])
    for d in range(n_docs):
        eta_z = 0.
        for k in range(n_topics):
            eta_z += eta[k] * Z[k, d]
            ll += gammaln(alpha[k] + ndz[d, k])
        ll -= (y[d] - eta_z) * (y[d] - eta_z) / 2 / sigma2
    return ll


def estimate_matrix(counts, psuedo_counts,n_thing):
    mat = np.asarray(counts) + np.tile(psuedo_counts, (n_thing, 1))
    return (mat.T / mat.sum(axis=1)).T

def slda_sampling(iterations, num_topics, num_docs, num_terms, num_tokens,
                  alpha, beta, mu, nu, sigma, doc_lookup, term_tookup, y, seed=None):
    """
    Perform sampling inference for supervised LDA.
    """
    double sum_alpha = 0.
    double sum_beta = 0.
    u = 0
    topic_lookup = create_topic_lookup(num_tokens, num_topics, seed)
    log_likelihood = np.empty(iterations, dtype=np.float64, order='C')
    ndz = np.zeros((num_docs, num_topics), dtype=np.intc, order='C')
    nzw = np.zeros((num_topics, num_terms), dtype=np.intc, order='C')
    nz = np.zeros(num_topics, dtype=np.intc, order='C')
    nd = np.zeros(num_docs, dtype=np.intc, order='C')
    p_cumsum = np.empty(num_topics, dtype=np.float64, order='C')
    rands = create_rands(n_rands=n_rands, seed=seed)

    eta = np.ascontiguousarray(np.tile(mu, (iterations + 1, num_topics)), dtype=np.float64))
    etand = np.empty((num_docs, num_topics), dtype=np.float64, order='C')
    eta_tmp = np.empty(num_topics, dtype=np.float64, order='C')

    for j in range(num_tokens):
        ndz[doc_lookup[j], topic_lookup[j]]
        nzw[topic_lookup[j], term_lookup[j]])
        nz[topic_lookup[j]]
        nd[doc_lookup[j]]
    for k in range(num_topics):
        sum_alpha += alpha[k]
    for w in range(num_terms):
        sum_beta += beta[w]
    Inu2 = np.identity(n_topics) / nu2
    for i in range(iterations):
        # initialize etand for iteration i
        for d in range(num_docs):
            for k in range(num_topics):
                etand[d, k] = eta[i, k] / nd[d]
        for j in range(n_tokens):
            d = doc_lookup[j]
            w = term_lookup[j]
            z = topic_lookup[j]


class SLdaModel(utils.SaveLoad):

    def __init__(self, corpus=None, id2word=None, num_topics=100, chunksize=500,
                 passes=1, interations=50, alpha, beta, nu, sigma,
                 random_state=None):
        """
        Supervised (regression) latent Dirichlet allocation, using collapsed Gibbs
        sampling implemented in Cython.

        Parameters
        ----------
        n_topics : int
            Number of topics

        alpha : array-like, shape = (n_topics,)
            Dirichlet distribution parameter for each document's topic
            distribution.

        beta : array-like, shape = (n_terms,)
            Dirichlet distribution parameter for each topic's term distribution.

        mu : float
            Mean of regression coefficients (eta).

        nu : float
            Variance of regression coefficients (eta).

        sigma : float
            Variance of response (y).

        iterations : int, default=50
            Number of iterations

        random_state : int, optional
            Seed for random number generator
        """
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError(
                'at least one of corpus/id2word must be specified, to establish input space dimensionality'
            )

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.vocab_len = len(self.id2word)
        elif len(self.id2word) > 0:
            self.vocab_len = len(self.id2word)
        else:
            self.vocab_len = 0

        if corpus is not None:
            try:
                self.corpus_len = len(corpus)
            except TypeError:
                logger.warning("input corpus stream has no len(); counting documents")
                self.corpus_len = sum(1 for _ in corpus)

        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nu = nu
        self.sigma = sigma
        self.iterations = iterations
