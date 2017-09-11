#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Scikit learn interface for gensim for easy use of gensim with scikit-learn
Follows scikit-learn API conventions
"""

import numpy as np
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim import matutils


class HdpTransformer(TransformerMixin, BaseEstimator):
    """
    Base HDP module
    """

    def __init__(self, id2word, max_chunks=None, max_time=None, chunksize=256, kappa=1.0, tau=64.0, K=15, T=150,
                 alpha=1, gamma=1, eta=0.01, scale=1.0, var_converge=0.0001, outputdir=None, random_state=None):
        """
        Sklearn api for HDP model. See gensim.models.HdpModel for parameter details.
        """
        self.gensim_model = None
        self.id2word = id2word
        self.max_chunks = max_chunks
        self.max_time = max_time
        self.chunksize = chunksize
        self.kappa = kappa
        self.tau = tau
        self.K = K
        self.T = T
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.scale = scale
        self.var_converge = var_converge
        self.outputdir = outputdir
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        Calls gensim.models.HdpModel
        """
        if sparse.issparse(X):
            corpus = matutils.Sparse2Corpus(X)
        else:
            corpus = X

        self.gensim_model = models.HdpModel(
            corpus=corpus, id2word=self.id2word, max_chunks=self.max_chunks,
            max_time=self.max_time, chunksize=self.chunksize, kappa=self.kappa, tau=self.tau,
            K=self.K, T=self.T, alpha=self.alpha, gamma=self.gamma, eta=self.eta, scale=self.scale,
            var_converge=self.var_converge, outputdir=self.outputdir, random_state=self.random_state
        )
        return self

    def transform(self, docs):
        """
        Takes a list of documents as input ('docs').
        Returns a matrix of topic distribution for the given document bow, where a_ij
        indicates (topic_i, topic_probability_j).
        The input `docs` should be in BOW format and can be a list of documents like : [ [(4, 1), (7, 1)], [(9, 1), (13, 1)], [(2, 1), (6, 1)] ]
        or a single document like : [(4, 1), (7, 1)]
        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        check = lambda x: [x] if isinstance(x[0], tuple) else x
        docs = check(docs)
        X = [[] for _ in range(0, len(docs))]

        max_num_topics = 0
        for k, v in enumerate(docs):
            X[k] = self.gensim_model[v]
            max_num_topics = max(max_num_topics, max(x[0] for x in X[k]) + 1)

        for k, v in enumerate(X):
            # returning dense representation for compatibility with sklearn but we should go back to sparse representation in the future
            dense_vec = matutils.sparse2full(v, max_num_topics)
            X[k] = dense_vec

        return np.reshape(np.array(X), (len(docs), max_num_topics))

    def partial_fit(self, X):
        """
        Train model over X.
        """
        if sparse.issparse(X):
            X = matutils.Sparse2Corpus(X)

        if self.gensim_model is None:
            self.gensim_model = models.HdpModel(
                id2word=self.id2word, max_chunks=self.max_chunks,
                max_time=self.max_time, chunksize=self.chunksize, kappa=self.kappa, tau=self.tau,
                K=self.K, T=self.T, alpha=self.alpha, gamma=self.gamma, eta=self.eta, scale=self.scale,
                var_converge=self.var_converge, outputdir=self.outputdir, random_state=self.random_state
            )

        self.gensim_model.update(corpus=X)
        return self
