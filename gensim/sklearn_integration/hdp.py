#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Scikit learn interface for gensim for easy use of gensim with scikit-learn
Follows scikit-learn API conventions
"""

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim.sklearn_integration import BaseSklearnWrapper


class HdpTransformer(BaseSklearnWrapper, TransformerMixin, BaseEstimator):
    """
    Base HDP module
    """

    def __init__(self, id2word, max_chunks=None, max_time=None,
            chunksize=256, kappa=1.0, tau=64.0, K=15, T=150, alpha=1,
            gamma=1, eta=0.01, scale=1.0, var_converge=0.0001,
            outputdir=None, random_state=None):
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
        self.gensim_model = models.HdpModel(corpus=X, id2word=self.id2word, max_chunks=self.max_chunks,
            max_time=self.max_time, chunksize=self.chunksize, kappa=self.kappa, tau=self.tau,
            K=self.K, T=self.T, alpha=self.alpha, gamma=self.gamma, eta=self.eta, scale=self.scale,
            var_converge=self.var_converge, outputdir=self.outputdir, random_state=self.random_state)
        return self

    def transform(self, docs):
        """
        """
        if self.gensim_model is None:
            raise NotFittedError("This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method.")

        # The input as array of array
        check = lambda x: [x] if isinstance(x[0], tuple) else x
        docs = check(docs)
        X = [[] for _ in range(0, len(docs))]

        for k, v in enumerate(docs):
            doc_topics = self.gensim_model[v]
            X[k] = doc_topics
        return X

    def partial_fit(self, X):
        raise NotImplementedError("'partial_fit' has not been implemented for HdpTransformer")
