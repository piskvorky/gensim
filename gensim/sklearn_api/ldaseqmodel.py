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
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim.sklearn_api import BaseTransformer


class LdaSeqTransformer(BaseTransformer, TransformerMixin, BaseEstimator):
    """
    Base LdaSeq module
    """

    def __init__(self, time_slice=None, id2word=None, alphas=0.01, num_topics=10,
                initialize='gensim', sstats=None, lda_model=None, obs_variance=0.5, chain_variance=0.005, passes=10,
                random_state=None, lda_inference_max_iter=25, em_min_iter=6, em_max_iter=20, chunksize=100):
        """
        Sklearn wrapper for LdaSeq model. See gensim.models.LdaSeqModel for parameter details.
        """
        self.gensim_model = None
        self.time_slice = time_slice
        self.id2word = id2word
        self.alphas = alphas
        self.num_topics = num_topics
        self.initialize = initialize
        self.sstats = sstats
        self.lda_model = lda_model
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance
        self.passes = passes
        self.random_state = random_state
        self.lda_inference_max_iter = lda_inference_max_iter
        self.em_min_iter = em_min_iter
        self.em_max_iter = em_max_iter
        self.chunksize = chunksize

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        Calls gensim.models.LdaSeqModel
        """
        self.gensim_model = models.LdaSeqModel(corpus=X, time_slice=self.time_slice, id2word=self.id2word,
            alphas=self.alphas, num_topics=self.num_topics, initialize=self.initialize, sstats=self.sstats,
            lda_model=self.lda_model, obs_variance=self.obs_variance, chain_variance=self.chain_variance,
            passes=self.passes, random_state=self.random_state, lda_inference_max_iter=self.lda_inference_max_iter,
            em_min_iter=self.em_min_iter, em_max_iter=self.em_max_iter, chunksize=self.chunksize)
        return self

    def transform(self, docs):
        """
        Return the topic proportions for the documents passed.
        The input `docs` should be in BOW format and can be a list of documents like : [ [(4, 1), (7, 1)], [(9, 1), (13, 1)], [(2, 1), (6, 1)] ]
        or a single document like : [(4, 1), (7, 1)]
        """
        if self.gensim_model is None:
            raise NotFittedError("This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method.")

        # The input as array of array
        check = lambda x: [x] if isinstance(x[0], tuple) else x
        docs = check(docs)
        X = [[] for _ in range(0, len(docs))]

        for k, v in enumerate(docs):
            transformed_author = self.gensim_model[v]
            # Everything should be equal in length
            if len(transformed_author) != self.num_topics:
                transformed_author.extend([1e-12] * (self.num_topics - len(transformed_author)))
            X[k] = transformed_author

        return np.reshape(np.array(X), (len(docs), self.num_topics))
