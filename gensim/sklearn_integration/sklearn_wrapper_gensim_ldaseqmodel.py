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

from gensim import models
from gensim.sklearn_integration import base_sklearn_wrapper


class SklLdaSeqModel(base_sklearn_wrapper.BaseSklearnWrapper, TransformerMixin, BaseEstimator):
    """
    Base LdaSeq module
    """

    def __init__(self, time_slice=None, id2word=None, alphas=0.01, num_topics=10,
                initialize='gensim', sstats=None, lda_model=None, obs_variance=0.5, chain_variance=0.005, passes=10,
                random_state=None, lda_inference_max_iter=25, em_min_iter=6, em_max_iter=20, chunksize=100):
        """
        Sklearn wrapper for LdaSeq model. Class derived from gensim.models.LdaSeqModel
        """
        self.__model = None
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

    def get_params(self, deep=True):
        """
        Returns all parameters as dictionary.
        """
        return {"time_slice": self.time_slice, "id2word": self.id2word,
                "alphas": self.alphas, "num_topics": self.num_topics, "initialize": self.initialize,
                "sstats": self.sstats, "lda_model": self.lda_model, "obs_variance": self.obs_variance,
                "chain_variance": self.chain_variance, "passes": self.passes, "random_state": self.random_state,
                "lda_inference_max_iter": self.lda_inference_max_iter, "em_min_iter": self.em_min_iter,
                "em_max_iter": self.em_max_iter, "chunksize": self.chunksize}

    def set_params(self, **parameters):
        """
        Set all parameters.
        """
        super(SklLdaSeqModel, self).set_params(**parameters)

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        Calls gensim.models.LdaSeqModel
        """
        self.__model = models.LdaSeqModel(corpus=X, time_slice=self.time_slice, id2word=self.id2word,
            alphas=self.alphas, num_topics=self.num_topics, initialize=self.initialize, sstats=self.sstats,
            lda_model=self.lda_model, obs_variance=self.obs_variance, chain_variance=self.chain_variance,
            passes=self.passes, random_state=self.random_state, lda_inference_max_iter=self.lda_inference_max_iter,
            em_min_iter=self.em_min_iter, em_max_iter=self.em_max_iter, chunksize=self.chunksize)

    def transform(self, doc):
        """
        Return the topic proportions for the document passed.
        """
        return self.__model[doc]

    def partial_fit(self, X):
        raise NotImplementedError("'partial_fit' has not been implemented for the LDA Seq model")
