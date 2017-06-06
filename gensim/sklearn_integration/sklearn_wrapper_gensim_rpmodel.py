#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
"""
Scikit learn interface for gensim for easy use of gensim with scikit-learn
Follows scikit-learn API conventions
"""
# import numpy as np

from gensim import models
# from gensim import matutils
# from gensim.sklearn_integration import base_sklearn_wrapper
# from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator


class SklearnWrapperRpModel(models.RpModel, TransformerMixin, BaseEstimator):
    """
    Base RP module
    """

    def __init__(self, corpus, id2word=None, num_topics=300):
        """
        Sklearn wrapper for RP model. Class derived from gensim.models.RpModel.
        """
        self.corpus = corpus
        self.id2word = id2word
        self.num_topics = num_topics

        # if 'fit' function is not used, then 'corpus' is given in init
        if self.corpus:
            models.RpModel.__init__(self, self.corpus=corpus, self.id2word=id2word, self.num_topics=num_topics)

    def get_params(self, deep=True):
        """
        Returns all parameters as dictionary.
        """
        return {}

    def set_params(self, **parameters):
        """
        Set all parameters.
        """
        super(SklearnWrapperRpModel, self).set_params(**parameters)

    def fit(self, X, y=None):
        """
        """
        pass

    def transform(self, docs):
        """
        """
        pass

    def partial_fit(self, X):
        """
        """
        pass
