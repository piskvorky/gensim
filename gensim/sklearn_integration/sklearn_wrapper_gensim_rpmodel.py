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


class SklRpModel(base_sklearn_wrapper.BaseSklearnWrapper, TransformerMixin, BaseEstimator):
    """
    Base RP module
    """

    def __init__(self, id2word=None, num_topics=300):
        """
        Sklearn wrapper for RP model. Class derived from gensim.models.RpModel.
        """
        self.__model = None
        self.corpus = None
        self.id2word = id2word
        self.num_topics = num_topics

    def get_params(self, deep=True):
        """
        Returns all parameters as dictionary.
        """
        return {"corpus": self.corpus, "id2word": self.id2word, "num_topics": self.num_topics}

    def set_params(self, **parameters):
        """
        Set all parameters.
        """
        super(SklRpModel, self).set_params(**parameters)

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        Calls gensim.models.RpModel
        >>>gensim.models.RpModel(corpus=self.corpus, id2word=self.id2word, num_topics=self.num_topics)
        """
        self.corpus = X
        self.__model = models.RpModel(corpus=self.corpus, id2word=self.id2word, num_topics=self.num_topics)

    def transform(self, doc):
        """
        Take document/corpus as input.
        Return RP representation of the input document/corpus.
        """
        return self.__model[doc]

    def partial_fit(self, X):
        raise NotImplementedError("'partial_fit' has not been implemented for the RandomProjections model")
