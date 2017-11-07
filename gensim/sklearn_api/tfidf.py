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

import gensim
from gensim.models import TfidfModel


class TfIdfTransformer(TransformerMixin, BaseEstimator):
    """
    Base Tf-Idf module
    """

    def __init__(self, id2word=None, dictionary=None, wlocal=gensim.utils.identity,
                 wglobal=gensim.models.tfidfmodel.df2idf, normalize=True):
        """
        Sklearn wrapper for Tf-Idf model.
        """
        self.gensim_model = None
        self.id2word = id2word
        self.dictionary = dictionary
        self.wlocal = wlocal
        self.wglobal = wglobal
        self.normalize = normalize

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        """
        self.gensim_model = TfidfModel(corpus=X, id2word=self.id2word, dictionary=self.dictionary,
            wlocal=self.wlocal, wglobal=self.wglobal, normalize=self.normalize)
        return self

    def transform(self, docs):
        """
        Return the transformed documents after multiplication with the tf-idf matrix.
        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # input as python lists
        if isinstance(docs[0], tuple):
            docs = [docs]
        return [self.gensim_model[doc] for doc in docs]
