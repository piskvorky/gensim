#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Scikit learn interface for gensim for easy use of gensim with scikit-learn
Follows scikit-learn API conventions
"""

from numpy import integer
from six import string_types, integer_types
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim.sklearn_api import BaseTransformer


class D2VTransformer(BaseTransformer, TransformerMixin, BaseEstimator):
    """
    Base Doc2Vec module
    """

    def __init__(self, dm_mean=None, dm=1, dbow_words=0, dm_concat=0,
                dm_tag_count=1, docvecs=None, docvecs_mapfile=None,
                comment=None, trim_rule=None, **other_params):
        """
        Sklearn wrapper for Doc2Vec model. Class derived from gensim.models.Doc2Vec
        """
        self.gensim_model = None
        self.dm_mean = dm_mean
        self.dm = dm
        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        self.docvecs = docvecs
        self.docvecs_mapfile = docvecs_mapfile
        self.comment = comment
        self.trim_rule = trim_rule
        self.other_params = other_params

    def get_params(self, deep=True):
        """
        Returns all parameters as dictionary.
        """
        model_params = {"dm_mean": self.dm_mean, "dm": self.dm, "dbow_words": self.dbow_words,
        "dm_concat": self.dm_concat, "dm_tag_count": self.dm_tag_count, "docvecs": self.docvecs,
        "docvecs_mapfile": self.docvecs_mapfile, "comment": self.comment, "trim_rule": self.trim_rule}

        model_params.update(self.other_params)
        return model_params

    def set_params(self, **parameters):
        """
        Set all parameters.
        """
        super(D2VTransformer, self).set_params(**parameters)
        return self

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        Calls gensim.models.Doc2Vec
        """
        self.gensim_model = models.Doc2Vec(documents=X, dm_mean=self.dm_mean, dm=self.dm,
            dbow_words=self.dbow_words, dm_concat=self.dm_concat, dm_tag_count=self.dm_tag_count,
            docvecs=self.docvecs, docvecs_mapfile=self.docvecs_mapfile, comment=self.comment,
            trim_rule=self.trim_rule, **self.other_params)
        return self

    def transform(self, docs):
        """
        Return the vector representations for the input list of documents.
        """
        if self.gensim_model is None:
            raise NotFittedError("This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method.")

        # The input as array of array
        check = lambda x: [x] if isinstance(x, string_types + integer_types + (integer,)) else x
        docs = check(docs)
        X = [[] for _ in range(0, len(docs))]

        for k, v in enumerate(docs):
            doc_vec = self.gensim_model[v]
            X[k] = doc_vec

        return np.reshape(np.array(X), (len(docs), self.gensim_model.vector_size))

    def partial_fit(self, X):
        raise NotImplementedError("'partial_fit' has not been implemented for D2VTransformer")
