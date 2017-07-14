#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Scikit learn interface for gensim for easy use of gensim with scikit-learn
Follows scikit-learn API conventions
"""

from six import string_types
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim.corpora import Dictionary
from gensim.sklearn_integration import BaseSklearnWrapper
from gensim.utils import tokenize


class Text2BowTransformer(BaseSklearnWrapper, TransformerMixin, BaseEstimator):
    """
    Base Text2Bow module
    """

    def __init__(self, prune_at=2000000, tokenizer=tokenize):
        """
        Sklearn wrapper for Text2Bow model.
        """
        self.gensim_model = None
        self.prune_at = prune_at
        self.tokenizer = tokenizer

    def get_params(self, deep=True):
        """
        Returns all parameters as dictionary.
        """
        return {"prune_at": self.prune_at}

    def set_params(self, **parameters):
        """
        Set all parameters.
        """
        super(Text2BowTransformer, self).set_params(**parameters)
        return self

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        """
        tokenized_docs = list(map(lambda x: list(self.tokenizer(x)), X))
        self.gensim_model = Dictionary(documents=tokenized_docs, prune_at=self.prune_at)
        return self

    def transform(self, docs):
        """
        Return the BOW format for the input documents.
        """
        if self.gensim_model is None:
            raise NotFittedError("This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method.")

        # input as python lists
        check = lambda x: [x] if isinstance(x, string_types) else x
        docs = check(docs)
        tokenized_docs = list(map(lambda x: list(self.tokenizer(x)), docs))
        X = [[] for _ in range(0, len(tokenized_docs))]

        for k, v in enumerate(tokenized_docs):
            bow_val = self.gensim_model.doc2bow(v)
            X[k] = bow_val

        return X

    def partial_fit(self, X):
        if self.gensim_model is None:
            self.gensim_model = Dictionary(prune_at=self.prune_at)

        tokenized_docs = list(map(lambda x: list(self.tokenizer(x)), X))
        self.gensim_model.add_documents(tokenized_docs)
        return self
