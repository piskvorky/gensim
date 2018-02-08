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
from gensim.utils import tokenize


class Text2BowTransformer(TransformerMixin, BaseEstimator):
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

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        """
        tokenized_docs = [list(self.tokenizer(x)) for x in X]
        self.gensim_model = Dictionary(documents=tokenized_docs, prune_at=self.prune_at)
        return self

    def transform(self, docs):
        """
        Return the BOW format for the input documents.
        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # input as python lists
        if isinstance(docs, string_types):
            docs = [docs]
        tokenized_docs = (list(self.tokenizer(doc)) for doc in docs)
        return [self.gensim_model.doc2bow(doc) for doc in tokenized_docs]

    def partial_fit(self, X):
        if self.gensim_model is None:
            self.gensim_model = Dictionary(prune_at=self.prune_at)

        tokenized_docs = [list(self.tokenizer(x)) for x in X]
        self.gensim_model.add_documents(tokenized_docs)
        return self
