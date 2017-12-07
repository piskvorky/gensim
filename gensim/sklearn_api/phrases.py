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

from gensim import models


class PhrasesTransformer(TransformerMixin, BaseEstimator):
    """
    Base Phrases module
    """

    def __init__(self, min_count=5, threshold=10.0, max_vocab_size=40000000,
                 delimiter=b'_', progress_per=10000, scoring='default'):
        """
        Sklearn wrapper for Phrases model.
        """
        self.gensim_model = None
        self.min_count = min_count
        self.threshold = threshold
        self.max_vocab_size = max_vocab_size
        self.delimiter = delimiter
        self.progress_per = progress_per
        self.scoring = scoring

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        """
        self.gensim_model = models.Phrases(
            sentences=X, min_count=self.min_count, threshold=self.threshold,
            max_vocab_size=self.max_vocab_size, delimiter=self.delimiter,
            progress_per=self.progress_per, scoring=self.scoring
        )
        return self

    def transform(self, docs):
        """
        Return the input documents to return phrase tokens.
        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # input as python lists
        if isinstance(docs[0], string_types):
            docs = [docs]
        return [self.gensim_model[doc] for doc in docs]

    def partial_fit(self, X):
        if self.gensim_model is None:
            self.gensim_model = models.Phrases(
                sentences=X, min_count=self.min_count, threshold=self.threshold,
                max_vocab_size=self.max_vocab_size, delimiter=self.delimiter,
                progress_per=self.progress_per, scoring=self.scoring
            )

        self.gensim_model.add_vocab(X)
        return self
