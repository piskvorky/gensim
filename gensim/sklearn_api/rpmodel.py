#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Chinmaya Pancholi <chinmayapancholi13@gmail.com>
# Copyright (C) 2017 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Scikit learn interface for gensim for easy use of gensim with scikit-learn
Follows scikit-learn API conventions
"""

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim import matutils


class RpTransformer(TransformerMixin, BaseEstimator):
    """
    Base RP module
    """

    def __init__(self, id2word=None, num_topics=300):
        """
        Sklearn wrapper for RP model. See gensim.models.RpModel for parameter details.
        """
        self.gensim_model = None
        self.id2word = id2word
        self.num_topics = num_topics

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        Calls gensim.models.RpModel
        """
        self.gensim_model = models.RpModel(corpus=X, id2word=self.id2word, num_topics=self.num_topics)
        return self

    def transform(self, docs):
        """
        Take documents/corpus as input.
        Return RP representation of the input documents/corpus.
        The input `docs` can correspond to multiple documents like
        [[(0, 1.0), (1, 1.0), (2, 1.0)],
        [(0, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0)]]
        or a single document like : [(0, 1.0), (1, 1.0), (2, 1.0)]
        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        if isinstance(docs[0], tuple):
            docs = [docs]
        # returning dense representation for compatibility with sklearn
        # but we should go back to sparse representation in the future
        presentation = [matutils.sparse2full(self.gensim_model[doc], self.num_topics) for doc in docs]
        return np.reshape(np.array(presentation), (len(docs), self.num_topics))
