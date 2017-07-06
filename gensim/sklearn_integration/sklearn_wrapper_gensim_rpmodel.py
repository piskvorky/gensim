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
from gensim.sklearn_integration import BaseSklearnWrapper


class RpTransformer(BaseSklearnWrapper, TransformerMixin, BaseEstimator):
    """
    Base RP module
    """

    def __init__(self, id2word=None, num_topics=300):
        """
        Sklearn wrapper for RP model. Class derived from gensim.models.RpModel.
        """
        self.gensim_model = None
        self.id2word = id2word
        self.num_topics = num_topics

    def get_params(self, deep=True):
        """
        Returns all parameters as dictionary.
        """
        return {"id2word": self.id2word, "num_topics": self.num_topics}

    def set_params(self, **parameters):
        """
        Set all parameters.
        """
        super(RpTransformer, self).set_params(**parameters)
        return self

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
        The input `docs` can correspond to multiple documents like : [ [(0, 1.0), (1, 1.0), (2, 1.0)], [(0, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0)] ]
        or a single document like : [(0, 1.0), (1, 1.0), (2, 1.0)]
        """
        if self.gensim_model is None:
            raise NotFittedError("This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method.")

        # The input as array of array
        check = lambda x: [x] if isinstance(x[0], tuple) else x
        docs = check(docs)
        X = [[] for _ in range(0, len(docs))]

        for k, v in enumerate(docs):
            transformed_doc = self.gensim_model[v]
            probs_docs = list(map(lambda x: x[1], transformed_doc))
            # Everything should be equal in length
            if len(probs_docs) != self.num_topics:
                probs_docs.extend([1e-12] * (self.num_topics - len(probs_docs)))
            X[k] = probs_docs

        return np.reshape(np.array(X), (len(docs), self.num_topics))

    def partial_fit(self, X):
        raise NotImplementedError("'partial_fit' has not been implemented for RpTransformer")
