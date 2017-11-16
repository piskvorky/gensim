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
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim import matutils


class LsiTransformer(TransformerMixin, BaseEstimator):
    """
    Base LSI module
    """

    def __init__(self, num_topics=200, id2word=None, chunksize=20000,
                 decay=1.0, onepass=True, power_iters=2, extra_samples=100):
        """
        Sklearn wrapper for LSI model. See gensim.model.LsiModel for parameter details.
        """
        self.gensim_model = None
        self.num_topics = num_topics
        self.id2word = id2word
        self.chunksize = chunksize
        self.decay = decay
        self.onepass = onepass
        self.extra_samples = extra_samples
        self.power_iters = power_iters

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        Calls gensim.models.LsiModel
        """
        if sparse.issparse(X):
            corpus = matutils.Sparse2Corpus(X)
        else:
            corpus = X

        self.gensim_model = models.LsiModel(
            corpus=corpus, num_topics=self.num_topics, id2word=self.id2word, chunksize=self.chunksize,
            decay=self.decay, onepass=self.onepass, power_iters=self.power_iters, extra_samples=self.extra_samples
        )
        return self

    def transform(self, docs):
        """
        Takes a list of documents as input ('docs').
        Returns a matrix of topic distribution for the given document bow, where a_ij
        indicates (topic_i, topic_probability_j).
        The input `docs` should be in BOW format and can be a list of documents like :
        [
            [(4, 1), (7, 1)],
            [(9, 1), (13, 1)], [(2, 1), (6, 1)]
        ]
        or a single document like : [(4, 1), (7, 1)]
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
        distribution = [matutils.sparse2full(self.gensim_model[doc], self.num_topics) for doc in docs]
        return np.reshape(np.array(distribution), (len(docs), self.num_topics))

    def partial_fit(self, X):
        """
        Train model over X.
        """
        if sparse.issparse(X):
            X = matutils.Sparse2Corpus(X)

        if self.gensim_model is None:
            self.gensim_model = models.LsiModel(
                num_topics=self.num_topics, id2word=self.id2word, chunksize=self.chunksize, decay=self.decay,
                onepass=self.onepass, power_iters=self.power_iters, extra_samples=self.extra_samples
            )

        self.gensim_model.add_documents(corpus=X)
        return self
