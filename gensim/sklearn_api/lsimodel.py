#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Chinmaya Pancholi <chinmayapancholi13@gmail.com>
# Copyright (C) 2017 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Scikit learn interface for `gensim.models.lsimodel`.

Follows scikit-learn API conventions to facilitate using gensim along with scikit-learn.

Examples
--------
Integrate with sklearn Pipelines:

    >>> model = LsiTransformer(num_topics=15, id2word=id2word)
    >>> clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
    >>> pipe = Pipeline([('features', model,), ('classifier', clf)])
    >>> pipe.fit(corpus, data.target)

"""

import numpy as np
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim import matutils


class LsiTransformer(TransformerMixin, BaseEstimator):
    """Base LSI module.

    Wraps :class:`~gensim.model.lsimodel.LsiModel`.
    For more information on the inner working please take a look at
    the original class.

    """

    def __init__(self, num_topics=200, id2word=None, chunksize=20000,
                 decay=1.0, onepass=True, power_iters=2, extra_samples=100):
        """Sklearn wrapper for LSI model.

        Parameters are propagated to the original models constructor. For an explanation
        please refer to :meth:`~gensim.models.lsimodel.LsiModel.__init__`

        Parameters
        ----------
        num_topics : int, optional
        id2word : dict of {int: str}, optional
        chunksize :  int, optional
        decay : float, optional
        onepass : bool, optional
        power_iters: int, optional
        extra_samples : int, optional

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
        Calls :meth:`~gensim.models.lsimodel.LsiModel`

        Parameters
        ----------
        X : iterable of iterable of (int, float)
            Stream of document vectors or sparse matrix of shape: [num_terms, num_documents].

        Returns
        -------
        :class:`~gensim.sklearn_api.lsimodel.LsiTransformer`
            The trained model.

        """
        if sparse.issparse(X):
            corpus = matutils.Sparse2Corpus(sparse=X, documents_columns=False)
        else:
            corpus = X

        self.gensim_model = models.LsiModel(
            corpus=corpus, num_topics=self.num_topics, id2word=self.id2word, chunksize=self.chunksize,
            decay=self.decay, onepass=self.onepass, power_iters=self.power_iters, extra_samples=self.extra_samples
        )
        return self

    def transform(self, docs):
        """Computes the topic distribution matrix

        Parameters
        ----------
        docs : iterable of iterable of (int, float)
            Stream of document vectors or sparse matrix of shape: [`num_terms`, num_documents].

        Returns
        -------
        list of (int, int)
            Topic distribution matrix of shape [num_docs, num_topics]

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
        """Train model over a potentially incomplete set of documents.

        This method can be used in two ways:
            1. On an unfitted model in which case the model is initialized and trained on `X`.
            2. On an already fitted model in which case the model is **further** trained on `X`.

        Parameters
        ----------
        X : iterable of iterable of (int, float)
            Stream of document vectors or sparse matrix of shape: [num_terms, num_documents].

        Returns
        -------
        :class:`~gensim.sklearn_api.lsimodel.LsiTransformer`
            The trained model.

        """
        if sparse.issparse(X):
            X = matutils.Sparse2Corpus(sparse=X, documents_columns=False)

        if self.gensim_model is None:
            self.gensim_model = models.LsiModel(
                num_topics=self.num_topics, id2word=self.id2word, chunksize=self.chunksize, decay=self.decay,
                onepass=self.onepass, power_iters=self.power_iters, extra_samples=self.extra_samples
            )

        self.gensim_model.add_documents(corpus=X)
        return self
