#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Scikit learn interface for gensim for easy use of gensim with scikit-learn
follows on scikit learn API conventions
"""

import numpy as np
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim import matutils
from gensim.sklearn_integration import base_sklearn_wrapper


class SklLdaModel(base_sklearn_wrapper.BaseSklearnWrapper, TransformerMixin, BaseEstimator):
    """
    Base LDA module
    """

    def __init__(
            self, num_topics=100, id2word=None,
            chunksize=2000, passes=1, update_every=1,
            alpha='symmetric', eta=None, decay=0.5, offset=1.0,
            eval_every=10, iterations=50, gamma_threshold=0.001,
            minimum_probability=0.01, random_state=None):
        """
        Sklearn wrapper for LDA model. derived class for gensim.model.LdaModel .
        """
        self.gensim_model = None
        self.num_topics = num_topics
        self.id2word = id2word
        self.chunksize = chunksize
        self.passes = passes
        self.update_every = update_every
        self.alpha = alpha
        self.eta = eta
        self.decay = decay
        self.offset = offset
        self.eval_every = eval_every
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold
        self.minimum_probability = minimum_probability
        self.random_state = random_state

    def get_params(self, deep=True):
        """
        Returns all parameters as dictionary.
        """
        return {"num_topics": self.num_topics, "id2word": self.id2word, "chunksize": self.chunksize,
                "passes": self.passes, "update_every": self.update_every, "alpha": self.alpha, "eta": self.eta,
                "decay": self.decay, "offset": self.offset, "eval_every": self.eval_every, "iterations": self.iterations,
                "gamma_threshold": self.gamma_threshold, "minimum_probability": self.minimum_probability,
                "random_state": self.random_state}

    def set_params(self, **parameters):
        """
        Set all parameters.
        """
        super(SklLdaModel, self).set_params(**parameters)

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        Calls gensim.models.LdaModel
        """
        if sparse.issparse(X):
            corpus = matutils.Sparse2Corpus(X)
        else:
            corpus = X

        self.gensim_model = models.LdaModel(corpus=corpus, num_topics=self.num_topics, id2word=self.id2word,
            chunksize=self.chunksize, passes=self.passes, update_every=self.update_every,
            alpha=self.alpha, eta=self.eta, decay=self.decay, offset=self.offset,
            eval_every=self.eval_every, iterations=self.iterations,
            gamma_threshold=self.gamma_threshold, minimum_probability=self.minimum_probability,
            random_state=self.random_state)
        return self

    def transform(self, docs, minimum_probability=None):
        """
        Takes as an list of input a documents (documents).
        Returns matrix of topic distribution for the given document bow, where a_ij
        indicates (topic_i, topic_probability_j).
        """
        if self.gensim_model is None:
            raise NotFittedError("This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method.")

        # The input as array of array
        check = lambda x: [x] if isinstance(x[0], tuple) else x
        docs = check(docs)
        X = [[] for _ in range(0, len(docs))]

        for k, v in enumerate(docs):
            doc_topics = self.gensim_model.get_document_topics(v, minimum_probability=minimum_probability)
            probs_docs = list(map(lambda x: x[1], doc_topics))
            # Everything should be equal in length
            if len(probs_docs) != self.num_topics:
                probs_docs.extend([1e-12]*(self.num_topics - len(probs_docs)))
            X[k] = probs_docs
        return np.reshape(np.array(X), (len(docs), self.num_topics))

    def partial_fit(self, X):
        """
        Train model over X.
        By default, 'online (single-pass)' mode is used for training the LDA model.
        Configure `passes` and `update_every` params at init to choose the mode among :
            - online (single-pass): update_every != None and passes == 1
            - online (multi-pass): update_every != None and passes > 1
            - batch: update_every == None
        """
        if sparse.issparse(X):
            X = matutils.Sparse2Corpus(X)

        if self.gensim_model is None:
            self.gensim_model = models.LdaModel(num_topics=self.num_topics, id2word=self.id2word,
                chunksize=self.chunksize, passes=self.passes, update_every=self.update_every,
                alpha=self.alpha, eta=self.eta, decay=self.decay, offset=self.offset,
                eval_every=self.eval_every, iterations=self.iterations, gamma_threshold=self.gamma_threshold,
                minimum_probability=self.minimum_probability, random_state=self.random_state)

        self.gensim_model.update(corpus=X)
        return self