#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
"""
Scikit learn interface for gensim for easy use of gensim with scikit-learn
follows on scikit learn API conventions
"""
import numpy as np

from gensim import models
from gensim import matutils
from gensim.sklearn_integration import base_sklearn_wrapper
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator


class SklearnWrapperLdaModel(models.LdaModel, base_sklearn_wrapper.BaseSklearnWrapper, TransformerMixin, BaseEstimator):
    """
    Base LDA module
    """

    def __init__(
            self, corpus=None, num_topics=100, id2word=None,
            chunksize=2000, passes=1, update_every=1,
            alpha='symmetric', eta=None, decay=0.5, offset=1.0,
            eval_every=10, iterations=50, gamma_threshold=0.001,
            minimum_probability=0.01, random_state=None):
        """
        Sklearn wrapper for LDA model. derived class for gensim.model.LdaModel .
        """
        self.corpus = corpus
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
        # if no fit function is used , then corpus is given in init
        if self.corpus:
            models.LdaModel.__init__(
                self, corpus=self.corpus, num_topics=self.num_topics, id2word=self.id2word,
                chunksize=self.chunksize, passes=self.passes, update_every=self.update_every,
                alpha=self.alpha, eta=self.eta, decay=self.decay, offset=self.offset,
                eval_every=self.eval_every, iterations=self.iterations,
                gamma_threshold=self.gamma_threshold, minimum_probability=self.minimum_probability,
                random_state=self.random_state)

    def get_params(self, deep=True):
        """
        Returns all parameters as dictionary.
        """
        return {"corpus": self.corpus, "num_topics": self.num_topics, "id2word": self.id2word,
                "chunksize": self.chunksize, "passes": self.passes,
                "update_every": self.update_every, "alpha": self.alpha, "eta": self.eta, "decay": self.decay,
                "offset": self.offset, "eval_every": self.eval_every, "iterations": self.iterations,
                "gamma_threshold": self.gamma_threshold, "minimum_probability": self.minimum_probability,
                "random_state": self.random_state}

    def set_params(self, **parameters):
        """
        Set all parameters.
        """
        super(SklearnWrapperLdaModel, self).set_params(**parameters)

    def fit(self, X, y=None):
        """
        For fitting corpus into the class object.
        Calls gensim.model.LdaModel:
        >>> gensim.models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, passes=passes, update_every=update_every, alpha=alpha, iterations=iterations, eta=eta, random_state=random_state)
        """
        if sparse.issparse(X):
            self.corpus = matutils.Sparse2Corpus(X)
        else:
            self.corpus = X

        models.LdaModel.__init__(
            self, corpus=self.corpus, num_topics=self.num_topics, id2word=self.id2word,
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
        # The input as array of array
        check = lambda x: [x] if isinstance(x[0], tuple) else x
        docs = check(docs)
        X = [[] for _ in range(0, len(docs))]

        for k, v in enumerate(docs):
            doc_topics = self.get_document_topics(v, minimum_probability=minimum_probability)
            probs_docs = list(map(lambda x: x[1], doc_topics))
            # Everything should be equal in length
            if len(probs_docs) != self.num_topics:
                probs_docs.extend([1e-12]*(self.num_topics - len(probs_docs)))
            X[k] = probs_docs
        return np.reshape(np.array(X), (len(docs), self.num_topics))

    def get_topic_dist(self, bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False):
        """
        Takes as an input a new document (bow).
        Returns the topic distribution for the given document bow, as a list of (topic_id, topic_probability) 2-tuples.
        """
        return self.get_document_topics(
            bow, minimum_probability=minimum_probability,
            minimum_phi_value=minimum_phi_value, per_word_topics=per_word_topics)

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

        self.update(corpus=X)
