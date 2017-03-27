#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
"""
Scikit learn interface for gensim for easy use of gensim with scikit-learn
Follows scikit-learn API conventions
"""
import numpy as np

from gensim import models
from gensim import matutils
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator

# accuracy defaults for the multi-pass stochastic algo
P2_EXTRA_DIMS = 100  # set to `None` for dynamic P2_EXTRA_DIMS=k
P2_EXTRA_ITERS = 2

class SklearnWrapperLsiModel(models.LsiModel, TransformerMixin, BaseEstimator):
    """
    Base LSI module
    """

    def __init__(self, corpus=None, num_topics=200, id2word=None, chunksize=20000,
                 decay=1.0, onepass=True, power_iters=P2_EXTRA_ITERS, extra_samples=P2_EXTRA_DIMS):
        """
        Sklearn wrapper for LSI model. Class derived from gensim.model.LsiModel.
        """
        self.corpus = corpus
        self.num_topics = num_topics
        self.id2word = id2word
        self.chunksize = chunksize
        self.decay = decay
        self.onepass = onepass
        self.extra_samples = extra_samples
        self.power_iters = power_iters

        # if 'fit' function is not used, then 'corpus' is given in init
        if self.corpus:
            models.LsiModel.__init__(self, corpus=self.corpus, num_topics=self.num_topics, id2word=self.id2word, chunksize=self.chunksize,
                 decay=self.decay, onepass=self.onepass, power_iters=self.power_iters, extra_samples=self.extra_samples)

    def get_params(self, deep=True):
        """
        Returns all parameters as dictionary.
        """
        return {"corpus": self.corpus, "num_topics": self.num_topics, "id2word": self.id2word,
                "chunksize": self.chunksize, "decay": self.decay, "onepass": self.onepass,
                "extra_samples": self.extra_samples, "power_iters": self.power_iters}

    def set_params(self, **parameters):
        """
        Set all parameters.
        """
        for parameter, value in parameters.items():
            self.parameter = value
        return self

    def fit(self, X,  y=None):
        """
        For fitting corpus into the class object.
        Calls gensim.model.LsiModel:
        >>>gensim.models.LsiModel(corpus=corpus, num_topics=num_topics, id2word=id2word, chunksize=chunksize, decay=decay, onepass=onepass, power_iters=power_iters, extra_samples=extra_samples)
        """
        if sparse.issparse(X):
            self.corpus = matutils.Sparse2Corpus(X)
        else:
            self.corpus = X

        models.LsiModel.__init__(self, corpus=self.corpus, num_topics=self.num_topics, id2word=self.id2word, chunksize=self.chunksize,
             decay=self.decay, onepass=self.onepass, power_iters=self.power_iters, extra_samples=self.extra_samples)
        return self

    def transform(self, docs):
        """
        Takes a list of documents as input ('docs').
        Returns a matrix of topic distribution for the given document bow, where a_ij
        indicates (topic_i, topic_probability_j).
        """
        # The input as array of array
        check = lambda x: [x] if isinstance(x[0], tuple) else x
        docs = check(docs)
        X = [[] for i in range(0,len(docs))];
        for k,v in enumerate(docs):
            doc_topics = self[v]
            probs_docs = list(map(lambda x: x[1], doc_topics))
            # Everything should be equal in length
            if len(probs_docs) != self.num_topics:
                probs_docs.extend([1e-12]*(self.num_topics - len(probs_docs)))
            X[k] = probs_docs
            probs_docs = []
        return np.reshape(np.array(X), (len(docs), self.num_topics))

    def partial_fit(self, X):
        """
        Train model over X.
        """
        if sparse.issparse(X):
            X = matutils.Sparse2Corpus(X)
        self.add_documents(corpus=X)