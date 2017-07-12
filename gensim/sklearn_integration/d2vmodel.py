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
from six import string_types
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim.sklearn_integration import BaseSklearnWrapper


class D2VTransformer(BaseSklearnWrapper, TransformerMixin, BaseEstimator):
    """
    Base Doc2Vec module
    """

    def __init__(self, dm_mean=None, dm=1, dbow_words=0, dm_concat=0,
                dm_tag_count=1, docvecs=None, docvecs_mapfile=None,
                comment=None, trim_rule=None, size=100, alpha=0.025,
                window=5, min_count=5, max_vocab_size=None, sample=1e-3,
                seed=1, workers=3, min_alpha=0.0001, hs=0, negative=5,
                cbow_mean=1, hashfxn=hash, iter=5, sorted_vocab=1,
                batch_words=10000):
        """
        Sklearn api for Doc2Vec model. See gensim.models.Doc2Vec and gensim.models.Word2Vec for parameter details.
        """
        self.gensim_model = None
        self.dm_mean = dm_mean
        self.dm = dm
        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        self.docvecs = docvecs
        self.docvecs_mapfile = docvecs_mapfile
        self.comment = comment
        self.trim_rule = trim_rule

        # attributes associated with gensim.models.Word2Vec
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.iter = iter
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words

    def get_params(self, deep=True):
        """
        Return all parameters as dictionary.
        """
        return {"dm_mean": self.dm_mean, "dm": self.dm, "dbow_words": self.dbow_words,
        "dm_concat": self.dm_concat, "dm_tag_count": self.dm_tag_count, "docvecs": self.docvecs,
        "docvecs_mapfile": self.docvecs_mapfile, "comment": self.comment, "trim_rule": self.trim_rule,
        "size": self.size, "alpha": self.alpha, "window": self.window, "min_count": self.min_count,
        "max_vocab_size": self.max_vocab_size, "sample": self.sample, "seed": self.seed,
        "workers": self.workers, "min_alpha": self.min_alpha, "hs": self.hs,
        "negative": self.negative, "cbow_mean": self.cbow_mean, "hashfxn": self.hashfxn,
        "iter": self.iter, "sorted_vocab": self.sorted_vocab, "batch_words": self.batch_words}

    def set_params(self, **parameters):
        """
        Set parameters
        """
        super(D2VTransformer, self).set_params(**parameters)
        return self

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        Calls gensim.models.Doc2Vec
        """
        self.gensim_model = models.Doc2Vec(documents=X, dm_mean=self.dm_mean, dm=self.dm,
            dbow_words=self.dbow_words, dm_concat=self.dm_concat, dm_tag_count=self.dm_tag_count,
            docvecs=self.docvecs, docvecs_mapfile=self.docvecs_mapfile, comment=self.comment,
            trim_rule=self.trim_rule, size=self.size, alpha=self.alpha, window=self.window,
            min_count=self.min_count, max_vocab_size=self.max_vocab_size, sample=self.sample,
            seed=self.seed, workers=self.workers, min_alpha=self.min_alpha, hs=self.hs,
            negative=self.negative, cbow_mean=self.cbow_mean, hashfxn=self.hashfxn,
            iter=self.iter, sorted_vocab=self.sorted_vocab, batch_words=self.batch_words)
        return self

    def transform(self, docs):
        """
        Return the vector representations for the input documents.
        The input `docs` should be a list of lists like : [ ['calculus', 'mathematical'], ['geometry', 'operations', 'curves'] ]
        or a single document like : ['calculus', 'mathematical']
        """
        if self.gensim_model is None:
            raise NotFittedError("This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method.")

        # The input as array of array
        check = lambda x: [x] if isinstance(x[0], string_types) else x
        docs = check(docs)
        X = [[] for _ in range(0, len(docs))]

        for k, v in enumerate(docs):
            doc_vec = self.gensim_model.infer_vector(v)
            X[k] = doc_vec

        return np.reshape(np.array(X), (len(docs), self.gensim_model.vector_size))

    def partial_fit(self, X):
        raise NotImplementedError("'partial_fit' has not been implemented for D2VTransformer")
