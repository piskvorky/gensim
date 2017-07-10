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
import six
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim.sklearn_api import BaseTransformer


class W2VTransformer(BaseTransformer, TransformerMixin, BaseEstimator):
    """
    Base Word2Vec module
    """

    def __init__(self, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=10000):
        """
        Sklearn wrapper for Word2Vec model. Class derived from gensim.models.Word2Vec
        """
        self.gensim_model = None
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words

    def get_params(self, deep=True):
        """
        Returns all parameters as dictionary.
        """
        return {"size": self.size, "alpha": self.alpha, "window": self.window, "min_count": self.min_count,
        "max_vocab_size": self.max_vocab_size, "sample": self.sample, "seed": self.seed,
        "workers": self.workers, "min_alpha": self.min_alpha, "sg": self.sg, "hs": self.hs,
        "negative": self.negative, "cbow_mean": self.cbow_mean, "hashfxn": self.hashfxn,
        "iter": self.iter, "null_word": self.null_word, "trim_rule": self.trim_rule,
        "sorted_vocab": self.sorted_vocab, "batch_words": self.batch_words}

    def set_params(self, **parameters):
        """
        Set all parameters.
        """
        super(W2VTransformer, self).set_params(**parameters)
        return self

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        Calls gensim.models.Word2Vec
        """
        self.gensim_model = models.Word2Vec(sentences=X, size=self.size, alpha=self.alpha,
            window=self.window, min_count=self.min_count, max_vocab_size=self.max_vocab_size,
            sample=self.sample, seed=self.seed, workers=self.workers, min_alpha=self.min_alpha,
            sg=self.sg, hs=self.hs, negative=self.negative, cbow_mean=self.cbow_mean,
            hashfxn=self.hashfxn, iter=self.iter, null_word=self.null_word, trim_rule=self.trim_rule,
            sorted_vocab=self.sorted_vocab, batch_words=self.batch_words)
        return self

    def transform(self, words):
        """
        Return the word-vectors for the input list of words.
        """
        if self.gensim_model is None:
            raise NotFittedError("This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method.")

        # The input as array of array
        check = lambda x: [x] if isinstance(x, six.string_types) else x
        words = check(words)
        X = [[] for _ in range(0, len(words))]

        for k, v in enumerate(words):
            word_vec = self.gensim_model[v]
            X[k] = word_vec

        return np.reshape(np.array(X), (len(words), self.size))

    def partial_fit(self, X):
        raise NotImplementedError("'partial_fit' has not been implemented for W2VTransformer")
