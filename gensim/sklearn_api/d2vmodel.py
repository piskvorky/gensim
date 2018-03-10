#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Scikit learn interface for gensim for easy use of gensim with scikit-learn
Follows scikit-learn API conventions
"""
import warnings
import numpy as np
from six import string_types
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim.models import doc2vec


class D2VTransformer(TransformerMixin, BaseEstimator):
    """
    Base Doc2Vec module
    """

    def __init__(self, dm_mean=None, dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1, docvecs=None,
                 docvecs_mapfile=None, comment=None, trim_rule=None, size=None, vector_size=100, alpha=0.025, window=5,
                 min_count=5, max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001, hs=0, negative=5,
                 cbow_mean=1, hashfxn=hash, iter=None, epochs=5, sorted_vocab=1, batch_words=10000):
        """
        Sklearn api for Doc2Vec model. See gensim.models.Doc2Vec and gensim.models.Word2Vec for parameter details.
        """
        
        if iter is not None:
            warnings.warn("The parameter `iter` is deprecated, will be removed in 4.0.0, use `epochs` instead.")
            epochs = iter

        if size is not None:
            warnings.warn("The parameter `size` is deprecated, will be removed in 4.0.0, use `vector_size` instead.")
            vector_size = size
          
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
        self.vector_size = vector_size
        self.size = vector_size
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
        self.epochs = epochs
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        
    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        Calls gensim.models.Doc2Vec
        """
        if isinstance(X[0], doc2vec.TaggedDocument):
            d2v_sentences = X
        else:
            d2v_sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(X)]
        self.gensim_model = models.Doc2Vec(
            documents=d2v_sentences, dm_mean=self.dm_mean, dm=self.dm,
            dbow_words=self.dbow_words, dm_concat=self.dm_concat, dm_tag_count=self.dm_tag_count,
            docvecs=self.docvecs, docvecs_mapfile=self.docvecs_mapfile, comment=self.comment,
            trim_rule=self.trim_rule, size=self.vector_size, alpha=self.alpha, window=self.window,
            min_count=self.min_count, max_vocab_size=self.max_vocab_size, sample=self.sample,
            seed=self.seed, workers=self.workers, min_alpha=self.min_alpha, hs=self.hs,
            negative=self.negative, cbow_mean=self.cbow_mean, hashfxn=self.hashfxn,
            epochs=self.epochs, sorted_vocab=self.sorted_vocab, batch_words=self.batch_words
        )
        return self

    def transform(self, docs):
        """
        Return the vector representations for the input documents.
        The input `docs` should be a list of lists like
        [['calculus', 'mathematical'],
        ['geometry', 'operations', 'curves']]
        or a single document like : ['calculus', 'mathematical']
        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        if isinstance(docs[0], string_types):
            docs = [docs]
        vectors = [self.gensim_model.infer_vector(doc) for doc in docs]
        return np.reshape(np.array(vectors), (len(docs), self.gensim_model.vector_size))
