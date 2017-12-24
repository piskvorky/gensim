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


class AuthorTopicTransformer(TransformerMixin, BaseEstimator):
    """
    Base AuthorTopic module
    """

    def __init__(self, num_topics=100, id2word=None, author2doc=None, doc2author=None,
                 chunksize=2000, passes=1, iterations=50, decay=0.5, offset=1.0,
                 alpha='symmetric', eta='symmetric', update_every=1, eval_every=10,
                 gamma_threshold=0.001, serialized=False, serialization_path=None,
                 minimum_probability=0.01, random_state=None):
        """
        Sklearn wrapper for AuthorTopic model. See gensim.models.AuthorTopicModel for parameter details.
        """
        self.gensim_model = None
        self.num_topics = num_topics
        self.id2word = id2word
        self.author2doc = author2doc
        self.doc2author = doc2author
        self.chunksize = chunksize
        self.passes = passes
        self.iterations = iterations
        self.decay = decay
        self.offset = offset
        self.alpha = alpha
        self.eta = eta
        self.update_every = update_every
        self.eval_every = eval_every
        self.gamma_threshold = gamma_threshold
        self.serialized = serialized
        self.serialization_path = serialization_path
        self.minimum_probability = minimum_probability
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the model according to the given training data.
        Calls gensim.models.AuthorTopicModel
        """
        self.gensim_model = models.AuthorTopicModel(
            corpus=X, num_topics=self.num_topics, id2word=self.id2word,
            author2doc=self.author2doc, doc2author=self.doc2author, chunksize=self.chunksize, passes=self.passes,
            iterations=self.iterations, decay=self.decay, offset=self.offset, alpha=self.alpha, eta=self.eta,
            update_every=self.update_every, eval_every=self.eval_every, gamma_threshold=self.gamma_threshold,
            serialized=self.serialized, serialization_path=self.serialization_path,
            minimum_probability=self.minimum_probability, random_state=self.random_state
        )
        return self

    def transform(self, author_names):
        """
        Return topic distribution for input authors as a list of
        (topic_id, topic_probabiity) 2-tuples.
        """
        # The input as array of array
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        if not isinstance(author_names, list):
            author_names = [author_names]
        # returning dense representation for compatibility with sklearn
        # but we should go back to sparse representation in the future
        topics = [matutils.sparse2full(self.gensim_model[author_name], self.num_topics) for author_name in author_names]
        return np.reshape(np.array(topics), (len(author_names), self.num_topics))

    def partial_fit(self, X, author2doc=None, doc2author=None):
        """
        Train model over X.
        """
        if self.gensim_model is None:
            self.gensim_model = models.AuthorTopicModel(
                corpus=X, num_topics=self.num_topics, id2word=self.id2word,
                author2doc=self.author2doc, doc2author=self.doc2author, chunksize=self.chunksize, passes=self.passes,
                iterations=self.iterations, decay=self.decay, offset=self.offset, alpha=self.alpha, eta=self.eta,
                update_every=self.update_every, eval_every=self.eval_every, gamma_threshold=self.gamma_threshold,
                serialized=self.serialized, serialization_path=self.serialization_path,
                minimum_probability=self.minimum_probability, random_state=self.random_state
            )

        self.gensim_model.update(corpus=X, author2doc=author2doc, doc2author=doc2author)
        return self
