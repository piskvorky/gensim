#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Chinmaya Pancholi <chinmayapancholi13@gmail.com>
# Copyright (C) 2017 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Scikit learn interface for :class:`~gensim.models.rpmodel.RpModel`.

Follows scikit-learn API conventions to facilitate using gensim along with scikit-learn.

Examples
--------
>>> from gensim.sklearn_api.rpmodel import RpTransformer
>>> from gensim.test.utils import common_dictionary, common_corpus
>>>
>>> # Initialize and fit the model.
>>> model = RpTransformer(id2word=common_dictionary).fit(common_corpus)
>>>
>>> # Use the trained model to transform a document.
>>> result = model.transform(common_corpus[3])

"""

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim import matutils


class RpTransformer(TransformerMixin, BaseEstimator):
    """Base Word2Vec module, wraps :class:`~gensim.models.rpmodel.RpModel`.

    For more information please have a look to `Random projection <https://en.wikipedia.org/wiki/Random_projection>`_.

    """
    def __init__(self, id2word=None, num_topics=300):
        """

        Parameters
        ----------
        id2word : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Mapping `token_id` -> `token`, will be determined from corpus if `id2word == None`.
        num_topics : int, optional
            Number of dimensions.

        """
        self.gensim_model = None
        self.id2word = id2word
        self.num_topics = num_topics

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : iterable of list of (int, number)
            Input corpus in BOW format.

        Returns
        -------
        :class:`~gensim.sklearn_api.rpmodel.RpTransformer`
            The trained model.

        """
        self.gensim_model = models.RpModel(corpus=X, id2word=self.id2word, num_topics=self.num_topics)
        return self

    def transform(self, docs):
        """Find the Random Projection factors for `docs`.

        Parameters
        ----------
        docs : {iterable of iterable of (int, int), list of (int, number)}
            Document or documents to be transformed in BOW format.

        Returns
        -------
        numpy.ndarray of shape [`len(docs)`, `num_topics`]
            RP representation for each input document.

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
        presentation = [matutils.sparse2full(self.gensim_model[doc], self.num_topics) for doc in docs]
        return np.reshape(np.array(presentation), (len(docs), self.num_topics))
