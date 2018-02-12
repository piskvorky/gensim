#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Scikit learn interface for `gensim.models.phrases`.

Follows scikit-learn API conventions to facilitate using gensim along with scikit-learn.

"""

from six import string_types
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models


class PhrasesTransformer(TransformerMixin, BaseEstimator):
    """Base Phrases module

    Wraps :class:`~gensim.models.phrases.Phrases`.
    For more information on the inner workings please take a look at
    the original class.

    """

    def __init__(self, min_count=5, threshold=10.0, max_vocab_size=40000000,
                 delimiter=b'_', progress_per=10000, scoring='default'):
        """Sklearn wrapper for Phrases model.

        Parameters are propagated to the original models constructor. For an explanation
        please refer to :meth:`~gensim.models.phrases.Phrases.__init__`

        Parameters
        ----------
        min_count : int
        threshold : float
        max_vocab_size : int
        delimiter : str
        progress_per : int
        scoring : str or callable

        """
        self.gensim_model = None
        self.min_count = min_count
        self.threshold = threshold
        self.max_vocab_size = max_vocab_size
        self.delimiter = delimiter
        self.progress_per = progress_per
        self.scoring = scoring

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : iterable of list of str
            Sequence of sentences to be used for training the model.

        Returns
        -------
        :class:`~gensim.sklearn_api.phrases.PhrasesTransformer`
            The trained model.

        """
        self.gensim_model = models.Phrases(
            sentences=X, min_count=self.min_count, threshold=self.threshold,
            max_vocab_size=self.max_vocab_size, delimiter=self.delimiter,
            progress_per=self.progress_per, scoring=self.scoring
        )
        return self

    def transform(self, docs):
        """Transform the input documents into phrase tokens.

        Words in the sentence will be joined by u`_`.

        Parameters
        ----------
        docs : iterable of list of str
            Sequence of sentences to be used transformed.

        Returns
        -------
        iterable of str
            Phrase representation for each of the input sentences.

        """

        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # input as python lists
        if isinstance(docs[0], string_types):
            docs = [docs]
        return [self.gensim_model[doc] for doc in docs]

    def partial_fit(self, X):
        """Train model over a potentially incomplete set of sentences.

        This method can be used in two ways:
            1. On an unfitted model in which case the model is initialized and trained on `X`.
            2. On an already fitted model in which case the X sentences are **added** to the vocabulary.

        Parameters
        ----------
        X : iterable of list of str
            Sequence of sentences to be used for training the model.

        Returns
        -------
        :class:`~gensim.sklearn_api.phrases.PhrasesTransformer`
            The trained model.

        """

        if self.gensim_model is None:
            self.gensim_model = models.Phrases(
                sentences=X, min_count=self.min_count, threshold=self.threshold,
                max_vocab_size=self.max_vocab_size, delimiter=self.delimiter,
                progress_per=self.progress_per, scoring=self.scoring
            )

        self.gensim_model.add_vocab(X)
        return self
