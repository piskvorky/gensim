#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Scikit learn interface for :class:`~gensim.corpora.dictionary.Dictionary`.

Follows scikit-learn API conventions to facilitate using gensim along with scikit-learn.

Examples
--------
>>> from gensim.sklearn_api import Text2BowTransformer
>>>
>>> # Get a corpus as an iterable of unicode strings.
>>> texts = [u'complier system computer', u'loading computer system']
>>>
>>> # Create a transformer..
>>> model = Text2BowTransformer()
>>>
>>> # Use sklearn-style `fit_transform` to get the BOW representation of each document.
>>> model.fit_transform(texts)
[[(0, 1), (1, 1), (2, 1)], [(1, 1), (2, 1), (3, 1)]]

"""
from six import string_types
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim.corpora import Dictionary
from gensim.utils import tokenize


class Text2BowTransformer(TransformerMixin, BaseEstimator):
    """Base Text2Bow module , wraps :class:`~gensim.corpora.dictionary.Dictionary`.

    For more information on the inner workings please take a look at the original class.

    """
    def __init__(self, prune_at=2000000, tokenizer=tokenize):
        """
        Parameters
        ----------
        prune_at : int, optional
            Total number of unique words. Dictionary will keep not more than `prune_at` words.
        tokenizer : callable (str -> list of str), optional
            A callable to split a document into a list of each terms, default is :func:`gensim.utils.tokenize`.

        """
        self.gensim_model = None
        self.prune_at = prune_at
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : iterable of str
            A collection of documents used for training the model.

        Returns
        -------
        :class:`~gensim.sklearn_api.text2bow.Text2BowTransformer`
            The trained model.

        """
        tokenized_docs = [list(self.tokenizer(x)) for x in X]
        self.gensim_model = Dictionary(documents=tokenized_docs, prune_at=self.prune_at)
        return self

    def transform(self, docs):
        """Get the BOW format for the `docs`.

        Parameters
        ----------
        docs : {iterable of str, str}
            A collection of documents to be transformed.

        Returns
        -------
        iterable of list (int, int) 2-tuples.
            The BOW representation of each document.

        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # input as python lists
        if isinstance(docs, string_types):
            docs = [docs]
        tokenized_docs = (list(self.tokenizer(doc)) for doc in docs)
        return [self.gensim_model.doc2bow(doc) for doc in tokenized_docs]

    def partial_fit(self, X):
        """Train model over a potentially incomplete set of documents.

        This method can be used in two ways:
            1. On an unfitted model in which case the dictionary is initialized and trained on `X`.
            2. On an already fitted model in which case the dictionary is **expanded** by `X`.

        Parameters
        ----------
        X : iterable of str
            A collection of documents used to train the model.

        Returns
        -------
        :class:`~gensim.sklearn_api.text2bow.Text2BowTransformer`
            The trained model.

        """
        if self.gensim_model is None:
            self.gensim_model = Dictionary(prune_at=self.prune_at)

        tokenized_docs = [list(self.tokenizer(x)) for x in X]
        self.gensim_model.add_documents(tokenized_docs)
        return self
