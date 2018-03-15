#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Scikit learn interface for `gensim.models.phrases.Phrases`.

Follows scikit-learn API conventions to facilitate using gensim along with scikit-learn.

Examples
--------
>>> from gensim.sklearn_api.phrases import PhrasesTransformer
>>>
>>> # Create the model. Make sure no term is ignored and combinations seen 3+ times are captured.
>>> m = PhrasesTransformer(min_count=1, threshold=3)
>>> texts = [
...   ['I', 'love', 'computer', 'science'],
...   ['computer', 'science', 'is', 'my', 'passion'],
...   ['I', 'studied', 'computer', 'science']
... ]
>>>
>>> # Use sklearn fit_transform to see the transformation.
>>> # Since computer and science were seen together 3+ times they are considered a phrase.
>>> assert ['I', 'love', 'computer_science'] == m.fit_transform(texts)[0]

"""
from six import string_types
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models


class PhrasesTransformer(TransformerMixin, BaseEstimator):
    """Base Phrases module, wraps :class:`~gensim.models.phrases.Phrases`.

    For more information, please have a look to `Mikolov, et. al: "Efficient Estimation of Word Representations in
    Vector Space" <https://arxiv.org/pdf/1301.3781.pdf>`_ and `Gerlof Bouma: "Normalized (Pointwise) Mutual Information
    in Collocation Extraction" <https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf>`_.

    """
    def __init__(self, min_count=5, threshold=10.0, max_vocab_size=40000000,
                 delimiter=b'_', progress_per=10000, scoring='default'):
        """

        Parameters
        ----------
        min_count : int, optional
            Terms with a count lower than this will be ignored
        threshold : float, optional
            Only phrases scoring above this will be accepted, see `scoring` below.
        max_vocab_size : int, optional
            Maximum size of the vocabulary. Used to control pruning of less common words, to keep memory under control.
            The default of 40M needs about 3.6GB of RAM.
        delimiter : str, optional
            Character used to join collocation tokens, should be a byte string (e.g. b'_').
        progress_per : int, optional
            Training will report to the logger every that many phrases are learned.
        scoring : str or function, optional
            Specifies how potential phrases are scored for comparison to the `threshold`
            setting. `scoring` can be set with either a string that refers to a built-in scoring function,
            or with a function with the expected parameter names. Two built-in scoring functions are available
            by setting `scoring` to a string:

                * 'default': Explained in `Mikolov, et. al: "Efficient Estimation of Word Representations
                  in Vector Space" <https://arxiv.org/pdf/1301.3781.pdf>`_.
                * 'npmi': Explained in `Gerlof Bouma: "Normalized (Pointwise) Mutual Information in Collocation
                  Extraction" <https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf>`_.

            'npmi' is more robust when dealing with common words that form part of common bigrams, and
            ranges from -1 to 1, but is slower to calculate than the default.

            To use a custom scoring function, create a function with the following parameters and set the `scoring`
            parameter to the custom function, see :func:`~gensim.models.phrases.original_scorer` as example.
            You must define all the parameters (but can use only part of it):

                * worda_count: number of occurrences in `sentences` of the first token in the phrase being scored
                * wordb_count: number of occurrences in `sentences` of the second token in the phrase being scored
                * bigram_count: number of occurrences in `sentences` of the phrase being scored
                * len_vocab: the number of unique tokens in `sentences`
                * min_count: the `min_count` setting of the Phrases class
                * corpus_word_count: the total number of (non-unique) tokens in `sentences`

            A scoring function without any of these parameters (even if the parameters are not used) will
            raise a ValueError on initialization of the Phrases class. The scoring function must be pickleable.

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

        Words in the sentence will be joined by `self.delimiter`.

        Parameters
        ----------
        docs : {iterable of list of str, list of str}
            Sequence of documents to be used transformed.

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
