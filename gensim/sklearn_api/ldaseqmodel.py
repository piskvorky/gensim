#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Chinmaya Pancholi <chinmayapancholi13@gmail.com>
# Copyright (C) 2017 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Scikit learn interface for :class:`~gensim.models.ldaseqmodel.LdaSeqModel`.

Follows scikit-learn API conventions to facilitate using gensim along with scikit-learn.

Examples
--------

>>> from gensim.test.utils import common_corpus, common_dictionary
>>> from gensim.sklearn_api.ldaseqmodel import LdaSeqTransformer
>>>
>>> # Create a sequential LDA transformer to extract 2 topics from the common corpus.
>>> # Divide the work into 3 unequal time slices.
>>> model = LdaSeqTransformer(id2word=common_dictionary, num_topics=2, time_slice=[3, 4, 2], initialize='gensim')
>>>
>>> # Each document almost entirely belongs to one of the two topics.
>>> transformed_corpus = model.fit_transform(common_corpus)

"""
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models


class LdaSeqTransformer(TransformerMixin, BaseEstimator):
    """Base Sequential LDA module.

    Wraps :class:`~gensim.models.ldaseqmodel.LdaSeqModel`.
    For more information on the inner workings please take a look at
    the original class.


    """

    def __init__(self, time_slice=None, id2word=None, alphas=0.01, num_topics=10, initialize='gensim', sstats=None,
                 lda_model=None, obs_variance=0.5, chain_variance=0.005, passes=10, random_state=None,
                 lda_inference_max_iter=25, em_min_iter=6, em_max_iter=20, chunksize=100):
        """Sklearn wrapper for  :class:`~gensim.models.ldaseqmodel.LdaSeqModel` model.

        Parameters
        ----------
        time_slice : list of int, optional
            Contains the number of documents in each time-slice.
        id2word : dict of (int, str)
            Mapping from an ID to the word it represents in the vocabulary.
        alphas : float
            The prior probability of each topic.
        num_topics : int
            Number of latent topics to be discovered in the corpus.
        initialize : str {'gensim', 'own', 'ldamodel'}
            Controls the initialization of the DTM model. Supports three different modes:
                * 'gensim', default: Uses gensim's own LDA initialization.
                * 'own': Uses your own initialization matrix of an LDA model that has been previously trained.
                * 'lda_model': Use a previously used LDA model, passing it through the `lda_model` argument.
        sstats : np.ndarray of shape (vocab_len, `num_topics`)
            If `initialize` is set to 'own' this will be used to initialize the DTM model.
        lda_model : :class:`~gensim.models.ldamodel.LdaModel`
            If `initialize` is set to 'lda_model' this object will be used to create the `sstats` initialization matrix.
        obs_variance : float
            Observed variance used to approximate the true and forward variance as shown in
            `David M. Blei, John D. Lafferty: "Dynamic Topic Models"
            <https://mimno.infosci.cornell.edu/info6150/readings/398.pdf>`_.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve.
        passes : int
            Number of passes over the corpus for the initial :class:`~gensim.models.ldamodel.LdaModel`
        random_state : {np.random.RandomState, int}
            Can be a np.random.RandomState object, or the seed to generate one. Used for reproducibility of results.
        lda_inference_max_iter : int
            Maximum number of iterations in the inference step of the LDA training.
        en_min_iter : int
            Minimum number of iterations until converge of the Expectation-Maximization algorithm
        en_max_iter : int
            Maximum number of iterations until converge of the Expectation-Maximization algorithm
        chunksize : int
            Number of documents in the corpus do be processed in in a chunk.

        """
        self.gensim_model = None
        self.time_slice = time_slice
        self.id2word = id2word
        self.alphas = alphas
        self.num_topics = num_topics
        self.initialize = initialize
        self.sstats = sstats
        self.lda_model = lda_model
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance
        self.passes = passes
        self.random_state = random_state
        self.lda_inference_max_iter = lda_inference_max_iter
        self.em_min_iter = em_min_iter
        self.em_max_iter = em_max_iter
        self.chunksize = chunksize

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {iterable of iterable of (int, int), scipy.sparse matrix}
            A collection of documents in BOW format used for training the model.

        Returns
        -------
        :class:`~gensim.sklearn_api.ldaseqmodel.LdaSeqTransformer`
            The trained model.

        """
        self.gensim_model = models.LdaSeqModel(
            corpus=X, time_slice=self.time_slice, id2word=self.id2word,
            alphas=self.alphas, num_topics=self.num_topics, initialize=self.initialize, sstats=self.sstats,
            lda_model=self.lda_model, obs_variance=self.obs_variance, chain_variance=self.chain_variance,
            passes=self.passes, random_state=self.random_state, lda_inference_max_iter=self.lda_inference_max_iter,
            em_min_iter=self.em_min_iter, em_max_iter=self.em_max_iter, chunksize=self.chunksize
        )
        return self

    def transform(self, docs):
        """

        Parameters
        ----------
        docs : {iterable of iterable of (int, int), scipy.sparse matrix}
            A collection of documents in BOW format to be transformed.

        Returns
        -------
        np.ndarray of shape (`len(docs)`, `num_topics`)
            The topic representation of each document.

        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        if isinstance(docs[0], tuple):
            docs = [docs]
        proportions = [self.gensim_model[doc] for doc in docs]
        return np.reshape(np.array(proportions), (len(docs), self.num_topics))
