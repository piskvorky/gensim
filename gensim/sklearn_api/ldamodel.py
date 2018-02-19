#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Chinmaya Pancholi <chinmayapancholi13@gmail.com>
# Copyright (C) 2017 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Scikit learn interface for :class:`~gensim.models.ldamodel.LdaModel`.

Follows scikit-learn API conventions to facilitate using gensim along with scikit-learn.

Examples
--------


"""

import numpy as np
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim import matutils


class LdaTransformer(TransformerMixin, BaseEstimator):
    """Base LDA module.

    Wraps :class:`~gensim.models.ldamodel.LdaModel`.
    For more information on the inner workings please take a look at
    the original class.

    """

    def __init__(self, num_topics=100, id2word=None, chunksize=2000, passes=1, update_every=1, alpha='symmetric',
                 eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001,
                 minimum_probability=0.01, random_state=None, scorer='perplexity', dtype=np.float32):

        """Sklearn wrapper for LDA model.
        Based on [1]_.

        Notes
        -----
        Configure `passes` and `update_every` params to choose the mode among:

            - online (single-pass): update_every != None and passes == 1
            - online (multi-pass): update_every != None and passes > 1
            - batch: update_every == None

        By default, 'online (single-pass)' mode is used for training the LDA model.

        References
        ----------
        .. [1] Matthew D. Hoffman, David M. Blei, Francis Bach, "Online Learning for Latent Dirichlet Allocation",
               NIPS'10 Proceedings of the 23rd International Conference on Neural Information Processing Systems -
               Volume 1 Pages 856-864, https://www.di.ens.fr/~fbach/mdhnips2010.pdf

        Parameters
        ----------
        num_topics : int, optional
            The number of requested latent topics to be extracted from the training corpus.
        id2word : dict of (int, str), optional
            Mapping from integer ID to words in the corpus. Used to determine vocabulary size and logging.
        chunksize : int, optional
            If `distributed` is True, this is the number of documents to be handled in each worker job.
        passes : int, optional
            Number of passes through the corpus during online training.
        update_every : int, optional
            Number of documents to be iterated through for each update.
            Set to 0 for batch learning, > 1 for online iterative learning.
        alpha : {np.array, str}, optional
            Can be set to an 1D array of length equal to the number of expected topics that expresses
            our a-priori belief for the each topics' probability.
            Alternatively default prior selecting strategies can be employed by supplying a string:
                'asymmetric': Uses a fixed normalized assymetric prior of `1.0 / topicno`.
                'default': Learns an assymetric prior from the corpus.
        eta : {float, np.array, str}, optional
            A-priori belief on word probability. This can be:
                a scalar for a symmetric prior over topic/word probability
                a vector : of length num_words to denote an assymetric user defined probability for each word.
                a matrix of shape (`num_topics`, num_words) to assign a probability for each word condition on each topic.
                the string 'auto' to learn the asymmetric prior from the data.
        decay : float, optional
            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten
            when each new document is examined. Corresponds to Kappa from [1]_.
        offset : float, optional
            Hyperparameter that controls how much we will slow down the first steps the first few iterations.
            Corresponds to Tau_0 from [1]_.
        eval_every : int, optional
            Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.
        iterations : int, optional
            Maximum number of iterations through the corpus when infering the topic distribution of a corpus.
        gamma_threshold : float, optional
            Minimum change in the value of the gamma parameters to continue iterating.
        minimum_probability : float, optional
            Topics with a probability lower than this threshold will be filtered out.
        random_state : {np.random.RandomState, int}, optional
            Either a randomState object or a seed to generate one. Useful for reproducibility.
        scorer : str, optional
            Method to compute a score reflecting how well the model has fit the input corpus.
            Allowed values are:
                'perplexity': Minimize the model's perplexity.
                'mass_u': Use :class:`~gensim.models.coherencemodel.CoherenceModel` to compute a topics coherence.
        dtype : type, optional
            Data-type to use during calculations inside model. All inputs are also converted.
            Available types: `numpy.float16`, `numpy.float32`, `numpy.float64`.

        """

        self.gensim_model = None
        self.num_topics = num_topics
        self.id2word = id2word
        self.chunksize = chunksize
        self.passes = passes
        self.update_every = update_every
        self.alpha = alpha
        self.eta = eta
        self.decay = decay
        self.offset = offset
        self.eval_every = eval_every
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold
        self.minimum_probability = minimum_probability
        self.random_state = random_state
        self.scorer = scorer
        self.dtype = dtype

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {iterable of iterable of (int, int), scipy.sparse matrix}
            A collection of documents in BOW format used for training the model.

        Returns
        -------
        :class:`~gensim.sklearn_api.ldamodel.LdaTransformer`
            The trained model.

        """
        if sparse.issparse(X):
            corpus = matutils.Sparse2Corpus(sparse=X, documents_columns=False)
        else:
            corpus = X

        self.gensim_model = models.LdaModel(
            corpus=corpus, num_topics=self.num_topics, id2word=self.id2word,
            chunksize=self.chunksize, passes=self.passes, update_every=self.update_every,
            alpha=self.alpha, eta=self.eta, decay=self.decay, offset=self.offset,
            eval_every=self.eval_every, iterations=self.iterations,
            gamma_threshold=self.gamma_threshold, minimum_probability=self.minimum_probability,
            random_state=self.random_state, dtype=self.dtype
        )
        return self

    def transform(self, docs):
        """Return the BOW format for the input documents.

        Parameters
        ----------
        docs : iterable of iterable of (int, int)
            A collection of documents in BOW format to be transformed.

        Returns
        -------
        np.array of shape (`len(docs)`, `num_topics`)
            The topic distribution for each input document.

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
        distribution = [matutils.sparse2full(self.gensim_model[doc], self.num_topics) for doc in docs]
        return np.reshape(np.array(distribution), (len(docs), self.num_topics))

    def partial_fit(self, X):
        """Train model over a potentially incomplete set of documents.

        Uses the parameters set in the constructor.
        This method can be used in two ways:
            1. On an unfitted model in which case the model is initialized and trained on `X`.
            2. On an already fitted model in which case the model is **updated** by `X`.

        Parameters
        ----------
        X : {iterable of iterable of (int, int), scipy.sparse matrix}
            A collection of documents in BOW format used for training the model.

        Returns
        -------
        :class:`~gensim.sklearn_api.ldamodel.LdaTransformer`
            The trained model.

        """
        if sparse.issparse(X):
            X = matutils.Sparse2Corpus(sparse=X, documents_columns=False)

        if self.gensim_model is None:
            self.gensim_model = models.LdaModel(
                num_topics=self.num_topics, id2word=self.id2word,
                chunksize=self.chunksize, passes=self.passes, update_every=self.update_every,
                alpha=self.alpha, eta=self.eta, decay=self.decay, offset=self.offset,
                eval_every=self.eval_every, iterations=self.iterations, gamma_threshold=self.gamma_threshold,
                minimum_probability=self.minimum_probability, random_state=self.random_state,
                dtype=self.dtype
            )

        self.gensim_model.update(corpus=X)
        return self

    def score(self, X, y=None):
        """Compute score reflecting how well the model has fitted for the input data.

        The scoring method is set using the `scorer` argument in :meth:`~gensim.sklearn_api.ldamodel.LdaTransformer`.
        Higher score is better.

        Parameters
        ----------
        X : iterable of iterable of (int, int)
            Input corpus in BOW format.

        Returns
        -------
        float
            The score computed based on the selected method.

        """
        if self.scorer == 'perplexity':
            corpus_words = sum(cnt for document in X for _, cnt in document)
            subsample_ratio = 1.0
            perwordbound = \
                self.gensim_model.bound(X, subsample_ratio=subsample_ratio) / (subsample_ratio * corpus_words)
            return -1 * np.exp2(-perwordbound)  # returning (-1*perplexity) to select model with minimum value
        elif self.scorer == 'u_mass':
            goodcm = models.CoherenceModel(model=self.gensim_model, corpus=X, coherence=self.scorer, topn=3)
            return goodcm.get_coherence()
        else:
            raise ValueError("Invalid value {} supplied for `scorer` param".format(self.scorer))
