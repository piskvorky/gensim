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

        Parameters
        ----------

        num_topics : int, optional
            The number of requested latent topics to be extracted from the training corpus.
        id2word : dict of (int, str), optional
            Mapping from integer ID to words in the corpus. Used to determine vocabulary size and logging.
        chunksize : int, optional
            If `distributed` is True, this is the number of documents to be handled in each worker job.
        passes : int
            Number of passes through the corpus during online training.
        update_every : int
            Number of documents to be iterated through for each update. Set to 0 for batch learning, > 1 for online iterative learning.
        alpha : {np.array, str}
            Can be set to an 1D array of length equal to the number of expected topics that expresses
            our a-priori belief for the each topics' probability.
            Alternatively default prior selecting strategies can be employed by supplying a string:

                'asymmetric': Uses a fixed normalized assymetric prior of `1.0 / topicno`.
                'default': Learns an assymetric prior from the corpus.
        eta : {np.array, str}

        decay
        offset
        eval_every
        iterations
        gamma_threshold
        minimum_probability
        random_state
        scorer
        dtype


        `alpha` and `eta` are hyperparameters that affect sparsity of the document-topic
        (theta) and topic-word (lambda) distributions. Both default to a symmetric
        1.0/num_topics prior.

        `alpha` can be set to an explicit array = prior of your choice. It also
        support special values of 'asymmetric' and 'auto': the former uses a fixed
        normalized asymmetric 1.0/topicno prior, the latter learns an asymmetric
        prior directly from your data.

        `eta` can be a scalar for a symmetric prior over topic/word
        distributions, or a vector of shape num_words, which can be used to
        impose (user defined) asymmetric priors over the word distribution.
        It also supports the special value 'auto', which learns an asymmetric
        prior over words directly from your data. `eta` can also be a matrix
        of shape num_topics x num_words, which can be used to impose
        asymmetric priors over the word distribution on a per-topic basis
        (can not be learned from data).

        Turn on `distributed` to force distributed computing
        (see the `web tutorial <http://radimrehurek.com/gensim/distributed.html>`_
        on how to set up a cluster of machines for gensim).

        Calculate and log perplexity estimate from the latest mini-batch every
        `eval_every` model updates (setting this to 1 slows down training ~2x;
        default is 10 for better performance). Set to None to disable perplexity estimation.

        `decay` and `offset` parameters are the same as Kappa and Tau_0 in
        Hoffman et al, respectively.

        `minimum_probability` controls filtering the topics returned for a document (bow).

        `random_state` can be a np.random.RandomState object or the seed for one.

        `callbacks` a list of metric callbacks to log/visualize evaluation metrics of topic model during training.

        `dtype` is data-type to use during calculations inside model. All inputs are also converted to this dtype.
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
        """
        Fit the model according to the given training data.
        Calls gensim.models.LdaModel
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
        """
        Takes a list of documents as input ('docs').
        Returns a matrix of topic distribution for the given document bow, where a_ij
        indicates (topic_i, topic_probability_j).
        The input `docs` should be in BOW format and can be a list of documents like
        [[(4, 1), (7, 1)],
        [(9, 1), (13, 1)], [(2, 1), (6, 1)]]
        or a single document like : [(4, 1), (7, 1)]
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
        """
        Train model over X.
        By default, 'online (single-pass)' mode is used for training the LDA model.
        Configure `passes` and `update_every` params at init to choose the mode among :

            - online (single-pass): update_every != None and passes == 1
            - online (multi-pass): update_every != None and passes > 1
            - batch: update_every == None

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
        """
        Compute score reflecting how well the model has fit for the input data.
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
            raise ValueError("Invalid value of `scorer` param supplied")
