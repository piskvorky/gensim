#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Scikit learn interface for :class:`~gensim.models.hdpmodel.HdpModel`.

Follows scikit-learn API conventions to facilitate using gensim along with scikit-learn.


Examples
--------
>>> from gensim.test.utils import common_dictionary, common_corpus
>>> from gensim.sklearn_api import HdpTransformer
>>>
>>> # Lets extract the distribution of each document in topics
>>> model = HdpTransformer(id2word=common_dictionary)
>>> distr = model.fit_transform(common_corpus)

"""
import numpy as np
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim import matutils


class HdpTransformer(TransformerMixin, BaseEstimator):
    """Base HDP module, wraps :class:`~gensim.models.hdpmodel.HdpModel`.

    For more information on the inner workings please take a look at
    the original class. The inner workings of this class heavily depends on `Wang, Paisley, Blei: "Online Variational
    Inference for the Hierarchical Dirichlet Process, JMLR (2011)"
    <http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf>`_.

    """

    def __init__(self, id2word, max_chunks=None, max_time=None, chunksize=256, kappa=1.0, tau=64.0, K=15, T=150,
                 alpha=1, gamma=1, eta=0.01, scale=1.0, var_converge=0.0001, outputdir=None, random_state=None):
        """

        Parameters
        ----------
        id2word : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Mapping between a words ID and the word itself in the vocabulary.
        max_chunks : int, optional
            Upper bound on how many chunks to process.It wraps around corpus beginning in another corpus pass,
            if there are not enough chunks in the corpus.
        max_time : int, optional
            Upper bound on time in seconds for which model will be trained.
        chunksize : int, optional
            Number of documents to be processed by the model in each mini-batch.
        kappa : float, optional
            Learning rate, see `Wang, Paisley, Blei: "Online Variational Inference for the Hierarchical Dirichlet
            Process, JMLR (2011)" <http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf>`_.
        tau : float, optional
            Slow down parameter, see `Wang, Paisley, Blei: "Online Variational Inference for the Hierarchical
            Dirichlet Process, JMLR (2011)" <http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf>`_.
        K : int, optional
            Second level truncation level, see `Wang, Paisley, Blei: "Online Variational Inference for the Hierarchical
            Dirichlet Process, JMLR (2011)" <http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf>`_.
        T : int, optional
            Top level truncation level, see `Wang, Paisley, Blei: "Online Variational Inference for the Hierarchical
            Dirichlet  Process, JMLR (2011)" <http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf>`_.
        alpha : int, optional
            Second level concentration, see `Wang, Paisley, Blei: "Online Variational Inference for the Hierarchical
            Dirichlet  Process, JMLR (2011)" <http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf>`_.
        gamma : int, optional
            First level concentration, see `Wang, Paisley, Blei: "Online Variational Inference for the Hierarchical
            Dirichlet  Process, JMLR (2011)" <http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf>`_.
        eta : float, optional
            The topic Dirichlet, see `Wang, Paisley, Blei: "Online Variational Inference for the Hierarchical
            Dirichlet  Process, JMLR (2011)" <http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf>`_.
        scale : float, optional
            Weights information from the mini-chunk of corpus to calculate rhot.
        var_converge : float, optional
            Lower bound on the right side of convergence. Used when updating variational parameters
            for a single document.
        outputdir : str, optional
            Path to a directory where topic and options information will be stored.
        random_state : int, optional
            Seed used to create a :class:`~np.random.RandomState`. Useful for obtaining reproducible results.

        """
        self.gensim_model = None
        self.id2word = id2word
        self.max_chunks = max_chunks
        self.max_time = max_time
        self.chunksize = chunksize
        self.kappa = kappa
        self.tau = tau
        self.K = K
        self.T = T
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.scale = scale
        self.var_converge = var_converge
        self.outputdir = outputdir
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {iterable of iterable of (int, int), scipy.sparse matrix}
            A collection of documents in BOW format used for training the model.

        Returns
        -------
        :class:`~gensim.sklearn_api.hdp.HdpTransformer`
            The trained model.

        """
        if sparse.issparse(X):
            corpus = matutils.Sparse2Corpus(sparse=X, documents_columns=False)
        else:
            corpus = X

        self.gensim_model = models.HdpModel(
            corpus=corpus, id2word=self.id2word, max_chunks=self.max_chunks,
            max_time=self.max_time, chunksize=self.chunksize, kappa=self.kappa, tau=self.tau,
            K=self.K, T=self.T, alpha=self.alpha, gamma=self.gamma, eta=self.eta, scale=self.scale,
            var_converge=self.var_converge, outputdir=self.outputdir, random_state=self.random_state
        )
        return self

    def transform(self, docs):
        """Infer a matrix of topic distribution for the given document bow, where a_ij
        indicates (topic_i, topic_probability_j).

        Parameters
        ----------
        docs : {iterable of list of (int, number), list of (int, number)}
            Document or sequence of documents in BOW format.

        Returns
        -------
        numpy.ndarray of shape [`len(docs), num_topics`]
            Topic distribution for `docs`.

        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        if isinstance(docs[0], tuple):
            docs = [docs]
        distribution, max_num_topics = [], 0

        for doc in docs:
            topicd = self.gensim_model[doc]
            distribution.append(topicd)
            max_num_topics = max(max_num_topics, max(topic[0] for topic in topicd) + 1)

        # returning dense representation for compatibility with sklearn
        # but we should go back to sparse representation in the future
        distribution = [matutils.sparse2full(t, max_num_topics) for t in distribution]
        return np.reshape(np.array(distribution), (len(docs), max_num_topics))

    def partial_fit(self, X):
        """Train model over a potentially incomplete set of documents.

        Uses the parameters set in the constructor.
        This method can be used in two ways:
        * On an unfitted model in which case the model is initialized and trained on `X`.
        * On an already fitted model in which case the model is **updated** by `X`.

        Parameters
        ----------
        X : {iterable of iterable of (int, int), scipy.sparse matrix}
            A collection of documents in BOW format used for training the model.

        Returns
        -------
        :class:`~gensim.sklearn_api.hdp.HdpTransformer`
            The trained model.

        """
        if sparse.issparse(X):
            X = matutils.Sparse2Corpus(sparse=X, documents_columns=False)

        if self.gensim_model is None:
            self.gensim_model = models.HdpModel(
                id2word=self.id2word, max_chunks=self.max_chunks,
                max_time=self.max_time, chunksize=self.chunksize, kappa=self.kappa, tau=self.tau,
                K=self.K, T=self.T, alpha=self.alpha, gamma=self.gamma, eta=self.eta, scale=self.scale,
                var_converge=self.var_converge, outputdir=self.outputdir, random_state=self.random_state
            )

        self.gensim_model.update(corpus=X)
        return self
