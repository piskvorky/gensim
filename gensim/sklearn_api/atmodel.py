#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Chinmaya Pancholi <chinmayapancholi13@gmail.com>
# Copyright (C) 2017 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Scikit learn interface for :class:`~gensim.models.atmodel.AuthorTopicModel`.

Follows scikit-learn API conventions to facilitate using gensim along with scikit-learn.

Examples
--------
.. sourcecode:: pycon

    >>> from gensim.test.utils import common_dictionary, common_corpus
    >>> from gensim.sklearn_api.atmodel import AuthorTopicTransformer
    >>>
    >>> # Pass a mapping from authors to the documents they contributed to.
    >>> author2doc = {
    ...     'john': [0, 1, 2, 3, 4, 5, 6],
    ...     'jane': [2, 3, 4, 5, 6, 7, 8],
    ...     'jack': [0, 2, 4, 6, 8]
    ... }
    >>>
    >>> # Lets use the model to discover 2 different topics.
    >>> model = AuthorTopicTransformer(id2word=common_dictionary, author2doc=author2doc, num_topics=2, passes=100)
    >>>
    >>> # In which of those 2 topics does jack mostly contribute to?
    >>> topic_dist = model.fit(common_corpus).transform('jack')

"""
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim import matutils


class AuthorTopicTransformer(TransformerMixin, BaseEstimator):
    """Base Author Topic module, wraps :class:`~gensim.models.atmodel.AuthorTopicModel`.

    The model's internal workings are heavily based on  `"The Author-Topic Model for Authors and Documents",
    Osen-Zvi et. al 2004 <https://mimno.infosci.cornell.edu/info6150/readings/398.pdf>`_.

    """
    def __init__(self, num_topics=100, id2word=None, author2doc=None, doc2author=None,
                 chunksize=2000, passes=1, iterations=50, decay=0.5, offset=1.0,
                 alpha='symmetric', eta='symmetric', update_every=1, eval_every=10,
                 gamma_threshold=0.001, serialized=False, serialization_path=None,
                 minimum_probability=0.01, random_state=None):
        """

        Parameters
        ----------
        num_topics : int, optional
            Number of requested latent topics to be extracted from the training corpus.
        id2word : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Mapping from a words' ID to the word itself. Used to determine the vocabulary size, as well as for debugging
            and topic printing.
        author2doc : dict of (str, list of int), optional
            Maps an authors name to a list of document IDs where has has contributed.
            Either `author2doc` or `doc2author` **must be supplied**.
        doc2author : dict of (int, list of str)
            Maps a document (using its ID) to a list of author names that contributed to it.
            Either `author2doc` or `doc2author` **must be supplied**.
        chunksize : int, optional
            Number of documents to be processed by the model in each mini-batch.
        passes : int, optional
            Number of times the model can make a pass over the corpus during training.
        iterations : int, optional
            Maximum number of times the model before convergence during the M step of the EM algorithm.
        decay : float, optional
            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten
            when each new document is examined. Corresponds to Kappa from `"The Author-Topic Model for Authors
            and Documents", Osen-Zvi et. al 2004 <https://mimno.infosci.cornell.edu/info6150/readings/398.pdf>`_.
        offset : float, optional
            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.
            Corresponds to Tau_0 from `"The Author-Topic Model for Authors and Documents", Osen-Zvi et. al 2004
            <https://mimno.infosci.cornell.edu/info6150/readings/398.pdf>`_.
        alpha : {np.ndarray, str}, optional
            Can be set to an 1D array of length equal to the number of expected topics that expresses
            our a-priori belief for the each topics' probability.
            Alternatively default prior selecting strategies can be employed by supplying a string:

                * 'asymmetric': Uses a fixed normalized asymmetric prior of `1.0 / topicno`.
                * 'auto': Learns an asymmetric prior from the corpus.
        eta : {float, np.array, str}, optional
            A-priori belief on word probability, this can be:

                * scalar for a symmetric prior over topic/word probability,
                * vector of length num_words to denote an asymmetric user defined probability for each word,
                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination,
                * the string 'auto' to learn the asymmetric prior from the data.
        update_every : int, optional
            Number of mini-batches between each model update.
        eval_every : int, optional
            Number of updates between two log perplexity estimates.
            Set to None to disable perplexity estimation.
        gamma_threshold : float, optional
            Minimum change in the value of the gamma parameters to continue iterating.
        serialized : bool, optional
            Indicates whether the input corpora to the model are simple in-memory lists (`serialized = False`)
            or saved to the hard-drive (`serialized = True`). Note that this behaviour is quite different from
            other Gensim models. If your data is too large to fit in to memory, use this functionality.
        serialization_path : str, optional
            Path to file that used for storing the serialized object, **must be supplied if `serialized = True`**.
            An existing file *cannot* be overwritten, either delete the old file or choose a different name.
        minimum_probability : float, optional
            Topics with a probability lower than this threshold will be filtered out.
        random_state : {np.random.RandomState, int}, optional
            Either a randomState object or a seed to generate one. Useful for reproducibility.

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
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : iterable of list of (int, number)
            Sequence of documents in BoW format.

        Returns
        -------
        :class:`~gensim.sklearn_api.atmodel.AuthorTopicTransformer`
            The trained model.

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
        """Infer the topic probabilities for each author.

        Parameters
        ----------
        author_names : {iterable of str, str}
            Author name or sequence of author names whose topics will be identified.

        Returns
        -------
        numpy.ndarray
            Topic distribution for each input author.

        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of arrays
        if not isinstance(author_names, list):
            author_names = [author_names]
        # returning dense representation for compatibility with sklearn
        # but we should go back to sparse representation in the future
        topics = [matutils.sparse2full(self.gensim_model[author_name], self.num_topics) for author_name in author_names]
        return np.reshape(np.array(topics), (len(author_names), self.num_topics))

    def partial_fit(self, X, author2doc=None, doc2author=None):
        """Train model over a potentially incomplete set of documents.

        This method can be used in two ways:
        * On an unfitted model in which case the model is initialized and trained on `X`.
        * On an already fitted model in which case the model is **updated** by `X`.


        Parameters
        ----------
        X : iterable of list of (int, number)
            Sequence of documents in BoW format.
        author2doc : dict of (str, list of int), optional
            Maps an authors name to a list of document IDs where has has contributed.
            Either `author2doc` or `doc2author` **must be supplied**.
        doc2author : dict of (int, list of str)
            Maps a document (using its ID) to a list of author names that contributed to it.
            Either `author2doc` or `doc2author` **must be supplied**.

        Returns
        -------
        :class:`~gensim.sklearn_api.atmodel.AuthorTopicTransformer`
            The trained model.

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
