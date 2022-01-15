#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Based on Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>

"""Lda Sequence model, inspired by `David M. Blei, John D. Lafferty: "Dynamic Topic Models"
<https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.
The original C/C++ implementation can be found on `blei-lab/dtm <https://github.com/blei-lab/dtm>`_.


TODO: The next steps to take this forward would be:

#. Include DIM mode. Most of the infrastructure for this is in place.
#. See if LdaPost can be replaced by LdaModel completely without breaking anything.
#. Try and make it distributed, especially around the E and M step.
#. Remove all C/C++ coding style/syntax.

Examples
--------

Set up a model using have 30 documents, with 5 in the first time-slice, 10 in the second, and 15 in the third

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_corpus
    >>> from gensim.models import LdaSeqModel
    >>>
    >>> ldaseq = LdaSeqModel(corpus=common_corpus, time_slice=[2, 4, 3], num_topics=2, chunksize=1)

Persist a model to disk and reload it later

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> temp_file = datapath("model")
    >>> ldaseq.save(temp_file)
    >>>
    >>> # Load a potentially pre-trained model from disk.
    >>> ldaseq = LdaSeqModel.load(temp_file)

Access the document embeddings generated from the DTM

.. sourcecode:: pycon

    >>> doc = common_corpus[1]
    >>>
    >>> embedding = ldaseq[doc]

"""

import logging

import numpy as np
from gensim import utils, matutils
from gensim.models import ldamodel
from .ldaseq_sslm_inner import fit_sslm, sslm_counts_init
from .ldaseq_posterior_inner import fit_lda_post

logger = logging.getLogger(__name__)


class LdaSeqModel(utils.SaveLoad):
    """Estimate Dynamic Topic Model parameters based on a training corpus."""

    def __init__(
            self, corpus=None, time_slice=None, id2word=None, alphas=0.01, num_topics=10,
            initialize='gensim', sstats=None, lda_model=None, obs_variance=0.5, chain_variance=0.005, passes=10,
            random_state=None, lda_inference_max_iter=25, em_min_iter=6, em_max_iter=20, chunksize=100,
    ):
        """

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
            If not given, the model is left untrained (presumably because you want to call
            :meth:`~gensim.models.ldamodel.LdaSeqModel.update` manually).
        time_slice : list of int, optional
            Number of documents in each time-slice. Each time slice could for example represent a year's published
            papers, in case the corpus comes from a journal publishing over multiple years.
            It is assumed that `sum(time_slice) == num_documents`.
        id2word : dict of (int, str), optional
            Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for
            debugging and topic printing.
        alphas : float, optional
            The prior probability for the model.
        num_topics : int, optional
            The number of requested latent topics to be extracted from the training corpus.
        initialize : {'gensim', 'own', 'ldamodel'}, optional
            Controls the initialization of the DTM model. Supports three different modes:
                * 'gensim': Uses gensim's LDA initialization.
                * 'own': Uses your own initialization matrix of an LDA model that has been previously trained.
                * 'lda_model': Use a previously used LDA model, passing it through the `lda_model` argument.
        sstats : numpy.ndarray , optional
            Sufficient statistics used for initializing the model if `initialize == 'own'`. Corresponds to matrix
            beta in the linked paper for time slice 0, expected shape (`self.vocab_len`, `num_topics`).
        lda_model : :class:`~gensim.models.ldamodel.LdaModel`
            Model whose sufficient statistics will be used to initialize the current object if `initialize == 'gensim'`.
        obs_variance : float, optional
            Observed variance used to approximate the true and forward variance as shown in
            `David M. Blei, John D. Lafferty: "Dynamic Topic Models"
            <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.
        chain_variance : float, optional
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.
        passes : int, optional
            Number of passes over the corpus for the initial :class:`~gensim.models.ldamodel.LdaModel`
        random_state : {numpy.random.RandomState, int}, optional
            Can be a np.random.RandomState object, or the seed to generate one. Used for reproducibility of results.
        lda_inference_max_iter : int, optional
            Maximum number of iterations in the inference step of the LDA training.
        em_min_iter : int, optional
            Minimum number of iterations until converge of the Expectation-Maximization algorithm
        em_max_iter : int, optional
            Maximum number of iterations until converge of the Expectation-Maximization algorithm.
        chunksize : int, optional
            Number of documents in the corpus do be processed in in a chunk.

        """
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError(
                'at least one of corpus/id2word must be specified, to establish input space dimensionality'
            )

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.vocab_len = len(self.id2word)
        elif self.id2word:
            self.vocab_len = len(self.id2word)
        else:
            self.vocab_len = 0

        if corpus is not None:
            try:
                self.corpus_len = len(corpus)
            except TypeError:
                logger.warning("input corpus stream has no len(); counting documents")
                self.corpus_len = sum(1 for _ in corpus)

        self.time_slice = time_slice
        if self.time_slice is not None:
            self.num_time_slices = len(time_slice)

        self.num_topics = num_topics
        self.num_time_slices = len(time_slice)
        self.alphas = np.full(num_topics, alphas)

        # topic_chains contains for each topic a 'state space language model' object
        # which in turn has information about each topic
        # the sslm class is described below and contains information
        # on topic-word probabilities and doc-topic probabilities.
        self.topic_chains = [
            sslm(
                num_time_slices=self.num_time_slices, vocab_len=self.vocab_len,
                chain_variance=chain_variance, obs_variance=obs_variance
            ) for i in range(num_topics)]

        # if a corpus and time_slice is provided, depending on the user choice of initializing LDA, we start DTM.
        if corpus is not None and time_slice is not None:

            self.max_doc_len = max(len(line) for line in corpus)

            if initialize == 'gensim':
                lda_model = ldamodel.LdaModel(
                    corpus, id2word=self.id2word, num_topics=self.num_topics,
                    passes=passes, alpha=self.alphas, random_state=random_state,
                    dtype=np.float64
                )
                self.sstats = np.transpose(lda_model.state.sstats)
            if initialize == 'ldamodel':
                self.sstats = np.transpose(lda_model.state.sstats)
            if initialize == 'own':
                self.sstats = sstats

            # initialize model from sstats
            self.init_ldaseq_ss(chain_variance, obs_variance, self.alphas, self.sstats)

            # fit DTM
            self.fit(corpus, lda_inference_max_iter, em_min_iter, em_max_iter, chunksize)

    def init_ldaseq_ss(self, topic_chain_variance, topic_obs_variance, alpha, init_suffstats):
        """Initialize State Space Language Model, topic-wise.

        Parameters
        ----------
        topic_chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve.
        topic_obs_variance : float
            Observed variance used to approximate the true and forward variance as shown in
            `David M. Blei, John D. Lafferty: "Dynamic Topic Models"
            <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.
        alpha : float
            The prior probability for the model.
        init_suffstats : numpy.ndarray
            Sufficient statistics used for initializing the model, expected shape (`self.vocab_len`, `num_topics`).

        """
        # TODO why do we pass this alpha if it is already attr?
        self.alphas = alpha
        for k, chain in enumerate(self.topic_chains):
            # а что мы сюда копируем? получается наша chain для каждого топика?
            sstats = init_suffstats[:, k]
            chain.sslm_counts_init(topic_obs_variance, topic_chain_variance, sstats)

    def fit(self, corpus, lda_inference_max_iter, em_min_iter, em_max_iter, chunksize):
        """Fit a LDA Sequence model (DTM).

        This method will iteratively setup LDA models and perform EM steps until the sufficient statistics convergence,
        or until the maximum number of iterations is reached. Because the true posterior is intractable, an
        appropriately tight lower bound must be used instead. This function will optimize this bound, by minimizing
        its true Kullback-Liebler Divergence with the true posterior.

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
        lda_inference_max_iter : int
            Maximum number of iterations for the inference step of LDA.
        em_min_iter : int
            Minimum number of time slices to be inspected.
        em_max_iter : int
            Maximum number of time slices to be inspected.
        chunksize : int
            Number of documents to be processed in each chunk.

        Returns
        -------
        float
            The highest lower bound for the true posterior produced after all iterations.

       """
        LDASQE_EM_THRESHOLD = 1e-4
        # if bound is low, then we increase iterations.
        LOWER_ITER = 10
        ITER_MULT_LOW = 2
        MAX_ITER = 500

        bound = 0
        convergence = LDASQE_EM_THRESHOLD + 1
        iter_ = 0

        # setting up memory buffer which are used on every iteration of a cycle below
        gammas = np.zeros((self.corpus_len, self.num_topics))
        lhoods = np.zeros((self.corpus_len, self.num_topics + 1))

        # initiate sufficient statistics buffer
        topic_suffstats = [np.zeros((self.vocab_len, self.num_time_slices))
                           for topic in range(self.num_topics)]

        # main optimization cycle
        while iter_ < em_min_iter or ((convergence > LDASQE_EM_THRESHOLD) and iter_ <= em_max_iter):

            logger.info(" EM iter %i", iter_)
            logger.info("E Step")

            old_bound = bound

            # initiate sufficient statistics (resetting buffers from previous interation)
            for topic_stat in topic_suffstats:
                topic_stat[:] = 0.0

            # resetting buffer from previous iteration
            gammas[:] = 0.0
            lhoods[:] = 0.0

            # compute the likelihood of a sequential corpus under an LDA
            # seq model and find the evidence lower bound. This is the E - Step
            bound, gammas = \
                self.lda_seq_infer(corpus, topic_suffstats, gammas, lhoods, iter_, lda_inference_max_iter, chunksize)
            self.gammas = gammas

            logger.info("M Step")

            # fit the variational distribution. This is the M - Step
            topic_bound = self.fit_lda_seq_topics(topic_suffstats)
            bound += topic_bound

            if (bound - old_bound) < 0:
                # if max_iter is too low, increase iterations.
                if lda_inference_max_iter < LOWER_ITER:
                    lda_inference_max_iter *= ITER_MULT_LOW
                logger.info("Bound went down, increasing iterations to %i", lda_inference_max_iter)

            # check for convergence
            convergence = np.fabs((bound - old_bound) / old_bound)

            if convergence < LDASQE_EM_THRESHOLD:
                lda_inference_max_iter = MAX_ITER
                logger.info("Starting final iterations, max iter is %i", lda_inference_max_iter)
                convergence = 1.0

            logger.info("iteration %i iteration lda seq bound is %f convergence is %f", iter_, bound, convergence)

            iter_ += 1

        return bound

    def lda_seq_infer(self, corpus, topic_suffstats, gammas, lhoods,
                      iter_, lda_inference_max_iter, chunksize):
        """Inference (or E-step) for the lower bound EM optimization.

        This is used to set up the gensim :class:`~gensim.models.ldamodel.LdaModel` to be used for each time-slice.
        It also allows for Document Influence Model code to be written in.

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
        topic_suffstats : numpy.ndarray
            Sufficient statistics for time slice 0, used for initializing the model if `initialize == 'own'`,
            expected shape (`self.vocab_len`, `num_topics`).
        gammas : numpy.ndarray
            Topic weight variational parameters for each document. If not supplied, it will be inferred from the model.
        lhoods : list of float
            The total log probability lower bound for each topic. Corresponds to the phi variational parameters in the
            linked paper.
        iter_ : int
            Current iteration.
        lda_inference_max_iter : int
            Maximum number of iterations for the inference step of LDA.
        chunksize : int
            Number of documents to be processed in each chunk.

        Returns
        -------
        (float, list of float)
            The first value is the highest lower bound for the true posterior.
            The second value is the list of optimized dirichlet variational parameters for the approximation of
            the posterior.

        """
        bound = 0.0

        lda = ldamodel.LdaModel(num_topics=self.num_topics, alpha=self.alphas, id2word=self.id2word, dtype=np.float64)

        lda.topics = np.zeros((self.vocab_len, self.num_topics))

        ldapost = LdaPost(max_doc_len=self.max_doc_len, num_topics=self.num_topics, lda=lda)
        bound, gammas = self.infer_dtm_seq(
            corpus, topic_suffstats, gammas, lhoods, lda,
            ldapost, iter_, bound, lda_inference_max_iter, chunksize
        )

        return bound, gammas

    def infer_dtm_seq(self, corpus, topic_suffstats, gammas, lhoods, lda,
                      ldapost, iter_, bound, lda_inference_max_iter, chunksize):
        """Compute the likelihood of a sequential corpus under an LDA seq model, and reports the likelihood bound.

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
        topic_suffstats : numpy.ndarray
            Sufficient statistics of the current model, expected shape (`self.vocab_len`, `num_topics`).
        gammas : numpy.ndarray
            Topic weight variational parameters for each document. If not supplied, it will be inferred from the model.
        lhoods : list of float of length `self.num_topics`
            The total log probability bound for each topic. Corresponds to phi from the linked paper.
        lda : :class:`~gensim.models.ldamodel.LdaModel`
            The trained LDA model of the previous iteration.
        ldapost : :class:`~gensim.models.ldaseqmodel.LdaPost`
            Posterior probability variables for the given LDA model. This will be used as the true (but intractable)
            posterior.
        iter_ : int
            The current iteration.
        bound : float
            The LDA bound produced after all iterations.
        lda_inference_max_iter : int
            Maximum number of iterations for the inference step of LDA.
        chunksize : int
            Number of documents to be processed in each chunk.

        Returns
        -------
        (float, list of float)
            The first value is the highest lower bound for the true posterior.
            The second value is the list of optimized dirichlet variational parameters for the approximation of
            the posterior.

        """
        doc_index = 0  # overall doc_index in corpus
        time = 0  # current time-slice
        doc_num = 0  # doc-index in current time-slice
        lda = self.make_lda_seq_slice(lda, time)  # create lda_seq slice

        time_slice = np.cumsum(np.array(self.time_slice))

        for chunk_no, chunk in enumerate(utils.grouper(corpus, chunksize)):
            # iterates chunk size for constant memory footprint
            for doc in chunk:
                # this is used to update the time_slice and create a new lda_seq slice every new time_slice
                if doc_index > time_slice[time]:
                    time += 1
                    lda = self.make_lda_seq_slice(lda, time)  # create lda_seq slice
                    doc_num = 0

                gam = gammas[doc_index]
                lhood = lhoods[doc_index]

                ldapost.gamma = gam
                ldapost.lhood = lhood
                ldapost.doc = doc

                # TODO: replace fit_lda_post with appropriate ldamodel functions, if possible.
                if iter_ == 0:
                    doc_lhood = ldapost.fit_lda_post(
                        doc_num, time, None, lda_inference_max_iter=lda_inference_max_iter
                    )
                else:
                    doc_lhood = ldapost.fit_lda_post(
                        doc_num, time, self, lda_inference_max_iter=lda_inference_max_iter
                    )

                if topic_suffstats is not None:
                    topic_suffstats = ldapost.update_lda_seq_ss(time, doc, topic_suffstats)

                gammas[doc_index] = ldapost.gamma
                bound += doc_lhood
                doc_index += 1
                doc_num += 1

        return bound, gammas

    def make_lda_seq_slice(self, lda, time):
        """Update the LDA model topic-word values using time slices.

        Parameters
        ----------

        lda : :class:`~gensim.models.ldamodel.LdaModel`
            The stationary model to be updated
        time : int
            The time slice assigned to the stationary model.

        Returns
        -------
        lda : :class:`~gensim.models.ldamodel.LdaModel`
            The stationary model updated to reflect the passed time slice.

        """
        for k in range(self.num_topics):
            lda.topics[:, k] = self.topic_chains[k].e_log_prob[:, time]

        lda.alpha = np.copy(self.alphas)
        return lda

    def fit_lda_seq_topics(self, topic_suffstats):
        """Fit the sequential model topic-wise.

        Parameters
        ----------
        topic_suffstats : numpy.ndarray
            Sufficient statistics of the current model, expected shape (`self.vocab_len`, `num_topics`).

        Returns
        -------
        float
            The sum of the optimized lower bounds for all topics.

        """
        lhood = 0

        for k, chain in enumerate(self.topic_chains):
            logger.info("Fitting topic number %i", k)
            lhood_term = chain.fit_sslm(topic_suffstats[k])
            lhood += lhood_term

        return lhood

    def print_topic_times(self, topic, top_terms=20):
        """Get the most relevant words for a topic, for each timeslice. This can be used to inspect the evolution of a
        topic through time.

        Parameters
        ----------
        topic : int
            The index of the topic.
        top_terms : int, optional
            Number of most relevant words associated with the topic to be returned.

        Returns
        -------
        list of list of str
            Top `top_terms` relevant terms for the topic for each time slice.

        """
        topics = []
        for time in range(self.num_time_slices):
            topics.append(self.print_topic(topic, time, top_terms))

        return topics

    def print_topics(self, time=0, top_terms=20):
        """Get the most relevant words for every topic.

        Parameters
        ----------
        time : int, optional
            The time slice in which we are interested in (since topics evolve over time, it is expected that the most
            relevant words will also gradually change).
        top_terms : int, optional
            Number of most relevant words to be returned for each topic.

        Returns
        -------
        list of list of (str, float)
            Representation of all topics. Each of them is represented by a list of pairs of words and their assigned
            probability.

        """
        return [self.print_topic(topic, time, top_terms) for topic in range(self.num_topics)]

    def print_topic(self, topic, time=0, top_terms=20):
        """Get the list of words most relevant to the given topic.

        Parameters
        ----------
        topic : int
            The index of the topic to be inspected.
        time : int, optional
            The time slice in which we are interested in (since topics evolve over time, it is expected that the most
            relevant words will also gradually change).
        top_terms : int, optional
            Number of words associated with the topic to be returned.

        Returns
        -------
        list of (str, float)
            The representation of this topic. Each element in the list includes the word itself, along with the
            probability assigned to it by the topic.

        """
        topic = self.topic_chains[topic].e_log_prob
        topic = np.transpose(topic)
        topic = np.exp(topic[time])
        topic = topic / topic.sum()
        bestn = matutils.argsort(topic, top_terms, reverse=True)
        beststr = [(self.id2word[id_], topic[id_]) for id_ in bestn]
        return beststr

    def doc_topics(self, doc_number):
        """Get the topic mixture for a document.

        Uses the priors for the dirichlet distribution that approximates the true posterior with the optimal
        lower bound, and therefore requires the model to be already trained.


        Parameters
        ----------
        doc_number : int
            Index of the document for which the mixture is returned.

        Returns
        -------
        list of length `self.num_topics`
            Probability for each topic in the mixture (essentially a point in the `self.num_topics - 1` simplex.

        """
        doc_topic = self.gammas / self.gammas.sum(axis=1)[:, np.newaxis]
        return doc_topic[doc_number]

    def dtm_vis(self, time, corpus):
        """Get the information needed to visualize the corpus model at a given time slice, using the pyLDAvis format.

        Parameters
        ----------
        time : int
            The time slice we are interested in.
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            The corpus we want to visualize at the given time slice.

        Returns
        -------
        doc_topics : list of length `self.num_topics`
            Probability for each topic in the mixture (essentially a point in the `self.num_topics - 1` simplex.
        topic_term : numpy.ndarray
            The representation of each topic as a multinomial over words in the vocabulary,
            expected shape (`num_topics`, vocabulary length).
        doc_lengths : list of int
            The number of words in each document. These could be fixed, or drawn from a Poisson distribution.
        term_frequency : numpy.ndarray
            The term frequency matrix (denoted as beta in the original Blei paper). This could also be the TF-IDF
            representation of the corpus, expected shape (number of documents, length of vocabulary).
        vocab : list of str
            The set of unique terms existing in the cropuse's vocabulary.

        """
        doc_topic = self.gammas / self.gammas.sum(axis=1)[:, np.newaxis]

        def normalize(x):
            return x / x.sum()

        topic_term = [
            normalize(np.exp(chain.e_log_prob.T[time]))
            for k, chain in enumerate(self.topic_chains)
        ]

        doc_lengths = []
        term_frequency = np.zeros(self.vocab_len)
        for doc_no, doc in enumerate(corpus):
            doc_lengths.append(len(doc))

            for term, freq in doc:
                term_frequency[term] += freq

        vocab = [self.id2word[i] for i in range(len(self.id2word))]

        return doc_topic, np.array(topic_term), doc_lengths, term_frequency, vocab

    def dtm_coherence(self, time):
        """Get the coherence for each topic.

        Can be used to measure the quality of the model, or to inspect the convergence through training via a callback.

        Parameters
        ----------
        time : int
            The time slice.

        Returns
        -------
        list of list of str
            The word representation for each topic, for each time slice. This can be used to check the time coherence
            of topics as time evolves: If the most relevant words remain the same then the topic has somehow
            converged or is relatively static, if they change rapidly the topic is evolving.

        """
        coherence_topics = []
        for topics in self.print_topics(time):
            coherence_topic = []
            for word, dist in topics:
                coherence_topic.append(word)
            coherence_topics.append(coherence_topic)

        return coherence_topics

    def __getitem__(self, doc):
        """Get the topic mixture for the given document, using the inferred approximation of the true posterior.

        Parameters
        ----------
        doc : list of (int, float)
            The doc in BOW format. Can be an unseen document.

        Returns
        -------
        list of float
            Probabilities for each topic in the mixture. This is essentially a point in the `num_topics - 1` simplex.

        """
        lda_model = ldamodel.LdaModel(
            num_topics=self.num_topics, alpha=self.alphas, id2word=self.id2word, dtype=np.float64)
        lda_model.topics = np.zeros((self.vocab_len, self.num_topics))
        ldapost = LdaPost(num_topics=self.num_topics, max_doc_len=len(doc), lda=lda_model, doc=doc)

        time_lhoods = []
        for time in range(self.num_time_slices):
            lda_model = self.make_lda_seq_slice(lda_model, time)  # create lda_seq slice
            lhood = ldapost.fit_lda_post(0, time, self)
            time_lhoods.append(lhood)

        doc_topic = ldapost.gamma / ldapost.gamma.sum()
        # should even the likelihoods be returned?
        return doc_topic


class sslm(utils.SaveLoad):
    """Encapsulate the inner State Space Language Model for DTM.

    Some important attributes of this class:

        * `obs` is a matrix containing the document to topic ratios.
        * `e_log_prob` is a matrix containing the topic to word ratios.
        * `mean` contains the mean values to be used for inference for each word for a time slice.
        * `variance` contains the variance values to be used for inference of word in a time slice.
        * `fwd_mean` and`fwd_variance` are the forward posterior values for the mean and the variance.
        * `zeta` is an extra variational parameter with a value for each time slice.

    """

    def __init__(self, vocab_len=None, num_time_slices=None, obs_variance=0.5, chain_variance=0.005):
        self.vocab_len = vocab_len
        self.num_time_slices = num_time_slices
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance

        # setting up matrices
        self.obs = np.zeros((vocab_len, num_time_slices), dtype=np.float64)
        self.e_log_prob = np.zeros((vocab_len, num_time_slices), dtype=np.float64)
        self.mean = np.zeros((vocab_len, num_time_slices + 1), dtype=np.float64)
        self.fwd_mean = np.zeros((vocab_len, num_time_slices + 1), dtype=np.float64)
        self.fwd_variance = np.zeros((vocab_len, num_time_slices + 1), dtype=np.float64)
        self.variance = np.zeros((vocab_len, num_time_slices + 1), dtype=np.float64)
        self.zeta = np.zeros(num_time_slices, dtype=np.float64)

    def sslm_counts_init(self, obs_variance, chain_variance, sstats):
        """Initialize the State Space Language Model with LDA sufficient statistics.

        Called for each topic-chain and initializes initial mean, variance and Topic-Word probabilities
        for the first time-slice.

        Parameters
        ----------
        obs_variance : float, optional
            Observed variance used to approximate the true and forward variance.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.
        sstats : numpy.ndarray
            Sufficient statistics of the LDA model. Corresponds to matrix beta in the linked paper for time slice 0,
            expected shape (`self.vocab_len`, `num_topics`).

        """

        sslm_counts_init(self, obs_variance, chain_variance, sstats)

    def fit_sslm(self, sstats):
        """Fits variational distribution.

        This is essentially the m-step.
        Maximizes the approximation of the true posterior for a particular topic using the provided sufficient
        statistics. Updates the values using :meth:`~gensim.models.ldaseqmodel.sslm.update_obs` and
        :meth:`~gensim.models.ldaseqmodel.sslm.compute_expected_log_prob`.

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the
            current time slice, expected shape (`self.vocab_len`, `num_topics`).

        Returns
        -------
        float
            The lower bound for the true posterior achieved using the fitted approximate distribution.

        """

        return fit_sslm(self, sstats)


class LdaPost(utils.SaveLoad):
    """Posterior values associated with each set of documents.

    TODO: use **Hoffman, Blei, Bach: Online Learning for Latent Dirichlet Allocation, NIPS 2010.**
    to update phi, gamma. End game would be to somehow replace LdaPost entirely with LdaModel.

    """

    def __init__(self, doc=None, lda=None, max_doc_len=None, num_topics=None, gamma=None, lhood=None):
        """Initialize the posterior value structure for the given LDA model.

        Parameters
        ----------
        doc : list of (int, int)
            A BOW representation of the document. Each element in the list is a pair of a word's ID and its number
            of occurences in the document.
        lda : :class:`~gensim.models.ldamodel.LdaModel`, optional
            The underlying LDA model.
        max_doc_len : int, optional
            The maximum number of words in a document.
        num_topics : int, optional
            Number of topics discovered by the LDA model.
        gamma : numpy.ndarray, optional
            Topic weight variational parameters for each document. If not supplied, it will be inferred from the model.
        lhood : float, optional
            The log likelihood lower bound.

        """
        self.doc = doc
        self.lda = lda
        self.gamma = gamma
        self.lhood = lhood
        if self.gamma is None:
            self.gamma = np.zeros(num_topics)
        if self.lhood is None:
            self.lhood = np.zeros(num_topics + 1)

        if max_doc_len is not None and num_topics is not None:
            self.phi = np.zeros((max_doc_len, num_topics))
            self.log_phi = np.zeros((max_doc_len, num_topics))

    def fit_lda_post(self, doc_number, time, ldaseq, LDA_INFERENCE_CONVERGED=1e-8,
                     lda_inference_max_iter=25, g=None, g3_matrix=None, g4_matrix=None, g5_matrix=None):
        """Posterior inference for lda.

        Parameters
        ----------
        doc_number : int
            The documents number.
        time : int
            Time slice.
        ldaseq : object
            Unused.
        LDA_INFERENCE_CONVERGED : float
            Epsilon value used to check whether the inference step has sufficiently converged.
        lda_inference_max_iter : int
            Maximum number of iterations in the inference step.
        g : object
            Unused. Will be useful when the DIM model is implemented.
        g3_matrix: object
            Unused. Will be useful when the DIM model is implemented.
        g4_matrix: object
            Unused. Will be useful when the DIM model is implemented.
        g5_matrix: object
            Unused. Will be useful when the DIM model is implemented.

        Returns
        -------
        float
            The optimal lower bound for the true posterior using the approximate distribution.
        """

        return np.array(fit_lda_post(self, doc_number, time, ldaseq, LDA_INFERENCE_CONVERGED,
                                     lda_inference_max_iter, g, g3_matrix, g4_matrix, g5_matrix))

    def update_lda_seq_ss(self, time, doc, topic_suffstats):
        """Update lda sequence sufficient statistics from an lda posterior.

        This is very similar to the :meth:`~gensim.models.ldaseqmodel.LdaPost.update_gamma` method and uses
        the same formula.

        Parameters
        ----------
        time : int
            The time slice.
        doc : list of (int, float)
            Unused but kept here for backwards compatibility. The document set in the constructor (`self.doc`) is used
            instead.
        topic_suffstats : list of float
            Sufficient statistics for each topic.

        Returns
        -------
        list of float
            The updated sufficient statistics for each topic.

        """
        num_topics = self.lda.num_topics

        for k in range(num_topics):
            topic_ss = topic_suffstats[k]
            n = 0
            for word_id, count in self.doc:
                topic_ss[word_id][time] += count * self.phi[n][k]
                n += 1
            topic_suffstats[k] = topic_ss

        return topic_suffstats