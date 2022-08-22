#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Based on Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>

"""Lda Sequence model, inspired by
`David M. Blei, John D. Lafferty: "Dynamic Topic Models"
<https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.
The original C/C++ implementation can be found on `blei-lab/dtm <https://github.com/blei-lab/dtm>`_.


TODO: The next steps to take this forward would be:

#. Include DIM mode. Most of the infrastructure for this is in place.
#. See if LdaPost can be replaced by LdaModel completely without breaking anything.
#. Heavy lifting going on in the Sslm class - efforts can be made to cythonise mathematical methods, in particular,
   update_obs and the optimization takes a lot time.
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
from scipy.special import digamma, gammaln
from scipy import optimize

from gensim import utils, matutils
from gensim.models import ldamodel


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
        self.topic_chains = []
        for topic in range(num_topics):
            sslm_ = sslm(
                num_time_slices=self.num_time_slices, vocab_len=self.vocab_len, num_topics=self.num_topics,
                chain_variance=chain_variance, obs_variance=obs_variance
            )
            self.topic_chains.append(sslm_)

        # the following are class variables which are to be integrated during Document Influence Model
        self.top_doc_phis = None
        self.influence = None
        self.renormalized_influence = None
        self.influence_sum_lgl = None

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
            self.fit_lda_seq(corpus, lda_inference_max_iter, em_min_iter, em_max_iter, chunksize)

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
        self.alphas = alpha
        for k, chain in enumerate(self.topic_chains):
            sstats = init_suffstats[:, k]
            sslm.sslm_counts_init(chain, topic_obs_variance, topic_chain_variance, sstats)

            # initialize the below matrices only if running DIM
            # ldaseq.topic_chains[k].w_phi_l = np.zeros((ldaseq.vocab_len, ldaseq.num_time_slices))
            # ldaseq.topic_chains[k].w_phi_sum = np.zeros((ldaseq.vocab_len, ldaseq.num_time_slices))
            # ldaseq.topic_chains[k].w_phi_sq = np.zeros((ldaseq.vocab_len, ldaseq.num_time_slices))

    def fit_lda_seq(self, corpus, lda_inference_max_iter, em_min_iter, em_max_iter, chunksize):
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

        num_topics = self.num_topics
        vocab_len = self.vocab_len
        data_len = self.num_time_slices
        corpus_len = self.corpus_len

        bound = 0
        convergence = LDASQE_EM_THRESHOLD + 1
        iter_ = 0

        while iter_ < em_min_iter or ((convergence > LDASQE_EM_THRESHOLD) and iter_ <= em_max_iter):

            logger.info(" EM iter %i", iter_)
            logger.info("E Step")
            # TODO: bound is initialized to 0
            old_bound = bound

            # initiate sufficient statistics
            topic_suffstats = []
            for topic in range(num_topics):
                topic_suffstats.append(np.zeros((vocab_len, data_len)))

            # set up variables
            gammas = np.zeros((corpus_len, num_topics))
            lhoods = np.zeros((corpus_len, num_topics + 1))
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
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        bound = 0.0

        lda = ldamodel.LdaModel(num_topics=num_topics, alpha=self.alphas, id2word=self.id2word, dtype=np.float64)
        lda.topics = np.zeros((vocab_len, num_topics))
        ldapost = LdaPost(max_doc_len=self.max_doc_len, num_topics=num_topics, lda=lda)

        model = "DTM"
        if model == "DTM":
            bound, gammas = self.inferDTMseq(
                corpus, topic_suffstats, gammas, lhoods, lda,
                ldapost, iter_, bound, lda_inference_max_iter, chunksize
            )
        elif model == "DIM":
            self.InfluenceTotalFixed(corpus)
            bound, gammas = self.inferDIMseq(
                corpus, topic_suffstats, gammas, lhoods, lda,
                ldapost, iter_, bound, lda_inference_max_iter, chunksize
            )

        return bound, gammas

    def inferDTMseq(self, corpus, topic_suffstats, gammas, lhoods, lda,
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
                    doc_lhood = LdaPost.fit_lda_post(
                        ldapost, doc_num, time, None, lda_inference_max_iter=lda_inference_max_iter
                    )
                else:
                    doc_lhood = LdaPost.fit_lda_post(
                        ldapost, doc_num, time, self, lda_inference_max_iter=lda_inference_max_iter
                    )

                if topic_suffstats is not None:
                    topic_suffstats = LdaPost.update_lda_seq_ss(ldapost, time, doc, topic_suffstats)

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
            lhood_term = sslm.fit_sslm(chain, topic_suffstats[k])
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
            lhood = LdaPost.fit_lda_post(ldapost, 0, time, self)
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

    def __init__(self, vocab_len=None, num_time_slices=None, num_topics=None, obs_variance=0.5, chain_variance=0.005):
        self.vocab_len = vocab_len
        self.num_time_slices = num_time_slices
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance
        self.num_topics = num_topics

        # setting up matrices
        self.obs = np.zeros((vocab_len, num_time_slices))
        self.e_log_prob = np.zeros((vocab_len, num_time_slices))
        self.mean = np.zeros((vocab_len, num_time_slices + 1))
        self.fwd_mean = np.zeros((vocab_len, num_time_slices + 1))
        self.fwd_variance = np.zeros((vocab_len, num_time_slices + 1))
        self.variance = np.zeros((vocab_len, num_time_slices + 1))
        self.zeta = np.zeros(num_time_slices)

        # the following are class variables which are to be integrated during Document Influence Model
        self.m_update_coeff = None
        self.mean_t = None
        self.variance_t = None
        self.influence_sum_lgl = None
        self.w_phi_l = None
        self.w_phi_sum = None
        self.w_phi_l_sq = None
        self.m_update_coeff_g = None

    def update_zeta(self):
        """Update the Zeta variational parameter.

        Zeta is described in the appendix and is equal to sum (exp(mean[word] + Variance[word] / 2)),
        over every time-slice. It is the value of variational parameter zeta which maximizes the lower bound.

        Returns
        -------
        list of float
            The updated zeta values for each time slice.

        """
        for j, val in enumerate(self.zeta):
            self.zeta[j] = np.sum(np.exp(self.mean[:, j + 1] + self.variance[:, j + 1] / 2))
        return self.zeta

    def compute_post_variance(self, word, chain_variance):
        r"""Get the variance, based on the
        `Variational Kalman Filtering approach for Approximate Inference (section 3.1)
        <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.

        This function accepts the word to compute variance for, along with the associated sslm class object,
        and returns the `variance` and the posterior approximation `fwd_variance`.

        Notes
        -----
        This function essentially computes Var[\beta_{t,w}] for t = 1:T

        .. :math::

            fwd\_variance[t] \equiv E((beta_{t,w}-mean_{t,w})^2 |beta_{t}\ for\ 1:t) =
            (obs\_variance / fwd\_variance[t - 1] + chain\_variance + obs\_variance ) *
            (fwd\_variance[t - 1] + obs\_variance)

        .. :math::

            variance[t] \equiv E((beta_{t,w}-mean\_cap_{t,w})^2 |beta\_cap_{t}\ for\ 1:t) =
            fwd\_variance[t - 1] + (fwd\_variance[t - 1] / fwd\_variance[t - 1] + obs\_variance)^2 *
            (variance[t - 1] - (fwd\_variance[t-1] + obs\_variance))

        Parameters
        ----------
        word: int
            The word's ID.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The first returned value is the variance of each word in each time slice, the second value is the
            inferred posterior variance for the same pairs.

        """
        INIT_VARIANCE_CONST = 1000

        T = self.num_time_slices
        variance = self.variance[word]
        fwd_variance = self.fwd_variance[word]
        # forward pass. Set initial variance very high
        fwd_variance[0] = chain_variance * INIT_VARIANCE_CONST
        for t in range(1, T + 1):
            if self.obs_variance:
                c = self.obs_variance / (fwd_variance[t - 1] + chain_variance + self.obs_variance)
            else:
                c = 0
            fwd_variance[t] = c * (fwd_variance[t - 1] + chain_variance)

        # backward pass
        variance[T] = fwd_variance[T]
        for t in range(T - 1, -1, -1):
            if fwd_variance[t] > 0.0:
                c = np.power((fwd_variance[t] / (fwd_variance[t] + chain_variance)), 2)
            else:
                c = 0
            variance[t] = (c * (variance[t + 1] - chain_variance)) + ((1 - c) * fwd_variance[t])

        return variance, fwd_variance

    def compute_post_mean(self, word, chain_variance):
        """Get the mean, based on the `Variational Kalman Filtering approach for Approximate Inference (section 3.1)
        <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.

        Notes
        -----
        This function essentially computes E[\beta_{t,w}] for t = 1:T.

        .. :math::

            Fwd_Mean(t) ≡  E(beta_{t,w} | beta_ˆ 1:t )
            = (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance ) * fwd_mean[t - 1] +
            (1 - (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance)) * beta

        .. :math::

            Mean(t) ≡ E(beta_{t,w} | beta_ˆ 1:T )
            = fwd_mean[t - 1] + (obs_variance / fwd_variance[t - 1] + obs_variance) +
            (1 - obs_variance / fwd_variance[t - 1] + obs_variance)) * mean[t]

        Parameters
        ----------
        word: int
            The word's ID.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The first returned value is the mean of each word in each time slice, the second value is the
            inferred posterior mean for the same pairs.

        """
        T = self.num_time_slices
        obs = self.obs[word]
        fwd_variance = self.fwd_variance[word]
        mean = self.mean[word]
        fwd_mean = self.fwd_mean[word]

        # forward
        fwd_mean[0] = 0
        for t in range(1, T + 1):
            c = self.obs_variance / (fwd_variance[t - 1] + chain_variance + self.obs_variance)
            fwd_mean[t] = c * fwd_mean[t - 1] + (1 - c) * obs[t - 1]

        # backward pass
        mean[T] = fwd_mean[T]
        for t in range(T - 1, -1, -1):
            if chain_variance == 0.0:
                c = 0.0
            else:
                c = chain_variance / (fwd_variance[t] + chain_variance)
            mean[t] = c * fwd_mean[t] + (1 - c) * mean[t + 1]
        return mean, fwd_mean

    def compute_expected_log_prob(self):
        """Compute the expected log probability given values of m.

        The appendix describes the Expectation of log-probabilities in equation 5 of the DTM paper;
        The below implementation is the result of solving the equation and is implemented as in the original
        Blei DTM code.

        Returns
        -------
        numpy.ndarray of float
            The expected value for the log probabilities for each word and time slice.

        """
        for (w, t), val in np.ndenumerate(self.e_log_prob):
            self.e_log_prob[w][t] = self.mean[w][t + 1] - np.log(self.zeta[t])
        return self.e_log_prob

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
        W = self.vocab_len
        T = self.num_time_slices

        log_norm_counts = np.copy(sstats)
        log_norm_counts /= sum(log_norm_counts)
        log_norm_counts += 1.0 / W
        log_norm_counts /= sum(log_norm_counts)
        log_norm_counts = np.log(log_norm_counts)

        # setting variational observations to transformed counts
        self.obs = (np.repeat(log_norm_counts, T, axis=0)).reshape(W, T)
        # set variational parameters
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance

        # compute post variance, mean
        for w in range(W):
            self.variance[w], self.fwd_variance[w] = self.compute_post_variance(w, self.chain_variance)
            self.mean[w], self.fwd_mean[w] = self.compute_post_mean(w, self.chain_variance)

        self.zeta = self.update_zeta()
        self.e_log_prob = self.compute_expected_log_prob()

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
        W = self.vocab_len
        bound = 0
        old_bound = 0
        sslm_fit_threshold = 1e-6
        sslm_max_iter = 2
        converged = sslm_fit_threshold + 1

        # computing variance, fwd_variance
        self.variance, self.fwd_variance = \
            (np.array(x) for x in zip(*(self.compute_post_variance(w, self.chain_variance) for w in range(W))))

        # column sum of sstats
        totals = sstats.sum(axis=0)
        iter_ = 0

        model = "DTM"
        if model == "DTM":
            bound = self.compute_bound(sstats, totals)
        if model == "DIM":
            bound = self.compute_bound_fixed(sstats, totals)

        logger.info("initial sslm bound is %f", bound)

        while converged > sslm_fit_threshold and iter_ < sslm_max_iter:
            iter_ += 1
            old_bound = bound
            self.obs, self.zeta = self.update_obs(sstats, totals)

            if model == "DTM":
                bound = self.compute_bound(sstats, totals)
            if model == "DIM":
                bound = self.compute_bound_fixed(sstats, totals)

            converged = np.fabs((bound - old_bound) / old_bound)
            logger.info("iteration %i iteration lda seq bound is %f convergence is %f", iter_, bound, converged)

        self.e_log_prob = self.compute_expected_log_prob()
        return bound

    def compute_bound(self, sstats, totals):
        """Compute the maximized lower bound achieved for the log probability of the true posterior.

        Uses the formula presented in the appendix of the DTM paper (formula no. 5).

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the first
            time slice, expected shape (`self.vocab_len`, `num_topics`).
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.

        Returns
        -------
        float
            The maximized lower bound.

        """
        w = self.vocab_len
        t = self.num_time_slices

        term_1 = 0
        term_2 = 0
        term_3 = 0

        val = 0
        ent = 0

        chain_variance = self.chain_variance
        # computing mean, fwd_mean
        self.mean, self.fwd_mean = \
            (np.array(x) for x in zip(*(self.compute_post_mean(w, self.chain_variance) for w in range(w))))
        self.zeta = self.update_zeta()

        val = sum(self.variance[w][0] - self.variance[w][t] for w in range(w)) / 2 * chain_variance

        logger.info("Computing bound, all times")

        for t in range(1, t + 1):
            term_1 = 0.0
            term_2 = 0.0
            ent = 0.0
            for w in range(w):

                m = self.mean[w][t]
                prev_m = self.mean[w][t - 1]

                v = self.variance[w][t]

                # w_phi_l is only used in Document Influence Model; the values are always zero in this case
                # w_phi_l = sslm.w_phi_l[w][t - 1]
                # exp_i = np.exp(-prev_m)
                # term_1 += (np.power(m - prev_m - (w_phi_l * exp_i), 2) / (2 * chain_variance)) -
                # (v / chain_variance) - np.log(chain_variance)

                term_1 += \
                    (np.power(m - prev_m, 2) / (2 * chain_variance)) - (v / chain_variance) - np.log(chain_variance)
                term_2 += sstats[w][t - 1] * m
                ent += np.log(v) / 2  # note the 2pi's cancel with term1 (see doc)

            term_3 = -totals[t - 1] * np.log(self.zeta[t - 1])
            val += term_2 + term_3 + ent - term_1

        return val

    def update_obs(self, sstats, totals):
        """Optimize the bound with respect to the observed variables.

        TODO:
        This is by far the slowest function in the whole algorithm.
        Replacing or improving the performance of this would greatly speed things up.

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the first
            time slice, expected shape (`self.vocab_len`, `num_topics`).
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.

        Returns
        -------
        (numpy.ndarray of float, numpy.ndarray of float)
            The updated optimized values for obs and the zeta variational parameter.

        """

        OBS_NORM_CUTOFF = 2
        STEP_SIZE = 0.01
        TOL = 1e-3

        W = self.vocab_len
        T = self.num_time_slices

        runs = 0
        mean_deriv_mtx = np.zeros((T, T + 1))

        norm_cutoff_obs = None
        for w in range(W):
            w_counts = sstats[w]
            counts_norm = 0
            # now we find L2 norm of w_counts
            for i in range(len(w_counts)):
                counts_norm += w_counts[i] * w_counts[i]

            counts_norm = np.sqrt(counts_norm)

            if counts_norm < OBS_NORM_CUTOFF and norm_cutoff_obs is not None:
                obs = self.obs[w]
                norm_cutoff_obs = np.copy(obs)
            else:
                if counts_norm < OBS_NORM_CUTOFF:
                    w_counts = np.zeros(len(w_counts))

                # TODO: apply lambda function
                for t in range(T):
                    mean_deriv_mtx[t] = self.compute_mean_deriv(w, t, mean_deriv_mtx[t])

                deriv = np.zeros(T)
                args = self, w_counts, totals, mean_deriv_mtx, w, deriv
                obs = self.obs[w]
                model = "DTM"

                if model == "DTM":
                    # slowest part of method
                    obs = optimize.fmin_cg(
                        f=f_obs, fprime=df_obs, x0=obs, gtol=TOL, args=args, epsilon=STEP_SIZE, disp=0
                    )
                if model == "DIM":
                    pass
                runs += 1

                if counts_norm < OBS_NORM_CUTOFF:
                    norm_cutoff_obs = obs

                self.obs[w] = obs

        self.zeta = self.update_zeta()

        return self.obs, self.zeta

    def compute_mean_deriv(self, word, time, deriv):
        """Helper functions for optimizing a function.

        Compute the derivative of:

        .. :math::

            E[\beta_{t,w}]/d obs_{s,w} for t = 1:T.

        Parameters
        ----------
        word : int
            The word's ID.
        time : int
            The time slice.
        deriv : list of float
            Derivative for each time slice.

        Returns
        -------
        list of float
            Mean derivative for each time slice.

        """

        T = self.num_time_slices
        fwd_variance = self.variance[word]

        deriv[0] = 0

        # forward pass
        for t in range(1, T + 1):
            if self.obs_variance > 0.0:
                w = self.obs_variance / (fwd_variance[t - 1] + self.chain_variance + self.obs_variance)
            else:
                w = 0.0
            val = w * deriv[t - 1]
            if time == t - 1:
                val += (1 - w)
            deriv[t] = val

        for t in range(T - 1, -1, -1):
            if self.chain_variance == 0.0:
                w = 0.0
            else:
                w = self.chain_variance / (fwd_variance[t] + self.chain_variance)
            deriv[t] = w * deriv[t] + (1 - w) * deriv[t + 1]

        return deriv

    def compute_obs_deriv(self, word, word_counts, totals, mean_deriv_mtx, deriv):
        """Derivation of obs which is used in derivative function `df_obs` while optimizing.

        Parameters
        ----------
        word : int
            The word's ID.
        word_counts : list of int
            Total word counts for each time slice.
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.
        mean_deriv_mtx : list of float
            Mean derivative for each time slice.
        deriv : list of float
            Mean derivative for each time slice.

        Returns
        -------
        list of float
            Mean derivative for each time slice.

        """

        # flag
        init_mult = 1000

        T = self.num_time_slices

        mean = self.mean[word]
        variance = self.variance[word]

        # only used for DIM mode
        # w_phi_l = self.w_phi_l[word]
        # m_update_coeff = self.m_update_coeff[word]

        # temp_vector holds temporary zeta values
        self.temp_vect = np.zeros(T)

        for u in range(T):
            self.temp_vect[u] = np.exp(mean[u + 1] + variance[u + 1] / 2)

        for t in range(T):
            mean_deriv = mean_deriv_mtx[t]
            term1 = 0
            term2 = 0
            term3 = 0
            term4 = 0

            for u in range(1, T + 1):
                mean_u = mean[u]
                mean_u_prev = mean[u - 1]
                dmean_u = mean_deriv[u]
                dmean_u_prev = mean_deriv[u - 1]

                term1 += (mean_u - mean_u_prev) * (dmean_u - dmean_u_prev)
                term2 += (word_counts[u - 1] - (totals[u - 1] * self.temp_vect[u - 1] / self.zeta[u - 1])) * dmean_u

                model = "DTM"
                if model == "DIM":
                    # do some stuff
                    pass

            if self.chain_variance:
                term1 = - (term1 / self.chain_variance)
                term1 = term1 - (mean[0] * mean_deriv[0]) / (init_mult * self.chain_variance)
            else:
                term1 = 0.0

            deriv[t] = term1 + term2 + term3 + term4

        return deriv


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

        # the following are class variables which are to be integrated during Document Influence Model

        self.doc_weight = None
        self.renormalized_doc_weight = None

    def update_phi(self, doc_number, time):
        """Update variational multinomial parameters, based on a document and a time-slice.

        This is done based on the original Blei-LDA paper, where:
        log_phi := beta * exp(Ψ(gamma)), over every topic for every word.

        TODO: incorporate lee-sueng trick used in
        **Lee, Seung: Algorithms for non-negative matrix factorization, NIPS 2001**.

        Parameters
        ----------
        doc_number : int
            Document number. Unused.
        time : int
            Time slice. Unused.

        Returns
        -------
        (list of float, list of float)
            Multinomial parameters, and their logarithm, for each word in the document.

        """
        num_topics = self.lda.num_topics
        # digamma values
        dig = np.zeros(num_topics)

        for k in range(num_topics):
            dig[k] = digamma(self.gamma[k])

        n = 0   # keep track of iterations for phi, log_phi
        for word_id, count in self.doc:
            for k in range(num_topics):
                self.log_phi[n][k] = dig[k] + self.lda.topics[word_id][k]

            log_phi_row = self.log_phi[n]
            phi_row = self.phi[n]

            # log normalize
            v = log_phi_row[0]
            for i in range(1, len(log_phi_row)):
                v = np.logaddexp(v, log_phi_row[i])

            # subtract every element by v
            log_phi_row = log_phi_row - v
            phi_row = np.exp(log_phi_row)
            self.log_phi[n] = log_phi_row
            self.phi[n] = phi_row
            n += 1  # increase iteration

        return self.phi, self.log_phi

    def update_gamma(self):
        """Update variational dirichlet parameters.

        This operations is described in the original Blei LDA paper:
        gamma = alpha + sum(phi), over every topic for every word.

        Returns
        -------
        list of float
            The updated gamma parameters for each word in the document.

        """
        self.gamma = np.copy(self.lda.alpha)
        n = 0  # keep track of number of iterations for phi, log_phi
        for word_id, count in self.doc:
            phi_row = self.phi[n]
            for k in range(self.lda.num_topics):
                self.gamma[k] += phi_row[k] * count
            n += 1

        return self.gamma

    def init_lda_post(self):
        """Initialize variational posterior. """
        total = sum(count for word_id, count in self.doc)
        self.gamma.fill(self.lda.alpha[0] + float(total) / self.lda.num_topics)
        self.phi[:len(self.doc), :] = 1.0 / self.lda.num_topics
        # doc_weight used during DIM
        # ldapost.doc_weight = None

    def compute_lda_lhood(self):
        """Compute the log likelihood bound.

        Returns
        -------
        float
            The optimal lower bound for the true posterior using the approximate distribution.

        """
        num_topics = self.lda.num_topics
        gamma_sum = np.sum(self.gamma)

        # to be used in DIM
        # sigma_l = 0
        # sigma_d = 0

        lhood = gammaln(np.sum(self.lda.alpha)) - gammaln(gamma_sum)
        self.lhood[num_topics] = lhood

        # influence_term = 0
        digsum = digamma(gamma_sum)

        model = "DTM"  # noqa:F841
        for k in range(num_topics):
            # below code only to be used in DIM mode
            # if ldapost.doc_weight is not None and (model == "DIM" or model == "fixed"):
            #     influence_topic = ldapost.doc_weight[k]
            #     influence_term = \
            #           - ((influence_topic * influence_topic + sigma_l * sigma_l) / 2.0 / (sigma_d * sigma_d))

            e_log_theta_k = digamma(self.gamma[k]) - digsum
            lhood_term = \
                (self.lda.alpha[k] - self.gamma[k]) * e_log_theta_k + \
                gammaln(self.gamma[k]) - gammaln(self.lda.alpha[k])
            # TODO: check why there's an IF
            n = 0
            for word_id, count in self.doc:
                if self.phi[n][k] > 0:
                    lhood_term += \
                        count * self.phi[n][k] * (e_log_theta_k + self.lda.topics[word_id][k] - self.log_phi[n][k])
                n += 1
            self.lhood[k] = lhood_term
            lhood += lhood_term
            # in case of DIM add influence term
            # lhood += influence_term

        return lhood

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

        self.init_lda_post()
        # sum of counts in a doc
        total = sum(count for word_id, count in self.doc)

        model = "DTM"
        if model == "DIM":
            # if in DIM then we initialise some variables here
            pass

        lhood = self.compute_lda_lhood()
        lhood_old = 0
        converged = 0
        iter_ = 0

        # first iteration starts here
        iter_ += 1
        lhood_old = lhood
        self.gamma = self.update_gamma()

        model = "DTM"

        if model == "DTM" or sslm is None:
            self.phi, self.log_phi = self.update_phi(doc_number, time)
        elif model == "DIM" and sslm is not None:
            self.phi, self.log_phi = self.update_phi_fixed(doc_number, time, sslm, g3_matrix, g4_matrix, g5_matrix)

        lhood = self.compute_lda_lhood()
        converged = np.fabs((lhood_old - lhood) / (lhood_old * total))

        while converged > LDA_INFERENCE_CONVERGED and iter_ <= lda_inference_max_iter:

            iter_ += 1
            lhood_old = lhood
            self.gamma = self.update_gamma()
            model = "DTM"

            if model == "DTM" or sslm is None:
                self.phi, self.log_phi = self.update_phi(doc_number, time)
            elif model == "DIM" and sslm is not None:
                self.phi, self.log_phi = self.update_phi_fixed(doc_number, time, sslm, g3_matrix, g4_matrix, g5_matrix)

            lhood = self.compute_lda_lhood()
            converged = np.fabs((lhood_old - lhood) / (lhood_old * total))

        return lhood

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


# the following functions are used in update_obs as the objective function.
def f_obs(x, *args):
    """Function which we are optimising for minimizing obs.

    Parameters
    ----------
    x : list of float
        The obs values for this word.
    sslm : :class:`~gensim.models.ldaseqmodel.sslm`
        The State Space Language Model for DTM.
    word_counts : list of int
        Total word counts for each time slice.
    totals : list of int of length `len(self.time_slice)`
        The totals for each time slice.
    mean_deriv_mtx : list of float
        Mean derivative for each time slice.
    word : int
        The word's ID.
    deriv : list of float
        Mean derivative for each time slice.

    Returns
    -------
    list of float
        The value of the objective function evaluated at point `x`.

    """
    sslm, word_counts, totals, mean_deriv_mtx, word, deriv = args
    # flag
    init_mult = 1000

    T = len(x)
    val = 0
    term1 = 0
    term2 = 0

    # term 3 and 4 for DIM
    term3 = 0
    term4 = 0

    sslm.obs[word] = x
    sslm.mean[word], sslm.fwd_mean[word] = sslm.compute_post_mean(word, sslm.chain_variance)

    mean = sslm.mean[word]
    variance = sslm.variance[word]

    # only used for DIM mode
    # w_phi_l = sslm.w_phi_l[word]
    # m_update_coeff = sslm.m_update_coeff[word]

    for t in range(1, T + 1):
        mean_t = mean[t]
        mean_t_prev = mean[t - 1]

        val = mean_t - mean_t_prev
        term1 += val * val
        term2 += word_counts[t - 1] * mean_t - totals[t - 1] * np.exp(mean_t + variance[t] / 2) / sslm.zeta[t - 1]

        model = "DTM"
        if model == "DIM":
            # stuff happens
            pass

    if sslm.chain_variance > 0.0:

        term1 = - (term1 / (2 * sslm.chain_variance))
        term1 = term1 - mean[0] * mean[0] / (2 * init_mult * sslm.chain_variance)
    else:
        term1 = 0.0

    final = -(term1 + term2 + term3 + term4)

    return final


def df_obs(x, *args):
    """Derivative of the objective function which optimises obs.

    Parameters
    ----------
    x : list of float
        The obs values for this word.
    sslm : :class:`~gensim.models.ldaseqmodel.sslm`
        The State Space Language Model for DTM.
    word_counts : list of int
        Total word counts for each time slice.
    totals : list of int of length `len(self.time_slice)`
        The totals for each time slice.
    mean_deriv_mtx : list of float
        Mean derivative for each time slice.
    word : int
        The word's ID.
    deriv : list of float
        Mean derivative for each time slice.

    Returns
    -------
    list of float
        The derivative of the objective function evaluated at point `x`.

    """
    sslm, word_counts, totals, mean_deriv_mtx, word, deriv = args

    sslm.obs[word] = x
    sslm.mean[word], sslm.fwd_mean[word] = sslm.compute_post_mean(word, sslm.chain_variance)

    model = "DTM"
    if model == "DTM":
        deriv = sslm.compute_obs_deriv(word, word_counts, totals, mean_deriv_mtx, deriv)
    elif model == "DIM":
        deriv = sslm.compute_obs_deriv_fixed(
            p.word, p.word_counts, p.totals, p.sslm, p.mean_deriv_mtx, deriv)  # noqa:F821

    return np.negative(deriv)
