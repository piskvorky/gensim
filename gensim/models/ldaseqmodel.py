#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Based on Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>


"""

Inspired by the Blei's original DTM code and paper.
Original DTM C/C++ code: https://github.com/blei-lab/dtm
DTM Paper: https://www.cs.princeton.edu/~blei/papers/BleiLafferty2006a.pdf


TODO:
The next steps to take this forward would be:

    1) Include DIM mode. Most of the infrastructure for this is in place.
    2) See if LdaPost can be replaced by LdaModel completely without breaking anything.
    3) Heavy lifting going on in the sslm class - efforts can be made to cythonise mathematical methods.
        - in particular, update_obs and the optimization takes a lot time.
    4) Try and make it distributed, especially around the E and M step.
    5) Remove all C/C++ coding style/syntax.
"""

from gensim import utils, matutils
from gensim.models import ldamodel
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
import logging

logger = logging.getLogger('gensim.models.ldaseqmodel')

class LdaSeqModel(utils.SaveLoad):
    """
    The constructor estimates Dynamic Topic Model parameters based
    on a training corpus.
    If we have 30 documents, with 5 in the first time-slice, 10 in the second, and 15 in the third, we would
    set up our model like this:

    >>> ldaseq = LdaSeqModel(corpus=corpus, time_slice= [5, 10, 15], num_topics=5)

    Model persistency is achieved through inheriting utils.SaveLoad.

    >>> ldaseq.save("ldaseq")

    saves the model to disk.
    """

    def __init__(self, corpus=None, time_slice=None, id2word=None, alphas=0.01, num_topics=10,
                initialize='gensim', sstats=None, lda_model=None, obs_variance=0.5, chain_variance=0.005, passes=10,
                random_state=None, lda_inference_max_iter=25, em_min_iter=6, em_max_iter=20, chunksize=100):
        """
        `corpus` is any iterable gensim corpus

        `time_slice` as described above is a list which contains the number of documents in each time-slice

        `id2word` is a mapping from word ids (integers) to words (strings). It is used to determine the vocabulary size and printing topics.

        `alphas`  is a prior of your choice and should be a double or float value. default is 0.01

        `num_topics` is the number of requested latent topics to be extracted from the training corpus.

        `initalize` allows the user to decide how he wants to initialise the DTM model. Default is through gensim LDA.
        You can use your own sstats of an LDA model previously trained as well by specifying 'own' and passing a np matrix through sstats.
        If you wish to just pass a previously used LDA model, pass it through `lda_model`
        Shape of sstats is (vocab_len, num_topics)

        `chain_variance` is a constant which dictates how the beta values evolve - it is a gaussian parameter defined in the
        beta distribution.

        `passes` is the number of passes of the initial LdaModel.

        `random_state` can be a np.random.RandomState object or the seed for one, for the LdaModel.
        """
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.vocab_len = len(self.id2word)
        elif len(self.id2word) > 0:
            self.vocab_len = len(self.id2word)
        else:
            self.vocab_len = 0

        if corpus is not None:
            try:
                self.corpus_len = len(corpus)
            except:
                logger.warning("input corpus stream has no len(); counting documents")
                self.corpus_len = sum(1 for _ in corpus)

        self.time_slice = time_slice
        if self.time_slice is not None:
            self.num_time_slices = len(time_slice)

        max_doc_len = 0
        for line_no, line in enumerate(corpus):
            if len(line) > max_doc_len:
                max_doc_len = len(line)
        self.max_doc_len = max_doc_len

        self.num_topics = num_topics
        self.num_time_slices = len(time_slice)
        self.alphas = np.full(num_topics, alphas)

        # topic_chains contains for each topic a 'state space language model' object which in turn has information about each topic
        # the sslm class is described below and contains information on topic-word probabilities and doc-topic probabilities.
        self.topic_chains = []
        for topic in range(0, num_topics):
            sslm_ = sslm(num_time_slices=self.num_time_slices, vocab_len=self.vocab_len, num_topics=self.num_topics, chain_variance=chain_variance, obs_variance=obs_variance)
            self.topic_chains.append(sslm_)

        # the following are class variables which are to be integrated during Document Influence Model
        self.top_doc_phis = None
        self.influence = None
        self.renormalized_influence = None
        self.influence_sum_lgl = None

        # if a corpus and time_slice is provided, depending on the user choice of initializing LDA, we start DTM.
        if corpus is not None and time_slice is not None:
            if initialize == 'gensim':
                lda_model = ldamodel.LdaModel(corpus, id2word=self.id2word, num_topics=self.num_topics, passes=passes, alpha=self.alphas, random_state=random_state)
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
        """
        Method to initialize State Space Language Model, topic wise.
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
        """
        fit an lda sequence model:

        for each time period:
            set up lda model with E[log p(w|z)] and \alpha
            for each document:
                perform posterior inference
                update sufficient statistics/likelihood

        maximize topics

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
            for topic in range(0, num_topics):
                topic_suffstats.append(np.resize(np.zeros(vocab_len * data_len), (vocab_len, data_len)))

            # set up variables
            gammas = np.resize(np.zeros(corpus_len * num_topics), (corpus_len, num_topics))
            lhoods = np.resize(np.zeros(corpus_len * num_topics + 1), (corpus_len, num_topics + 1))
            # compute the likelihood of a sequential corpus under an LDA
            # seq model and find the evidence lower bound. This is the E - Step
            bound, gammas = self.lda_seq_infer(corpus, topic_suffstats, gammas, lhoods, iter_, lda_inference_max_iter, chunksize)
            self.gammas = gammas

            logger.info("M Step")

            # fit the variational distribution. This is the M - Step
            topic_bound = self.fit_lda_seq_topics(topic_suffstats)
            bound += topic_bound

            if ((bound - old_bound) < 0):
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


    def lda_seq_infer(self, corpus, topic_suffstats, gammas, lhoods, iter_, lda_inference_max_iter, chunksize):
        """
        Inference or E- Step.
        This is used to set up the gensim LdaModel to be used for each time-slice.
        It also allows for Document Influence Model code to be written in.
        """
        num_topics = self.num_topics
        vocab_len = self.vocab_len
        bound = 0.0

        lda = ldamodel.LdaModel(num_topics=num_topics, alpha=self.alphas, id2word=self.id2word)
        lda.topics = np.array(np.split(np.zeros(vocab_len * num_topics), vocab_len))
        ldapost = LdaPost(max_doc_len=self.max_doc_len, num_topics=num_topics, lda=lda)

        model = "DTM"
        if model == "DTM":
            bound, gammas = self.inferDTMseq(corpus, topic_suffstats, gammas, lhoods, lda, ldapost, iter_, bound, lda_inference_max_iter, chunksize)
        elif model == "DIM":
            self.InfluenceTotalFixed(corpus)
            bound, gammas = self.inferDIMseq(corpus, topic_suffstats, gammas, lhoods, lda, ldapost, iter_, bound, lda_inference_max_iter, chunksize)

        return bound, gammas


    def inferDTMseq(self, corpus, topic_suffstats, gammas, lhoods, lda, ldapost, iter_, bound, lda_inference_max_iter, chunksize):
        """
        Computes the likelihood of a sequential corpus under an LDA seq model, and return the likelihood bound.
        Need to pass the LdaSeq model, corpus, sufficient stats, gammas and lhoods matrices previously created,
        and LdaModel and LdaPost class objects.
        """
        doc_index = 0 # overall doc_index in corpus
        time = 0 # current time-slice
        doc_num = 0  # doc-index in current time-lice
        num_topics = self.num_topics
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
                    doc_lhood = LdaPost.fit_lda_post(ldapost, doc_num, time, None, lda_inference_max_iter=lda_inference_max_iter)
                else:
                    doc_lhood = LdaPost.fit_lda_post(ldapost, doc_num, time, self, lda_inference_max_iter=lda_inference_max_iter)

                if topic_suffstats is not None:
                    topic_suffstats = LdaPost.update_lda_seq_ss(ldapost, time, doc, topic_suffstats)

                gammas[doc_index] = ldapost.gamma
                bound += doc_lhood
                doc_index += 1
                doc_num += 1

        return bound, gammas


    def make_lda_seq_slice(self, lda, time):
        """
        set up the LDA model topic-word values with that of ldaseq.
        """
        for k in range(0, self.num_topics):
            lda.topics[:, k] = np.copy(self.topic_chains[k].e_log_prob[:, time])

        lda.alpha = np.copy(self.alphas)
        return lda


    def fit_lda_seq_topics(self, topic_suffstats):
        """
        Fit lda sequence topic wise.
        """
        lhood = 0
        lhood_term = 0

        for k, chain in enumerate(self.topic_chains):
            logger.info("Fitting topic number %i", k)
            lhood_term = sslm.fit_sslm(chain, topic_suffstats[k])
            lhood += lhood_term

        return lhood


    def print_topic_times(self, topic, top_terms=20):
        """
        Prints one topic showing each time-slice.
        """
        topics = []
        for time in range(0, self.num_time_slices):
            topics.append(self.print_topic(topic, time, top_terms))

        return topics


    def print_topics(self, time=0, top_terms=20):
        """
        Prints all topics in a particular time-slice.
        """
        topics =[]
        for topic in range(0, self.num_topics):
            topics.append(self.print_topic(topic, time, top_terms))
        return topics


    def print_topic(self, topic, time=0, top_terms=20):
        """
        Topic is the topic number
        Time is for a particular time_slice
        top_terms is the number of terms to display
        """
        topic = self.topic_chains[topic].e_log_prob
        topic = np.transpose(topic)
        topic = np.exp(topic[time])
        topic = topic / topic.sum()
        bestn = matutils.argsort(topic, top_terms, reverse=True)
        beststr = [(self.id2word[id_], round(topic[id_], 3)) for id_ in bestn]
        return beststr


    def doc_topics(self, doc_number):
        """
        On passing the LdaSeqModel trained ldaseq object, the doc_number of your document in the corpus,
        it returns the doc-topic probabilities of that document.
        """
        doc_topic = np.copy(self.gammas)
        doc_topic /= doc_topic.sum(axis=1)[:, np.newaxis]
        return doc_topic[doc_number]


    def dtm_vis(self, time, corpus):
        """
        returns term_frequency, vocab, doc_lengths, topic-term distributions and doc_topic distributions, specified by pyLDAvis format.
        all of these are needed to visualise topics for DTM for a particular time-slice via pyLDAvis.
        input parameter is the year to do the visualisation.
        """
        doc_topic = np.copy(self.gammas)
        doc_topic /= doc_topic.sum(axis=1)[:, np.newaxis]

        topic_term = [np.exp(np.transpose(chain.e_log_prob)[time]) / np.exp(np.transpose(chain.e_log_prob)[time]).sum() for k, chain in enumerate(self.topic_chains)]

        doc_lengths = [len(doc) for doc_no, doc in enumerate(corpus)]

        term_frequency = np.zeros(self.vocab_len)
        for doc_no, doc in enumerate(corpus):
            for pair in doc:
                term_frequency[pair[0]] += pair[1]

        vocab = [self.id2word[i] for i in range(0, len(self.id2word))]
        # returns np arrays for doc_topic proportions, topic_term proportions, and document_lengths, term_frequency.
        # these should be passed to the `pyLDAvis.prepare` method to visualise one time-slice of DTM topics.
        return doc_topic, np.array(topic_term), doc_lengths, term_frequency, vocab


    def dtm_coherence(self, time):
        """
        returns all topics of a particular time-slice without probabilitiy values for it to be used
        for either "u_mass" or "c_v" coherence.
        """
        coherence_topics = []
        for topics in self.print_topics(time):
            coherence_topic = []
            for word, dist in topics:
                coherence_topic.append(word)
            coherence_topics.append(coherence_topic)

        return coherence_topics

    def __getitem__(self, doc):
        """
        Similar to the LdaModel __getitem__ function, it returns topic proportions of a document passed.
        """
        lda_model = ldamodel.LdaModel(num_topics=self.num_topics, alpha=self.alphas, id2word=self.id2word)
        lda_model.topics = np.array(np.split(np.zeros(self.vocab_len * self.num_topics), self.vocab_len))
        ldapost = LdaPost(num_topics=self.num_topics, max_doc_len=len(doc), lda=lda_model, doc=doc)

        time_lhoods = []
        for time in range(0, self.num_time_slices):
            lda_model = self.make_lda_seq_slice(lda_model, time)  # create lda_seq slice
            lhood = LdaPost.fit_lda_post(ldapost, 0, time, self)
            time_lhoods.append(lhood)

        doc_topic = ldapost.gamma / ldapost.gamma.sum()
        # should even the likelihoods be returned?
        return doc_topic

# endclass LdaSeqModel


class sslm(utils.SaveLoad):
    """
    The sslm class is the State Space Language Model for DTM and contains the following information:
    `obs` values contain the doc - topic ratios
    `e_log_prob` contains topic - word ratios
    `mean`, `fwd_mean` contains the mean values to be used for inference for each word for a time_slice
    `variance`, `fwd_variance` contains the variance values to be used for inference for each word in a time_slice
    `fwd_mean`, `fwd_variance` are the forward posterior values.
    `zeta` is an extra variational parameter with a value for each time-slice
    """
    def __init__(self, vocab_len=None, num_time_slices=None, num_topics=None, obs_variance=0.5, chain_variance=0.005):
        self.vocab_len = vocab_len
        self.num_time_slices = num_time_slices
        self.obs_variance = obs_variance
        self.chain_variance= chain_variance
        self.num_topics = num_topics

        # setting up matrices
        self.obs = np.array(np.split(np.zeros(num_time_slices * vocab_len), vocab_len))
        self.e_log_prob = np.array(np.split(np.zeros(num_time_slices * vocab_len), vocab_len))
        self.mean = np.array(np.split(np.zeros((num_time_slices + 1) * vocab_len), vocab_len))
        self.fwd_mean = np.array(np.split(np.zeros((num_time_slices + 1) * vocab_len), vocab_len))
        self.fwd_variance = np.array(np.split(np.zeros((num_time_slices + 1) * vocab_len), vocab_len))
        self.variance = np.array(np.split(np.zeros((num_time_slices + 1) * vocab_len), vocab_len))
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
        """
        Updates the Zeta Variational Parameter.
        Zeta is described in the appendix and is equal to sum (exp(mean[word] + Variance[word] / 2)), over every time-slice.
        It is the value of variational parameter zeta which maximizes the lower bound.
        """
        for j, val in enumerate(self.zeta):
            self.zeta[j] = np.sum(np.exp(self.mean[:, j + 1] + self.variance[:, j + 1] / 2))
        return self.zeta


    def compute_post_variance(self, word, chain_variance):
        """
        Based on the Variational Kalman Filtering approach for Approximate Inference [https://www.cs.princeton.edu/~blei/papers/BleiLafferty2006a.pdf]
        This function accepts the word to compute variance for, along with the associated sslm class object, and returns variance and fwd_variance
        Computes Var[\beta_{t,w}] for t = 1:T

        Fwd_Variance(t) ≡ E((beta_{t,w} − mean_{t,w})^2 |beta_{t} for 1:t)
        = (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance ) * (fwd_variance[t - 1] + obs_variance)

        Variance(t) ≡ E((beta_{t,w} − mean_cap{t,w})^2 |beta_cap{t} for 1:t)
        = fwd_variance[t - 1] + (fwd_variance[t - 1] / fwd_variance[t - 1] + obs_variance)^2 * (variance[t - 1] - (fwd_variance[t-1] + obs_variance))

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
                c  = 0
            variance[t] = (c * (variance[t + 1] - chain_variance)) + ((1 - c) * fwd_variance[t])

        return variance, fwd_variance


    def compute_post_mean(self, word, chain_variance):
        """
        Based on the Variational Kalman Filtering approach for Approximate Inference [https://www.cs.princeton.edu/~blei/papers/BleiLafferty2006a.pdf]
        This function accepts the word to compute mean for, along with the associated sslm class object, and returns mean and fwd_mean
        Essentially a forward-backward to compute E[\beta_{t,w}] for t = 1:T.

        Fwd_Mean(t) ≡  E(beta_{t,w} | beta_ˆ 1:t )
        = (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance ) * fwd_mean[t - 1] + (1 - (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance)) * beta

        Mean(t) ≡ E(beta_{t,w} | beta_ˆ 1:T )
        = fwd_mean[t - 1] + (obs_variance / fwd_variance[t - 1] + obs_variance) + (1 - obs_variance / fwd_variance[t - 1] + obs_variance)) * mean[t]

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
        """
        Compute the expected log probability given values of m.
        The appendix describes the Expectation of log-probabilities in equation 5 of the DTM paper;
        The below implementation is the result of solving the equation and is as implemented in the original Blei DTM code.
        """
        for (w, t), val in np.ndenumerate(self.e_log_prob):
            self.e_log_prob[w][t] = self.mean[w][t + 1] - np.log(self.zeta[t])
        return self.e_log_prob


    def sslm_counts_init(self, obs_variance, chain_variance, sstats):
        """
        Initialize State Space Language Model with LDA sufficient statistics.
        Called for each topic-chain and initializes intial mean, variance and Topic-Word probabilities for the first time-slice.
        """
        W = self.vocab_len
        T = self.num_time_slices

        log_norm_counts = np.copy(sstats)
        log_norm_counts = log_norm_counts / sum(log_norm_counts)
        log_norm_counts = log_norm_counts + 1.0 / W
        log_norm_counts = log_norm_counts / sum(log_norm_counts)
        log_norm_counts = np.log(log_norm_counts)

        # setting variational observations to transformed counts
        self.obs = (np.repeat(log_norm_counts, T, axis=0)).reshape(W, T)
        # set variational parameters
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance

        # compute post variance, mean
        for w in range(0, W):
            self.variance[w], self.fwd_variance[w] = self.compute_post_variance(w, self.chain_variance)
            self.mean[w], self.fwd_mean[w] = self.compute_post_mean(w, self.chain_variance)

        self.zeta = self.update_zeta()
        self.e_log_prob = self.compute_expected_log_prob()


    def fit_sslm(self, sstats):
        """
        Fits variational distribution.
        This is essentially the m-step.
        Accepts the sstats for a particular topic for input and maximizes values for that topic.
        Updates the values in the update_obs() and compute_expected_log_prob methods.
        """
        W = self.vocab_len
        bound = 0
        old_bound = 0
        sslm_fit_threshold = 1e-6
        sslm_max_iter = 2
        converged = sslm_fit_threshold + 1

        totals = np.zeros(sstats.shape[1])

        # computing variance, fwd_variance
        self.variance, self.fwd_variance = map(np.array, list(zip(*[self.compute_post_variance(w, self.chain_variance) for w in range(0, W)])))

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
        """
        Compute log probability bound.
        Forumula is as described in appendix of DTM by Blei. (formula no. 5)
        """
        W = self.vocab_len
        T = self.num_time_slices

        term_1 = 0
        term_2 = 0
        term_3 = 0

        val = 0
        ent = 0

        chain_variance = self.chain_variance
        # computing mean, fwd_mean
        self.mean, self.fwd_mean = map(np.array, (zip(*[self.compute_post_mean(w, self.chain_variance) for w in range(0, W)])))
        self.zeta = self.update_zeta()

        for w in range(0, W):
            val += (self.variance[w][0] - self.variance[w][T]) / 2 * chain_variance

        logger.info("Computing bound, all times")

        for t in range(1, T + 1):
            term_1 = 0.0
            term_2 = 0.0
            ent = 0.0
            for w in range(0, W):

                m = self.mean[w][t]
                prev_m = self.mean[w][t - 1]

                v = self.variance[w][t]

                # w_phi_l is only used in Document Influence Model; the values are aleays zero in this case
                # w_phi_l = sslm.w_phi_l[w][t - 1]
                # exp_i = np.exp(-prev_m)
                # term_1 += (np.power(m - prev_m - (w_phi_l * exp_i), 2) / (2 * chain_variance)) - (v / chain_variance) - np.log(chain_variance)

                term_1 += (np.power(m - prev_m, 2) / (2 * chain_variance)) - (v / chain_variance) - np.log(chain_variance)
                term_2 += sstats[w][t - 1] * m
                ent += np.log(v) / 2  # note the 2pi's cancel with term1 (see doc)

            term_3 = -totals[t - 1] * np.log(self.zeta[t - 1])
            val += term_2 + term_3 + ent - term_1

        return val


    def update_obs(self, sstats, totals):
        """
        Function to perform optimization of obs. Parameters are suff_stats set up in the fit_sslm method.

        TODO:
        This is by far the slowest function in the whole algorithm.
        Replacing or improving the performance of this would greatly speed things up.
        """

        OBS_NORM_CUTOFF = 2
        STEP_SIZE = 0.01
        TOL = 1e-3

        W = self.vocab_len
        T = self.num_time_slices

        runs = 0
        mean_deriv_mtx = np.resize(np.zeros(T * (T + 1)), (T, T + 1))

        norm_cutoff_obs = None
        for w in range(0, W):
            w_counts = sstats[w]
            counts_norm = 0
            # now we find L2 norm of w_counts
            for i in range(0, len(w_counts)):
                counts_norm += w_counts[i] * w_counts[i]

            counts_norm = np.sqrt(counts_norm)

            if counts_norm < OBS_NORM_CUTOFF and norm_cutoff_obs is not None:
                obs = self.obs[w]
                norm_cutoff_obs = np.copy(obs)
            else:
                if counts_norm < OBS_NORM_CUTOFF:
                    w_counts = np.zeros(len(w_counts))

                # TODO: apply lambda function
                for t in range(0, T):
                    mean_deriv = mean_deriv_mtx[t]
                    mean_deriv = self.compute_mean_deriv(w, t, mean_deriv)
                    mean_deriv_mtx[t] = mean_deriv

                deriv = np.zeros(T)
                args = self, w_counts, totals, mean_deriv_mtx, w, deriv
                obs = self.obs[w]
                model = "DTM"

                if model == "DTM":
                    # slowest part of method
                    obs = optimize.fmin_cg(f=f_obs, fprime=df_obs, x0=obs, gtol=TOL, args=args, epsilon=STEP_SIZE, disp=0)
                if model == "DIM":
                    pass
                runs += 1

                if counts_norm < OBS_NORM_CUTOFF:
                    norm_cutoff_obs = obs

                self.obs[w] = obs

        self.zeta = self.update_zeta()

        return self.obs, self.zeta


    def compute_mean_deriv(self, word, time, deriv):
        """
        Used in helping find the optimum function.
        computes derivative of E[\beta_{t,w}]/d obs_{s,w} for t = 1:T.
        put the result in deriv, allocated T+1 vector
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
        """
        Derivation of obs which is used in derivative function [df_obs] while optimizing.
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

        for u in range(0, T):
            self.temp_vect[u] = np.exp(mean[u + 1] + variance[u + 1] / 2)

        for t in range(0, T):
            mean_deriv = mean_deriv_mtx[t]
            term1 = 0
            term2 = 0
            term3 = 0
            term4 = 0

            for u in range(1, T + 1):
                mean_u = mean[u]
                variance_u_prev = variance[u - 1]
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
# endclass sslm

class LdaPost(utils.SaveLoad):

    """
    Posterior values associated with each set of documents.
    TODO: use **Hoffman, Blei, Bach: Online Learning for Latent Dirichlet Allocation, NIPS 2010.**
    to update phi, gamma. End game would be to somehow replace LdaPost entirely with LdaModel.
    """

    def __init__(self, doc=None, lda=None, max_doc_len=None, num_topics=None, gamma=None, lhood=None):

        self.doc = doc
        self.lda = lda
        self.gamma = gamma
        self.lhood = lhood
        if self.gamma is None:
            self.gamma = np.zeros(num_topics)
        if self.lhood is None:
            self.lhood = np.zeros(num_topics + 1)

        if max_doc_len is not None and num_topics is not None:
            self.phi = np.resize(np.zeros(max_doc_len * num_topics), (max_doc_len, num_topics))
            self.log_phi = np.resize(np.zeros(max_doc_len * num_topics), (max_doc_len, num_topics))

        # the following are class variables which are to be integrated during Document Influence Model

        self.doc_weight = None
        self.renormalized_doc_weight = None


    def update_phi(self, doc_number, time):
        """
        Update variational multinomial parameters, based on a document and a time-slice.
        This is done based on the original Blei-LDA paper, where:
        log_phi := beta * exp(Ψ(gamma)), over every topic for every word.

        TODO: incorporate lee-sueng trick used in **Lee, Seung: Algorithms for non-negative matrix factorization, NIPS 2001**.
        """
        num_topics = self.lda.num_topics
        # digamma values
        dig = np.zeros(num_topics)

        for k in range(0, num_topics):
            dig[k] = digamma(self.gamma[k])

        n = 0   # keep track of iterations for phi, log_phi
        for word_id, count in self.doc:
            for k in range(0, num_topics):
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
            n +=1 # increase iteration

        return self.phi, self.log_phi


    def update_gamma(self):
        """
        update variational dirichlet parameters as described in the original Blei LDA paper:
        gamma = alpha + sum(phi), over every topic for every word.
        """
        self.gamma = np.copy(self.lda.alpha)
        n = 0 # keep track of number of iterations for phi, log_phi
        for word_id, count in self.doc:
            phi_row = self.phi[n]
            for k in range(0, self.lda.num_topics):
                self.gamma[k] += phi_row[k] * count
            n += 1

        return self.gamma


    def init_lda_post(self):
        """
        Initialize variational posterior, does not return anything.
        """
        total = sum(count for word_id, count in self.doc)
        self.gamma.fill(self.lda.alpha[0] + float(total) / self.lda.num_topics)
        self.phi[:len(self.doc),:] = 1.0 / self.lda.num_topics
        # doc_weight used during DIM
        # ldapost.doc_weight = None


    def compute_lda_lhood(self):
        """
        compute the likelihood bound
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

        model = "DTM"
        for k in range(0, num_topics):
            # below code only to be used in DIM mode
            # if ldapost.doc_weight is not None and (model == "DIM" or model == "fixed"):
            #     influence_topic = ldapost.doc_weight[k]
            #     influence_term = - ((influence_topic * influence_topic + sigma_l * sigma_l) / 2.0 / (sigma_d * sigma_d))

            e_log_theta_k = digamma(self.gamma[k]) - digsum
            lhood_term = (self.lda.alpha[k] - self.gamma[k]) * e_log_theta_k + gammaln(self.gamma[k]) - gammaln(self.lda.alpha[k])
            # TODO: check why there's an IF
            n = 0
            for word_id, count in self.doc:
                if self.phi[n][k] > 0:
                    lhood_term += count * self.phi[n][k] * (e_log_theta_k + self.lda.topics[word_id][k] - self.log_phi[n][k])
                n += 1
            self.lhood[k] = lhood_term
            lhood += lhood_term
            # in case of DIM add influence term
            # lhood += influence_term

        return lhood

    def fit_lda_post(self, doc_number, time, ldaseq, LDA_INFERENCE_CONVERGED=1e-8,
                    lda_inference_max_iter=25, g=None, g3_matrix=None, g4_matrix=None, g5_matrix=None):
        """
        Posterior inference for lda.
        g, g3, g4 and g5 are matrices used in Document Influence Model and not used currently.
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
        """
        Update lda sequence sufficient statistics from an lda posterior.
        This is very similar to the update_gamma method and uses the same formula.
        """
        num_topics = self.lda.num_topics

        for k in range(0, num_topics):
            topic_ss = topic_suffstats[k]
            n = 0
            for word_id, count in self.doc:
                topic_ss[word_id][time] += count * self.phi[n][k]
                n += 1
            topic_suffstats[k] = topic_ss

        return topic_suffstats
# endclass LdaPost


# the following functions are used in update_obs as the function to optimize
def f_obs(x, *args):
    """
    Function which we are optimising for minimizing obs.
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
        var_t_prev = variance[t - 1]

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

    """
    Derivative of function which optimises obs.
    """
    sslm, word_counts, totals, mean_deriv_mtx, word, deriv = args

    sslm.obs[word] = x
    sslm.mean[word], sslm.fwd_mean[word] = sslm.compute_post_mean(word, sslm.chain_variance)

    model = "DTM"
    if model == "DTM":
        deriv = sslm.compute_obs_deriv(word, word_counts, totals, mean_deriv_mtx, deriv)
    elif model == "DIM":
        deriv = sslm.compute_obs_deriv_fixed(p.word, p.word_counts, p.totals, p.sslm, p.mean_deriv_mtx, deriv)

    return np.negative(deriv)
