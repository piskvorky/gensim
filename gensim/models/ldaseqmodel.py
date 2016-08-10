#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Based on Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>


"""

This is the class which is used to help with Dynamic Topic Modelling of a corpus.
Inspired by the Blei's original DTM code and paper. 
DTM C/C++ code: https://github.com/blei-lab/dtm
DTM Paper: https://www.cs.princeton.edu/~blei/papers/BleiLafferty2006a.pdf


"""

from gensim import interfaces, utils, matutils
from gensim.models import ldamodel
import numpy
import math
from scipy.special import digamma
from scipy import optimize


class seq_corpus(utils.SaveLoad):

    """
    `seq_corpus` is basically a wrapper class which contains information about the corpus.
    `vocab_len` is the length of the vocabulary.
    `max_doc_len` is the maximum number of terms a single document has.
    `num_time_slices` is the number of sequences, i.e number of time-slices.
    `corpus_len` is the number of documents present.
    `time_slice` is a list or numpy array which the user must provide which contains the number of documents in each time-slice.
    `corpus` is any iterable gensim corpus.

    """
    def __init__(self, corpus=None, time_slice=None, id2word=None):


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

        self.corpus = corpus
        if self.corpus is not None:
            self.corpus_len = len(corpus)

        self.time_slice = time_slice
        if self.time_slice is not None:
            self.num_time_slices = len(time_slice)

        max_doc_len = 0
        for line_no, line in enumerate(corpus):
            if len(line) > max_doc_len:
                max_doc_len = len(line)
        self.max_doc_len = max_doc_len

# endclass seq_corpus

class Doc(utils.SaveLoad):
    """
    The doc class contains information used for each document. 

    """
    def __init__(self, nterms=None, word=None, count=None, total=None):

        self.nterms = nterms
        self.word = word
        self.count = count
        self.total = total

# endclass Doc

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
                initialize='gensim', sstats=None,  lda_model=None, obs_variance=0.5, chain_variance=0.005):
        """
        `corpus` is any iterable gensim corpus

        `time_slice` as described above is a list which contains the number of documents in each time-slice

        `id2word` is a mapping from word ids (integers) to words (strings). It is used to determine the vocabulary size and printing topics.

        `alphas`  is a prior of your choice and should be a double or float value. default is 0.01

        `num_topics` is the number of requested latent topics to be extracted from the training corpus.

        `initalize` allows the user to decide how he wants to initialise the DTM model. Default is through gensim LDA.
        if `initalize` is 'blei-lda', then we will use the python port of blei's orignal LDA code.
        You can use your own sstats of an LDA model previously trained as well by specifying 'own' and passing a numpy matrix through sstats.
        If you wish to just pass a previously used LDA model, pass it through `lda_model` 
        Shape of sstats is (vocab_len, num_topics)

        `chain_variance` is a constant which dictates how the beta values evolve - it is a gaussian parameter defined in the
        beta distribution. 

        """
 
        if corpus is not None:
            self.corpus = seq_corpus(corpus=corpus, id2word=id2word, time_slice=time_slice)
            self.vocab_len = len(self.corpus.id2word)


        self.num_topics = num_topics
        self.num_time_slices = len(time_slice)
        self.alphas = numpy.full(num_topics, alphas)

        #topic_chains contains for each topic a 'state space language model' object which in turn has information about each topic
        #the sslm class is described below and contains information on topic-word probabilities and doc-topic probabilities.
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
        if self.corpus is not None and time_slice is not None:
            if initialize == 'gensim':
                lda_model = ldamodel.LdaModel(corpus, id2word=self.corpus.id2word, num_topics=self.num_topics, passes=10, alpha=self.alphas)
                self.sstats = numpy.transpose(lda_model.state.sstats)
            if initialize == 'ldamodel':
                self.sstats = numpy.transpose(lda_model.state.sstats)
            if initialize == 'own':
                self.sstats = sstats
            if initialize == 'blei-lda':
                self.sstats = lda_sstats(self.corpus, self.num_topics, self.vocab_len, self.alphas)
            # initialize model from sstats
            init_ldaseq_ss(self, chain_variance, obs_variance, self.alphas, self.sstats)

            # fit DTM
            fit_lda_seq(self, self.corpus)

# endclass LdaSeqModel

class sslm(utils.SaveLoad):
    """
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

        self.obs = numpy.array(numpy.split(numpy.zeros(num_time_slices * vocab_len), vocab_len))
        self.e_log_prob = numpy.array(numpy.split(numpy.zeros(num_time_slices * vocab_len), vocab_len))
        self.mean = numpy.array(numpy.split(numpy.zeros((num_time_slices + 1) * vocab_len), vocab_len))
        self.fwd_mean = numpy.array(numpy.split(numpy.zeros((num_time_slices + 1) * vocab_len), vocab_len))
        self.fwd_variance = numpy.array(numpy.split(numpy.zeros((num_time_slices + 1) * vocab_len), vocab_len))
        self.variance = numpy.array(numpy.split(numpy.zeros((num_time_slices + 1) * vocab_len), vocab_len))
        self.zeta = numpy.zeros(num_time_slices)

        # the following are class variables which are to be integrated during Document Influence Model
        self.m_update_coeff = None
        self.mean_t = None
        self.variance_t = None
        self.influence_sum_lgl = None
        self.w_phi_l = None
        self.w_phi_sum = None
        self.w_phi_l_sq = None
        self.m_update_coeff_g = None

# endclass sslm

class LdaPost(utils.SaveLoad):

    """
    Posterior values associated with each set of documents.
    """

    def __init__(self, doc=None, lda=None, max_doc_len=None, num_topics=None, gamma=None, lhood=None):

        self.doc = doc
        self.lda = lda
        self.gamma = gamma
        self.lhood = lhood

        if max_doc_len is not None and num_topics is not None:
            self.phi = numpy.resize(numpy.zeros(max_doc_len * num_topics), (max_doc_len, num_topics))
            self.log_phi = numpy.resize(numpy.zeros(max_doc_len * num_topics), (max_doc_len, num_topics))

        # the following are class variables which are to be integrated during Document Influence Model

        self.doc_weight = None
        self.renormalized_doc_weight = None

# endclass LdaState

def update_zeta(sslm):

    """
    Updates the Zeta Variational Parameter.
    Zeta is described in the appendix and is equal to sum (exp(mean[word] + Variance[word] / 2)), over every time-slice.
    It is the value of variational parameter zeta which maximizes the lower bound.
    """

    vocab_len = sslm.vocab_len
    num_time_slices = sslm.num_time_slices
    sslm.zeta.fill(0)

    for i in range(0, vocab_len):
        for j in range(0, num_time_slices):

            m = sslm.mean[i][j + 1]
            v = sslm.variance[i][j + 1]
            val = numpy.exp(m + v/2)
            sslm.zeta[j] = sslm.zeta[j] + val  

    return sslm.zeta

def compute_post_variance(word, sslm, chain_variance):

    """
    Based on the Variational Kalman Filtering approach for Approximate Inference [https://www.cs.princeton.edu/~blei/papers/BleiLafferty2006a.pdf]
    This function accepts the word to compute variance for, along with the associated sslm class object, and returns variance and fwd_variance
    Computes Var[\beta_{t,w}] for t = 1:T

    Fwd_Variance(t) ≡ E((beta_{t,w} − mean_{t,w})^2 |beta_{t} for 1:t) 
    = (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance ) * (fwd_variance[t - 1] + obs_variance)
    
    Variance(t) ≡ E((beta_{t,w} − mean_cap{t,w})^2 |beta_cap{t} for 1:t) 
    = fwd_variance[t - 1] + (fwd_variance[t - 1] / fwd_variance[t - 1] + obs_variance)^2 * (variance[t - 1] - (fwd_variance[t-1] + obs_variance))    

    """
    INIT_VARIANCE = 1000

    T = sslm.num_time_slices
    variance = sslm.variance[word]
    fwd_variance = sslm.fwd_variance[word]

    # forward pass. Set initial variance very high
    fwd_variance[0] = chain_variance * INIT_VARIANCE
    
    for t in range(1, T + 1):
        if sslm.obs_variance:
            w = sslm.obs_variance / (fwd_variance[t - 1] + chain_variance + sslm.obs_variance)
        else:
            w = 0
        fwd_variance[t] = w * (fwd_variance[t - 1] + chain_variance)

    # backward pass 
    variance[T] = fwd_variance[T]
    for t in range(T - 1, -1, -1):
        if fwd_variance[t] > 0.0:
            w = numpy.power((fwd_variance[t] / (fwd_variance[t] + chain_variance)), 2)
        else:
            w = 0
        variance[t] = (w * (variance[t + 1] - chain_variance)) + ((1 - w) * fwd_variance[t])

    return variance, fwd_variance
    

 
def compute_post_mean(word, sslm, chain_variance):

    """
    Based on the Variational Kalman Filtering approach for Approximate Inference [https://www.cs.princeton.edu/~blei/papers/BleiLafferty2006a.pdf]
    This function accepts the word to compute mean for, along with the associated sslm class object, and returns mean and fwd_mean
    Essentially a forward-backward to compute E[\beta_{t,w}] for t = 1:T.

    Fwd_Mean(t) ≡  E(beta_{t,w} | beta_ˆ 1:t ) 
    = (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance ) * fwd_mean[t - 1] + (1 - (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance)) * beta
    
    Mean(t) ≡ E(beta_{t,w} | beta_ˆ 1:T ) 
    = fwd_mean[t - 1] + (obs_variance / fwd_variance[t - 1] + obs_variance) + (1 - obs_variance / fwd_variance[t - 1] + obs_variance)) * mean[t]    

    """


    T = sslm.num_time_slices

    obs = sslm.obs[word]
    fwd_variance = sslm.fwd_variance[word]

    mean = sslm.mean[word]
    fwd_mean = sslm.fwd_mean[word]

    # forward 
    fwd_mean[0] = 0
    for t in range(1, T + 1):
        w = sslm.obs_variance / (fwd_variance[t - 1] + chain_variance + sslm.obs_variance)
        fwd_mean[t] = w * fwd_mean[t - 1] + (1 - w) * obs[t - 1]

    # backward pass
    mean[T] = fwd_mean[T]
    for t in range(T - 1, -1, -1):
        if chain_variance == 0.0:
            w = 0.0
        else:
            w = chain_variance / (fwd_variance[t] + chain_variance)
        mean[t] = w * fwd_mean[t] + (1 - w) * mean[t + 1]

    return mean, fwd_mean


def update_phi(doc, time, ldapost):

    """
    Update variational multinomial parameters, based on a document and a time-slice.
    This is done based on the original Blei-LDA paper, where:
    log_phi := beta * exp(Ψ(gamma)), over every topic for every word.
    
    TODO: incorporate lee-sueng trick used in **Lee, Seung: Algorithms for non-negative matrix factorization, NIPS 2001**.
    """

    K = ldapost.lda.num_topics
    N = ldapost.doc.nterms

    dig = numpy.zeros(K)

    for k in range(0, K):
        dig[k] = digamma(ldapost.gamma[k])

    for n in range(0, N):
        w = ldapost.doc.word[n]
        for k in range(0, K):
            ldapost.log_phi[n][k] = dig[k] + ldapost.lda.topics[w][k]

        log_phi_row = ldapost.log_phi[n]
        phi_row = ldapost.phi[n]

        # log normalize
        v = log_phi_row[0]
        for i in range(1, len(log_phi_row)):
            v = numpy.logaddexp(v, log_phi_row[i])

        for i in range(0, len(log_phi_row)):
            log_phi_row[i] = log_phi_row[i] - v

        for k in range(0, K):
            phi_row[k] = numpy.exp(log_phi_row[k])

        ldapost.log_phi[n] = log_phi_row
        ldapost.phi[n] = phi_row

    return ldapost.phi, ldapost.log_phi

def update_gamma(ldapost):

    """
    update variational dirichlet parameters as described in the original Blei LDA paper:
    gamma = alpha + sum(phi), over every topic for every word.

    """

    K = ldapost.lda.num_topics
    N = ldapost.doc.nterms

    ldapost.gamma = numpy.copy(ldapost.lda.alpha)

    for n in range(0, N):
        phi_row = ldapost.phi[n]
        count = ldapost.doc.count[n]

        for k in range(0, K):
            ldapost.gamma[k] += phi_row[k] * count

    return ldapost.gamma

def compute_expected_log_prob(sslm):

    """
    Compute the expected log probability given values of m.
    The appendix describes the Expectation of log-probabilities in equation 5 of the DTM paper;
    The below implementation is the result of solving the equation and is as implemented in the original Blei DTM code.
    """

    W = sslm.vocab_len
    T = sslm.num_time_slices
    for t in range(0, T):
        for w in range(0, W):
            sslm.e_log_prob[w][t] = sslm.mean[w][t + 1] - numpy.log(sslm.zeta[t])

    return sslm.e_log_prob


def sslm_counts_init(sslm, obs_variance, chain_variance, sstats):

    """
    Initialize State Space Language Model with LDA sufficient statistics.
    """

    W = sslm.vocab_len
    T = sslm.num_time_slices

    log_norm_counts = numpy.copy(sstats)
    log_norm_counts = log_norm_counts / sum(log_norm_counts)

    log_norm_counts = log_norm_counts + 1.0 / W
    log_norm_counts = log_norm_counts / sum(log_norm_counts)
    log_norm_counts = numpy.log(log_norm_counts)
    

    # setting variational observations to transformed counts
    for t in range(0, T):
        sslm.obs[:,t] = log_norm_counts

    # set variational parameters
    sslm.obs_variance = obs_variance
    sslm.chain_variance = chain_variance

    # compute post variance
    for w in range(0, W):
       sslm.variance, sslm.fwd_variance = compute_post_variance(w, sslm, sslm.chain_variance)

    for w in range(0, W):
       sslm.mean, sslm.fwd_mean = compute_post_mean(w, sslm, sslm.chain_variance)

    sslm.zeta = update_zeta(sslm)
    sslm.e_log_prob = compute_expected_log_prob(sslm)

def init_ldaseq_ss(ldaseq, topic_chain_variance, topic_obs_variance, alpha, init_suffstats):

    """
    Method to initialize State Space Language Model, topic wise.
    """

    ldaseq.alphas = alpha
    for k in range(0, ldaseq.num_topics):
        sstats = init_suffstats[:,k]
        sslm_counts_init(ldaseq.topic_chains[k], topic_obs_variance, topic_chain_variance, sstats)

        # initialize the below matrices only if running DIM
        # ldaseq.topic_chains[k].w_phi_l = numpy.zeros((ldaseq.vocab_len, ldaseq.num_time_slices))
        # ldaseq.topic_chains[k].w_phi_sum = numpy.zeros((ldaseq.vocab_len, ldaseq.num_time_slices))
        # ldaseq.topic_chains[k].w_phi_sq = numpy.zeros((ldaseq.vocab_len, ldaseq.num_time_slices))

def fit_lda_seq(ldaseq, seq_corpus):
    """
    fit an lda sequence model:
   
    for each time period
        set up lda model with E[log p(w|z)] and \alpha
        for each document
            perform posterior inference
            update sufficient statistics/likelihood
   
    maximize topics

   """
  
    LDA_INFERENCE_MAX_ITER = 25
    LDASQE_EM_THRESHOLD = 1e-4
    LDA_SEQ_MIN_ITER = 6
    LDA_SEQ_MAX_ITER = 20

    num_topics = ldaseq.num_topics
    vocab_len = ldaseq.vocab_len
    data_len = seq_corpus.num_time_slices
    corpus_len = seq_corpus.corpus_len
    
    bound = 0
    convergence = LDASQE_EM_THRESHOLD + 1

    iter_ = 0

    while iter_ < LDA_SEQ_MIN_ITER or ((convergence > LDASQE_EM_THRESHOLD) and iter_ <= LDA_SEQ_MAX_ITER):

        print (" EM iter " , iter_)
        print ("E Step")

        old_bound = bound
        # initiate sufficient statistics
        topic_suffstats = []
        for num_topics in range(0, num_topics):
            topic_suffstats.append(numpy.resize(numpy.zeros(vocab_len * data_len), (vocab_len, data_len)))

        # set up variables
        gammas = numpy.resize(numpy.zeros(corpus_len * num_topics), (corpus_len, num_topics))
        lhoods = numpy.resize(numpy.zeros(corpus_len * num_topics + 1), (corpus_len, num_topics + 1))
        # compute the likelihood of a sequential corpus under an LDA

        # seq model and find the evidence lower bound. This is the E - Step
        bound = lda_seq_infer(ldaseq, seq_corpus, topic_suffstats, gammas, lhoods, iter_)

        print ("M Step")

        # fit the variational distribution
        topic_bound = fit_lda_seq_topics(ldaseq, topic_suffstats)
        bound += topic_bound


        if ((bound - old_bound) < 0):
            if LDA_INFERENCE_MAX_ITER < 10:
                LDA_INFERENCE_MAX_ITER *= 2                
            print ("Bound went down, increasing iterations to" , LDA_INFERENCE_MAX_ITER)

        # check for convergence
        convergence = numpy.fabs((bound - old_bound) / old_bound)

        if convergence < LDASQE_EM_THRESHOLD:

            LDA_INFERENCE_MAX_ITER = 500
            print ("Starting final iterations, max iter is", LDA_INFERENCE_MAX_ITER)
            convergence = 1.0

        print (iter_, "iteration lda seq bound is", bound, ", convergence is ", convergence)

        iter_ += 1

    return bound


def lda_seq_infer(ldaseq, seq_corpus, topic_suffstats, gammas, lhoods, iter_):

    """
    Inference or E- Step.
    This is used to set up the gensim LdaModel to be used for each time-slice. 
    It also allows for Document Influence Model code to be written in.
    """

    num_topics = ldaseq.num_topics
    vocab_len = ldaseq.vocab_len
    bound = 0.0
    
    lda = ldamodel.LdaModel(num_topics=num_topics, alpha=ldaseq.alphas, id2word=seq_corpus.id2word)
    lda.topics = numpy.array(numpy.split(numpy.zeros(vocab_len * num_topics), vocab_len))
    ldapost = LdaPost(max_doc_len=seq_corpus.max_doc_len, num_topics=num_topics, lda=lda)

    model = "DTM"
    if model == "DTM":
        bound = inferDTMseq(daseq, seq_corpus, topic_suffstats, gammas, lhoods, lda, ldapost)
    elif model == "DIM":
        InfluenceTotalFixed(ldaseq, seq_corpus);
        bound = inferDIMseq(ldaseq, seq_corpus, topic_suffstats, gammas, lhoods, lda, ldapost)

    return bound


def inferDTMseq(ldaseq, seq_corpus, topic_suffstats, gammas, lhoods, lda, ldapost):

    """
    Computes the likelihood of a sequential corpus under an LDA seq model, and return the likelihood bound.
    Need to pass the LdaSeq model, seq_corpus, sufficient stats, gammas and lhoods matrices previously created,
    and LdaModel and LdaPost class objects.
    """

    doc_index = 0 # overall doc_index in corpus
    time = 0 # current time-slice
    doc_num = 0  # doc-index in current time-lice
    num_topics = ldaseq.num_topics
    make_lda_seq_slice(lda, ldaseq, time)  # create lda_seq slice

    time_slice = numpy.cumsum(numpy.array(ldaseq.time_slice))

    for line_no, line in enumerate(seq_corpus.corpus):
        # this is used to update the time_slice and create a new lda_seq slice every new time_slice
        if doc_index > time_slice[t]:
            time += 1
            make_lda_seq_slice(lda, ldaseq, time)    
            doc_num = 0

        gam = gammas[doc_index]
        lhood = lhoods[doc_index]

        doc_ = line

        nterms = len(doc_)
        words = []
        counts = []
        totals = 0
        for word_id, count in doc_:
            words.append(int(word_id))
            counts.append(int(count))
            totals += int(count)

        doc = Doc(word=words, count=counts, total=totals, nterms=int(nterms))
        ldapost.gamma = gam
        ldapost.lhood = lhood
        ldapost.doc = doc

        if iter_ == 0:
            doc_lhood = fit_lda_post(doc_num, time, ldapost, None, None, None, None, None)
        else:
            doc_lhood = fit_lda_post(doc_num, time, ldapost, ldaseq, None, None, None, None)
       

        if topic_suffstats != None:
            topic_suffstats = update_lda_seq_ss(time, doc, ldapost, topic_suffstats)

        bound += doc_lhood
        doc_index += 1
        doc_num += 1

    return bound

def fit_lda_post(doc_number, time, ldapost, ldaseq, g, g3_matrix, g4_matrix, g5_matrix):

    """
    Posterior inference for lda.
    """

    LDA_INFERENCE_CONVERGED = 1e-8
    LDA_INFERENCE_MAX_ITER = 25

    init_lda_post(ldapost)

    model = "DTM"
    if model == "DIM":
        # if in DIM then we initialise some variables here
        pass

    lhood = compute_lda_lhood(ldapost)
    lhood_old = 0
    converged = 0
    iter_ = 0

    # first iteration starts here
    iter_ += 1
    lhood_old = lhood
    ldapost.gamma = update_gamma(ldapost)

    model = "DTM"

    if model == "DTM" or sslm is None:
        ldapost.phi, ldapost.log_phi = update_phi(doc_number, time, ldapost)
    elif model == "DIM" and sslm is not None:
        ldapost.phi, ldapost.log_phi = update_phi_fixed(doc_number, time, ldapost, sslm, g3_matrix, g4_matrix, g5_matrix)

    lhood = compute_lda_lhood(ldapost)
    converged = numpy.fabs((lhood_old - lhood) / (lhood_old * ldapost.doc.total))


    while converged > LDA_INFERENCE_CONVERGED and iter_ <= LDA_INFERENCE_MAX_ITER:

        iter_ += 1
        lhood_old = lhood
        ldapost.gamma = update_gamma(ldapost)
        model = "DTM"

        if model == "DTM" or sslm is None:
            ldapost.phi, ldapost.log_phi  = update_phi(doc_number, time, ldapost)
        elif model == "DIM" and sslm is not None:
            ldapost.phi, ldapost.log_phi  = update_phi_fixed(doc_number, time, ldapost, sslm, g3_matrix, g4_matrix, g5_matrix)

        lhood = compute_lda_lhood(ldapost)
        converged = numpy.fabs((lhood_old - lhood) / (lhood_old * ldapost.doc.total))

    return lhood


def make_lda_seq_slice(lda, ldaseq, time):

    """
    set up the LDA model topic-word values with that of ldaseq.
    """

    num_topics = ldaseq.num_topics
    for k in range(0, num_topics):
        lda.topics[:,k] = numpy.copy(ldaseq.topic_chains[k].e_log_prob[:,time])

    lda.alpha = numpy.copy(ldaseq.alphas)

    return

def update_lda_seq_ss(time, doc, ldapost, topic_suffstats):

    """
    Update lda sequence sufficient statistics from an lda posterior.
    """

    num_topics = numpy.shape(ldapost.phi)[1]
    nterms = doc.nterms

    for k in range(0, num_topics):
        topic_ss = topic_suffstats[k]
        for n in range(0, nterms):
            w = doc.word[n]
            c = doc.count[n]
            topic_ss[w][time] = topic_ss[w][time] + c * ldapost.phi[n][k]

        topic_suffstats[k] = topic_ss

    return topic_suffstats

def init_lda_post(ldapost):

    """
    Initialize variational posterior, does not return anything.
    """
    num_topics = ldapost.lda.num_topics
    nterms = ldapost.doc.nterms

    for k in range(0, num_topics):
        ldapost.gamma[k] = ldapost.lda.alpha[k] + float(ldapost.doc.total) / K
        for n in range(0, nterms):
            ldapost.phi[n][k] = 1.0 / K
   
    # doc_weight used during DIM
    # ldapost.doc_weight = None

    return

def compute_lda_lhood(ldapost):
    """
    compute the likelihood bound
    """

    K = ldapost.lda.num_topics
    N = ldapost.doc.nterms
    gamma_sum = numpy.sum(ldapost.gamma)

    # TODO: flags
    FLAGS_sigma_l = 0
    FLAGS_sigma_d = 0 

    lhood = math.lgamma(numpy.sum(ldapost.lda.alpha)) - math.lgamma(gamma_sum)
    ldapost.lhood[K] = lhood

    # influence_term = 0
    digsum = digamma(gamma_sum)

    model = "DTM"
    for k in range(0, K):
        # below code only to be used in DIM mode
        # if ldapost.doc_weight is not None and (model == "DIM" or model == "fixed"):
        #     influence_topic = ldapost.doc_weight[k]
        #     influence_term = - ((influence_topic * influence_topic + FLAGS_sigma_l * FLAGS_sigma_l) / 2.0 / (FLAGS_sigma_d * FLAGS_sigma_d))

        e_log_theta_k = digamma(ldapost.gamma[k]) - digsum
        lhood_term = (ldapost.lda.alpha[k] - ldapost.gamma[k]) * e_log_theta_k + math.lgamma(ldapost.gamma[k]) - math.lgamma(ldapost.lda.alpha[k])

        for n in range(0, N):
            if ldapost.phi[n][k] > 0:
                lhood_term += ldapost.doc.count[n] *  ldapost.phi[n][k] * (e_log_theta_k + ldapost.lda.topics[ldapost.doc.word[n]][k] - ldapost.log_phi[n][k])

        ldapost.lhood[k] = lhood_term
        lhood += lhood_term
        # lhood += influence_term

    return lhood


def fit_lda_seq_topics(ldaseq, topic_suffstats):
    """
    Fit lda sequence topic wise.
    """
    lhood = 0
    lhood_term = 0
    K = ldaseq.num_topics

    for k in range(0, K):
        print ("Fitting topic number" , k)
        lhood_term = fit_sslm(ldaseq.topic_chains[k], topic_suffstats[k])
        lhood += lhood_term

    return lhood

def fit_sslm(sslm, counts):

    """
    Fit variational distribution.
    """

    W = sslm.vocab_len
    bound = 0
    old_bound = 0
    sslm_fit_threshold = 1e-6
    sslm_max_iter = 2
    converged = sslm_fit_threshold + 1

    totals = numpy.zeros(counts.shape[1])

    for w in range(0, W):
        sslm.variance, sslm.fwd_variance = compute_post_variance(w, sslm, sslm.chain_variance)
    
    # column sum of counts
    totals = counts.sum(axis=0)
    iter_ = 0

    model = "DTM"
    if model == "DTM":
        bound = compute_bound(counts, totals, sslm)
    if model == "DIM":
        bound = compute_bound_fixed(counts, totals, sslm)

    print ("initial sslm bound is " , bound)

    while converged > sslm_fit_threshold and iter_ < sslm_max_iter:
        iter_ += 1
        old_bound = bound
        sslm.obs, sslm.zeta = update_obs(counts, totals, sslm)


        if model == "DTM":
            bound = compute_bound(counts, totals, sslm)
        if model == "DIM":
            bound = compute_bound_fixed(counts, totals, sslm)

        converged = numpy.fabs((bound - old_bound) / old_bound)

        print (iter_, " iteration lda seq bound is ", bound, " convergence is", converged)

    sslm.e_log_prob = compute_expected_log_prob(sslm)

    return bound


def compute_bound(word_counts, totals, sslm):

    """
    Compute log probability bound. 
    Forumula is as described in appendix of DTM.
    """
    W = sslm.vocab_len
    T = sslm.num_time_slices

    term_1 = 0
    term_2 = 0
    term_3 = 0

    val = 0
    ent = 0

    chain_variance = sslm.chain_variance

    for w in range(0, W):
        sslm.mean, sslm.fwd_mean = compute_post_mean(w, sslm, chain_variance)

    sslm.zeta = update_zeta(sslm)

    for w in range(0, W):
        val += (sslm.variance[w][0] - sslm.variance[w][T]) / 2 * chain_variance

    print ("Computing bound, all times")

    for t in range(1, T + 1):
        term_1 = 0.0
        term_2 = 0.0
        ent = 0.0
        for w in range(0, W):

            m = sslm.mean[w][t]
            prev_m = sslm.mean[w][t - 1]

            v = sslm.variance[w][t]

            # w_phi_l is only used in Document Influence Model; the values are aleays zero in this case
            # w_phi_l = sslm.w_phi_l[w][t - 1]
            # exp_i = numpy.exp(-prev_m)
            # term_1 += (numpy.power(m - prev_m - (w_phi_l * exp_i), 2) / (2 * chain_variance)) - (v / chain_variance) - numpy.log(chain_variance)
            
            term_1 += (numpy.power(m - prev_m, 2) / (2 * chain_variance)) - (v / chain_variance) - numpy.log(chain_variance)
            term_2 += word_counts[w][t - 1] * m
            ent += numpy.log(v) / 2  # note the 2pi's cancel with term1 (see doc)

        term_3 =  -totals[t - 1] * numpy.log(sslm.zeta[t - 1])
        val += term_2 + term_3 + ent - term_1

    return val
    

def update_obs(word_counts, totals, sslm):

    """
    Fucntion to perform optimization
    """

    OBS_NORM_CUTOFF = 2
    STEP_SIZE = 0.01
    TOL = 1e-3


    W = sslm.vocab_len
    T = sslm.num_time_slices

    runs = 0
    mean_deriv_mtx = numpy.resize(numpy.zeros(T * (T + 1)), (T, T + 1))

    norm_cutoff_obs = None
    for w in range(0, W):
        w_counts = word_counts[w]
        counts_norm = 0
        # now we find L2 norm of w_counts
        for i in range(0, len(w_counts)):
            counts_norm += w_counts[i] * w_counts[i]

        counts_norm = numpy.sqrt(counts_norm)

        if counts_norm < OBS_NORM_CUTOFF and norm_cutoff_obs is not None:
            obs = sslm.obs[w]
            norm_cutoff_obs = numpy.copy(obs)
        else: 
            if counts_norm < OBS_NORM_CUTOFF:
                w_counts = numpy.zeros(len(w_counts))

            for t in range(0, T):
                mean_deriv = mean_deriv_mtx[t]
                mean_deriv = compute_mean_deriv(w, t, sslm, mean_deriv)
                mean_deriv_mtx[t] = mean_deriv

            deriv = numpy.zeros(T)
            args = sslm, w_counts, totals, mean_deriv_mtx, w, deriv
            obs = sslm.obs[w]
            model = "DTM"

            if model == "DTM":
                obs = optimize.fmin_cg(f=f_obs, fprime=df_obs, x0=obs, gtol=TOL, args=args, epsilon=STEP_SIZE, disp=0)
            if model == "DIM":
                pass
            runs += 1

            if counts_norm < OBS_NORM_CUTOFF:
                norm_cutoff_obs = obs

            sslm.obs[w] = obs

    sslm.zeta = update_zeta(sslm)
    
    return sslm.obs, sslm.zeta

 
def compute_mean_deriv(word, time, sslm, deriv):

    """
    Used in helping find the optimum function.
    computes derivative of E[\beta_{t,w}]/d obs_{s,w} for t = 1:T.
    put the result in deriv, allocated T+1 vector
    """

    T = sslm.num_time_slices
    fwd_variance = sslm.variance[word]

    deriv[0] = 0

    # forward pass
    for t in range(1, T + 1):
        if sslm.obs_variance > 0.0:
            w = sslm.obs_variance / (fwd_variance[t - 1] + sslm.chain_variance + sslm.obs_variance)
        else:
            w = 0.0

        val = w * deriv[t - 1]
        if time == t - 1:
            val += (1 - w)

        deriv[t]= val

    for t in range(T - 1, -1, -1):
        if sslm.chain_variance == 0.0:
            w = 0.0
        else:
            w = sslm.chain_variance / (fwd_variance[t] + sslm.chain_variance)
        deriv[t] = w * deriv[t] + (1 - w) * deriv[t + 1]

    return deriv

def f_obs(x, *args):

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
    sslm.mean, sslm.fwd_mean = compute_post_mean(word, sslm, sslm.chain_variance)

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
        term2 += word_counts[t - 1] * mean_t - totals[t - 1] * numpy.exp(mean_t + variance[t] / 2) / sslm.zeta[t - 1]

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


def compute_obs_deriv(word, word_counts, totals, sslm, mean_deriv_mtx, deriv):

    # flag
    init_mult = 1000

    T = sslm.num_time_slices

    mean = sslm.mean[word]
    variance = sslm.variance[word]

    # only used for DIM mode
    # w_phi_l = sslm.w_phi_l[word]
    # m_update_coeff = sslm.m_update_coeff[word]

    sslm.temp_vect = numpy.zeros(T)

    for u in range(0, T):
        sslm.temp_vect[u] = numpy.exp(mean[u + 1] + variance[u + 1] / 2)

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

            term2 += (word_counts[u - 1] - (totals[u - 1] * sslm.temp_vect[u - 1] / sslm.zeta[u - 1])) * dmean_u

            model = "DTM"
            if model == "DIM":
                # do some stuff
                pass

        if sslm.chain_variance:
            term1 = - (term1 / sslm.chain_variance)
            term1 = term1 - (mean[0] * mean_deriv[0]) / (init_mult * sslm.chain_variance)
        else:
            term1 = 0.0

        deriv[t] = term1 + term2 + term3 + term4
    
    return deriv

def df_obs(x, *args):

    sslm, word_counts, totals, mean_deriv_mtx, word, deriv = args

    sslm.obs[word] = x
    sslm.mean, sslm.fwd_mean = compute_post_mean(word, sslm, sslm.chain_variance)

    model = "DTM"
    if model == "DTM":
        deriv = compute_obs_deriv(word, word_counts, totals, sslm, mean_deriv_mtx, deriv)
    elif model == "DIM":
        deriv = compute_obs_deriv_fixed(p.word, p.word_counts, p.totals, p.sslm, p.mean_deriv_mtx, deriv)

    return numpy.negative(deriv)



# the following code replicates Blei's original LDA, ported to python. 
# idea is to let user initialise LDA sstats through this instead of gensim LDA if wanted.

def lda_sstats(seq_corpus, num_topics, num_terms, alpha):

    lda_model = mockLDA(num_topics=num_topics, num_terms=num_terms)
    lda_model.alpha = alpha # this will have shape equal to  number of topics
    lda_ss = initialize_ss_random(seq_corpus, num_topics)
    lda_m_step(lda_model, lda_ss, seq_corpus, num_topics)
    em_iter = 10
    lda_em(lda_model, lda_ss, seq_corpus, em_iter, num_topics)

    return lda_ss

def initialize_ss_random(seq_corpus, num_topics):

    N = seq_corpus.num_terms
    K = num_topics

    topic = numpy.array(numpy.split(numpy.zeros(N * K), N))

    for n in range(0, N):
        for k in range(0, K):
            topic[n][k] = numpy.random.random() + 0.5 / seq_corpus.num_docs + 4.0

    return topic

def lda_m_step(lda_model, lda_ss, seq_corpus, num_topics):

    K = num_topics
    W = seq_corpus.num_terms
    lhood = 0
    for k in range(0, K):

        ss_k = lda_ss[:,k]
        log_p = lda_model.topics[:,k]

        LDA_VAR_BAYES = True
        if LDA_VAR_BAYES is True:

            numpy.copyto(log_p, ss_k)
            log_p = log_p / sum(log_p)
            log_p = numpy.log(log_p)

        else:
            pass

        lda_model.topics[:,k] = log_p

    return lhood

def lda_em(lda_model, lda_ss, seq_corpus, max_iter, num_topics):

    LDA_EM_CONVERGED = 5e-5
    LDA_INFERENCE_CONVERGED = 1e-8

    iter_ = 0 
    lhood = lda_e_step(lda_model, seq_corpus, lda_ss, num_topics)
    old_lhood = 0
    converged = 0
    m_lhood = lda_m_step(lda_model, lda_ss, seq_corpus, num_topics)

    # do step starts
    iter_ += 1
    old_lhood = lhood
    e_lhood = lda_e_step(lda_model, seq_corpus, lda_ss, num_topics)
    m_lhood = lda_m_step(lda_model, lda_ss, seq_corpus, num_topics)
    lhood = e_lhood + m_lhood
    converged = (old_lhood - lhood) / old_lhood

    while (converged > LDA_EM_CONVERGED or iter_ <= 5) and iter_ < max_iter:

        iter_ += 1
        old_lhood = lhood
        e_lhood = lda_e_step(lda_model, seq_corpus, lda_ss, num_topics)
        m_lhood = lda_m_step(lda_model, lda_ss, seq_corpus, num_topics)
        lhood = e_lhood + m_lhood
        converged = (old_lhood - lhood) / old_lhood
        print converged

    return lhood



def lda_e_step(lda_model, seq_corpus, lda_ss, num_topics):

    K = num_topics

    if lda_ss is not None:
        lda_ss.fill(0)

    ldapost = LdaPost(max_doc_len=seq_corpus.max_doc_len, num_topics=K, lda=lda_model)
    ldapost.gamma = numpy.zeros(K)
    ldapost.lhood = numpy.zeros(K + 1)

    lhood = 0

    for line_no, line in enumerate(seq_corpus.corpus):

        doc_ = line

        nterms = len(doc_)
        words = []
        counts = []
        totals = 0
        for word_id, count in doc_:
            words.append(int(word_id))
            counts.append(int(count))
            totals += int(count)

        doc = Doc(word=words, count=counts, total=totals, nterms=int(nterms))
        ldapost.doc = doc
        lhood += fit_lda_post(d, 0, ldapost, None, None, None, None, None)

        if lda_ss is not None:
            for k in range(0, K):
                for n in range(0, ldapost.doc.nterms):
                    lda_ss[ldapost.doc.word[n]][k] += ldapost.phi[n][k] * ldapost.doc.count[n]

    return lhood


def print_topics(ldaseq, topic, time=0, top_terms=20):
    """
    Topic is the topic numner
    Time is for a particular time_slice
    top_terms is the number of terms to display
    """
    topic = ldaseq.topic_chains[topic].e_log_prob[time]
    topic = numpy.transpose(topic)
    topic = topic / topic.sum()
    bestn = matutils.argsort(topic, top_terms, reverse=True)
    beststr = [(round(topic[id_], 3), ldaseq.corpus.id2word[id_]) for id_ in bestn]

    return beststr




# the fdf used in optimising obs. Can use if we figure a way to use an optimization function which requires this

# def fdf_obs(x, params, f, df):
#     p = params
#     model = "DTM"
#     if model == "DTM":
#         f = f_obs(x, params)
#         compute_obs_deriv(p.word, p.word_counts, p.totals, p.sslm, p.mean_deriv_mtx, df)
#     elif model == "DIM":
#         f = f_obs_multiplt(x, params)
#         compute_obs_deriv_fixed(p.word, p.word_counts, p.totals, p.sslm, p.mean_deriv_mtx, df)
#     for i in range(0, len(df)):
#         df[i] = - df[i]


