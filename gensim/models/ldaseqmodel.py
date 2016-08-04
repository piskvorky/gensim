#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Based on Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>


"""

This is the class which is used to help with Dynamic Topic Modelling of a corpus.
It is a work in progress and will change largely throughout the course of development.
Inspired by the Blei's original DTM code and paper. TODO: add links

As of now, the LdaSeqModel and SSLM classes mimic the structures of the same name in the Blei DTM code.
Few mathematical helper functions will be made and tested.

"""

from gensim import interfaces, utils, matutils
from gensim.models import ldamodel
import numpy
import math
from scipy.special import digamma
from scipy import optimize
import sys


class seq_corpus(utils.SaveLoad):

    """
    seq_corpus is basically a wrapper class which contains information about the corpus.
    num_terms is the length of the vocabulary.
    max_nterms is the maximum number of terms a single document has.
    num_sequences is the number of sequences, i.e number of time-slices.
    num_docs is the number of documents present.
    time_slice is a list or numpy array which the user must provide which contains the number of documents in each time-slice.
    corpus is any iterable gensim corpus.

    """
    def __init__(self, num_terms=None, max_nterms=None, num_sequences=None, num_docs=None, corpus=None, time_slice=None, id2word=None):


        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0

        self.corpus = corpus
        if self.corpus is not None:
            self.num_docs = len(corpus)

        self.time_slice = time_slice
        if self.time_slice is not None:
            self.num_sequences = len(time_slice)

        # need to still figure out way to get max_nterms    
        self.max_nterms = max_nterms


class Doc(utils.SaveLoad):
    """
    The doc class contains information used for each document. 

    """
    def __init__(self, nterms=None, word=None, count=None, total=None):

        self.nterms = nterms
        self.word = word
        self.count = count
        self.total = total


class LdaSeqModel(utils.SaveLoad):
    """
    Class which contains information of our whole DTM model.
    Topic chains contains for each topic a 'state space language model' object which in turn has information about each topic

    """

    def __init__(self, corpus=None, num_topics=10, id2word=None, num_sequences=None, num_terms=None, alphas=None):
        

        self.corpus = corpus
        self.num_topics = num_topics
        self.num_sequences = num_sequences
        self.num_terms = num_terms
        self.topic_chains = topic_chains
        if self.topic_chains is None:
            for topic in range(0, num_topics):
                sslm_ = sslm(num_sequences=num_sequences, num_terms=num_terms, num_topics=num_topics)
                topic_chains.append(sslm_)

        # the following are class variables which are to be integrated during Document Influence Model
        self.top_doc_phis = None
        self.influence = None
        self.renormalized_influence = None
        self.influence_sum_lgl = None

class sslm(utils.SaveLoad):
    """
    obs values contain the doc - topic ratios
    e_log_prob contains topic - word ratios
    mean, fwd_mean contains the mean values to be used for inference for each word for a time_slice
    variance, fwd_variance contains the variance values to be used for inference for each word in a time_slice

    """
    def __init__(self, num_terms=None, num_sequences=None, obs_variance=0.5, chain_variance=0.005):


        self.num_terms = num_terms
        self.num_sequences = num_sequences
        self.obs_variance = obs_variance
        self.chain_variance= chain_variance

        self.obs = numpy.array(numpy.split(numpy.zeros(num_sequences * num_terms), num_terms))
        self.e_log_prob = numpy.array(numpy.split(numpy.zeros(num_sequences * num_terms), num_terms))
        self.mean = numpy.array(numpy.split(numpy.zeros((num_sequences + 1) * num_terms), num_terms))
        self.fwd_mean = numpy.array(numpy.split(numpy.zeros((num_sequences + 1) * num_terms), num_terms))
        self.fwd_variance = numpy.array(numpy.split(numpy.zeros((num_sequences + 1) * num_terms), num_terms))
        self.variance = numpy.array(numpy.split(numpy.zeros((num_sequences + 1) * num_terms), num_terms))
        self.zeta = numpy.zeros(num_sequences)

        # the following are class variables which are to be integrated during Document Influence Model
        self.m_update_coeff = None
        self.mean_t = None
        self.variance_t = None
        self.influence_sum_lgl = None
        self.w_phi_l = None
        self.w_phi_sum = None
        self.w_phi_l_sq = None
        self.m_update_coeff_g = None

        # temp_vect 
        self.temp_vect = None

class Lda_Post(utils.SaveLoad):
    """
    Posterior values associated with each set of documents document

    """

    def __init__(self, doc=None, lda=None, max_nterms=None, num_topics=None, gamma=None, lhood=None):

        self.doc = doc
        self.lda = lda
        self.gamma = gamma
        self.lhood = lhood

        if max_nterms is not None and num_topics is not None:
            self.phi = numpy.resize(numpy.zeros(max_nterms * num_topics), (max_nterms, num_topics))
            self.log_phi = numpy.resize(numpy.zeros(max_nterms * num_topics), (max_nterms, num_topics))

        # the following are class variables which are to be integrated during Document Influence Model

        self.doc_weight = None
        self.renormalized_doc_weight = None

def update_zeta(sslm):

    num_terms = sslm.num_terms
    num_sequences = sslm.num_sequences
    # making zero and updating
    sslm.zeta.fill(0)
    for i in range(0, num_terms):
        for j in range(0, num_sequences):
            m = sslm.mean[i][j + 1]
            v = sslm.variance[i][j + 1]
            val = numpy.exp(m + v/2)
            sslm.zeta[j] = sslm.zeta[j] + val  
    return

def compute_post_variance(word, sslm, chain_variance):

    T = sslm.num_sequences
    variance = sslm.variance[word] # pick wordth row
    fwd_variance = sslm.fwd_variance[word] # pick wordth row

    # forward pass. Set initial variance very high
    fwd_variance[0] = chain_variance * 1000
    
    for t in range(1, T + 1):
        if sslm.obs_variance:
            w = sslm.obs_variance / (fwd_variance[t - 1] + chain_variance + sslm.obs_variance)
        else:
            w = 0
        fwd_variance[t] = w * (fwd_variance[t-1] + chain_variance)

    # backward pass 
    variance[T] = fwd_variance[T]
    for t in range(T - 1, -1, -1):
        if fwd_variance[t] > 0.0:
            w = numpy.power((fwd_variance[t] / (fwd_variance[t] + chain_variance)), 2)
        else:
            w = 0
        variance[t] = (w * (variance[t + 1] - chain_variance)) + ((1 - w) * fwd_variance[t])

    sslm.variance[word] = variance
    sslm.fwd_variance[word] = fwd_variance
    return
    

def compute_post_mean(word, sslm, chain_variance):

    T = sslm.num_sequences
    obs = sslm.obs[word] # wordth row
    mean = sslm.mean[word]
    fwd_mean = sslm.fwd_mean[word]
    fwd_variance = sslm.fwd_variance[word]

    # forward 
    fwd_mean[0] = 0
    for t in range(1, T + 1):
        w = sslm.obs_variance / (fwd_variance[t - 1] + chain_variance + sslm.obs_variance)
        fwd_mean[t] = w * fwd_mean[t - 1] + (1 - w) * obs[t - 1]
        if fwd_mean[t] is None:
            # error message
            pass

    # backward pass
    mean[T] = fwd_mean[T]
    for t in range(T - 1, -1, -1):
        if chain_variance == 0.0:
            w = 0.0
        else:
            w = chain_variance / (fwd_variance[t] + chain_variance)
        mean[t] = w * fwd_mean[t] + (1 - w) * mean[t + 1]
        if mean[t] is None:
            # error message
            pass
 
    return

def compute_expected_log_prob(sslm):

    W = sslm.num_terms
    T = sslm.num_sequences
    for t in range(0, T):
        for w in range(0, W):
            sslm.e_log_prob[w][t] = sslm.mean[w][t + 1] - numpy.log(sslm.zeta[t])
    return


def sslm_counts_init(sslm, obs_variance, chain_variance, sstats):

    W = sslm.num_terms
    T = sslm.num_sequences

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
       compute_post_variance(w, sslm, sslm.chain_variance)

    for w in range(0, W):
       compute_post_mean(w, sslm, sslm.chain_variance)

    update_zeta(sslm)
    compute_expected_log_prob(sslm)

def init_ldaseq_ss(ldaseq, topic_chain_variance, topic_obs_variance, alpha, init_suffstats):

    ldaseq.alphas = alpha
    for k in range(0, ldaseq.num_topics):

        sstats = init_suffstats[:,k]
        sslm_counts_init(ldaseq.topic_chains[k], topic_obs_variance, topic_chain_variance, sstats)

        # initialize the below matrices only if running DIM
        # ldaseq.topic_chains[k].w_phi_l = numpy.zeros((ldaseq.num_terms, ldaseq.num_sequences))
        # ldaseq.topic_chains[k].w_phi_sum = numpy.zeros((ldaseq.num_terms, ldaseq.num_sequences))
        # ldaseq.topic_chains[k].w_phi_sq = numpy.zeros((ldaseq.num_terms, ldaseq.num_sequences))

def fit_lda_seq(ldaseq, seq_corpus):

    K = ldaseq.num_topics
    W = ldaseq.num_terms
    data_len = seq_corpus.num_sequences
    num_docs = seq_corpus.num_docs
    
    LDA_INFERENCE_MAX_ITER = 25

    bound = 0
    heldout_bound = 0
    ldasqe_em_threshold = 1e-4
    convergence = ldasqe_em_threshold + 1

    iter_ = 0
    final_iters_flag = 0 
    last_iter = 0

    # this is a flag/input do something about it
    lda_seq_min_iter = 6
    lda_seq_max_iter = 20
    
    while iter_ < lda_seq_min_iter or ((final_iters_flag is 0 or convergence > ldasqe_em_threshold) and iter_ <= lda_seq_max_iter):
        if not (iter_ < lda_seq_min_iter or ((final_iters_flag is 0 or convergence > ldasqe_em_threshold) and iter_ <= lda_seq_max_iter)):
            last_iter = 1

        # log
        print (" EM iter " , iter_)
        print ("E Step")

        # writing to file
        em_log.write(str(bound) + "\t" + str(convergence))
        old_bound = bound

        # initiate sufficient statistics
        topic_suffstats = []
        for k in range(0, K):
            topic_suffstats.append(numpy.resize(numpy.zeros(W * data_len), (W, data_len)))

        # set up variables
        gammas = numpy.resize(numpy.zeros(num_docs * K), (num_docs, K))
        lhoods = numpy.resize(numpy.zeros(num_docs * K + 1), (num_docs, K + 1))

        bound = lda_seq_infer(ldaseq, seq_corpus, topic_suffstats, gammas, lhoods, iter_, last_iter)

        print ("M Step")

        topic_bound = fit_lda_seq_topics(ldaseq, topic_suffstats)
        bound += topic_bound


        if ((bound - old_bound) < 0):
            if (LDA_INFERENCE_MAX_ITER == 1):
                LDA_INFERENCE_MAX_ITER = 2
            if (LDA_INFERENCE_MAX_ITER == 2):
                LDA_INFERENCE_MAX_ITER = 5
            if (LDA_INFERENCE_MAX_ITER == 5):
                LDA_INFERENCE_MAX_ITER = 10
            if (LDA_INFERENCE_MAX_ITER == 10):
                LDA_INFERENCE_MAX_ITER = 20
            print ("Bound went down, increasing it to" , LDA_INFERENCE_MAX_ITER)

        # check for convergence
        convergence = numpy.fabs((bound - old_bound) / old_bound)

        if convergence < ldasqe_em_threshold:
            final_iters_flag = 1
            LDA_INFERENCE_MAX_ITER = 500
            print ("Starting final iterations, max iter is", LDA_INFERENCE_MAX_ITER)
            convergence = 1.0

        print (iter_, "iteration lda seq bound is", bound, ", convergence is ", convergence)

        iter_ += 1

    return bound


def lda_seq_infer(ldaseq, seq_corpus, topic_suffstats, gammas, lhoods, iter_, last_iter):

    K = ldaseq.num_topics
    W = ldaseq.num_terms
    bound = 0.0
    
    lda = ldamodel.LdaModel(num_topics=K, alpha=ldaseq.alphas, id2word=seq_corpus.id2word)
    lda_post = Lda_Post(max_nterms=seq_corpus.max_nterms, num_topics=K, lda=lda)


    model = "DTM"
    if model == "DTM":
        inferDTMseq(K, ldaseq, seq_corpus, topic_suffstats, gammas, lhoods, iter_, last_iter, lda, lda_post, bound)
    elif model == "DIM":
        InfluenceTotalFixed(ldaseq, seq_corpus);
        inferDIMseq(K, ldaseq, seq_corpus, topic_suffstats, gammas, lhoods, iter_, last_iter, lda, lda_post, bound)

    return bound

def inferDTMseq(K, ldaseq, seq_corpus, topic_suffstats, gammas, lhoods, iter_, last_iter, lda, lda_post, bound):

    def cumsum(it):
        total = 0
        for x in it:
            total += x
            yield total

    doc_index = 0
    t = 0
    d = 0
    make_lda_seq_slice(lda, ldaseq, t)  


    time_slice = list(cumsum(ldaseq.time_slice))

    for line_no, line in enumerate(seq_corpus.corpus):
        if doc_index > time_slice[t]:
            t += 1
            make_lda_seq_slice(lda, ldaseq, t)    
            d = 0

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
        lda_post.gamma = gam
        lda_post.lhood = lhood
        lda_post.doc = doc

        if iter_ == 0:
            doc_lhood = fit_lda_post(d, t, lda_post, None, None, None, None, None)
        else:
            doc_lhood = fit_lda_post(d, t, lda_post, ldaseq, None, None, None, None)
       

        if topic_suffstats != None:
            update_lda_seq_ss(t, doc, lda_post, topic_suffstats)

        bound += doc_lhood
        doc_index += 1
        d += 1

    return

def fit_lda_post(doc_number, time, lda_post, ldaseq, g, g3_matrix, g4_matrix, g5_matrix):


    init_lda_post(lda_post)


    model = "DTM"
    if model == "DIM":
        # if in DIM then we initialise some variables here
        pass

    lhood = compute_lda_lhood(lda_post)
    lhood_old = 0
    converged = 0
    iter_ = 0
    LDA_INFERENCE_CONVERGED = 1e-8
    LDA_INFERENCE_MAX_ITER = 25


    iter_ += 1
    lhood_old = lhood
    update_gamma(lda_post)

    model = "DTM"

    if model == "DTM" or sslm is None:
        update_phi(doc_number, time, lda_post, sslm, g)
    elif model == "DIM" and sslm is not None:
        update_phi_fixed(doc_number, time, lda_post, sslm, g3_matrix, g4_matrix, g5_matrix)

    lhood = compute_lda_lhood(lda_post)
    converged = numpy.fabs((lhood_old - lhood) / (lhood_old * lda_post.doc.total))


    while converged > LDA_INFERENCE_CONVERGED and iter_ <= LDA_INFERENCE_MAX_ITER:

        iter_ += 1
        lhood_old = lhood
        update_gamma(lda_post)
        model = "DTM"

        if model == "DTM" or sslm is None:
            update_phi(doc_number, time, lda_post, sslm, g)
        elif model == "DIM" and sslm is not None:
            update_phi_fixed(doc_number, time, lda_post, sslm, g3_matrix, g4_matrix, g5_matrix)

        lhood = compute_lda_lhood(lda_post)
        converged = numpy.fabs((lhood_old - lhood) / (lhood_old * lda_post.doc.total))

    return lhood


def make_lda_seq_slice(lda, ldaseq, time):

    K = ldaseq.num_topics

    for k in range(0, K):
        lda.topics[:,k] = numpy.copy(ldaseq.topic_chains[k].e_log_prob[:,time])

    lda.alpha = numpy.copy(ldaseq.alphas)

    return

def update_lda_seq_ss(time, doc, lda_post, topic_suffstats):

    K = numpy.shape(lda_post.phi)[1]
    N = doc.nterms

    for k in range(0, K):
        topic_ss = topic_suffstats[k]
        for n in range(0, N):
            w = doc.word[n]
            c = doc.count[n]
            topic_ss[w][time] = topic_ss[w][time] + c * lda_post.phi[n][k]

        topic_suffstats[k] = topic_ss

    return

def init_lda_post(lda_post):

    K = lda_post.lda.num_topics
    N = lda_post.doc.nterms
    for k in range(0, K):
        lda_post.gamma[k] = lda_post.lda.alpha[k] + float(lda_post.doc.total) / K
        for n in range(0, N):
            lda_post.phi[n][k] = 1.0 / K
   
    # doc_weight used during DIM
    # lda_post.doc_weight = None

    return

def compute_lda_lhood(lda_post):
    

    K = lda_post.lda.num_topics
    N = lda_post.doc.nterms
    gamma_sum = numpy.sum(lda_post.gamma)

    # figure out how to do flags
    FLAGS_sigma_l = 0
    FLAGS_sigma_d = 0 

    lhood = math.lgamma(numpy.sum(lda_post.lda.alpha)) - math.lgamma(gamma_sum)
    lda_post.lhood[K] = lhood

    # influence_term = 0
    digsum = digamma(gamma_sum)

    model = "DTM"
    for k in range(0, K):
        # below code only to be used in DIM mode
        # if lda_post.doc_weight is not None and (model == "DIM" or model == "fixed"):
        #     influence_topic = lda_post.doc_weight[k]
        #     influence_term = - ((influence_topic * influence_topic + FLAGS_sigma_l * FLAGS_sigma_l) / 2.0 / (FLAGS_sigma_d * FLAGS_sigma_d))
           

        e_log_theta_k = digamma(lda_post.gamma[k]) - digsum
        lhood_term = (lda_post.lda.alpha[k] - lda_post.gamma[k]) * e_log_theta_k + math.lgamma(lda_post.gamma[k]) - math.lgamma(lda_post.lda.alpha[k])

        for n in range(0, N):
            if lda_post.phi[n][k] > 0:
                lhood_term += lda_post.doc.count[n] *  lda_post.phi[n][k] * (e_log_theta_k + lda_post.lda.topics[lda_post.doc.word[n]][k] - lda_post.log_phi[n][k])

        lda_post.lhood[k] = lhood_term
        lhood += lhood_term
        lhood += influence_term

    return lhood

# update variational multinomial parameters
def update_phi(doc, time, lda_post, ldaseq, g):

    K = lda_post.lda.num_topics
    N = lda_post.doc.nterms

    dig = numpy.zeros(K)

    for k in range(0, K):
        dig[k] = digamma(lda_post.gamma[k])

    for n in range(0, N):
        w = lda_post.doc.word[n]
        for k in range(0, K):
            lda_post.log_phi[n][k] = dig[k] + lda_post.lda.topics[w][k]

        log_phi_row = lda_post.log_phi[n]
        phi_row = lda_post.phi[n]

        # log normalize
        v = log_phi_row[0]
        for i in range(1, len(log_phi_row)):
            v = numpy.logaddexp(v, log_phi_row[i])

        for i in range(0, len(log_phi_row)):
            log_phi_row[i] = log_phi_row[i] - v

        for k in range(0, K):
            phi_row[k] = numpy.exp(log_phi_row[k])

        lda_post.log_phi[n] = log_phi_row
        lda_post.phi[n] = phi_row

    return

# update variational dirichlet parameters
def update_gamma(lda_post):

    K = lda_post.lda.num_topics
    N = lda_post.doc.nterms

    lda_post.gamma = numpy.copy(lda_post.lda.alpha)

    for n in range(0, N):
        phi_row = lda_post.phi[n]
        count = lda_post.doc.count[n]

        for k in range(0, K):
            lda_post.gamma[k] += phi_row[k] * count

    return

def fit_lda_seq_topics(ldaseq, topic_suffstats):
    lhood = 0
    lhood_term = 0
    K = ldaseq.num_topics

    for k in range(0, K):
        print ("Fitting topic number" , k)
        lhood_term = fit_sslm(ldaseq.topic_chains[k], topic_suffstats[k])
        lhood += lhood_term

    return lhood

def fit_sslm(sslm, counts):

    W = sslm.num_terms
    bound = 0
    old_bound = 0
    sslm_fit_threshold = 1e-6
    sslm_max_iter = 2
    converged = sslm_fit_threshold + 1

    totals = numpy.zeros(counts.shape[1])

    for w in range(0, W):
        compute_post_variance(w, sslm, sslm.chain_variance)

    totals = col_sum(counts, totals)
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
        update_obs(counts, totals, sslm)


        if model == "DTM":
            bound = compute_bound(counts, totals, sslm)
        if model == "DIM":
            bound = compute_bound_fixed(counts, totals, sslm)

        converged = numpy.fabs((bound - old_bound) / old_bound)

        print (iter_, " iteration lda seq bound is ", bound, " convergence is", converged)

    compute_expected_log_prob(sslm)

    return bound


def col_sum(matrix, vector):

    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            vector[j] = vector[j] + matrix[i][j]

    return vector

def compute_bound(word_counts, totals, sslm):

    W = sslm.num_terms
    T = sslm.num_sequences

    term_1 = 0
    term_2 = 0
    term_3 = 0

    val = 0
    ent = 0

    chain_variance = sslm.chain_variance

    for w in range(0, W):
        compute_post_mean(w, sslm, chain_variance)

    update_zeta(sslm)

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
    
# fucntion to perform optimization
def update_obs(word_counts, totals, sslm):


    OBS_NORM_CUTOFF = 2

    W = sslm.num_terms
    T = sslm.num_sequences

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
                compute_mean_deriv(w, t, sslm, mean_deriv)
                mean_deriv_mtx[t] = mean_deriv

            deriv = numpy.zeros(T)
            args = sslm, w_counts, totals, mean_deriv_mtx, w, deriv
            obs = sslm.obs[w]
            step_size = 0.01
            tol = 1e-3
            model = "DTM"

            if model == "DTM":
                obs = optimize.fmin_cg(f=f_obs, fprime=df_obs, x0=obs, gtol=tol, args=args, epsilon=step_size, disp=0)
            if model == "DIM":
                pass
            runs += 1

            if counts_norm < OBS_NORM_CUTOFF:
                norm_cutoff_obs = obs

            sslm.obs[w] = obs

    update_zeta(sslm)
    return


 # compute d E[\beta_{t,w}]/d obs_{s,w} for t = 1:T.
 # put the result in deriv, allocated T+1 vector
 
def compute_mean_deriv(word, time, sslm, deriv):

    T = sslm.num_sequences
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

    return

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
    compute_post_mean(word, sslm, sslm.chain_variance)

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

    T = sslm.num_sequences

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
    
    return

def df_obs(x, *args):

    sslm, word_counts, totals, mean_deriv_mtx, word, deriv = args

    sslm.obs[word] = x
    compute_post_mean(word, sslm, sslm.chain_variance)

    model = "DTM"
    if model == "DTM":
        compute_obs_deriv(word, word_counts, totals, sslm, mean_deriv_mtx, deriv)
    elif model == "DIM":
        compute_obs_deriv_fixed(p.word, p.word_counts, p.totals, p.sslm, p.mean_deriv_mtx, deriv)

    return numpy.negative(deriv)



# the following code replicates Blei's original LDA, ported to python. 
# idea is to let user initialise LDA sstats through this instead of gensim LDA if wanted.

def lda_sstats(seq_corpus, num_topics, num_terms, alpha):

    lda_model = ldamodel.LdaModel(num_topics=num_topics, id2word=seq_corpus.id2word)
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

            lop_p = numpy.copy(ss_k)
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

    return lhood


def lda_e_step(lda_model, seq_corpus, lda_ss, num_topics):

    K = num_topics

    if lda_ss is not None:
        lda_ss.fill(0)

    lda_post = Lda_Post(max_nterms=seq_corpus.max_nterms, num_topics=K, lda=lda_model)
    lda_post.gamma = numpy.zeros(K)
    lda_post.lhood = numpy.zeros(K + 1)

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
        lda_post.doc = doc
        lhood += fit_lda_post(d, 0, lda_post, None, None, None, None, None)

        if lda_ss is not None:
            for k in range(0, K):
                for n in range(0, lda_post.doc.nterms):
                    lda_ss[lda_post.doc.word[n]][k] += lda_post.phi[n][k] * lda_post.doc.count[n]

    return lhood

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


