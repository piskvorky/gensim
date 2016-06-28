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

class seq_corpus(utils.SaveLoad):
    def __init__(self, num_terms=0, max_nterms=0, length=0, num_doc=0, corpuses=0):
        self.num_terms = num_terms
        self.max_nterms = max_nterms
        self.length = length
        self.num_docs = num_docs

        # list of corpus class objects
        self.corpuses = corpuses

class LdaSeqModel(utils.SaveLoad):
    def __init__(self, corpus=None, num_topics=10, id2word=None, num_sequence=None, num_terms=None, alphas=None, top_doc_phis=None,
     topic_chains=None, influence=None, influence_sum_lgl=None, renormalized_influence=None):
        # store user-supplied parameters
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

        if self.num_terms == 0:
            raise ValueError("cannot compute DTM over an empty collection (no terms)")

        self.num_topics = num_topics
        self.num_sequence = num_sequence
        self.alphas = alphas
        self.topic_chains = []
        for topic in range(0, num_topics):
            topic_chains.append(sslm)

        self.top_doc_phis = top_doc_phis

        # influence values as of now not using
        self.influence = influence
        self.renormalized_influence = renormalized_influence
        self.influence_sum_lgl = influence_sum_lgl

class sslm(utils.SaveLoad):
  def __init__(self, num_terms=None, num_sequence=None, obs=None, obs_variance=0.5, chain_variance=0.005, fwd_variance=None,
               mean=None, variance=None,  zeta=None, e_log_prob=None, fwd_mean=None, m_update_coeff=None,
               mean_t=None, variance_t=None, influence_sum_lgl=None, w_phi_l=None, w_phi_sum=None, w_phi_l_sq=None,  m_update_coeff_g=None):

        self.obs = obs
        self.zeta = zeta # array equal to number of sequences
        self.mean = mean # matrix of dimensions num_terms * (num_of sequences + 1)
        self.variance = variance # matrix of dimensions num_terms * (num_of sequences + 1)
        self.num_terms = num_terms
        self.num_sequence = num_sequence
        self.obs_variance = obs_variance
        self.chain_variance= chain_variance
        self.fwd_variance = fwd_variance
        self.fwd_mean = fwd_mean
        self.e_log_prob = e_log_prob
        self.m_update_coeff = m_update_coeff
        self.mean_t = mean_t
        self.variance_t = variance_t
        self.influence_sum_lgl = influence_sum_lgl
        self.w_phi_l = w_phi_l
        self.w_phi_sum = w_phi_sum
        self.w_phi_l_sq = w_phi_l_sq
        self.m_update_coeff_g = m_update_coeff_g

class lda_post(utils.SaveLoad):
    def __init__(self, doc=None, lda=None, phi=None, log_phi=None, gamma=None, lhood=None, doc_weight=None, renormalized_doc_weight=None):
        return

class opt_params(utils.SaveLoad):
    def __init__(sslm, word_counts, totals, mean_deriv_mtx, word):
        return

def update_zeta(sslm):
    # setting limits 
    num_terms = sslm.obs.shape[0] # this is word length (our example, 562)
    num_sequence = sslm.obs.shape[1] # this is number of sequeces
    # making zero and updating
    sslm.zeta.fill(0)
    for i in range(0, num_terms):
        for j in range(0, num_sequence):
            m = sslm.mean[i][j + 1]
            v = sslm.variance[i][j + 1]
            val = numpy.exp(m + v/2)
            sslm.zeta[j] = sslm.zeta[j] + val  
    return

def compute_post_variance(word , sslm, chain_variance):
    T = sslm.num_sequence
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

    T = sslm.num_sequence
    obs = sslm.obs[word] # wordth row
    mean = sslm.mean[word]
    fwd_mean = sslm.fwd_mean[word]
    fwd_variance = sslm.fwd_variance[word]

    # forward 
    fwd_mean[0] = 0
    for t in range(1, T + 1):
            # assert(fabs(vget(&fwd_variance, t-1) +
            # chain_variance + var->obs_variance) > 0.0);
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

    sslm.mean[word] = mean
    sslm.fwd_mean[word] = fwd_mean    
    return

def compute_expected_log_prob(sslm):

    W = sslm.num_terms
    T = sslm.num_sequence
    for t in range(0, T):
        for w in range(0, W):
            sslm.e_log_prob[w][t] = sslm.mean[w][t + 1] - numpy.log(sslm.zeta[t])
    return


def sslm_counts_init(sslm, obs_variance, chain_variance, sstats):

    W = sslm.num_terms
    T = sslm.num_sequence

    log_norm_counts = sstats
    log_norm_counts = log_norm_counts / sum(log_norm_counts)

    log_norm_counts = log_norm_counts + 1.0/W
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

def init_ldaseq_ss(ldaseq, lda, alpha, topic_chain_variance, topic_obs_variance):
    ldaseq.alpha = alpha
    for k in range(0, ldaseq.num_topics):
        sstats = lda.state.sstats[k]
        sslm_counts_init(ldaseq.topic_chains[k], topic_obs_variance, topic_chain_variance, sstats)

        # dont't need to initialize here, but writing for reference
        ldaseq.topic_chains[k].w_phi_l = numpy.zeros((ldaseq.num_terms, ldaseq.num_sequence))
        ldaseq.topic_chains[k].w_phi_sum = numpy.zeros((ldaseq.num_terms, ldaseq.num_sequence))
        ldaseq.topic_chains[k].w_phi_sq = numpy.zeros((ldaseq.num_terms, ldaseq.num_sequence))

def fit_lda_seq(ldaseq, seq_corpus):
    K = ldaseq.num_topics
    W = ldaseq.num_terms
    data_len = seq_corpus.length
    no_docs = seq_corpus.no_docs
    
    # heldout_gammas = NULL
    # heldout_llhood = NULL

    bound = 0
    heldout_bound = 0
    ldasqe_em_threshold = 1e-4
    convergence = ldasqe_em_threshold + 1

    # make directory
    em_log = open("em_log.dat", "w")
    gammas_file = open("gammas.dat", "w")
    lhoods_file = open("lhoods.dat", "w")

    iter_ = 0
    final_iters_flag = 0 
    last_iter = 0

    # this is a flag/input do something about it
    lda_seq_min_iter = 0
    lda_seq_max_iter = 0
    
    while iter_ < lda_seq_min_iter or ((final_iters_flag is 0 or convergence > ldasqe_em_threshold) and iter_ <= lda_seq_max_iter):
        if not (iter_ < lda_sequence_min_iter or ((final_iters_flag is 0 or convergence > ldasqe_em_threshold) and iter_ <= lda_seq_max_iter)):
            last_iter = 1

        # log
        print (" EM iter " , iter_)
        print ("E Step")

        # writing to file
        em_log.write(str(bound) + "\t" + str(convergence))
        old_bound = bound

        # initiate sufficient statistics
        topic_suffstats = numpy.zeros(K)
        for k in range(0, K):
            topic_suffstats[k] = numpy.resize(numpy.zeros(W * data_len), (W, data_len))

        # set up variables
        gammas = numpy.resize(numpy.zeros(no_docs * K), (no_docs, K))
        lhoods = numpy.resize(numpy.zeros(no_docs * K + 1), (no_docs, K + 1))

        bound = lda_seq_infer(ldaseq, seq_corpus, topic_suffstats, gammas, lhoods, iter_, last_iter)

        # figure out how to write to file here
        # TODO save to file for command line
        gammas_file.write(gammas)
        lhoods_file.write(lhoods)

        print ("M Step")

        topic_bound = fit_lda_seq_topics(ldaseq, topic_suffstats)
        bound += topic_bound

        write_lda_seq(ldaseq)

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

        print ("%d lda seq bound is = %d, convergence is %d", iter_, bound, convergence)

        iter_ += 1

    return bound


def lda_seq_infer(ldaseq, seq_corpus, topic_suffstats, gammas, lhoods, iter_, last_iter):

    K = ldaseq.num_topics
    W = ldaseq.num_terms
    bound = 0.0
    
    lda = ldamodel.LdaModel(num_topics=K)
    lda_post.phi = numpy.resize(numpy.zeros(seq_corpus.max_nterms * K), (seq_corpus.max_nterms, K))
    lda_post.log_phi = numpy.resize(numpy.zeros(seq_corpus.max_nterms * K), (seq_corpus.max_nterms, K))
    lda_post.model = lda

    model = "DTM"
    if model == "DTM":
        inferDTMseq(K, ldaseq, seq_corpus, topic_suffstats, gammas, lhoods, iter_, last_iter, lda, lda_post, bound)
    elif model == "DIM":
        InfluenceTotalFixed(ldaseq, seq_corpus);
        inferDIMseq(K, ldaseq, seq_corpus, topic_suffstats, gammas, lhoods, iter_, last_iter, lda, lda_post, bound)

    return bound

def inferDTMseq(K, ldaseq, seq_corpus, topic_suffstats, gammas, lhoods, iter_, last_iter, lda, lda_post, bound):

    doc_index = 0
    for t in range(0, seq_corpus.length):
        make_lda_seq_slice(lda, ldaseq, t)
        # what to do here
        ndocs = seq_corpus.corpuses[t].ndocs
        for d in range(0, ndocs):
            gam = gammas[doc_index]
            lhood = lhoods[doc_index]
            lda_post.gamma = gam
            lda_post.lhood = lhood
            lda_post.doc = seq_corpus.corpuses[t].doc[d]
            if iter_ == 0:
                doc_lhood = fit_lda_post(d, t, post, None, None, None, None, None)
            else:
                doc_lhood = fit_lda_post(d, t, post, model, None, None, None, None)
            if topic_suffstats != None:
                update_lda_seq_ss(t, seq_corpus.corpuses[t].doc[d], lda_post, topic_suffstats)
            bound += doc_lhood
            doc_index += 1
    return

def fit_lda_post():
    return

def make_lda_seq_slice():
    return

def update_lda_seq_ss():
    return


def fit_lda_seq_topics(ldaseq, topic_suffstats):
    lhood = 0
    lhood_term = 0
    K = ldaseq.num_topics

    for k in range(0, K):
        print ("Fitting topic number" , k)
        lhood_term = fit_sslm(ldaseq.topic_chains[k], topic_suffstats[k])
        lhood +=lhood_term

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
        update_obs(counts, totals, sslm_max_iter)

        if model == "DTM":
            bound = compute_bound(counts, totals, sslm)
        if model == "DIM":
            bound = compute_bound_fixed(counts, totals, sslm)

        converged = numpy.fabs((bound - old_bound) / old_bound)

        print ("%d lda seq bound is = %d, convergence is %d", iter_, bound, converged)

    compute_expected_log_prob(sslm)

    return bound


def col_sum(matrix, vector):
    for i in range(0, matrix.shape[1]):
        for j in range(0, matrix.shape[0]):
            vector[j] = vector[j] + matrix[i][j]

    return vector

def compute_bound(word_counts, totals, sslm):

    W = sslm.num_terms
    T = sslm.num_sequence

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
        for w in range(0, W):

            m = sslm.mean[w][t]
            prev_m = sslm.mean[w][t - 1]

            v = sslm.variance[w][t]

            # Values specifically related to document influence:
            # Note that our indices are off by 1 here.

            w_phi_l = sslm.w_phi_l[w][t - 1]
            exp_i = numpy.exp(numpy.negative(prev_m))

            term_1 += (numpy.power(m - prev_m - (w_phi_l * exp_i), 2) / (2 * chain_variance)) - (v / chain_variance) - numpy.log(chain_variance)

            term_2 += word_counts[w][t - 1] * m
            ent += numpy.log(v) / 2  # note the 2pi's cancel with term1 (see doc)

        term_3 +=  totals[t - 1] * numpy.log(sslm.zeta[t - 1])
        val += numpy.negative(term_1) + term_2 + term_3 + ent

    return val

# fucntion to perform optimization
# def update_obs(counts, totals, sslm):

#     W = sslm.num_terms
#     T = sslm.num_sequence

#     runs = 0

#     params = opt_params(var=sslm, totals=totals)
#     mean_deriv_mtx = numpy.resize(numpy.zeros(T * (T + 1)), (T, T + 1))

#     # for w in range(0, W):










    