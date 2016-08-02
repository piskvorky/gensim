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

# this is a mock LDA class to help with testing until this is figured out
class mockLDA(utils.SaveLoad):
    def __init__(self, num_topics=None, alpha=None, num_terms=None, topics=None):
        self.num_topics = num_topics
        self.num_terms = num_terms
        self.alpha = alpha
        if topics is None:
            self.topics = numpy.array(numpy.split(numpy.zeros(num_terms * num_topics), num_terms))
        elif topics is not None:
            self.topics = topics

# a mock document class to help with testing until this is figured out
class Doc(utils.SaveLoad):
    def __init__(self, nterms=None, word=None, count=None, total=None):
        self.nterms = nterms
        self.word = word
        self.count = count
        self.total = total

class seq_corpus(utils.SaveLoad):
    def __init__(self, num_terms=None, max_nterms=None, length=None, num_docs=None, corpuses=None, corpus=None):
        self.num_terms = num_terms
        self.max_nterms = max_nterms
        self.length = len(corpuses)
        self.num_docs = num_docs

        # list of corpus class objects
        self.corpuses = corpuses
        self.corpus = corpus

class LdaSeqModel(utils.SaveLoad):
    def __init__(self, corpus=None, num_topics=10, id2word=None, num_sequences=None, num_terms=None, alphas=None, top_doc_phis=None,
     topic_chains=[], influence=None, influence_sum_lgl=None, renormalized_influence=None):
        # store user-supplied parameters
        self.corpus = corpus
        self.id2word = id2word
        self.num_topics = num_topics
        self.num_sequences = num_sequences
        self.num_terms = num_terms
        self.alphas = alphas
        self.topic_chains = topic_chains
        if self.topic_chains is None:
            for topic in range(0, num_topics):
                sslm_ = sslm(num_sequences=num_sequences, num_terms=num_terms, num_topics=num_topics)
                topic_chains.append(sslm_)

        self.top_doc_phis = top_doc_phis
        # influence values as of now not using
        self.influence = influence
        self.renormalized_influence = renormalized_influence
        self.influence_sum_lgl = influence_sum_lgl

class sslm(utils.SaveLoad):
  def __init__(self, num_terms=None, num_sequences=None, obs=None, obs_variance=0.5, chain_variance=0.005, fwd_variance=None,
               mean=None, variance=None,  zeta=None, e_log_prob=None, fwd_mean=None, m_update_coeff=None, temp_vect=None,
               mean_t=None, variance_t=None, influence_sum_lgl=None, w_phi_l=None, w_phi_sum=None, w_phi_l_sq=None,  m_update_coeff_g=None):

        self.obs = obs
        self.zeta = zeta # array equal to number of sequences
        self.mean = mean # matrix of dimensions num_terms * (num_of sequences + 1)
        self.variance = variance # matrix of dimensions num_terms * (num_of sequences + 1)
        self.num_terms = num_terms
        self.num_sequences = num_sequences
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

        # temp_vect 
        self.temp_vect = temp_vect

class lda_post(utils.SaveLoad):
    def __init__(self, doc=None, lda=None, phi=None, log_phi=None, gamma=None, lhood=None, doc_weight=None, renormalized_doc_weight=None):
        self.doc = doc
        self.lda = lda
        self.phi = phi
        self.log_phi = log_phi
        self.gamma = gamma
        self.lhood = lhood
        self.doc_weight = doc_weight
        self.renormalized_doc_weight = renormalized_doc_weight

def make_seq_corpus(corpus, time_seq):
    split_corpus = []
    time_seq.insert(0, 0)
    for time in range(0, len(time_seq) - 1):
        time_seq[time + 1] = time_seq[time] + time_seq[time + 1]
        split_corpus.append(corpus[time_seq[time]:time_seq[time+1]])

    num_docs = len(corpus)
    length = len(split_corpus)
    # num_terms = len(corpus.dictionary)

    seq_corpus_ = seq_corpus(num_docs=num_docs, length=length, corpuses=split_corpus, corpus=corpus)

    return seq_corpus_

def update_zeta(sslm):
    # setting limits 
    # num_terms = sslm.obs.shape[0] # this is word length (our example, 562)
    # num_sequences = sslm.obs.shape[1] # this is number of sequeces
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

def compute_post_variance(word , sslm, chain_variance):
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

    # sslm.mean[word] = mean
    # sslm.fwd_mean[word] = fwd_mean    
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
        # dont't need to initialize here, but writing for reference
        ldaseq.topic_chains[k].w_phi_l = numpy.zeros((ldaseq.num_terms, ldaseq.num_sequences))
        ldaseq.topic_chains[k].w_phi_sum = numpy.zeros((ldaseq.num_terms, ldaseq.num_sequences))
        ldaseq.topic_chains[k].w_phi_sq = numpy.zeros((ldaseq.num_terms, ldaseq.num_sequences))

def fit_lda_seq(ldaseq, seq_corpus):

    K = ldaseq.num_topics
    W = ldaseq.num_terms
    data_len = seq_corpus.length
    num_docs = seq_corpus.num_docs
    
    # heldout_gammas = NULL
    # heldout_llhood = NULL
    LDA_INFERENCE_MAX_ITER = 25

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


        # figure out how to write to file here
        # TODO save to file for command line
        # gammas_file.write(gammas)
        # lhoods_file.write(lhoods)

        print ("M Step")

        topic_bound = fit_lda_seq_topics(ldaseq, topic_suffstats)
        bound += topic_bound



        # write_lda_seq(ldaseq)

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
    
    lda = mockLDA(num_topics=K, alpha=ldaseq.alphas, num_terms=W)
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
        ndocs = len(seq_corpus.corpuses[t])

        for d in range(0, ndocs):

            gam = gammas[doc_index]
            lhood = lhoods[doc_index]

            doc_ = seq_corpus.corpuses[t][d]
            nterms, word_id = doc_.split(' ', 1)
            words = []
            counts = []
            totals = 0

            for pair in word_id.split():
                word, count = pair.split(':')
                words.append(int(word))
                counts.append(int(count))
                totals += int(count)

            doc = Doc(word=words, count=counts, total=totals, nterms=int(nterms))
            lda_post.gamma = gam
            lda_post.lhood = lhood
            lda_post.doc = doc
            lda_post.lda = lda

            if iter_ == 0:
                doc_lhood = fit_lda_post(d, t, lda_post, None, None, None, None, None)
            else:
                doc_lhood = fit_lda_post(d, t, lda_post, ldaseq, None, None, None, None)
           



            if topic_suffstats != None:
                update_lda_seq_ss(t, doc, lda_post, topic_suffstats)

            bound += doc_lhood
            doc_index += 1

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

    # convert from a do-while look

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

    lda_post.doc_weight = None
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

    influence_term = 0
    digsum = digamma(gamma_sum)

    model = "DTM"
    for k in range(0, K):
        if lda_post.doc_weight is not None and (model == "DIM" or model == "fixed"):
            influence_topic = lda_post.doc_weight[k]
            influence_term = - ((influence_topic * influence_topic + FLAGS_sigma_l * FLAGS_sigma_l) / 2.0 / (FLAGS_sigma_d * FLAGS_sigma_d))

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

            # Values specifically related to document influence:
            # Note that our indices are off by 1 here.
            w_phi_l = sslm.w_phi_l[w][t - 1]
            exp_i = numpy.exp(-prev_m)
            term_1 += (numpy.power(m - prev_m - (w_phi_l * exp_i), 2) / (2 * chain_variance)) - (v / chain_variance) - numpy.log(chain_variance)
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
    w_phi_l = sslm.w_phi_l[word]
    m_update_coeff = sslm.m_update_coeff[word]

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

    sslm.temp_vect = numpy.zeros(T)

    for u in range(0, T):
        sslm.temp_vect[u] = numpy.exp(mean[u + 1] + variance[u + 1] / 2)

    w_phi_l = sslm.w_phi_l[word]
    m_update_coeff = sslm.m_update_coeff[word]

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

def lda_sstats(seq_corpus, num_topics, num_terms, alpha):

    lda_model = mockLDA(num_topics=num_topics, num_terms=num_terms)
    lda_model.alpha = alpha # this will have shape equal to  number of topics
    # lda_ss = initialize_ss_random(seq_corpus, num_topics)

    lda_ss = numpy.array(numpy.split(numpy.loadtxt("sstats_rand"), num_terms))

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

    lda_post.phi = numpy.resize(numpy.zeros(seq_corpus.max_nterms * K), (seq_corpus.max_nterms, K))
    lda_post.log_phi = numpy.resize(numpy.zeros(seq_corpus.max_nterms * K), (seq_corpus.max_nterms, K))
    lda_post.gamma = numpy.zeros(K)
    lda_post.lhood = numpy.zeros(K + 1)
    lda_post.lda = lda_model

    lhood = 0

    for d in range(0, seq_corpus.num_docs):

        doc_ = seq_corpus.corpus[d]
        nterms, word_id = doc_.split(' ', 1)
        words = []
        counts = []
        totals = 0

        for pair in word_id.split():
            word, count = pair.split(':')
            words.append(int(word))
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

