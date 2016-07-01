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
               mean=None, variance=None,  zeta=None, e_log_prob=None, fwd_mean=None, m_update_coeff=None, temp_vect=None,
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

        return

class opt_params(utils.SaveLoad):
    def __init__(self, sslm, word_counts, totals, mean_deriv_mtx, word):
        self.sslm = sslm
        self.word_counts 
        self.totals = totals
        self.mean_deriv_mtx = mean_deriv_mtx
        self.word = word

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
                doc_lhood = fit_lda_post(d, t, lda_post, None, None, None, None, None)
            else:
                doc_lhood = fit_lda_post(d, t, lda_post, ldaseq, None, None, None, None)
            if topic_suffstats != None:
                update_lda_seq_ss(t, seq_corpus.corpuses[t].doc[d], lda_post, topic_suffstats)
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

        # go through to this again
        converged = numpy.fabs((lhood_old - lhood) / lhood_old * lda_post.doc.total)

    return lhood


def make_lda_seq_slice(lda, ldaseq, time):

    K = ldaseq.num_topics

    for k in range(0, K):
        # s = ldaseq.topic_chains[k].e_log_prob[time]
        # d = lda.topics[k]
        # deep_copy(s, d)
        ldaseq.topic_chains[k].e_log_prob[time] = lda.topics[k]
    ldaseq.alpha = lda.alpha    

    return

def update_lda_seq_ss(time, doc, lda_post, topic_suffstats):

    K = numpy.size(lda_post.phi)[1].size[1]
    N = doc.nterms

    for k in range(0, K):
        topic_ss = topic_suffstats[k]
        for n in range(0, N):
            w = doc.word[n]
            c = doc.count[n]
            topic_ss[w][time] = topic_ss[w][time] + c * lda_post.phi[n][k]
    return

def init_lda_post(lda_post):
    K = lda_post.lda.num_topics
    N = lda_post.doc.nterms

    for k in range(0, K):
        lda_post.gamma[k] = lda_post.lda.alpha[k] + float(lda_post.doc.total) / k 
        for n in range(0, N):
            lda_post.phi[n][k] = 1.0 / K

    lda_post.doc_weight = None
    return

def compute_lda_lhood(lda_post):
    
    K = lda_post.lda.num_topics
    N = lda_post.doc.nterms
    gamma_sum = numpy.sum(lda_post.gamam)

    # figure out how to do flags
    FLAGS_sigma_l = 0
    FLAGS_sigma_d = 0 

    # need to find replacement for this gsl method
    lhood = gls_sf_lngamma(numpy.sum(lda_post.lda.alpha)) - gls_sf_lngamma(gamma_sum)
    lda_post.lhood[K] = lhood

    influence_term = 0
    # need to find replacement for this gsl method
    digsum = gsl_sf_psi(gamma_sum)

    model = "DTM"
    for k in range(0, K):
        if lda_post.doc_weight is not None and (model == "DIM" or model == "DTM"):
            influence_topic = lda_post.doc_weight[k]
            influence_term = - ((influence_topic * influence_topic + FLAGS_sigma_l * FLAGS_sigma_l) / 2.0 / (FLAGS_sigma_d * FLAGS_sigma_d))

        e_log_theta_k = gsl_sf_psi(lda_post.gamma[k]) - digsum

        # figure out what is this gsl stuff
        lhood_term = (lda_post.lda.alpha[k] - lda_post.gamma[k]) * e_log_theta_k + gls_sf_lngamma(lda_post.gamma[k]) - gls_sf_lngamma(lda_post.lda.alpha[k])

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
        dig[k] = gsl_sf_psi(lda_post.gamma[k])

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
            phi_row[i] = numpy.exp(log_phi_row[i])

    return

# update variational dirichlet parameters
def update_gamma(lda_post):

    K = lda_post.lda.num_topics
    N = lda_post.doc.nterms

    lda_post.gamma = lda_post.lda.alpha
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
def update_obs(word_counts, totals, sslm):

    OBS_NORM_CUTOFF = 2

    # used in optimize function but not sure what is happening
    f_val = None
    conv_val = None
    niter = None

    W = sslm.num_terms
    T = sslm.num_sequence

    runs = 0

    params = opt_params(var=sslm, totals=totals)
    mean_deriv_mtx = numpy.resize(numpy.zeros(T * (T + 1)), (T, T + 1))


    for w in range(0, W):
        w_counts = word_counts[w]

        counts_norm = 0
        # now we find L2 norm of w_counts
        for i in range(0, len(word_counts)):
            counts_norm += word_counts[i] * word_counts[i]

        if counts_norm < OBS_NORM_CUTOFF and norm_cutoff_obs is not None:
            obs = sslm.obs[w]
            # a memcopy is happening here
            norm_cutoff_obs = obs
        else: 
            if counts_norm < OBS_NORM_CUTOFF:
                w_counts = numpy.zeros(len(word_counts))

            for t in range(0, T):
                mean_deriv = mean_deriv_mtx[t]
                compute_mean_deriv(w, t, sslm, mean_deriv)

            params.word_counts = w_counts
            params.word = w
            params.mean_deriv_mtx = mean_deriv_mtx
            obs = sslm.obs[w]

            model = "DTM"
            if model == "DTM":
                optimize_fdf(T, obs, params, fdf_obs, df_obs, f_obs, f_val, conv_val, niter)
            if model == "DIM":
                optimize_fdf(T, obs, params, fdf_obs, df_obs, f_obs_fixed, f_val, conv_val, niter)

            runs += 1

            if counts_norm < OBS_NORM_CUTOFF:
                norm_cutoff_obs = obs

    update_zeta(sslm)
    return


 # compute d E[\beta_{t,w}]/d obs_{s,w} for t = 1:T.
 # put the result in deriv, allocated T+1 vector
 
def compute_mean_deriv(word, time, sslm, deriv):

    T = sslm.num_sequence
    fwd_variance = sslm.variance[w]

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

# maximize a function using it's derivative
def optimize_fdf(dim, x, params, fdf, df, f, f_val, conv_val, niter):
   
    MAX_ITER = 15
    # what is multimin?
    obj = gsl_multimin_function_fdf()
    obj.f = f
    obj.df = df
    obj.fdf = fdf
    obj.n = dim
    obj.params = params

    method = gsl_multimin_fdfminimizer_conjugate_fr;
    opt = gsl_multimin_fdfminimizer_alloc(method, dim);
    gsl_multimin_fdfminimizer_set(opt, obj, x, 0.01, 1e-3)

    iter_ = 0
    f_old = 0

    # convert from a do while here
    while converged > 1e-8 and iter_ < MAX_ITER:
        iter_ += 1
        status = gsl_multimin_fdfminimizer_iterate(opt)
        converged = numpy.fabs((f_old - opt.f) / (dim * f_old))
        f_old = opt.f

    # all of these are pointer values being reset, so should probably return them
    f_val = opt.f
    conv_val = converged
    niter = iter_

    return

def fdf_obs(x, params, f, df):

    p = params
    model = "DTM"

    if model == "DTM":
        f = f_obs(x, params)
        compute_obs_deriv(p.word, p.word_counts, p.totals, p.sslm, p.mean_deriv_mtx, df)
    elif model == "DIM":
        f = f_obs_multiplt(x, params)
        compute_obs_deriv_fixed(p.word, p.word_counts, p.totals, p.sslm, p.mean_deriv_mtx, df)

    for i in range(0, len(df)):
        df[i] = - df[i]

def df_obs(x, params, df):

    p = params
    p.sslm.obs[p.word] = x

    compute_post_mean(p.word, p.sslm, p.sslm.chain_variance)
    if model == "DTM":
        compute_obs_deriv(p.word, p.word_counts, p.totals, p.sslm, p.mean_deriv_mtx, df)
    elif model == "DIM":
        compute_obs_deriv_fixed(p.word, p.word_counts, p.totals, p.sslm, p.mean_deriv_mtx, df)

def f_obs(x, params):

    # flag
    init_mult = 1000

    T = len(x)
    val = 0
    term1 = 0
    term2 = 0

    # term 3 and 4 for DIM
    term3 = 0 
    term4 = 0

    p = params
    p.sslm.obs[p.word] = x
    compute_post_mean(p.word, p.sslm, p.sslm.chain_variance)

    mean = p.sslm.mean[p.word]
    variance = p.sslm.variance[p.word]
    w_phi_l = p.sslm.w_phi_l[p.word]
    m_update_coeff = p.sslm.m_update_coeff[p.word]

    for t in range(1, T + 1):
        mean_t = mean[t]
        mean_t_prev = mean[t - 1]
        var_t_prev = variance[t - 1]

        val = mean_t - mean_t_prev
        term1 += val * val
        term2 += p.word_counts[t - 1] * mean_t - p.totals[t - 1] * numpy.exp(mean_t + variance[t] / 2) / p.sslm.zeta[t - 1]

        model = "DTM"
        if model == "DIM":
            # stuff happens
            pass

    if p.sslm.chain_variance > 0.0:
        
        term1 = - (term1 / 2 * p.sslm.chain_variance)
        term1 = term1 - mean[0] * mean[0] / (2 * init_milt * p.sslm.chain_variance)
    else:
        term1 = 0.0

    return -(term1 + term2 + term3 + term4)


def compute_obs_deriv(word, word_counts, totals, sslm, mean_deriv_mtx, deriv):

    # flag
    init_mult = 1000

    T = sslm.num_sequence

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

            term1 += (mean_u - mean_u_prev) * (dmean_u * dmean_u_prev)

            term2 += (word_counts[u - 1] - (totals[u - 1] * sslm.temp_vect[u - 1] / sslm.zeta[u - 1])) * dmean_u

            model = "DTM"
            if model == "DIM":
                # do some stuff
                pass

        if sslm.chain_variance:
            term1 = - (term1 / sslm.chain_variance)
            term1 = term1 - (mean[0] * mean_deriv[0]) / init_mult * sslm.chain_variance
        else:
            term1 = 0.0

        deriv[t] = term1 + term2 + term3 + term4
    
    return
