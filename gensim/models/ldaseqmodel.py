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
import numpy

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
            raise ValueError("cannot compute LDA over an empty collection (no terms)")

        self.num_topics = num_topics
        self.num_sequence = num_sequence
        self.alphas = alphas
        self.topic_chains = topic_chains
        self.top_doc_phis = top_doc_phis

        # influence values as of now not using
        self.influence = influence
        self.renormalized_influence = renormalized_influence
        self.influence_sum_lgl = influence_sum_lgl

class sslm(utils.SaveLoad):
  def __init__(num_terms=None, num_sequence=None, obs=None, obs_variance=None, chain_variance=None, fwd_variance=None,
               mean=None, variance=None,  zeta=None, e_log_prob=None, fwd_mean=None, m_update_coeff=None,
               mean_t=None, variance_t=None, influence_sum_lgl=None, w_phi_l=None, w_phi_sum=None, w_phi_l_sq=None,  m_update_coeff_g=None):

        self.obs = obs
        self.zeta = zeta
        self.mean = mean
        self.variance = variance


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
