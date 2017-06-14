#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains functions to compute direct confirmation on a pair of words or word subsets.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

EPSILON = 1e-12  # Should be small. Value as suggested in paper.

def log_conditional_probability(segmented_topics, per_topic_postings, num_docs, aggregated=True):
    """
    This function calculates the log-conditional-probability measure
    which is used by coherence measures such as U_mass.
    This is defined as: m_lc(S_i) = log[(P(W', W*) + e) / P(W*)]

    Args:
    ----
    segmented_topics : Output from the segmentation module of the segmented topics. Is a list of list of tuples.
    per_topic_postings : Output from the probability_estimation module. Is a dictionary of the posting list of all topics.
    num_docs : Total number of documents in corresponding corpus.
    aggregated : Boolean value deciding whether aggregated coherence score is generated or non-aggregated scores.

    Returns:
    -------
    m_lc : List of log conditional probability measure on each set in segmented topics.
    """
    m_lc = []
    if aggregated:
        for s_i in segmented_topics:
            for w_prime, w_star in s_i:
                w_prime_docs = per_topic_postings[w_prime]
                w_star_docs = per_topic_postings[w_star]
                co_docs = w_prime_docs.intersection(w_star_docs)
                if  w_star_docs:
                    m_lc_i = np.log(((len(co_docs) / float(num_docs)) + EPSILON) / (len(w_star_docs) / float(num_docs)))
                else:
                    m_lc_i = 0.0
                m_lc.append(m_lc_i)
    else:
        for s_i in segmented_topics:
            m_lc_i = []
            for w_prime, w_star in s_i:
                w_prime_docs = per_topic_postings[w_prime]
                w_star_docs = per_topic_postings[w_star]
                co_docs = w_prime_docs.intersection(w_star_docs)
                if  w_star_docs:
                    score = np.log(((len(co_docs) / float(num_docs)) + EPSILON) / (len(w_star_docs) / float(num_docs)))
                    m_lc_i.append(score)
                else:
                    score = 0.0
                    m_lc_i.append(score)
            m_lc.append(m_lc_i)

    return m_lc

def log_ratio_measure(segmented_topics, per_topic_postings, num_docs, normalize=False):
    """
    If normalize=False:
        Popularly known as PMI.
        This function calculates the log-ratio-measure which is used by
        coherence measures such as c_v.
        This is defined as: m_lr(S_i) = log[(P(W', W*) + e) / (P(W') * P(W*))]

    If normalize=True:
        This function calculates the normalized-log-ratio-measure, popularly knowns as
        NPMI which is used by coherence measures such as c_v.
        This is defined as: m_nlr(S_i) = m_lr(S_i) / -log[P(W', W*) + e]

    Args:
    ----
    segmented topics : Output from the segmentation module of the segmented topics. Is a list of list of tuples.
    per_topic_postings : Output from the probability_estimation module. Is a dictionary of the posting list of all topics
    num_docs : Total number of documents in corpus. Used for calculating probability.

    Returns:
    -------
    m_lr : List of log ratio measures on each set in segmented topics.
    """
    m_lr = []
    for s_i in segmented_topics:
        for w_prime, w_star in s_i:
            w_prime_docs = per_topic_postings[w_prime]
            w_star_docs = per_topic_postings[w_star]
            co_docs = w_prime_docs.intersection(w_star_docs)
            if normalize:
                # For normalized log ratio measure
                numerator = log_ratio_measure([[(w_prime, w_star)]], per_topic_postings, num_docs)[0]
                co_doc_prob = len(co_docs) / float(num_docs)
                m_lr_i = numerator / (-np.log(co_doc_prob + EPSILON))
            else:
                # For log ratio measure without normalization
                numerator = (len(co_docs) / float(num_docs)) + EPSILON
                denominator = (len(w_prime_docs) / float(num_docs)) * (len(w_star_docs) / float(num_docs))
                m_lr_i = np.log(numerator / denominator)
            m_lr.append(m_lr_i)

    return m_lr
