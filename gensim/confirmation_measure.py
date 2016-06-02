#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains functions to compute confirmation on a pair of words or word subsets.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

EPSILON = 1e-12 # Should be small. Value as suggested in paper.

def log_conditional_probability(segmented_topics, per_topic_postings):
    """
    This function calculates the log-conditional-probability measure
    which is used by coherence measures such as U_mass.
    This is defined as: m_lc(S_i) = log[(P(W', W*) + e) / P(W*)]

    Args:
    ----
    segmented_topics : Output from the segmentation module of the segmented topics. Is a list of list of tuples.
    per_topic_postings : Output from the probability_estimation module. Is a dictionary of the posting list of all topics.

    Returns:
    -------
    m_lc : List of log conditional probability measure on each set in segmented topics.
    """
    m_lc = []
    for s_i in segmented_topics:
        for w_prime, w_star in s_i: # Have to generalise this to all kinds of sets. Currently suited for S_One_One style segmentation.
            w_prime_docs = per_topic_postings[w_prime]
            w_star_docs = per_topic_postings[w_star]
            co_docs = w_prime_docs.intersection(w_star_docs)
            m_lc_i = np.log((len(co_docs) + EPSILON) / len(w_star_docs))
            m_lc.append(m_lc_i)

    return m_lc
