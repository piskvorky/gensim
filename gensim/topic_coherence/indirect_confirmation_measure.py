#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains functions to compute confirmation on a pair of words or word subsets.

The formula used to compute indirect confirmation measure is:
                                _              _
m_sim(m, gamma)(W', W*) = s_sim(V_m,gamma(W'), V_m,gamma(W*))

where s_sim can be cosine, dice or jaccard similarity and
_
V_m,gamma(W') = {sigma(w' belonging to W') m(w_i, w_j) ^ gamma} where j = 1, ...., |W|

Here 'm' is the direct confirmation measure used.
"""

import logging
import numpy as np

from gensim.topic_coherence import direct_confirmation_measure
from gensim.matutils import cossim

logger = logging.getLogger(__name__)

def _make_seg(w_prime, w, per_topic_postings, measure, gamma, backtrack, num_docs):
    """
    Internal helper function to return context vectors for segmentations.
    """
    context_vectors = {}
    if isinstance(w_prime, np.ndarray):
        for w_j in w:
            for w_i in w_prime:
                if (w_i, w_j) not in backtrack:
                    backtrack[(w_i, w_j)] = measure([[(w_i, w_j)]], per_topic_postings, num_docs)[0]
                if w_j not in context_vectors:
                    context_vectors[w_j] = backtrack[(w_i, w_j)] ** gamma
                else:
                    context_vectors[w_j] += backtrack[(w_i, w_j)] ** gamma
    else:
        for w_j in w:
            if (w_prime, w_j) not in backtrack:
                backtrack[(w_prime, w_j)] = measure([[(w_prime, w_j)]], per_topic_postings, num_docs)[0]
            context_vectors[w_j] = backtrack[(w_prime, w_j)] ** gamma
    return (context_vectors, backtrack)

def cosine_similarity(topics, segmented_topics, per_topic_postings, measure, gamma, num_docs):
    """
    This function calculates the indirect cosine measure. Given context vectors
    _   _         _   _
    u = V(W') and w = V(W*) for the word sets of a pair S_i = (W', W*) indirect
                                                                _     _
    cosine measure is computed as the cosine similarity between u and w.

    Args:
    ----
    topics : Topics obtained from the trained topic model.
    segmented_topics : segmented_topics : Output from the segmentation module of the segmented topics. Is a list of list of tuples.
    per_topic_postings : per_topic_postings : Output from the probability_estimation module. Is a dictionary of the posting list of all topics.
    measure : String. Direct confirmation measure to be used. Supported values are "nlr" (normalized log ratio).
    gamma : Gamma value for computing W', W* vectors.
    num_docs : Total number of documents in corresponding corpus.
    """
    if measure == 'nlr':
        measure = direct_confirmation_measure.normalized_log_ratio_measure
    else:
        raise ValueError("The direct confirmation measure you entered is not currently supported.")
    backtrack = {}
    s_cos_sim = []
    for top_words, s_i in zip(topics, segmented_topics):
        for w_prime, w_star in s_i:
            w_prime_context_vectors, backtrack_i = _make_seg(w_prime, top_words, per_topic_postings, measure, gamma, backtrack, num_docs)
            backtrack.update(backtrack_i)
            w_star_context_vectors, backtrack_i = _make_seg(w_star, top_words, per_topic_postings, measure, gamma, backtrack, num_docs)
            backtrack.update(backtrack_i)
            s_cos_sim_i = cossim(w_prime_context_vectors.items(), w_star_context_vectors.items())
            s_cos_sim.append(s_cos_sim_i)

    return s_cos_sim
