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


def log_conditional_probability(segmented_topics, accumulator, with_std=False, with_support=False):
    """
    This function calculates the log-conditional-probability measure
    which is used by coherence measures such as U_mass.
    This is defined as: m_lc(S_i) = log[(P(W', W*) + e) / P(W*)]

    Args:
        segmented_topics (list): Output from the segmentation module of the segmented
            topics. Is a list of list of tuples.
        accumulator: word occurrence accumulator from probability_estimation.
        with_std (bool): True to also include standard deviation across topic segment
            sets in addition to the mean coherence for each topic; default is False.
        with_support (bool): True to also include support across topic segments. The
            support is defined as the number of pairwise similarity comparisons were
            used to compute the overall topic coherence.

    Returns:
        list : of log conditional probability measure for each topic.
    """
    topic_coherences = []
    num_docs = float(accumulator.num_docs)
    for s_i in segmented_topics:
        segment_sims = []
        for w_prime, w_star in s_i:
            try:
                w_star_count = accumulator[w_star]
                co_occur_count = accumulator[w_prime, w_star]
                m_lc_i = np.log(((co_occur_count / num_docs) + EPSILON) / (w_star_count / num_docs))
            except KeyError:
                m_lc_i = 0.0

            segment_sims.append(m_lc_i)

        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))

    return topic_coherences


def aggregate_segment_sims(segment_sims, with_std, with_support):
    """Compute various statistics from the segment similarities generated via
    set pairwise comparisons of top-N word lists for a single topic.

    Args:
        segment_sims (iterable): floating point similarity values to aggregate.
        with_std (bool): Set to True to include standard deviation.
        with_support (bool): Set to True to include number of elements in `segment_sims`
            as a statistic in the results returned.

    Returns:
        tuple: with (mean[, std[, support]])
    """
    mean = np.mean(segment_sims)
    stats = [mean]
    if with_std:
        stats.append(np.std(segment_sims))
    if with_support:
        stats.append(len(segment_sims))

    return stats[0] if len(stats) == 1 else tuple(stats)


def log_ratio_measure(
        segmented_topics, accumulator, normalize=False, with_std=False, with_support=False):
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
        segmented_topics (list): Output from the segmentation module of the segmented
            topics. Is a list of list of tuples.
        accumulator: word occurrence accumulator from probability_estimation.
        with_std (bool): True to also include standard deviation across topic segment
            sets in addition to the mean coherence for each topic; default is False.
        with_support (bool): True to also include support across topic segments. The
            support is defined as the number of pairwise similarity comparisons were
            used to compute the overall topic coherence.

    Returns:
        list : of log ratio measure for each topic.
    """
    topic_coherences = []
    num_docs = float(accumulator.num_docs)
    for s_i in segmented_topics:
        segment_sims = []
        for w_prime, w_star in s_i:
            w_prime_count = accumulator[w_prime]
            w_star_count = accumulator[w_star]
            co_occur_count = accumulator[w_prime, w_star]

            if normalize:
                # For normalized log ratio measure
                numerator = log_ratio_measure([[(w_prime, w_star)]], accumulator)[0]
                co_doc_prob = co_occur_count / num_docs
                m_lr_i = numerator / (-np.log(co_doc_prob + EPSILON))
            else:
                # For log ratio measure without normalization
                numerator = (co_occur_count / num_docs) + EPSILON
                denominator = (w_prime_count / num_docs) * (w_star_count / num_docs)
                m_lr_i = np.log(numerator / denominator)

            segment_sims.append(m_lr_i)

        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))

    return topic_coherences
