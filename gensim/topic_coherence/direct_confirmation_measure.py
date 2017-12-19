#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains functions to compute direct confirmation on a pair of words or word subsets."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Should be small. Value as suggested in paper http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf
EPSILON = 1e-12


def log_conditional_probability(segmented_topics, accumulator, with_std=False, with_support=False):
    """Calculate the log-conditional-probability measure which is used by coherence measures such as `U_mass`.
    This is defined as :math:`m_{lc}(S_i) = log \\frac{P(W', W^{*}) + \epsilon}{P(W^{*})}`.

    Parameters
    ----------
    segmented_topics : list of lists of (int, int)
        Output from the :func:`~gensim.topic_coherence.segmentation.s_one_pre`,
        :func:`~gensim.topic_coherence.segmentation.s_one_one`.
    accumulator : :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`
        Word occurrence accumulator from :mod:`gensim.topic_coherence.probability_estimation`.
    with_std : bool
        True to also include standard deviation across topic segment sets in addition to the mean coherence
        for each topic.
    with_support : bool
        True to also include support across topic segments. The support is defined as the number of pairwise
        similarity comparisons were used to compute the overall topic coherence.

    Returns
    -------
    list of float
        Log conditional probabilities measurement for each topic.

    Examples
    --------
    >>> from gensim.topic_coherence import direct_confirmation_measure, text_analysis
    >>> from collections import namedtuple
    >>>
    >>> # Create dictionary
    >>> id2token = {1: 'test', 2: 'doc'}
    >>> token2id = {v: k for k, v in id2token.items()}
    >>> dictionary = namedtuple('Dictionary', 'token2id, id2token')(token2id, id2token)
    >>>
    >>> # Initialize segmented topics and accumulator
    >>> segmentation = [[(1, 2)]]
    >>>
    >>> accumulator = text_analysis.InvertedIndexAccumulator({1, 2}, dictionary)
    >>> accumulator._inverted_index = {0: {2, 3, 4}, 1: {3, 5}}
    >>> accumulator._num_docs = 5
    >>>
    >>> # result should be ~ ln(1 / 2) = -0.693147181
    >>> result = direct_confirmation_measure.log_conditional_probability(segmentation, accumulator)[0]


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
    """Compute various statistics from the segment similarities generated via set pairwise comparisons
    of top-N word lists for a single topic.

    Parameters
    ----------
    segment_sims : iterable of float
        Similarity values to aggregate.
    with_std : bool
        Set to True to include standard deviation.
    with_support : bool
        Set to True to include number of elements in `segment_sims` as a statistic in the results returned.

    Returns
    -------
    (float[, float[, int]])
        Tuple with (mean[, std[, support]]).

    Examples
    ---------
    >>> from gensim.topic_coherence import direct_confirmation_measure
    >>>
    >>> segment_sims = [0.2, 0.5, 1., 0.05]
    >>> direct_confirmation_measure.aggregate_segment_sims(segment_sims, True, True)
    (0.4375, 0.36293077852394939, 4)
    >>> direct_confirmation_measure.aggregate_segment_sims(segment_sims, False, False)
    0.4375

    """
    mean = np.mean(segment_sims)
    stats = [mean]
    if with_std:
        stats.append(np.std(segment_sims))
    if with_support:
        stats.append(len(segment_sims))

    return stats[0] if len(stats) == 1 else tuple(stats)


def log_ratio_measure(segmented_topics, accumulator, normalize=False, with_std=False, with_support=False):
    """Compute log ratio measure for `segment_topics`.

    Parameters
    ----------
    segmented_topics : list of lists of (int, int)
        Output from the :func:`~gensim.topic_coherence.segmentation.s_one_pre`,
        :func:`~gensim.topic_coherence.segmentation.s_one_one`.
    accumulator : :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`
        Word occurrence accumulator from :mod:`gensim.topic_coherence.probability_estimation`.
    normalize : bool
        Details in the "Notes" section.
    with_std : bool
        True to also include standard deviation across topic segment sets in addition to the mean coherence
        for each topic.
    with_support : bool
        True to also include support across topic segments. The support is defined as the number of pairwise
        similarity comparisons were used to compute the overall topic coherence.

    Notes
    -----
    If `normalize=False`:
        Calculate the log-ratio-measure, popularly known as **PMI** which is used by coherence measures such as `c_v`.
        This is defined as :math:`m_{lr}(S_i) = log \\frac{P(W', W^{*}) + \epsilon}{P(W') * P(W^{*})}`

    If `normalize=True`:
        Calculate the normalized-log-ratio-measure, popularly knowns as **NPMI**
        which is used by coherence measures such as `c_v`.
        This is defined as :math:`m_{nlr}(S_i) = \\frac{m_{lr}(S_i)}{-log(P(W', W^{*}) + \epsilon)}`

    Returns
    -------
    list of float
        Log ratio measurements for each topic.

    Examples
    --------
    >>> from gensim.topic_coherence import direct_confirmation_measure, text_analysis
    >>> from collections import namedtuple
    >>>
    >>> # Create dictionary
    >>> id2token = {1: 'test', 2: 'doc'}
    >>> token2id = {v: k for k, v in id2token.items()}
    >>> dictionary = namedtuple('Dictionary', 'token2id, id2token')(token2id, id2token)
    >>>
    >>> # Initialize segmented topics and accumulator
    >>> segmentation = [[(1, 2)]]
    >>>
    >>> accumulator = text_analysis.InvertedIndexAccumulator({1, 2}, dictionary)
    >>> accumulator._inverted_index = {0: {2, 3, 4}, 1: {3, 5}}
    >>> accumulator._num_docs = 5
    >>>
    >>> # result should be ~ ln{(1 / 5) / [(3 / 5) * (2 / 5)]} = -0.182321557
    >>> result = direct_confirmation_measure.log_ratio_measure(segmentation, accumulator)[0]

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
