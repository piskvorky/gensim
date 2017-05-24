#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains functions to compute confirmation on a pair of words or word subsets. The advantage of indirect
confirmation measure is that it computes similarity of words in W' and W* with respect to direct confirmations to all words.
Eg. Suppose x and z are both competing brands of cars, which semantically support each other. However, both brands are
seldom mentioned together in documents in the reference corpus. But their confirmations to other words like “road”
or “speed” do strongly correlate. This would be reflected by an indirect confirmation measure. Thus, indirect confirmation
measures may capture semantic support that direct measures would miss.

The formula used to compute indirect confirmation measure is:

m_{sim}_{(m, \gamma)}(W', W*) = s_{sim}(\vec{V}^{\,}_{m,\gamma}(W'), \vec{V}^{\,}_{m,\gamma}(W*))

where s_sim can be cosine, dice or jaccard similarity and

\vec{V}^{\,}_{m,\gamma}(W') = \Bigg \{{\sum_{w_{i} \in W'}^{ } m(w_{i}, w_{j})^{\gamma}}\Bigg \}_{j = 1,...,|W|}

Here 'm' is the direct confirmation measure used.
"""

import logging
import numpy as np

from gensim.topic_coherence import direct_confirmation_measure
from gensim.matutils import cossim

logger = logging.getLogger(__name__)


def _present(w_prime_star, w, w_backtrack):
    """
    Internal helper function to return index of (w_prime_star, w) in w_backtrack.
    Return -1 if not present.
    """
    index = -1
    flag = 0
    for arr in w_backtrack:
        index += 1
        if np.all(w_prime_star == arr[0]) and np.all(w == arr[1]):
            flag += 1
            break
    if not flag:
        return -1
    return index


def _make_seg(w_prime, w, accumulator, measure, gamma, backtrack):
    """
    Internal helper function to return context vectors for segmentations.
    """
    context_vectors = {}
    if isinstance(w_prime, np.ndarray):
        for w_j in w:
            for w_i in w_prime:
                if (w_i, w_j) not in backtrack:
                    backtrack[(w_i, w_j)] = measure[0]([[(w_i, w_j)]], accumulator, measure[1])[0]
                if w_j not in context_vectors:
                    context_vectors[w_j] = backtrack[(w_i, w_j)] ** gamma
                else:
                    context_vectors[w_j] += backtrack[(w_i, w_j)] ** gamma
    else:
        for w_j in w:
            if (w_prime, w_j) not in backtrack:
                backtrack[(w_prime, w_j)] = measure[0]([[(w_prime, w_j)]], accumulator, measure[1])[0]
            context_vectors[w_j] = backtrack[(w_prime, w_j)] ** gamma

    return context_vectors, backtrack


def cosine_similarity(topics, segmented_topics, accumulator, measure, gamma):
    """
    This function calculates the indirect cosine measure. Given context vectors
    _   _         _   _
    u = V(W') and w = V(W*) for the word sets of a pair S_i = (W', W*) indirect
                                                                _     _
    cosine measure is computed as the cosine similarity between u and w. The formula used is:

    m_{sim}_{(m, \gamma)}(W', W*) = s_{sim}(\vec{V}^{\,}_{m,\gamma}(W'), \vec{V}^{\,}_{m,\gamma}(W*))

    where each vector \vec{V}^{\,}_{m,\gamma}(W') = \Bigg \{{\sum_{w_{i} \in W'}^{ } m(w_{i}, w_{j})^{\gamma}}\Bigg \}_{j = 1,...,|W|}

    Args:
    ----
    topics : Topics obtained from the trained topic model.
    segmented_topics : segmented_topics : Output from the segmentation module of the segmented topics. Is a list of list of tuples.
    per_topic_postings : Output from the probability_estimation module. Is a dictionary of the posting list of all topics.
    measure : String. Direct confirmation measure to be used. Supported values are "nlr" (normalized log ratio).
    gamma : Gamma value for computing W', W* vectors.
    num_docs : Total number of documents in corresponding corpus.

    Returns:
    -------
    s_cos_sim : array of cosine similarity of the context vectors for each segmentation
    """
    if measure == 'nlr':
        # make normalized log ratio measure tuple
        measure = (direct_confirmation_measure.log_ratio_measure, True)
    else:
        raise ValueError("The direct confirmation measure you entered is not currently supported.")
    backtrack = {}  # Backtracking dictionary for storing measure values of topic id tuples eg. (1, 2).
    """
    For backtracking context vectors, we will create a list called w_backtrack to store (w_prime, w) or
    (w_star, w) tuples and a corresponding list context_vector_backtrack which will create a
    mapping of (w_prime or w_star, w) ---> context_vector.
    """
    w_backtrack = []
    context_vector_backtrack = []
    s_cos_sim = []
    for top_words, s_i in zip(topics, segmented_topics):
        for w_prime, w_star in s_i:
            # Step 1. Check if (w_prime, top_words) tuple in w_backtrack.
            # Step 2. If yes, return corresponding context vector
            w_prime_index = _present(w_prime, top_words, w_backtrack)
            if w_backtrack and w_prime_index != -1:
                w_prime_context_vectors = context_vector_backtrack[w_prime_index]
            else:
                w_prime_context_vectors, backtrack_i = _make_seg(w_prime, top_words, accumulator, measure, gamma, backtrack)
                backtrack.update(backtrack_i)
                # Update backtracking lists
                w_backtrack.append((w_prime, top_words))
                context_vector_backtrack.append(w_prime_context_vectors)

            # Step 1. Check if (w_star, top_words) tuple in w_backtrack.
            # Step 2. If yes, check if corresponding w is the same
            w_star_index = _present(w_star, top_words, w_backtrack)
            if w_backtrack and w_star_index != -1:
                w_star_context_vectors = context_vector_backtrack[w_star_index]
            else:
                w_star_context_vectors, backtrack_i = _make_seg(w_star, top_words, accumulator, measure, gamma, backtrack)
                backtrack.update(backtrack_i)
                # Update all backtracking lists
                w_backtrack.append((w_star, top_words))
                context_vector_backtrack.append(w_star_context_vectors)

            s_cos_sim_i = cossim(w_prime_context_vectors.items(), w_star_context_vectors.items())
            s_cos_sim.append(s_cos_sim_i)

    return s_cos_sim
