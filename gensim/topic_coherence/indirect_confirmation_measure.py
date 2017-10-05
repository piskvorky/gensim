#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains functions to compute confirmation on a pair of words or word subsets.
The advantage of indirect confirmation measure is that it computes similarity of words in W' and
W* with respect to direct confirmations to all words. Eg. Suppose x and z are both competing
brands of cars, which semantically support each other. However, both brands are seldom mentioned
together in documents in the reference corpus. But their confirmations to other words like “road”
or “speed” do strongly correlate. This would be reflected by an indirect confirmation measure.
Thus, indirect confirmation measures may capture semantic support that direct measures would miss.

The formula used to compute indirect confirmation measure is:

m_{sim}_{(m, \gamma)}(W', W*) =
    s_{sim}(\vec{V}^{\,}_{m,\gamma}(W'), \vec{V}^{\,}_{m,\gamma}(W*))

where s_sim can be cosine, dice or jaccard similarity and

\vec{V}^{\,}_{m,\gamma}(W') =
    \Bigg \{{\sum_{w_{i} \in W'}^{ } m(w_{i}, w_{j})^{\gamma}}\Bigg \}_{j = 1,...,|W|}

Here 'm' is the direct confirmation measure used.
"""

import itertools
import logging

import numpy as np
import scipy.sparse as sps

from gensim.topic_coherence.direct_confirmation_measure import (
    aggregate_segment_sims, log_ratio_measure)

logger = logging.getLogger(__name__)


def word2vec_similarity(segmented_topics, accumulator, with_std=False, with_support=False):
    """For each topic segmentation, compute average cosine similarity using a
    WordVectorsAccumulator.

    Args:
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
        list : of word2vec cosine similarities per topic.
    """
    topic_coherences = []
    total_oov = 0

    for topic_index, topic_segments in enumerate(segmented_topics):
        segment_sims = []
        num_oov = 0
        for w_prime, w_star in topic_segments:
            if not hasattr(w_prime, '__iter__'):
                w_prime = [w_prime]
            if not hasattr(w_star, '__iter__'):
                w_star = [w_star]

            try:
                segment_sims.append(accumulator.ids_similarity(w_prime, w_star))
            except ZeroDivisionError:
                num_oov += 1

        if num_oov > 0:
            total_oov += 1
            logger.warning(
                "%d terms for topic %d are not in word2vec model vocabulary",
                num_oov, topic_index)
        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))

    if total_oov > 0:
        logger.warning("%d terms for are not in word2vec model vocabulary", total_oov)
    return topic_coherences


def cosine_similarity(
        segmented_topics, accumulator, topics, measure='nlr', gamma=1,
        with_std=False, with_support=False):
    """
    This function calculates the indirect cosine measure. Given context vectors
    u = V(W') and w = V(W*) for the word sets of a pair S_i = (W', W*) indirect
    cosine measure is computed as the cosine similarity between u and w.

    The formula used is:

        m_{sim}_{(m, \gamma)}(W', W*) =
            s_{sim}(\vec{V}^{\,}_{m,\gamma}(W'), \vec{V}^{\,}_{m,\gamma}(W*))

    where each vector:

        \vec{V}^{\,}_{m,\gamma}(W') =
            \Bigg \{{\sum_{w_{i} \in W'}^{ } m(w_{i}, w_{j})^{\gamma}}\Bigg \}_{j = 1,...,|W|}

    Args:
        segmented_topics : Output from the segmentation module of the segmented topics.
            Is a list of list of tuples.
        accumulator : Output from the probability_estimation module. Is an accumulator
            of word occurrences (see text_analysis module).
        topics : Topics obtained from the trained topic model.
        measure (str): Direct confirmation measure to be used. Supported values are
            "nlr" (normalized log ratio).
        gamma : Gamma value for computing W', W* vectors; default is 1.
        with_std (bool): True to also include standard deviation across topic segment
            sets in addition to the mean coherence for each topic; default is False.
        with_support (bool): True to also include support across topic segments. The
            support is defined as the number of pairwise similarity comparisons were
            used to compute the overall topic coherence.

    Returns:
        list : of indirect cosine similarity measure for each topic.

    """
    context_vectors = ContextVectorComputer(measure, topics, accumulator, gamma)

    topic_coherences = []
    for topic_words, topic_segments in zip(topics, segmented_topics):
        topic_words = tuple(topic_words)  # because tuples are hashable
        segment_sims = np.zeros(len(topic_segments))
        for i, (w_prime, w_star) in enumerate(topic_segments):
            w_prime_cv = context_vectors[w_prime, topic_words]
            w_star_cv = context_vectors[w_star, topic_words]
            segment_sims[i] = _cossim(w_prime_cv, w_star_cv)

        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))

    return topic_coherences


class ContextVectorComputer(object):
    """Lazily compute context vectors for topic segments."""

    def __init__(self, measure, topics, accumulator, gamma):
        if measure == 'nlr':
            self.similarity = _pair_npmi
        else:
            raise ValueError(
                "The direct confirmation measure you entered is not currently supported.")

        self.mapping = _map_to_contiguous(topics)
        self.vocab_size = len(self.mapping)
        self.accumulator = accumulator
        self.gamma = gamma
        self.sim_cache = {}  # Cache similarities between tokens (pairs of word ids), e.g. (1, 2)
        self.context_vector_cache = {}  # mapping from (segment, topic_words) --> context_vector

    def __getitem__(self, idx):
        return self.compute_context_vector(*idx)

    def compute_context_vector(self, segment_word_ids, topic_word_ids):
        """
        Step 1. Check if (segment_word_ids, topic_word_ids) context vector has been cached.
        Step 2. If yes, return corresponding context vector, else compute, cache, and return.
        """
        key = _key_for_segment(segment_word_ids, topic_word_ids)
        context_vector = self.context_vector_cache.get(key, None)
        if context_vector is None:
            context_vector = self._make_seg(segment_word_ids, topic_word_ids)
            self.context_vector_cache[key] = context_vector
        return context_vector

    def _make_seg(self, segment_word_ids, topic_word_ids):
        """Internal helper function to return context vectors for segmentations."""
        context_vector = sps.lil_matrix((self.vocab_size, 1))
        if not hasattr(segment_word_ids, '__iter__'):
            segment_word_ids = (segment_word_ids,)

        for w_j in topic_word_ids:
            idx = (self.mapping[w_j], 0)
            for pair in (tuple(sorted((w_i, w_j))) for w_i in segment_word_ids):
                if pair not in self.sim_cache:
                    self.sim_cache[pair] = self.similarity(pair, self.accumulator)

                context_vector[idx] += self.sim_cache[pair] ** self.gamma

        return context_vector.tocsr()


def _pair_npmi(pair, accumulator):
    """Compute normalized pairwise mutual information (NPMI) between a pair of words.
    The pair is an iterable of (word_id1, word_id2).
    """
    return log_ratio_measure([[pair]], accumulator, True)[0]


def _cossim(cv1, cv2):
    return cv1.T.dot(cv2)[0, 0] / (_magnitude(cv1) * _magnitude(cv2))


def _magnitude(sparse_vec):
    return np.sqrt(np.sum(sparse_vec.data ** 2))


def _map_to_contiguous(ids_iterable):
    uniq_ids = {}
    n = 0
    for id_ in itertools.chain.from_iterable(ids_iterable):
        if id_ not in uniq_ids:
            uniq_ids[id_] = n
            n += 1
    return uniq_ids


def _key_for_segment(segment, topic_words):
    """A segment may have a single number of an iterable of them."""
    segment_key = tuple(segment) if hasattr(segment, '__iter__') else segment
    return segment_key, topic_words
