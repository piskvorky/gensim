#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains functions to perform segmentation on a list of topics.
"""

import itertools
import logging

from gensim.models._coherence.text_analysis import (
    CorpusAccumulator, WordOccurrenceAccumulator, ParallelWordOccurrenceAccumulator, WordVectorsAccumulator
)

logger = logging.getLogger(__name__)


def p_boolean_document(corpus, segmented_topics):
    """This function performs the boolean document probability estimation.
    Boolean document estimates the probability of a single word as the number
    of documents in which the word occurs divided by the total number of documents.

    Args:
        corpus : The corpus of documents.
        segmented_topics : Output from the segmentation of topics. Could be simply topics too.

    Returns:
        accumulator : word occurrence accumulator instance that can be used to lookup token
            frequencies and co-occurrence frequencies.
    """
    top_ids = unique_ids_from_segments(segmented_topics)
    return CorpusAccumulator(top_ids).accumulate(corpus)


def p_boolean_sliding_window(texts, segmented_topics, dictionary, window_size, processes=1):
    """This function performs the boolean sliding window probability estimation.
    Boolean sliding window determines word counts using a sliding window. The window
    moves over  the documents one word token per step. Each step defines a new virtual
    document  by copying the window content. Boolean document is applied to these virtual
    documents to compute word probabilities.

    Args:
        texts : List of string sentences.
        segmented_topics : Output from the segmentation of topics. Could be simply topics too.
        dictionary : Gensim dictionary mapping of the tokens and ids.
        window_size : Size of the sliding window. 110 found out to be the ideal size for large corpora.

    Returns:
        accumulator : word occurrence accumulator instance that can be used to lookup token
            frequencies and co-occurrence frequencies.
    """
    top_ids = unique_ids_from_segments(segmented_topics)
    if processes <= 1:
        accumulator = WordOccurrenceAccumulator(top_ids, dictionary)
    else:
        accumulator = ParallelWordOccurrenceAccumulator(processes, top_ids, dictionary)
    logger.info("using %s to estimate probabilities from sliding windows", accumulator)
    return accumulator.accumulate(texts, window_size)


def p_word2vec(texts, segmented_topics, dictionary, window_size=None, processes=1, model=None):
    """Train word2vec model on `texts` if model is not None.
    Returns:
    ----
    accumulator: text accumulator with trained context vectors.
    """
    top_ids = unique_ids_from_segments(segmented_topics)
    accumulator = WordVectorsAccumulator(
        top_ids, dictionary, model, window=window_size, workers=processes)
    return accumulator.accumulate(texts, window_size)


def unique_ids_from_segments(segmented_topics):
    """Return the set of all unique ids in a list of segmented topics.

    Args:
        segmented_topics: list of tuples of (word_id_set1, word_id_set2). Each word_id_set
            is either a single integer, or a `numpy.ndarray` of integers.
    Returns:
        unique_ids : set of unique ids across all topic segments.
    """
    unique_ids = set()  # is a set of all the unique ids contained in topics.
    for s_i in segmented_topics:
        for word_id in itertools.chain.from_iterable(s_i):
            if hasattr(word_id, '__iter__'):
                unique_ids.update(word_id)
            else:
                unique_ids.add(word_id)

    return unique_ids
