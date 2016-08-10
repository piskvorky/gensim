#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains functions to perform segmentation on a list of topics.
"""

import logging
import numpy as np

from gensim.corpora import Dictionary

from itertools import chain, islice

logger = logging.getLogger(__name__)

def _ret_top_ids(segmented_topics):
    """
    Helper function to return a set of all the unique topic ids in segmented topics.
    """
    top_ids = set()  # is a set of all the unique ids contained in topics.
    for s_i in segmented_topics:
        for id in chain.from_iterable(s_i):
            if isinstance(id, np.ndarray):
                for i in id:
                    top_ids.add(i)
            else:
                top_ids.add(id)
    return top_ids

def p_boolean_document(corpus, segmented_topics):
    """
    This function performs the boolean document probability estimation. Boolean document estimates the probability
    of a single word as the number of documents in which the word occurs divided by the total number of documents.

    Args:
    ----
    corpus : The corpus of documents.
    segmented_topics : Output from the segmentation of topics. Could be simply topics too.

    Returns:
    -------
    per_topic_postings : Boolean document posting list for each unique topic id.
    num_docs : Total number of documents in corpus.
    """
    top_ids = _ret_top_ids(segmented_topics)
    # Perform boolean document now to create document word list.
    per_topic_postings = {}
    for id in top_ids:
        id_list = set()
        for n, document in enumerate(corpus):
            if id in frozenset(x[0] for x in document):
                id_list.add(n)
        per_topic_postings[id] = id_list
    num_docs = len(corpus)
    return (per_topic_postings, num_docs)

def p_boolean_sliding_window(texts, segmented_topics, dictionary, window_size):
    """
    This function performs the boolean sliding window probability estimation. Boolean sliding window
    determines word counts using a sliding window. The window moves over the documents one word token per step.
    Each step defines a new virtual document by copying the window content. Boolean document is applied to
    these virtual documents to compute word probabilities.

    Args:
    ----
    texts : List of string sentences.
    segmented_topics : Output from the segmentation of topics. Could be simply topics too.
    dictionary : Gensim dictionary mapping of the tokens and ids.
    window_size : Size of the sliding window. 110 found out to be the ideal size for large corpora.

    Returns:
    -------
    per_topic_postings : Boolean sliding window postings list of all the unique topic ids.
    window_id[0] : Total no of windows
    """
    top_ids = _ret_top_ids(segmented_topics)
    window_id = 0  # Each window assigned a window id.
    per_topic_postings = {}
    token2id_dict = dictionary.token2id
    def add_topic_posting(top_ids, window, per_topic_postings, window_id, token2id_dict):
        for word in window:
            word_id = token2id_dict[word]
            if word_id in top_ids:
                if word_id in per_topic_postings:
                    per_topic_postings[word_id].add(window_id)
                else:
                    per_topic_postings[word_id] = set([window_id])
        window_id += 1
        return (window_id, per_topic_postings)
    # Apply boolean sliding window to each document in texts.
    for document in texts:
        it = iter(document)
        window = tuple(islice(it, window_size))
        window_id, per_topic_postings = add_topic_posting(top_ids, window, per_topic_postings, window_id, token2id_dict)
        for elem in it:
            window = window[1:] + (elem,)
            window_id, per_topic_postings = add_topic_posting(top_ids, window, per_topic_postings, window_id, token2id_dict)

    return per_topic_postings, window_id
