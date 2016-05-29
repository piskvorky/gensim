#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains functions to perform segmentation on a list of topics.
"""

import logging
from itertools import chain

logger = logging.getLogger(__name__)

def P_Boolean_Document(corpus, segmented_topics): # Works only for S_One_One kind of segmentations
    """
    This function performs the boolean document probability estimation. Boolean document estimates the probability
    of a single word as the number of documents in which the word occurs divided by the total number of documents.

    Args:
    ----
    corpus : The corpus of documents.
    segmented_topics : Output from the segmentation of topics. Could be simply topics too.

    Returns:
    -------
    per_topic_prob : Boolean document probability estimation of all the unique topic ids.
    """
    top_ids = set() # is a set of all the unique ids contained in topics.
    for s_i in segmented_topics:
        for id in chain.from_iterable(s_i):
            top_ids.add(id)
    num_docs = len(corpus)
    # Perform boolean document now to create document word list.
    per_topic_prob = {}
    for id in top_ids:
        id_list = set()
        for n, document in enumerate(corpus):
            if id in frozenset(x[0] for x in document):
                id_list.add(n)
        per_topic_prob[id] = float(len(id_list)) / num_docs

    return per_topic_prob
