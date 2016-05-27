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

logger = logging.getLogger(__name__)

def S_One_Pre(topics):
    """
    This function performs S_One_Pre segmentation on a list of topics.
    S_One_Pre segmentation is defined as: S_One_Pre = {(W', W*) | W' = {w_i};
                                                                  W* = {w_j}; w_i, w_j belongs to W; i > j}
    Example usage:

        >>> topics = [np.array([9, 10, 7])
        >>> S_One_Pre(topics[0])
        {0: [(10, 9), (7, 9), (7, 10)]}

    Args:
    ----
    topics : list of topics obtained from an algorithm such as LDA. Is a list such as [array([ 9, 10, 11]), array([ 9, 10,  7]), ...]

    Returns:
    -------
    s_one_pre : Dictionary of list of (W', W*) tuples for all unique topic ids
    """
    s_one_pre = {}

    for t, top_words in enumerate(topics):
        s_one_pre_t = []
        for w_prime in top_words[1:]:
            w_prime_index = np.where(top_words == w_prime)[0] # To get index of w_star in top_words
            for w_star in top_words[:w_prime_index]:
                s_one_pre_t.append((w_prime, w_star))
        s_one_pre[t] = s_one_pre_t
    return s_one_pre
