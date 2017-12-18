#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains functions to perform segmentation on a list of topics."""

import logging

logger = logging.getLogger(__name__)


def s_one_pre(topics):
    """Performs s_one_pre segmentation on a list of topics.

    Notes
    -----
    s_one_pre segmentation is defined as
    :math:`s_{pre} = {(W', W^{*}) | W' = w_{i}; W^{*} = {w_j}; w_{i}, w_{j} \in W; i > j}`

    Parameters
    ----------
    topics : list of np.array
        list of topics obtained from an algorithm such as LDA. Is a list such as
        [array([ 9, 10, 11]), array([ 9, 10,  7]), ...]

    Returns
    -------
    list of list of (str, str).
        :math:`(W', W^{*})` for all unique topic ids.

    Examples
    --------
    >>> import numpy as np
    >>> from gensim.topic_coherence import segmentation
    >>> topics = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    >>> # should be [[(2, 1), (3, 1), (3, 2)], [(5, 4), (6, 4), (6, 5)]]
    >>> segmentation.s_one_pre(topics)

    """
    s_one_pre_res = []

    for top_words in topics:
        s_one_pre_t = []
        for w_prime_index, w_prime in enumerate(top_words[1:]):
            for w_star in top_words[:w_prime_index + 1]:
                s_one_pre_t.append((w_prime, w_star))
        s_one_pre_res.append(s_one_pre_t)

    return s_one_pre_res


def s_one_one(topics):
    """Perform s_one_one segmentation on a list of topics.
    s_one_one segmentation is defined as
    :math:`s_{one} = {(W', W^{*}) | W' = {w_i}; W^{*} = {w_j}; w_{i}, w_{j} \in W; i != j}` #TODO: neq - doesn't work

    Parameters
    ----------
    topics : list of np.array
        List of topics obtained from an algorithm such as LDA.
        Is a list such as [array([ 9, 10, 11]), array([ 9, 10,  7]), ...].

    Returns
    -------
    list of list of (str, str).
        :math:`(W', W^{*})` for all unique topic ids.

    Examples
    -------
    >>> import numpy as np
    >>> from gensim.topic_coherence import segmentation
    >>> topics = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    >>> # should be [[(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)], [(4, 5), (4, 6), (5, 4), (5, 6), (6, 4), (6, 5)]]
    >>> segmentation.s_one_pre(topics)
    """
    s_one_one_res = []

    for top_words in topics:
        s_one_one_t = []
        for w_prime_index, w_prime in enumerate(top_words):
            for w_star_index, w_star in enumerate(top_words):
                if w_prime_index == w_star_index:
                    continue
                else:
                    s_one_one_t.append((w_prime, w_star))
        s_one_one_res.append(s_one_one_t)

    return s_one_one_res


def s_one_set(topics):
    """Perform s_one_set segmentation on a list of topics.
    s_one_set segmentation is defined as :math:`s_{set} = {(W', W^{*}) | W' = {w_i}; w_{i} \in W; W^{*} = W}`

    Parameters
    ----------
    topics : list of np.array
        List of topics obtained from an algorithm such as LDA. Is a list such as
        [array([ 9, 10, 11]), array([ 9, 10,  7]), ...].

    Returns
    -------
    list of list of (str, str).
        :math:`(W', W^{*})` for all unique topic ids.

    Examples
    --------
    >>> import numpy as np
    >>> from gensim.topic_coherence import segmentation
    >>> topics = [np.array([9, 10, 7])]
    >>> # should be [[(9, array([ 9, 10,  7])), (10, array([ 9, 10,  7])), (7, array([ 9, 10,  7]))]]
    >>> segmentation.s_one_set(topics)

    """
    s_one_set_res = []

    for top_words in topics:
        s_one_set_t = []
        for w_prime in top_words:
            s_one_set_t.append((w_prime, top_words))
        s_one_set_res.append(s_one_set_t)

    return s_one_set_res
