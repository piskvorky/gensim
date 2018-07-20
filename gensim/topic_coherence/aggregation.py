#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains functions to perform aggregation on a list of values obtained from the confirmation measure."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def arithmetic_mean(confirmed_measures):
    """
    Perform the arithmetic mean aggregation on the output obtained from
    the confirmation measure module.

    Parameters
    ----------
    confirmed_measures : list of float
        List of calculated confirmation measure on each set in the segmented topics.

    Returns
    -------
    `numpy.float`
        Arithmetic mean of all the values contained in confirmation measures.

    Examples
    --------
    >>> from gensim.topic_coherence.aggregation import arithmetic_mean
    >>> arithmetic_mean([1.1, 2.2, 3.3, 4.4])
    2.75

    """
    return np.mean(confirmed_measures)
