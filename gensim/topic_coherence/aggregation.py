#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains functions to perform aggregation on a list of values
obtained from the confirmation measure.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

def arithmetic_mean(confirmed_measures):
    """
    This functoin performs the arithmetic mean aggregation on the output obtained from
    the confirmation measure module.

    Args:
        confirmed_measures : list of calculated confirmation measure on each set in the segmented topics.

    Returns:
        mean : Arithmetic mean of all the values contained in confirmation measures.
    """
    return np.mean(confirmed_measures)
