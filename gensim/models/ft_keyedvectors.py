#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from gensim.models.word2vec import KeyedVectors

import logging
logger = logging.getLogger(__name__)


class FastTextKeyedVectors(KeyedVectors):
    """
    Class to contain vectors, vocab and ngrams for the FastText training class and other methods not directly
    involved in training such as most_similar().
    Subclasses KeyedVectors to implement oov lookups, storing ngrams and other FastText specific methods

    """
    def __init__(self):
        super(FastTextKeyedVectors, self).__init__()
        self.syn0_all_norm = None
        self.ngrams = {}