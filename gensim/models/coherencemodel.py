#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Module for calculating topic coherence in python. This is the implementation of
the four stage topic coherence pipeline from http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf.
The four stage pipeline is basically:

Segmentation -> Probability Estimation -> Confirmation Measure -> Aggregation.

Implementation of this pipeline allows for the user to in essence "make" a
coherence measure of his/her choice by choosing a method in each of the pipelines.
"""

import logging

from gensim import interfaces
from gensim import segmentation, probability_estimation, confirmation_measure, aggregation

logger = logging.getLogger(__name__)


class CoherenceModel(interfaces.TransformationABC):
    """
    Objects of this class allow for building and maintaining a model for topic
    coherence.

    The main methods are:

    1. constructor, which initializes the four stage pipeline by accepting a coherence measure,
    2. the ``get_coherence()`` method, which returns the topic coherence.

    Model persistency is achieved via its load/save methods.
    """
    def __init__(self, corpus, topics, coherence='u_mass'): # Have to validate coherence input. Check for invalid pipeline methods.
        self.corpus = corpus
        self.topics = topics
        if coherence == 'u_mass':
            self.seg = segmentation.S_One_Pre
            self.prob = probability_estimation.P_Boolean_Document
            self.conf = confirmation_measure.Log_Conditional_Probability
            self.aggr = aggregation.Arithmetic_Mean

    def __str__(self):
        return "CoherenceModel(segmentation=%s, probability estimation=%s, confirmation measure=%s, aggregation=%s)" % (
            self.seg, self.prob, self.conf, self.aggr)

    def get_coherence(self):
        segmented_topics = self.seg(self.topics)
        per_topic_prob = self.prob(self.corpus, segmented_topics)
        confirmed_measures = self.conf(segmented_topics, per_topic_prob)
        return self.aggr(confirmed_measures)
