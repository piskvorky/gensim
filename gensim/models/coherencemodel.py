#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Module for calculating topic coherence in python. This is the implementation of
the four stage topic coherence pipeline from the paper [1].
The four stage pipeline is basically:

Segmentation -> Probability Estimation -> Confirmation Measure -> Aggregation.

Implementation of this pipeline allows for the user to in essence "make" a
coherence measure of his/her choice by choosing a method in each of the pipelines.

[1] Michael Roeder, Andreas Both and Alexander Hinneburg. Exploring the space of topic
coherence measures. http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf.
"""

import logging

from gensim import interfaces
from gensim.topic_coherence import (segmentation, probability_estimation,
                                    direct_confirmation_measure, indirect_confirmation_measure,
                                    aggregation)
from gensim.corpora import Dictionary
from gensim.matutils import argsort
from gensim.utils import is_corpus
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaVowpalWabbit

logger = logging.getLogger(__name__)


class CoherenceModel(interfaces.TransformationABC):
    """
    Objects of this class allow for building and maintaining a model for topic
    coherence.

    The main methods are:

    1. constructor, which initializes the four stage pipeline by accepting a coherence measure,
    2. the ``get_coherence()`` method, which returns the topic coherence.

    >>> cm = CoherenceModel(model=tm, corpus=corpus, coherence='u_mass')  # tm is the trained topic model
    >>> cm.get_coherence()

    Model persistency is achieved via its load/save methods.
    """
    def __init__(self, model, texts=None, corpus=None, dictionary=None, coherence='c_v'):
        """
        Args:
        ----
        model : Pre-trained topic model.
        texts : Tokenized texts. Needed for coherence models that use sliding window based probability estimator.
        corpus : Gensim document corpus.
        dictionary : Gensim dictionary mapping of id word to create corpus.
        coherence : Coherence measure to be used. Supported values are:
                    u_mass
                    c_v
        """
        if texts is None and corpus is None:
            raise ValueError("One of texts or corpus has to be provided.")
        if coherence == 'u_mass':
            if is_corpus(corpus)[0]:
                if dictionary is None:
                    if model.id2word[0] == 0:
                        raise ValueError("The associated dictionary should be provided with the corpus or 'id2word' for topic model"
                                         "should be set as the dictionary.")
                    else:
                        self.dictionary = model.id2word
                else:
                    self.dictionary = dictionary
                self.corpus = corpus
            elif texts is not None:
                self.texts = texts
                if dictionary is None:
                    self.dictionary = Dictionary(self.texts)
                else:
                    self.dictionary = dictionary
                self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
            else:
                raise ValueError("Either 'corpus' with 'dictionary' or 'texts' should be provided for %s coherence." % coherence)

        elif coherence == 'c_v':
            if texts is None:
                raise ValueError("'texts' should be provided for %s coherence." % coherence)
            else:
                self.texts = texts
                self.dictionary = Dictionary(self.texts)
                self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]

        else:
            raise ValueError("%s coherence is not currently supported." % coherence)

        self.model = model
        self.topics = self._get_topics()
        self.coherence = coherence
        # Set pipeline parameters:
        if self.coherence == 'u_mass':
            self.seg = segmentation.s_one_pre
            self.prob = probability_estimation.p_boolean_document
            self.conf = direct_confirmation_measure.log_conditional_probability
            self.aggr = aggregation.arithmetic_mean

        elif self.coherence == 'c_v':
            self.seg = segmentation.s_one_set
            self.prob = probability_estimation.p_boolean_sliding_window
            self.conf = indirect_confirmation_measure.cosine_similarity
            self.aggr = aggregation.arithmetic_mean

    def __str__(self):
        return "CoherenceModel(segmentation=%s, probability estimation=%s, confirmation measure=%s, aggregation=%s)" % (
            self.seg, self.prob, self.conf, self.aggr)

    def _get_topics(self):
        """Internal helper function to return topics from a trained topic model."""
        topics = []  # FIXME : Meant to work for LDAModel, LdaVowpalWabbit right now. Make it work for others.
        if isinstance(self.model, LdaModel):
            for topic in self.model.state.get_lambda():
                bestn = argsort(topic, topn=10, reverse=True)
                topics.append(bestn)
        elif isinstance(self.model, LdaVowpalWabbit):
            for topic in self.model._get_topics():
                bestn = argsort(topic, topn=10, reverse=True)
                topics.append(bestn)
        return topics

    def get_coherence(self):
        if self.coherence == 'u_mass':
            segmented_topics = self.seg(self.topics)
            per_topic_postings, num_docs = self.prob(self.corpus, segmented_topics)
            confirmed_measures = self.conf(segmented_topics, per_topic_postings, num_docs)
            return self.aggr(confirmed_measures)

        elif self.coherence == 'c_v':
            segmented_topics = self.seg(self.topics)
            per_topic_postings, num_windows = self.prob(texts=self.texts, segmented_topics=segmented_topics,
                                                        dictionary=self.dictionary, window_size=2)  # FIXME : Change window size to 110 finally.
            confirmed_measures = self.conf(self.topics, segmented_topics, per_topic_postings, 'nlr', 1, num_windows)
            return self.aggr(confirmed_measures)
