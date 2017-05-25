#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Module for calculating topic coherence in python. This is the implementation of
the four stage topic coherence pipeline from the paper [1]_.
The four stage pipeline is basically:

Segmentation -> Probability Estimation -> Confirmation Measure -> Aggregation.

Implementation of this pipeline allows for the user to in essence "make" a
coherence measure of his/her choice by choosing a method in each of the pipelines.

.. [1] Michael Roeder, Andreas Both and Alexander Hinneburg. Exploring the space of topic
coherence measures. http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf.
"""

import logging
from collections import namedtuple

import numpy as np

from gensim import interfaces
from gensim.matutils import argsort
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaVowpalWabbit, LdaMallet
from gensim.topic_coherence import (segmentation, probability_estimation,
                                    direct_confirmation_measure, indirect_confirmation_measure,
                                    aggregation)
from gensim.utils import is_corpus, FakeDict

logger = logging.getLogger(__name__)

boolean_document_based = ['u_mass']
sliding_window_based = ['c_v', 'c_uci', 'c_npmi']
make_pipeline = namedtuple('Coherence_Measure', 'seg, prob, conf, aggr')

coherence_dict = {
    'u_mass': make_pipeline(segmentation.s_one_pre,
                            probability_estimation.p_boolean_document,
                            direct_confirmation_measure.log_conditional_probability,
                            aggregation.arithmetic_mean),
    'c_v': make_pipeline(segmentation.s_one_set,
                         probability_estimation.p_boolean_sliding_window,
                         indirect_confirmation_measure.cosine_similarity,
                         aggregation.arithmetic_mean),
    'c_uci': make_pipeline(segmentation.s_one_one,
                           probability_estimation.p_boolean_sliding_window,
                           direct_confirmation_measure.log_ratio_measure,
                           aggregation.arithmetic_mean),
    'c_npmi': make_pipeline(segmentation.s_one_one,
                            probability_estimation.p_boolean_sliding_window,
                            direct_confirmation_measure.log_ratio_measure,
                            aggregation.arithmetic_mean),
}

sliding_windows_dict = {
    'c_v': 110,
    'c_uci': 10,
    'c_npmi': 10
}


class CoherenceModel(interfaces.TransformationABC):
    """
    Objects of this class allow for building and maintaining a model for topic
    coherence.

    The main methods are:

    1. constructor, which initializes the four stage pipeline by accepting a coherence measure,
    2. the ``get_coherence()`` method, which returns the topic coherence.

    One way of using this feature is through providing a trained topic model. A dictionary has to be explicitly
    provided if the model does not contain a dictionary already::

        cm = CoherenceModel(model=tm, corpus=corpus, coherence='u_mass')  # tm is the trained topic model
        cm.get_coherence()

    Another way of using this feature is through providing tokenized topics such as::

        topics = [['human', 'computer', 'system', 'interface'],
                  ['graph', 'minors', 'trees', 'eps']]
        cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass') # note that a dictionary has to be provided.
        cm.get_coherence()

    Model persistency is achieved via its load/save methods.
    """
    def __init__(self, model=None, topics=None, texts=None, corpus=None, dictionary=None, window_size=None,
                 coherence='c_v', topn=10):
        """
        Args:
        ----
        model : Pre-trained topic model. Should be provided if topics is not provided.
                Currently supports LdaModel, LdaMallet wrapper and LdaVowpalWabbit wrapper. Use 'topics'
                parameter to plug in an as yet unsupported model.
        topics : List of tokenized topics. If this is preferred over model, dictionary should be provided. eg::
                 topics = [['human', 'machine', 'computer', 'interface'],
                               ['graph', 'trees', 'binary', 'widths']]
        texts : Tokenized texts. Needed for coherence models that use sliding window based probability estimator, eg::
                texts = [['system', 'human', 'system', 'eps'],
                             ['user', 'response', 'time'],
                             ['trees'],
                             ['graph', 'trees'],
                             ['graph', 'minors', 'trees'],
                             ['graph', 'minors', 'survey']]
        corpus : Gensim document corpus.
        dictionary : Gensim dictionary mapping of id word to create corpus. If model.id2word is present, this is not needed.
                     If both are provided, dictionary will be used.
        window_size : Is the size of the window to be used for coherence measures using boolean sliding window as their
                      probability estimator. For 'u_mass' this doesn't matter.
                      If left 'None' the default window sizes are used which are:
                      'c_v' : 110
                      'c_uci' : 10
                      'c_npmi' : 10
        coherence : Coherence measure to be used. Supported values are:
                    'u_mass'
                    'c_v'
                    'c_uci' also popularly known as c_pmi
                    'c_npmi'
                    For 'u_mass' corpus should be provided. If texts is provided, it will be converted to corpus using the dictionary.
                    For 'c_v', 'c_uci' and 'c_npmi' texts should be provided. Corpus is not needed.
        topn : Integer corresponding to the number of top words to be extracted from each topic.
        """
        if model is None and topics is None:
            raise ValueError("One of model or topics has to be provided.")
        elif topics is not None and dictionary is None:
            raise ValueError("dictionary has to be provided if topics are to be used.")

        if texts is None and corpus is None:
            raise ValueError("One of texts or corpus has to be provided.")

        # Check if associated dictionary is provided.
        if dictionary is None:
            if isinstance(model.id2word, FakeDict):
                raise ValueError("The associated dictionary should be provided with the corpus or 'id2word' for topic model"
                                 " should be set as the associated dictionary.")
            else:
                self.dictionary = model.id2word
        else:
            self.dictionary = dictionary

        # Check for correct inputs for u_mass coherence measure.
        self.coherence = coherence
        if coherence in boolean_document_based:
            if is_corpus(corpus)[0]:
                self.corpus = corpus
            elif texts is not None:
                self.texts = texts
                self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
            else:
                raise ValueError("Either 'corpus' with 'dictionary' or 'texts' should be provided for %s coherence." % coherence)

        # Check for correct inputs for c_v coherence measure.
        elif coherence in sliding_window_based:
            self.window_size = window_size
            if self.window_size is None:
                self.window_size = sliding_windows_dict[self.coherence]
            if texts is None:
                raise ValueError("'texts' should be provided for %s coherence." % coherence)
            else:
                self.texts = texts
        else:
            raise ValueError("%s coherence is not currently supported." % coherence)

        self.topn = topn
        self.model = model
        if model is not None:
            self.topics = self._get_topics()
        elif topics is not None:
            self.topics = []
            for topic in topics:
                t_i = []
                for n, _ in enumerate(topic):
                    t_i.append(dictionary.token2id[topic[n]])
                self.topics.append(np.array(t_i))

        self._accumulator = None

    def __str__(self):
        return coherence_dict[self.coherence].__str__()

    def _get_topics(self):
        """Internal helper function to return topics from a trained topic model."""
        topics = []
        if isinstance(self.model, LdaModel):
            for topic in self.model.state.get_lambda():
                bestn = argsort(topic, topn=self.topn, reverse=True)
                topics.append(bestn)
        elif isinstance(self.model, LdaVowpalWabbit):
            for topic in self.model._get_topics():
                bestn = argsort(topic, topn=self.topn, reverse=True)
                topics.append(bestn)
        elif isinstance(self.model, LdaMallet):
            for topic in self.model.word_topics:
                bestn = argsort(topic, topn=self.topn, reverse=True)
                topics.append(bestn)
        else:
            raise ValueError("This topic model is not currently supported. Supported topic models are"
                             "LdaModel, LdaVowpalWabbit and LdaMallet.")
        return topics

    def get_coherence_per_topic(self):
        measure = coherence_dict[self.coherence]
        segmented_topics = measure.seg(self.topics)

        if self.coherence in boolean_document_based:
            self._accumulator = measure.prob(self.corpus, segmented_topics)
            return measure.conf(segmented_topics, self._accumulator)

        self._accumulator = measure.prob(texts=self.texts, segmented_topics=segmented_topics,
                                         dictionary=self.dictionary, window_size=self.window_size)
        if self.coherence == 'c_v':
            return measure.conf(self.topics, segmented_topics, self._accumulator, 'nlr', 1)
        else:
            normalize = self.coherence == 'c_npmi'
            return measure.conf(segmented_topics, self._accumulator, normalize=normalize)

    def aggregate_measures(self, confirmed_measures):
        measure = coherence_dict[self.coherence]
        return measure.aggr(confirmed_measures)

    def get_coherence(self):
        """Return coherence value based on pipeline parameters."""
        confirmed_measures = self.get_coherence_per_topic()
        return self.aggregate_measures(confirmed_measures)
