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
import multiprocessing as mp
from collections import namedtuple

import numpy as np

from gensim import interfaces
from gensim.matutils import argsort
from gensim.topic_coherence import (segmentation, probability_estimation,
                                    direct_confirmation_measure, indirect_confirmation_measure,
                                    aggregation)
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments
from gensim.utils import is_corpus, FakeDict

logger = logging.getLogger(__name__)

BOOLEAN_DOCUMENT_BASED = {'u_mass'}
SLIDING_WINDOW_BASED = {'c_v', 'c_uci', 'c_npmi', 'c_w2v'}

_make_pipeline = namedtuple('Coherence_Measure', 'seg, prob, conf, aggr')
COHERENCE_MEASURES = {
    'u_mass': _make_pipeline(
        segmentation.s_one_pre,
        probability_estimation.p_boolean_document,
        direct_confirmation_measure.log_conditional_probability,
        aggregation.arithmetic_mean
    ),
    'c_v': _make_pipeline(
        segmentation.s_one_set,
        probability_estimation.p_boolean_sliding_window,
        indirect_confirmation_measure.cosine_similarity,
        aggregation.arithmetic_mean
    ),
    'c_w2v': _make_pipeline(
        segmentation.s_one_set,
        probability_estimation.p_word2vec,
        indirect_confirmation_measure.word2vec_similarity,
        aggregation.arithmetic_mean
    ),
    'c_uci': _make_pipeline(
        segmentation.s_one_one,
        probability_estimation.p_boolean_sliding_window,
        direct_confirmation_measure.log_ratio_measure,
        aggregation.arithmetic_mean
    ),
    'c_npmi': _make_pipeline(
        segmentation.s_one_one,
        probability_estimation.p_boolean_sliding_window,
        direct_confirmation_measure.log_ratio_measure,
        aggregation.arithmetic_mean
    ),
}

SLIDING_WINDOW_SIZES = {
    'c_v': 110,
    'c_w2v': 5,
    'c_uci': 10,
    'c_npmi': 10,
    'u_mass': None
}


class CoherenceModel(interfaces.TransformationABC):
    """Objects of this class allow for building and maintaining a model for topic coherence.

    The main methods are:

    1. constructor, which initializes the four stage pipeline by accepting a coherence measure,
    2. the ``get_coherence()`` method, which returns the topic coherence.

    Pipeline phases can also be executed individually. Methods for doing this are:

    1. `segment_topics()`, which performs segmentation of the given topics into their comparison sets.
    2. `estimate_probabilities()`, which accumulates word occurrence stats from the given corpus or texts.
        The output of this is also cached on the `CoherenceModel`, so calling this method can be used as
        a precomputation step for the next phase.
    3. `get_coherence_per_topic()`, which uses the segmented topics and estimated probabilities to compute
        the coherence of each topic. This output can be used to rank topics in order of most coherent to
        least. Such a ranking is useful if the intended use case of a topic model is document exploration
        by a human. It is also useful for filtering out incoherent topics (keep top-n from ranked list).
    4. `aggregate_measures(topic_coherences)`, which uses the pipeline's aggregation method to compute
        the overall coherence from the topic coherences.

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
    def __init__(self, model=None, topics=None, texts=None, corpus=None, dictionary=None,
                 window_size=None, keyed_vectors=None, coherence='c_v', topn=10, processes=-1):
        """
        Args:
            model : Pre-trained topic model. Should be provided if topics is not provided.
                Currently supports LdaModel, LdaMallet wrapper and LdaVowpalWabbit wrapper. Use 'topics'
                parameter to plug in an as yet unsupported model.
            topics : List of tokenized topics. If this is preferred over model, dictionary should be provided.
                eg::

                    topics = [['human', 'machine', 'computer', 'interface'],
                               ['graph', 'trees', 'binary', 'widths']]

            texts : Tokenized texts. Needed for coherence models that use sliding window based probability estimator,
                eg::

                    texts = [['system', 'human', 'system', 'eps'],
                             ['user', 'response', 'time'],
                             ['trees'],
                             ['graph', 'trees'],
                             ['graph', 'minors', 'trees'],
                             ['graph', 'minors', 'survey']]

            corpus : Gensim document corpus.
            dictionary : Gensim dictionary mapping of id word to create corpus. If model.id2word is present,
                this is not needed. If both are provided, dictionary will be used.
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
                For 'u_mass' corpus should be provided. If texts is provided, it will be converted
                to corpus using the dictionary. For 'c_v', 'c_uci' and 'c_npmi' texts should be provided.
                Corpus is not needed.
            topn : Integer corresponding to the number of top words to be extracted from each topic.
            processes : number of processes to use for probability estimation phase; any value less than 1 will be
                interpreted to mean num_cpus - 1; default is -1.
        """
        if model is None and topics is None:
            raise ValueError("One of model or topics has to be provided.")
        elif topics is not None and dictionary is None:
            raise ValueError("dictionary has to be provided if topics are to be used.")

        self.keyed_vectors = keyed_vectors
        if keyed_vectors is None and texts is None and corpus is None:
            raise ValueError("One of texts or corpus has to be provided.")

        # Check if associated dictionary is provided.
        if dictionary is None:
            if isinstance(model.id2word, FakeDict):
                raise ValueError(
                    "The associated dictionary should be provided with the corpus or 'id2word'"
                    " for topic model should be set as the associated dictionary.")
            else:
                self.dictionary = model.id2word
        else:
            self.dictionary = dictionary

        # Check for correct inputs for u_mass coherence measure.
        self.coherence = coherence
        self.window_size = window_size
        if self.window_size is None:
            self.window_size = SLIDING_WINDOW_SIZES[self.coherence]
        self.texts = texts
        self.corpus = corpus

        if coherence in BOOLEAN_DOCUMENT_BASED:
            if is_corpus(corpus)[0]:
                self.corpus = corpus
            elif self.texts is not None:
                self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
            else:
                raise ValueError(
                    "Either 'corpus' with 'dictionary' or 'texts' should "
                    "be provided for %s coherence.", coherence)

        # Check for correct inputs for sliding window coherence measure.
        elif coherence == 'c_w2v' and keyed_vectors is not None:
            pass
        elif coherence in SLIDING_WINDOW_BASED:
            if self.texts is None:
                raise ValueError("'texts' should be provided for %s coherence.", coherence)
        else:
            raise ValueError("%s coherence is not currently supported.", coherence)

        self.topn = topn
        self._model = model
        self._accumulator = None
        self._topics = None
        self.topics = topics

        self.processes = processes if processes > 1 else max(1, mp.cpu_count() - 1)

    def __str__(self):
        return str(self.measure)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        if model is not None:
            new_topics = self._get_topics()
            self._update_accumulator(new_topics)
            self._topics = new_topics

    @property
    def measure(self):
        return COHERENCE_MEASURES[self.coherence]

    @property
    def topics(self):
        return self._topics

    @topics.setter
    def topics(self, topics):
        new_topics = None
        if self.model is not None:
            new_topics = self._get_topics()
            if topics is not None:
                logger.warning(
                    "Ignoring topics you are attempting to set in favor of model's topics: %s",
                    self.model)
        elif topics is not None:
            new_topics = []
            for topic in topics:
                topic_token_ids = np.array([self.dictionary.token2id[token] for token in topic])
                new_topics.append(topic_token_ids)

        self._update_accumulator(new_topics)
        self._topics = new_topics

    def _update_accumulator(self, new_topics):
        if self._relevant_ids_will_differ(new_topics):
            logger.debug("Wiping cached accumulator since it does not contain all relevant ids.")
            self._accumulator = None

    def _relevant_ids_will_differ(self, new_topics):
        if self._accumulator is None or not self._topics_differ(new_topics):
            return False

        new_set = unique_ids_from_segments(self.measure.seg(new_topics))
        return not self._accumulator.relevant_ids.issuperset(new_set)

    def _topics_differ(self, new_topics):
        return (new_topics is not None and
                self._topics is not None and
                not np.array_equal(new_topics, self._topics))

    def _get_topics(self):
        """Internal helper function to return topics from a trained topic model."""
        try:
            return [
                argsort(topic, topn=self.topn, reverse=True) for topic in
                self.model.get_topics()
            ]
        except AttributeError:
            raise ValueError(
                "This topic model is not currently supported. Supported topic models"
                " should implement the `get_topics` method.")

    def segment_topics(self):
        return self.measure.seg(self.topics)

    def estimate_probabilities(self, segmented_topics=None):
        """Accumulate word occurrences and co-occurrences from texts or corpus using
        the optimal method for the chosen coherence metric. This operation may take
        quite some time for the sliding window based coherence methods.
        """
        if segmented_topics is None:
            segmented_topics = self.segment_topics()

        if self.coherence in BOOLEAN_DOCUMENT_BASED:
            self._accumulator = self.measure.prob(self.corpus, segmented_topics)
        else:
            kwargs = dict(
                texts=self.texts, segmented_topics=segmented_topics,
                dictionary=self.dictionary, window_size=self.window_size,
                processes=self.processes)
            if self.coherence == 'c_w2v':
                kwargs['model'] = self.keyed_vectors

            self._accumulator = self.measure.prob(**kwargs)

        return self._accumulator

    def get_coherence_per_topic(self, segmented_topics=None, with_std=False):
        """Return list of coherence values for each topic based on pipeline parameters."""
        measure = self.measure
        if segmented_topics is None:
            segmented_topics = measure.seg(self.topics)
        if self._accumulator is None:
            self.estimate_probabilities(segmented_topics)

        if self.coherence in BOOLEAN_DOCUMENT_BASED or self.coherence == 'c_w2v':
            kwargs = dict(with_std=with_std)
        elif self.coherence == 'c_v':
            kwargs = dict(topics=self.topics, measure='nlr', gamma=1)
        else:
            kwargs = dict(normalize=(self.coherence == 'c_npmi'))

        return measure.conf(segmented_topics, self._accumulator, **kwargs)

    def aggregate_measures(self, topic_coherences):
        """Aggregate the individual topic coherence measures using
        the pipeline's aggregation function.
        """
        return self.measure.aggr(topic_coherences)

    def get_coherence(self):
        """Return coherence value based on pipeline parameters."""
        confirmed_measures = self.get_coherence_per_topic()
        return self.aggregate_measures(confirmed_measures)
