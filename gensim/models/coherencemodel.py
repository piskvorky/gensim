#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Calculate topic coherence for topic models. This is the implementation of the four stage topic coherence pipeline
from the paper `Michael Roeder, Andreas Both and Alexander Hinneburg: "Exploring the space of topic coherence measures"
<http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf>`_.
Typically, :class:`~gensim.models.coherencemodel.CoherenceModel` used for evaluation of topic models.

The four stage pipeline is basically:

    * Segmentation
    * Probability Estimation
    * Confirmation Measure
    * Aggregation

Implementation of this pipeline allows for the user to in essence "make" a coherence measure of his/her choice
by choosing a method in each of the pipelines.

See Also
--------
:mod:`gensim.topic_coherence`
    Internal functions for pipelines.

"""
import logging
import multiprocessing as mp
from collections import namedtuple

import numpy as np

from gensim import interfaces, matutils
from gensim import utils
from gensim.topic_coherence import (segmentation, probability_estimation,
                                    direct_confirmation_measure, indirect_confirmation_measure,
                                    aggregation)
from gensim.topic_coherence.probability_estimation import unique_ids_from_segments

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

    Examples
    ---------
    One way of using this feature is through providing a trained topic model. A dictionary has to be explicitly provided
    if the model does not contain a dictionary already

    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_corpus, common_dictionary
        >>> from gensim.models.ldamodel import LdaModel
        >>> from gensim.models.coherencemodel import CoherenceModel
        >>>
        >>> model = LdaModel(common_corpus, 5, common_dictionary)
        >>>
        >>> cm = CoherenceModel(model=model, corpus=common_corpus, coherence='u_mass')
        >>> coherence = cm.get_coherence()  # get coherence value

    Another way of using this feature is through providing tokenized topics such as:

    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_corpus, common_dictionary
        >>> from gensim.models.coherencemodel import CoherenceModel
        >>> topics = [
        ...     ['human', 'computer', 'system', 'interface'],
        ...     ['graph', 'minors', 'trees', 'eps']
        ... ]
        >>>
        >>> cm = CoherenceModel(topics=topics, corpus=common_corpus, dictionary=common_dictionary, coherence='u_mass')
        >>> coherence = cm.get_coherence()  # get coherence value

    """
    def __init__(self, model=None, topics=None, texts=None, corpus=None, dictionary=None,
                 window_size=None, keyed_vectors=None, coherence='c_v', topn=20, processes=-1):
        """

        Parameters
        ----------
        model : :class:`~gensim.models.basemodel.BaseTopicModel`, optional
            Pre-trained topic model, should be provided if topics is not provided.
            Currently supports :class:`~gensim.models.ldamodel.LdaModel`,
            :class:`~gensim.models.ldamulticore.LdaMulticore`, :class:`~gensim.models.wrappers.ldamallet.LdaMallet` and
            :class:`~gensim.models.wrappers.ldavowpalwabbit.LdaVowpalWabbit`.
            Use `topics` parameter to plug in an as yet unsupported model.
        topics : list of list of str, optional
            List of tokenized topics, if this is preferred over model - dictionary should be provided.
        texts : list of list of str, optional
            Tokenized texts, needed for coherence models that use sliding window based (i.e. coherence=`c_something`)
            probability estimator .
        corpus : iterable of list of (int, number), optional
            Corpus in BoW format.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Gensim dictionary mapping of id word to create corpus.
            If `model.id2word` is present, this is not needed. If both are provided, passed `dictionary` will be used.
        window_size : int, optional
            Is the size of the window to be used for coherence measures using boolean sliding window as their
            probability estimator. For 'u_mass' this doesn't matter.
            If None - the default window sizes are used which are: 'c_v' - 110, 'c_uci' - 10, 'c_npmi' - 10.
        coherence : {'u_mass', 'c_v', 'c_uci', 'c_npmi'}, optional
            Coherence measure to be used.
            Fastest method - 'u_mass', 'c_uci' also known as `c_pmi`.
            For 'u_mass' corpus should be provided, if texts is provided, it will be converted to corpus
            using the dictionary. For 'c_v', 'c_uci' and 'c_npmi' `texts` should be provided (`corpus` isn't needed)
        topn : int, optional
            Integer corresponding to the number of top words to be extracted from each topic.
        processes : int, optional
            Number of processes to use for probability estimation phase, any value less than 1 will be interpreted as
            num_cpus - 1.

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
            if isinstance(model.id2word, utils.FakeDict):
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
            if utils.is_corpus(corpus)[0]:
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

        self._topn = topn
        self._model = model
        self._accumulator = None
        self._topics = None
        self.topics = topics

        self.processes = processes if processes >= 1 else max(1, mp.cpu_count() - 1)

    @classmethod
    def for_models(cls, models, dictionary, topn=20, **kwargs):
        """Initialize a CoherenceModel with estimated probabilities for all of the given models.
        Use :meth:`~gensim.models.coherencemodel.CoherenceModel.for_topics` method.

        Parameters
        ----------
        models : list of :class:`~gensim.models.basemodel.BaseTopicModel`
            List of models to evaluate coherence of, each of it should implements
            :meth:`~gensim.models.basemodel.BaseTopicModel.get_topics` method.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
            Gensim dictionary mapping of id word.
        topn : int, optional
            Integer corresponding to the number of top words to be extracted from each topic.
        kwargs : object
            Sequence of arguments, see :meth:`~gensim.models.coherencemodel.CoherenceModel.for_topics`.

        Return
        ------
        :class:`~gensim.models.coherencemodel.CoherenceModel`
            CoherenceModel with estimated probabilities for all of the given models.

        Example
        -------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import common_corpus, common_dictionary
            >>> from gensim.models.ldamodel import LdaModel
            >>> from gensim.models.coherencemodel import CoherenceModel
            >>>
            >>> m1 = LdaModel(common_corpus, 3, common_dictionary)
            >>> m2 = LdaModel(common_corpus, 5, common_dictionary)
            >>>
            >>> cm = CoherenceModel.for_models([m1, m2], common_dictionary, corpus=common_corpus, coherence='u_mass')
        """
        topics = [cls.top_topics_as_word_lists(model, dictionary, topn) for model in models]
        kwargs['dictionary'] = dictionary
        kwargs['topn'] = topn
        return cls.for_topics(topics, **kwargs)

    @staticmethod
    def top_topics_as_word_lists(model, dictionary, topn=20):
        """Get `topn` topics as list of words.

        Parameters
        ----------
        model : :class:`~gensim.models.basemodel.BaseTopicModel`
            Pre-trained topic model.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
            Gensim dictionary mapping of id word.
        topn : int, optional
            Integer corresponding to the number of top words to be extracted from each topic.

        Return
        ------
        list of list of str
            Top topics in list-of-list-of-words format.

        """
        if not dictionary.id2token:
            dictionary.id2token = {v: k for k, v in dictionary.token2id.items()}

        str_topics = []
        for topic in model.get_topics():
            bestn = matutils.argsort(topic, topn=topn, reverse=True)
            beststr = [dictionary.id2token[_id] for _id in bestn]
            str_topics.append(beststr)
        return str_topics

    @classmethod
    def for_topics(cls, topics_as_topn_terms, **kwargs):
        """Initialize a CoherenceModel with estimated probabilities for all of the given topics.

        Parameters
        ----------
        topics_as_topn_terms : list of list of str
            Each element in the top-level list should be the list of topics for a model.
            The topics for the model should be a list of top-N words, one per topic.

        Return
        ------
        :class:`~gensim.models.coherencemodel.CoherenceModel`
            CoherenceModel with estimated probabilities for all of the given models.

        """
        if not topics_as_topn_terms:
            raise ValueError("len(topics) must be > 0.")
        if any(len(topic_lists) == 0 for topic_lists in topics_as_topn_terms):
            raise ValueError("found empty topic listing in `topics`")

        topn = 0
        for topic_list in topics_as_topn_terms:
            for topic in topic_list:
                topn = max(topn, len(topic))

        topn = min(kwargs.pop('topn', topn), topn)
        super_topic = utils.flatten(topics_as_topn_terms)

        logging.info(
            "Number of relevant terms for all %d models: %d",
            len(topics_as_topn_terms), len(super_topic))
        cm = CoherenceModel(topics=[super_topic], topn=len(super_topic), **kwargs)
        cm.estimate_probabilities()
        cm.topn = topn
        return cm

    def __str__(self):
        return str(self.measure)

    @property
    def model(self):
        """Get `self._model` field.

        Return
        ------
        :class:`~gensim.models.basemodel.BaseTopicModel`
            Used model.

        """
        return self._model

    @model.setter
    def model(self, model):
        """Set `self._model` field.

        Parameters
        ----------
        model : :class:`~gensim.models.basemodel.BaseTopicModel`
            Input model.

        """
        self._model = model
        if model is not None:
            new_topics = self._get_topics()
            self._update_accumulator(new_topics)
            self._topics = new_topics

    @property
    def topn(self):
        """Get number of top words `self._topn`.

        Return
        ------
        int
            Integer corresponding to the number of top words.

        """
        return self._topn

    @topn.setter
    def topn(self, topn):
        """Set number of top words `self._topn`.

        Parameters
        ----------
        topn : int
            Number of top words.

        """
        current_topic_length = len(self._topics[0])
        requires_expansion = current_topic_length < topn

        if self.model is not None:
            self._topn = topn
            if requires_expansion:
                self.model = self._model  # trigger topic expansion from model
        else:
            if requires_expansion:
                raise ValueError("Model unavailable and topic sizes are less than topn=%d" % topn)
            self._topn = topn  # topics will be truncated in getter

    @property
    def measure(self):
        """Make pipeline, according to `coherence` parameter value.

        Return
        ------
        namedtuple
            Pipeline that contains needed functions/method for calculated coherence.

        """
        return COHERENCE_MEASURES[self.coherence]

    @property
    def topics(self):
        """Get topics `self._topics`.

        Return
        ------
        list of list of str
            Topics as list of tokens.

        """
        if len(self._topics[0]) > self._topn:
            return [topic[:self._topn] for topic in self._topics]
        else:
            return self._topics

    @topics.setter
    def topics(self, topics):
        """Set topics `self._topics`.

        Parameters
        ----------
        topics : list of list of str
            Topics.

        """
        if topics is not None:
            new_topics = []
            for topic in topics:
                topic_token_ids = self._ensure_elements_are_ids(topic)
                new_topics.append(topic_token_ids)

            if self.model is not None:
                logger.warning(
                    "The currently set model '%s' may be inconsistent with the newly set topics",
                    self.model)
        elif self.model is not None:
            new_topics = self._get_topics()
            logger.debug("Setting topics to those of the model: %s", self.model)
        else:
            new_topics = None

        self._update_accumulator(new_topics)
        self._topics = new_topics

    def _ensure_elements_are_ids(self, topic):
        try:
            return np.array([self.dictionary.token2id[token] for token in topic])
        except KeyError:  # might be a list of token ids already, but let's verify all in dict
            topic = (self.dictionary.id2token[_id] for _id in topic)
            return np.array([self.dictionary.token2id[token] for token in topic])

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
        return (new_topics is not None
                and self._topics is not None
                and not np.array_equal(new_topics, self._topics))

    def _get_topics(self):
        """Internal helper function to return topics from a trained topic model."""
        return self._get_topics_from_model(self.model, self.topn)

    @staticmethod
    def _get_topics_from_model(model, topn):
        """Internal helper function to return topics from a trained topic model.

        Parameters
        ----------
        model : :class:`~gensim.models.basemodel.BaseTopicModel`
            Pre-trained topic model.
        topn : int
            Integer corresponding to the number of top words.

        Return
        ------
        list of :class:`numpy.ndarray`
            Topics matrix

        """
        try:
            return [
                matutils.argsort(topic, topn=topn, reverse=True) for topic in
                model.get_topics()
            ]
        except AttributeError:
            raise ValueError(
                "This topic model is not currently supported. Supported topic models"
                " should implement the `get_topics` method.")

    def segment_topics(self):
        """Segment topic, alias for `self.measure.seg(self.topics)`.

        Return
        ------
        list of list of pair
            Segmented topics.

        """
        return self.measure.seg(self.topics)

    def estimate_probabilities(self, segmented_topics=None):
        """Accumulate word occurrences and co-occurrences from texts or corpus using the optimal method for the chosen
        coherence metric.

        Notes
        -----
        This operation may take quite some time for the sliding window based coherence methods.

        Parameters
        ----------
        segmented_topics : list of list of pair, optional
            Segmented topics, typically produced by :meth:`~gensim.models.coherencemodel.CoherenceModel.segment_topics`.

        Return
        ------
        :class:`~gensim.topic_coherence.text_analysis.CorpusAccumulator`
            Corpus accumulator.

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

    def get_coherence_per_topic(self, segmented_topics=None, with_std=False, with_support=False):
        """Get list of coherence values for each topic based on pipeline parameters.

        Parameters
        ----------
        segmented_topics : list of list of (int, number)
            Topics.
        with_std : bool, optional
            True to also include standard deviation across topic segment sets in addition to the mean coherence
            for each topic.
        with_support : bool, optional
            True to also include support across topic segments. The support is defined as the number of pairwise
            similarity comparisons were used to compute the overall topic coherence.

        Return
        ------
        list of float
            Sequence of similarity measure for each topic.

        """
        measure = self.measure
        if segmented_topics is None:
            segmented_topics = measure.seg(self.topics)
        if self._accumulator is None:
            self.estimate_probabilities(segmented_topics)

        kwargs = dict(with_std=with_std, with_support=with_support)
        if self.coherence in BOOLEAN_DOCUMENT_BASED or self.coherence == 'c_w2v':
            pass
        elif self.coherence == 'c_v':
            kwargs['topics'] = self.topics
            kwargs['measure'] = 'nlr'
            kwargs['gamma'] = 1
        else:
            kwargs['normalize'] = (self.coherence == 'c_npmi')

        return measure.conf(segmented_topics, self._accumulator, **kwargs)

    def aggregate_measures(self, topic_coherences):
        """Aggregate the individual topic coherence measures using the pipeline's aggregation function.
        Use `self.measure.aggr(topic_coherences)`.

        Parameters
        ----------
        topic_coherences : list of float
            List of calculated confirmation measure on each set in the segmented topics.

        Returns
        -------
        float
            Arithmetic mean of all the values contained in confirmation measures.

        """
        return self.measure.aggr(topic_coherences)

    def get_coherence(self):
        """Get coherence value based on pipeline parameters.

        Returns
        -------
        float
            Value of coherence.

        """
        confirmed_measures = self.get_coherence_per_topic()
        return self.aggregate_measures(confirmed_measures)

    def compare_models(self, models):
        """Compare topic models by coherence value.

        Parameters
        ----------
        models : :class:`~gensim.models.basemodel.BaseTopicModel`
            Sequence of topic models.

        Returns
        -------
        list of (float, float)
            Sequence of pairs of average topic coherence and average coherence.

        """
        model_topics = [self._get_topics_from_model(model, self.topn) for model in models]
        return self.compare_model_topics(model_topics)

    def compare_model_topics(self, model_topics):
        """Perform the coherence evaluation for each of the models.

        Parameters
        ----------
        model_topics : list of list of str
            list of list of words for the model trained with that number of topics.

        Returns
        -------
        list of (float, float)
            Sequence of pairs of average topic coherence and average coherence.

        Notes
        -----
        This first precomputes the probabilities once, then evaluates coherence for each model.

        Since we have already precomputed the probabilities, this simply involves using the accumulated stats in the
        :class:`~gensim.models.coherencemodel.CoherenceModel` to perform the evaluations, which should be pretty quick.

        """
        orig_topics = self._topics
        orig_topn = self.topn

        try:
            coherences = self._compare_model_topics(model_topics)
        finally:
            self.topics = orig_topics
            self.topn = orig_topn

        return coherences

    def _compare_model_topics(self, model_topics):
        """Get average topic and model coherences.

        Parameters
        ----------
        model_topics : list of list of str
            Topics from the model.

        Returns
        -------
        list of (float, float)
            Sequence of pairs of average topic coherence and average coherence.

        """
        coherences = []
        last_topn_value = min(self.topn - 1, 4)
        topn_grid = list(range(self.topn, last_topn_value, -5))

        for model_num, topics in enumerate(model_topics):
            self.topics = topics

            # We evaluate at various values of N and average them. This is a more robust,
            # according to: http://people.eng.unimelb.edu.au/tbaldwin/pubs/naacl2016.pdf
            coherence_at_n = {}
            for n in topn_grid:
                self.topn = n
                topic_coherences = self.get_coherence_per_topic()

                # Let's record the coherences for each topic, as well as the aggregated
                # coherence across all of the topics.
                # Some of them may be nan (if all words were OOV), so do mean value imputation.
                filled_coherences = np.array(topic_coherences)
                filled_coherences[np.isnan(filled_coherences)] = np.nanmean(filled_coherences)
                coherence_at_n[n] = (topic_coherences, self.aggregate_measures(filled_coherences))

            topic_coherences, avg_coherences = zip(*coherence_at_n.values())
            avg_topic_coherences = np.vstack(topic_coherences).mean(0)
            model_coherence = np.mean(avg_coherences)
            logging.info("Avg coherence for model %d: %.5f" % (model_num, model_coherence))
            coherences.append((avg_topic_coherences, model_coherence))

        return coherences
