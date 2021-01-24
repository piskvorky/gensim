#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

r"""This module contains functions to compute confirmation on a pair of words or word subsets.

Notes
-----
The advantage of indirect confirmation measure is that it computes similarity of words in :math:`W'` and
:math:`W^{*}` with respect to direct confirmations to all words. Eg. Suppose `x` and `z` are both competing
brands of cars, which semantically support each other. However, both brands are seldom mentioned
together in documents in the reference corpus. But their confirmations to other words like “road”
or “speed” do strongly correlate. This would be reflected by an indirect confirmation measure.
Thus, indirect confirmation measures may capture semantic support that direct measures would miss.

The formula used to compute indirect confirmation measure is

.. math::

    \widetilde{m}_{sim(m, \gamma)}(W', W^{*}) = s_{sim}(\vec{v}^{\,}_{m,\gamma}(W'), \vec{v}^{\,}_{m,\gamma}(W^{*}))


where :math:`s_{sim}` can be cosine, dice or jaccard similarity and

.. math::

    \vec{v}^{\,}_{m,\gamma}(W') = \Bigg \{{\sum_{w_{i} \in W'}^{ } m(w_{i}, w_{j})^{\gamma}}\Bigg \}_{j = 1,...,|W|}

"""

import itertools
import logging

import numpy as np
import scipy.sparse as sps

from gensim.topic_coherence.direct_confirmation_measure import aggregate_segment_sims, log_ratio_measure

logger = logging.getLogger(__name__)


def word2vec_similarity(segmented_topics, accumulator, with_std=False, with_support=False):
    """For each topic segmentation, compute average cosine similarity using a
    :class:`~gensim.topic_coherence.text_analysis.WordVectorsAccumulator`.

    Parameters
    ----------
    segmented_topics : list of lists of (int, `numpy.ndarray`)
        Output from the :func:`~gensim.topic_coherence.segmentation.s_one_set`.
    accumulator : :class:`~gensim.topic_coherence.text_analysis.WordVectorsAccumulator` or
                  :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`
        Word occurrence accumulator.
    with_std : bool, optional
        True to also include standard deviation across topic segment sets
        in addition to the mean coherence for each topic.
    with_support : bool, optional
        True to also include support across topic segments. The support is defined as
        the number of pairwise similarity comparisons were used to compute the overall topic coherence.

    Returns
    -------
    list of (float[, float[, int]])
        Сosine word2vec similarities per topic (with std/support if `with_std`, `with_support`).

    Examples
    --------
    .. sourcecode:: pycon

        >>> import numpy as np
        >>> from gensim.corpora.dictionary import Dictionary
        >>> from gensim.topic_coherence import indirect_confirmation_measure
        >>> from gensim.topic_coherence import text_analysis
        >>>
        >>> # create segmentation
        >>> segmentation = [[(1, np.array([1, 2])), (2, np.array([1, 2]))]]
        >>>
        >>> # create accumulator
        >>> dictionary = Dictionary()
        >>> dictionary.id2token = {1: 'fake', 2: 'tokens'}
        >>> accumulator = text_analysis.WordVectorsAccumulator({1, 2}, dictionary)
        >>> _ = accumulator.accumulate([['fake', 'tokens'], ['tokens', 'fake']], 5)
        >>>
        >>> # should be (0.726752426218 0.00695475919227)
        >>> mean, std = indirect_confirmation_measure.word2vec_similarity(segmentation, accumulator, with_std=True)[0]

    """
    topic_coherences = []
    total_oov = 0

    for topic_index, topic_segments in enumerate(segmented_topics):
        segment_sims = []
        num_oov = 0
        for w_prime, w_star in topic_segments:
            if not hasattr(w_prime, '__iter__'):
                w_prime = [w_prime]
            if not hasattr(w_star, '__iter__'):
                w_star = [w_star]

            try:
                segment_sims.append(accumulator.ids_similarity(w_prime, w_star))
            except ZeroDivisionError:
                num_oov += 1

        if num_oov > 0:
            total_oov += 1
            logger.warning(
                "%d terms for topic %d are not in word2vec model vocabulary",
                num_oov, topic_index)
        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))

    if total_oov > 0:
        logger.warning("%d terms for are not in word2vec model vocabulary", total_oov)
    return topic_coherences


def cosine_similarity(segmented_topics, accumulator, topics, measure='nlr',
                      gamma=1, with_std=False, with_support=False):
    """Calculate the indirect cosine measure.

    Parameters
    ----------
    segmented_topics: list of lists of (int, `numpy.ndarray`)
        Output from the segmentation module of the segmented topics.
    accumulator: :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`
        Output from the probability_estimation module. Is an topics: Topics obtained from the trained topic model.
    measure : str, optional
        Direct confirmation measure to be used. Supported values are "nlr" (normalized log ratio).
    gamma: float, optional
        Gamma value for computing :math:`W'` and :math:`W^{*}` vectors.
    with_std : bool
        True to also include standard deviation across topic segment sets in addition to the mean coherence
        for each topic; default is False.
    with_support : bool
        True to also include support across topic segments. The support is defined as the number of pairwise similarity
        comparisons were used to compute the overall topic coherence.

    Returns
    -------
    list
        List of indirect cosine similarity measure for each topic.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.corpora.dictionary import Dictionary
        >>> from gensim.topic_coherence import indirect_confirmation_measure, text_analysis
        >>> import numpy as np
        >>>
        >>> # create accumulator
        >>> dictionary = Dictionary()
        >>> dictionary.id2token = {1: 'fake', 2: 'tokens'}
        >>> accumulator = text_analysis.InvertedIndexAccumulator({1, 2}, dictionary)
        >>> accumulator._inverted_index = {0: {2, 3, 4}, 1: {3, 5}}
        >>> accumulator._num_docs = 5
        >>>
        >>> # create topics
        >>> topics = [np.array([1, 2])]
        >>>
        >>> # create segmentation
        >>> segmentation = [[(1, np.array([1, 2])), (2, np.array([1, 2]))]]
        >>> obtained = indirect_confirmation_measure.cosine_similarity(segmentation, accumulator, topics, 'nlr', 1)
        >>> print(obtained[0])
        0.623018926945

    """
    context_vectors = ContextVectorComputer(measure, topics, accumulator, gamma)

    topic_coherences = []
    for topic_words, topic_segments in zip(topics, segmented_topics):
        topic_words = tuple(topic_words)  # because tuples are hashable
        segment_sims = np.zeros(len(topic_segments))
        for i, (w_prime, w_star) in enumerate(topic_segments):
            w_prime_cv = context_vectors[w_prime, topic_words]
            w_star_cv = context_vectors[w_star, topic_words]
            segment_sims[i] = _cossim(w_prime_cv, w_star_cv)

        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))

    return topic_coherences


class ContextVectorComputer:
    """Lazily compute context vectors for topic segments.

    Parameters
    ----------
    measure: str
        Confirmation measure.
    topics: list of numpy.array
        Topics.
    accumulator : :class:`~gensim.topic_coherence.text_analysis.WordVectorsAccumulator` or
                  :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`
        Word occurrence accumulator from probability_estimation.
    gamma: float
        Value for computing vectors.

    Attributes
    ----------
    sim_cache: dict
        Cache similarities between tokens (pairs of word ids), e.g. (1, 2).
    context_vector_cache: dict
        Mapping from (segment, topic_words) --> context_vector.

    Example
    -------
    .. sourcecode:: pycon

        >>> from gensim.corpora.dictionary import Dictionary
        >>> from gensim.topic_coherence import indirect_confirmation_measure, text_analysis
        >>> import numpy as np
        >>>
        >>> # create measure, topics
        >>> measure = 'nlr'
        >>> topics = [np.array([1, 2])]
        >>>
        >>> # create accumulator
        >>> dictionary = Dictionary()
        >>> dictionary.id2token = {1: 'fake', 2: 'tokens'}
        >>> accumulator = text_analysis.WordVectorsAccumulator({1, 2}, dictionary)
        >>> _ = accumulator.accumulate([['fake', 'tokens'], ['tokens', 'fake']], 5)
        >>> cont_vect_comp = indirect_confirmation_measure.ContextVectorComputer(measure, topics, accumulator, 1)
        >>> cont_vect_comp.mapping
        {1: 0, 2: 1}
        >>> cont_vect_comp.vocab_size
        2

    """

    def __init__(self, measure, topics, accumulator, gamma):

        if measure == 'nlr':
            self.similarity = _pair_npmi
        else:
            raise ValueError(
                "The direct confirmation measure you entered is not currently supported.")

        self.mapping = _map_to_contiguous(topics)
        self.vocab_size = len(self.mapping)
        self.accumulator = accumulator
        self.gamma = gamma
        self.sim_cache = {}
        self.context_vector_cache = {}

    def __getitem__(self, idx):
        return self.compute_context_vector(*idx)

    def compute_context_vector(self, segment_word_ids, topic_word_ids):
        """Check if (segment_word_ids, topic_word_ids) context vector has been cached.

        Parameters
        ----------
        segment_word_ids: list
            Ids of words in segment.
        topic_word_ids: list
            Ids of words in topic.
        Returns
        -------
        csr_matrix :class:`~scipy.sparse.csr`
            If context vector has been cached, then return corresponding context vector,
            else compute, cache, and return.

        """
        key = _key_for_segment(segment_word_ids, topic_word_ids)
        context_vector = self.context_vector_cache.get(key, None)
        if context_vector is None:
            context_vector = self._make_seg(segment_word_ids, topic_word_ids)
            self.context_vector_cache[key] = context_vector
        return context_vector

    def _make_seg(self, segment_word_ids, topic_word_ids):
        """Return context vectors for segmentation (Internal helper function).

        Parameters
        ----------
        segment_word_ids : iterable or int
            Ids of words in segment.
        topic_word_ids : list
            Ids of words in topic.
        Returns
        -------
        csr_matrix :class:`~scipy.sparse.csr`
            Matrix in Compressed Sparse Row format

        """
        context_vector = sps.lil_matrix((self.vocab_size, 1))
        if not hasattr(segment_word_ids, '__iter__'):
            segment_word_ids = (segment_word_ids,)

        for w_j in topic_word_ids:
            idx = (self.mapping[w_j], 0)
            for pair in (tuple(sorted((w_i, w_j))) for w_i in segment_word_ids):
                if pair not in self.sim_cache:
                    self.sim_cache[pair] = self.similarity(pair, self.accumulator)

                context_vector[idx] += self.sim_cache[pair] ** self.gamma

        return context_vector.tocsr()


def _pair_npmi(pair, accumulator):
    """Compute normalized pairwise mutual information (**NPMI**) between a pair of words.

    Parameters
    ----------
    pair : (int, int)
        The pair of words (word_id1, word_id2).
    accumulator : :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`
        Word occurrence accumulator from probability_estimation.

    Return
    ------
    float
        NPMI between a pair of words.

    """
    return log_ratio_measure([[pair]], accumulator, True)[0]


def _cossim(cv1, cv2):
    return cv1.T.dot(cv2)[0, 0] / (_magnitude(cv1) * _magnitude(cv2))


def _magnitude(sparse_vec):
    return np.sqrt(np.sum(sparse_vec.data ** 2))


def _map_to_contiguous(ids_iterable):
    uniq_ids = {}
    n = 0
    for id_ in itertools.chain.from_iterable(ids_iterable):
        if id_ not in uniq_ids:
            uniq_ids[id_] = n
            n += 1
    return uniq_ids


def _key_for_segment(segment, topic_words):
    """A segment may have a single number of an iterable of them."""
    segment_key = tuple(segment) if hasattr(segment, '__iter__') else segment
    return segment_key, topic_words
