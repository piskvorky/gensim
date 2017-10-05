#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
This module contains implementation of the individual components of
the topic coherence pipeline.
"""

import sys
import logging
import itertools
from collections import Counter
import multiprocessing as mp
import numpy as np
from gensim.utils import iter_windows, flatten
from gensim.models.word2vec import Word2Vec
from scipy import sparse as sps
from six import string_types, iteritems

logger = logging.getLogger(__name__)
EPSILON = 1e-12  # Should be small. Value as suggested in paper.


"""
This part contains functions to perform aggregation on a list of values
obtained from the confirmation measure.
"""


def arithmetic_mean(confirmed_measures):
    """
    This function performs the arithmetic mean aggregation on the output obtained from
    the confirmation measure module.

    Args:
        confirmed_measures : list of calculated confirmation measure on each set in the segmented topics.

    Returns:
        mean : Arithmetic mean of all the values contained in confirmation measures.
    """
    return np.mean(confirmed_measures)


"""
This part contains functions to compute direct confirmation on a pair of words or word subsets.
"""


def log_conditional_probability(segmented_topics, accumulator, with_std=False, with_support=False):
    """
    This function calculates the log-conditional-probability measure
    which is used by coherence measures such as U_mass.
    This is defined as: m_lc(S_i) = log[(P(W', W*) + e) / P(W*)]

    Args:
        segmented_topics (list): Output from the segmentation module of the segmented
            topics. Is a list of list of tuples.
        accumulator: word occurrence accumulator from probability_estimation.
        with_std (bool): True to also include standard deviation across topic segment
            sets in addition to the mean coherence for each topic; default is False.
        with_support (bool): True to also include support across topic segments. The
            support is defined as the number of pairwise similarity comparisons were
            used to compute the overall topic coherence.

    Returns:
        list : of log conditional probability measure for each topic.
    """
    topic_coherences = []
    num_docs = float(accumulator.num_docs)
    for s_i in segmented_topics:
        segment_sims = []
        for w_prime, w_star in s_i:
            try:
                w_star_count = accumulator[w_star]
                co_occur_count = accumulator[w_prime, w_star]
                m_lc_i = np.log(((co_occur_count / num_docs) + EPSILON) / (w_star_count / num_docs))
            except KeyError:
                m_lc_i = 0.0

            segment_sims.append(m_lc_i)

        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))

    return topic_coherences


def aggregate_segment_sims(segment_sims, with_std, with_support):
    """Compute various statistics from the segment similarities generated via
    set pairwise comparisons of top-N word lists for a single topic.

    Args:
        segment_sims (iterable): floating point similarity values to aggregate.
        with_std (bool): Set to True to include standard deviation.
        with_support (bool): Set to True to include number of elements in `segment_sims`
            as a statistic in the results returned.

    Returns:
        tuple: with (mean[, std[, support]])
    """
    mean = np.mean(segment_sims)
    stats = [mean]
    if with_std:
        stats.append(np.std(segment_sims))
    if with_support:
        stats.append(len(segment_sims))

    return stats[0] if len(stats) == 1 else tuple(stats)


def log_ratio_measure(
        segmented_topics, accumulator, normalize=False, with_std=False, with_support=False):
    """
    If normalize=False:
        Popularly known as PMI.
        This function calculates the log-ratio-measure which is used by
        coherence measures such as c_v.
        This is defined as: m_lr(S_i) = log[(P(W', W*) + e) / (P(W') * P(W*))]

    If normalize=True:
        This function calculates the normalized-log-ratio-measure, popularly knowns as
        NPMI which is used by coherence measures such as c_v.
        This is defined as: m_nlr(S_i) = m_lr(S_i) / -log[P(W', W*) + e]

    Args:
        segmented_topics (list): Output from the segmentation module of the segmented
            topics. Is a list of list of tuples.
        accumulator: word occurrence accumulator from probability_estimation.
        with_std (bool): True to also include standard deviation across topic segment
            sets in addition to the mean coherence for each topic; default is False.
        with_support (bool): True to also include support across topic segments. The
            support is defined as the number of pairwise similarity comparisons were
            used to compute the overall topic coherence.

    Returns:
        list : of log ratio measure for each topic.
    """
    topic_coherences = []
    num_docs = float(accumulator.num_docs)
    for s_i in segmented_topics:
        segment_sims = []
        for w_prime, w_star in s_i:
            w_prime_count = accumulator[w_prime]
            w_star_count = accumulator[w_star]
            co_occur_count = accumulator[w_prime, w_star]

            if normalize:
                # For normalized log ratio measure
                numerator = log_ratio_measure([[(w_prime, w_star)]], accumulator)[0]
                co_doc_prob = co_occur_count / num_docs
                m_lr_i = numerator / (-np.log(co_doc_prob + EPSILON))
            else:
                # For log ratio measure without normalization
                numerator = (co_occur_count / num_docs) + EPSILON
                denominator = (w_prime_count / num_docs) * (w_star_count / num_docs)
                m_lr_i = np.log(numerator / denominator)

            segment_sims.append(m_lr_i)

        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))

    return topic_coherences


"""
This part contains functions to compute confirmation on a pair of words or word subsets.
The advantage of indirect confirmation measure is that it computes similarity of words in W' and
W* with respect to direct confirmations to all words. Eg. Suppose x and z are both competing
brands of cars, which semantically support each other. However, both brands are seldom mentioned
together in documents in the reference corpus. But their confirmations to other words like “road”
or “speed” do strongly correlate. This would be reflected by an indirect confirmation measure.
Thus, indirect confirmation measures may capture semantic support that direct measures would miss.

The formula used to compute indirect confirmation measure is:

m_{sim}_{(m, \gamma)}(W', W*) =
    s_{sim}(\vec{V}^{\,}_{m,\gamma}(W'), \vec{V}^{\,}_{m,\gamma}(W*))

where s_sim can be cosine, dice or jaccard similarity and

\vec{V}^{\,}_{m,\gamma}(W') =
    \Bigg \{{\sum_{w_{i} \in W'}^{ } m(w_{i}, w_{j})^{\gamma}}\Bigg \}_{j = 1,...,|W|}

Here 'm' is the direct confirmation measure used.
"""


def word2vec_similarity(segmented_topics, accumulator, with_std=False, with_support=False):
    """For each topic segmentation, compute average cosine similarity using a
    WordVectorsAccumulator.

    Args:
        segmented_topics (list): Output from the segmentation module of the segmented
            topics. Is a list of list of tuples.
        accumulator: word occurrence accumulator from probability_estimation.
        with_std (bool): True to also include standard deviation across topic segment
            sets in addition to the mean coherence for each topic; default is False.
        with_support (bool): True to also include support across topic segments. The
            support is defined as the number of pairwise similarity comparisons were
            used to compute the overall topic coherence.

    Returns:
        list : of word2vec cosine similarities per topic.

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


def cosine_similarity(segmented_topics, accumulator, topics, measure='nlr', gamma=1,
                      with_std=False, with_support=False):
    r"""
    This function calculates the indirect cosine measure.
    Given context vectors u = V(W') and w = V(W*) for the
    word sets of a pair S_i = (W', W*) indirect cosine measure
    is computed as the cosine similarity between u and w.
    The formula used is
    m_{sim}_{(m, \gamma)}(W', W*) = s_{sim}(\vec{V}^{\,}_{m,\gamma}(W'), \vec{V}^{\,}_{m,\gamma}(W*))
    where each vector
    \vec{V}^{\,}_{m,\gamma}(W') = \Bigg \{{\sum_{w_{i} \in W'}^{ } m(w_{i}, w_{j})^{\gamma}}\Bigg \}_{j = 1,...,|W|}

    Args:
        segmented_topics: Output from the segmentation module of the
            segmented topics. Is a list of list of tuples.
        accumulator: Output from the probability_estimation module. Is an
            accumulator of word occurrences (see text_analysis module).
        topics: Topics obtained from the trained topic model.
        measure (str): Direct confirmation measure to be used. Supported
            values are "nlr" (normalized log ratio).
        gamma: Gamma value for computing W', W* vectors; default is 1.
        with_std (bool): True to also include standard deviation across topic
            segment sets in addition to the mean coherence for each topic;
            default is False.
        with_support (bool): True to also include support across topic segments.
            The support is defined as the number of pairwise similarity
            comparisons were used to compute the overall topic coherence.
    Returns:
        list: of indirect cosine similarity measure for each topic.
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


class ContextVectorComputer(object):
    """Lazily compute context vectors for topic segments."""

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
        self.sim_cache = {}  # Cache similarities between tokens (pairs of word ids), e.g. (1, 2)
        self.context_vector_cache = {}  # mapping from (segment, topic_words) --> context_vector

    def __getitem__(self, idx):
        return self.compute_context_vector(*idx)

    def compute_context_vector(self, segment_word_ids, topic_word_ids):
        """
        Step 1. Check if (segment_word_ids, topic_word_ids) context vector has been cached.
        Step 2. If yes, return corresponding context vector, else compute, cache, and return.
        """
        key = _key_for_segment(segment_word_ids, topic_word_ids)
        context_vector = self.context_vector_cache.get(key, None)
        if context_vector is None:
            context_vector = self._make_seg(segment_word_ids, topic_word_ids)
            self.context_vector_cache[key] = context_vector
        return context_vector

    def _make_seg(self, segment_word_ids, topic_word_ids):
        """Internal helper function to return context vectors for segmentations."""
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
    """Compute normalized pairwise mutual information (NPMI) between a pair of words.
    The pair is an iterable of (word_id1, word_id2).
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


"""
This part contains functions to perform segmentation on a list of topics.
"""


def p_boolean_document(corpus, segmented_topics):
    """This function performs the boolean document probability estimation.
    Boolean document estimates the probability of a single word as the number
    of documents in which the word occurs divided by the total number of documents.

    Args:
        corpus : The corpus of documents.
        segmented_topics : Output from the segmentation of topics. Could be simply topics too.

    Returns:
        accumulator : word occurrence accumulator instance that can be used to lookup token
            frequencies and co-occurrence frequencies.
    """
    top_ids = unique_ids_from_segments(segmented_topics)
    return CorpusAccumulator(top_ids).accumulate(corpus)


def p_boolean_sliding_window(texts, segmented_topics, dictionary, window_size, processes=1):
    """This function performs the boolean sliding window probability estimation.
    Boolean sliding window determines word counts using a sliding window. The window
    moves over  the documents one word token per step. Each step defines a new virtual
    document  by copying the window content. Boolean document is applied to these virtual
    documents to compute word probabilities.

    Args:
        texts : List of string sentences.
        segmented_topics : Output from the segmentation of topics. Could be simply topics too.
        dictionary : Gensim dictionary mapping of the tokens and ids.
        window_size : Size of the sliding window. 110 found out to be the ideal size for large corpora.

    Returns:
        accumulator : word occurrence accumulator instance that can be used to lookup token
            frequencies and co-occurrence frequencies.
    """
    top_ids = unique_ids_from_segments(segmented_topics)
    if processes <= 1:
        accumulator = WordOccurrenceAccumulator(top_ids, dictionary)
    else:
        accumulator = ParallelWordOccurrenceAccumulator(processes, top_ids, dictionary)
    logger.info("using %s to estimate probabilities from sliding windows", accumulator)
    return accumulator.accumulate(texts, window_size)


def p_word2vec(texts, segmented_topics, dictionary, window_size=None, processes=1, model=None):
    """Train word2vec model on `texts` if model is not None.
    Returns:
    ----
    accumulator: text accumulator with trained context vectors.
    """
    top_ids = unique_ids_from_segments(segmented_topics)
    accumulator = WordVectorsAccumulator(
        top_ids, dictionary, model, window=window_size, workers=processes)
    return accumulator.accumulate(texts, window_size)


def unique_ids_from_segments(segmented_topics):
    """Return the set of all unique ids in a list of segmented topics.

    Args:
        segmented_topics: list of tuples of (word_id_set1, word_id_set2). Each word_id_set
            is either a single integer, or a `numpy.ndarray` of integers.
    Returns:
        unique_ids : set of unique ids across all topic segments.
    """
    unique_ids = set()  # is a set of all the unique ids contained in topics.
    for s_i in segmented_topics:
        for word_id in itertools.chain.from_iterable(s_i):
            if hasattr(word_id, '__iter__'):
                unique_ids.update(word_id)
            else:
                unique_ids.add(word_id)

    return unique_ids


"""
This part contains functions to perform segmentation on a list of topics.
"""


def s_one_pre(topics):
    """
    This function performs s_one_pre segmentation on a list of topics.
    s_one_pre segmentation is defined as: s_one_pre = {(W', W*) | W' = {w_i}; W* = {w_j}; w_i, w_j belongs to W; i > j}
    Example:

        >>> topics = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        >>> s_one_pre(topics)
        [[(2, 1), (3, 1), (3, 2)], [(5, 4), (6, 4), (6, 5)]]

    Args:
        topics : list of topics obtained from an algorithm such as LDA. Is a list such as [array([ 9, 10, 11]), array([ 9, 10,  7]), ...]

    Returns:
        s_one_pre_res : list of list of (W', W*) tuples for all unique topic ids
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
    """
    This function performs s_one_one segmentation on a list of topics.
    s_one_one segmentation is defined as: s_one_one = {(W', W*) | W' = {w_i}; W* = {w_j}; w_i, w_j belongs to W; i != j}
    Example:

        >>> topics = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        >>> s_one_pre(topics)
        [[(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)], [(4, 5), (4, 6), (5, 4), (5, 6), (6, 4), (6, 5)]]

    Args:
        topics : list of topics obtained from an algorithm such as LDA. Is a list such as [array([ 9, 10, 11]), array([ 9, 10,  7]), ...]

    Returns:
        s_one_one_res : list of list of (W', W*) tuples for all unique topic ids
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
    """
    This function performs s_one_set segmentation on a list of topics.
    s_one_set segmentation is defined as: s_one_set = {(W', W*) | W' = {w_i}; w_i belongs to W; W* = W}
    Example:
        >>> topics = [np.array([9, 10, 7])
        >>> s_one_set(topics)
        [[(9, array([ 9, 10,  7])),
          (10, array([ 9, 10,  7])),
          (7, array([ 9, 10,  7]))]]

    Args:
        topics : list of topics obtained from an algorithm such as LDA. Is a list such as [array([ 9, 10, 11]), array([ 9, 10,  7]), ...]

    Returns:
        s_one_set_res : list of list of (W', W*) tuples for all unique topic ids.
    """
    s_one_set_res = []

    for top_words in topics:
        s_one_set_t = []
        for w_prime in top_words:
            s_one_set_t.append((w_prime, top_words))
        s_one_set_res.append(s_one_set_t)

    return s_one_set_res


"""
This part contains classes for analyzing the texts of a corpus to accumulate
statistical information about word occurrences.
"""


def _ids_to_words(ids, dictionary):
    """Convert an iterable of ids to their corresponding words using a dictionary.
    This function abstracts away the differences between the HashDictionary and the standard one.

    Args:
        ids: list of list of tuples, where each tuple contains (token_id, iterable of token_ids).
            This is the format returned by the topic_coherence.segmentation functions.
    """
    if not dictionary.id2token:  # may not be initialized in the standard gensim.corpora.Dictionary
        setattr(dictionary, 'id2token', {v: k for k, v in dictionary.token2id.items()})

    top_words = set()
    for word_id in ids:
        word = dictionary.id2token[word_id]
        if isinstance(word, set):
            top_words = top_words.union(word)
        else:
            top_words.add(word)

    return top_words


class BaseAnalyzer(object):
    """Base class for corpus and text analyzers."""

    def __init__(self, relevant_ids):
        self.relevant_ids = relevant_ids
        self._vocab_size = len(self.relevant_ids)
        self.id2contiguous = {word_id: n for n, word_id in enumerate(self.relevant_ids)}
        self.log_every = 1000
        self._num_docs = 0

    @property
    def num_docs(self):
        return self._num_docs

    @num_docs.setter
    def num_docs(self, num):
        self._num_docs = num
        if self._num_docs % self.log_every == 0:
            logger.info(
                "%s accumulated stats from %d documents",
                self.__class__.__name__, self._num_docs)

    def analyze_text(self, text, doc_num=None):
        raise NotImplementedError("Base classes should implement analyze_text.")

    def __getitem__(self, word_or_words):
        if isinstance(word_or_words, string_types) or not hasattr(word_or_words, '__iter__'):
            return self.get_occurrences(word_or_words)
        else:
            return self.get_co_occurrences(*word_or_words)

    def get_occurrences(self, word_id):
        """Return number of docs the word occurs in, once `accumulate` has been called."""
        return self._get_occurrences(self.id2contiguous[word_id])

    def _get_occurrences(self, word_id):
        raise NotImplementedError("Base classes should implement occurrences")

    def get_co_occurrences(self, word_id1, word_id2):
        """Return number of docs the words co-occur in, once `accumulate` has been called."""
        return self._get_co_occurrences(self.id2contiguous[word_id1], self.id2contiguous[word_id2])

    def _get_co_occurrences(self, word_id1, word_id2):
        raise NotImplementedError("Base classes should implement co_occurrences")


class UsesDictionary(BaseAnalyzer):
    """A BaseAnalyzer that uses a Dictionary, hence can translate tokens to counts.
    The standard BaseAnalyzer can only deal with token ids since it doesn't have the token2id
    mapping.
    """

    def __init__(self, relevant_ids, dictionary):
        super(UsesDictionary, self).__init__(relevant_ids)
        self.relevant_words = _ids_to_words(self.relevant_ids, dictionary)
        self.dictionary = dictionary
        self.token2id = dictionary.token2id

    def get_occurrences(self, word):
        """Return number of docs the word occurs in, once `accumulate` has been called."""
        try:
            word_id = self.token2id[word]
        except KeyError:
            word_id = word
        return self._get_occurrences(self.id2contiguous[word_id])

    def _word2_contiguous_id(self, word):
        try:
            word_id = self.token2id[word]
        except KeyError:
            word_id = word
        return self.id2contiguous[word_id]

    def get_co_occurrences(self, word1, word2):
        """Return number of docs the words co-occur in, once `accumulate` has been called."""
        word_id1 = self._word2_contiguous_id(word1)
        word_id2 = self._word2_contiguous_id(word2)
        return self._get_co_occurrences(word_id1, word_id2)


class InvertedIndexBased(BaseAnalyzer):
    """Analyzer that builds up an inverted index to accumulate stats."""

    def __init__(self, *args):
        super(InvertedIndexBased, self).__init__(*args)
        self._inverted_index = np.array([set() for _ in range(self._vocab_size)])

    def _get_occurrences(self, word_id):
        return len(self._inverted_index[word_id])

    def _get_co_occurrences(self, word_id1, word_id2):
        s1 = self._inverted_index[word_id1]
        s2 = self._inverted_index[word_id2]
        return len(s1.intersection(s2))

    def index_to_dict(self):
        contiguous2id = {n: word_id for word_id, n in iteritems(self.id2contiguous)}
        return {contiguous2id[n]: doc_id_set for n, doc_id_set in enumerate(self._inverted_index)}


class CorpusAccumulator(InvertedIndexBased):
    """Gather word occurrence stats from a corpus by iterating over its BoW representation."""

    def analyze_text(self, text, doc_num=None):
        doc_words = frozenset(x[0] for x in text)
        top_ids_in_doc = self.relevant_ids.intersection(doc_words)
        for word_id in top_ids_in_doc:
            self._inverted_index[self.id2contiguous[word_id]].add(self._num_docs)

    def accumulate(self, corpus):
        for document in corpus:
            self.analyze_text(document)
            self.num_docs += 1
        return self


class WindowedTextsAnalyzer(UsesDictionary):
    """Gather some stats about relevant terms of a corpus by iterating over windows of texts."""

    def __init__(self, relevant_ids, dictionary):
        """
        Args:
            relevant_ids: the set of words that occurrences should be accumulated for.
            dictionary: Dictionary instance with mappings for the relevant_ids.
        """
        super(WindowedTextsAnalyzer, self).__init__(relevant_ids, dictionary)
        self._none_token = self._vocab_size  # see _iter_texts for use of none token

    def accumulate(self, texts, window_size):
        relevant_texts = self._iter_texts(texts)
        windows = iter_windows(relevant_texts, window_size, ignore_below_size=False, include_doc_num=True)

        for doc_num, virtual_document in windows:
            self.analyze_text(virtual_document, doc_num)
            self.num_docs += 1
        return self

    def _iter_texts(self, texts):
        dtype = np.uint16 if np.iinfo(np.uint16).max >= self._vocab_size else np.uint32
        for text in texts:
            if self.text_is_relevant(text):
                yield np.array([
                    self.id2contiguous[self.token2id[w]] if w in self.relevant_words
                    else self._none_token
                    for w in text], dtype=dtype)

    def text_is_relevant(self, text):
        """Return True if the text has any relevant words, else False."""
        for word in text:
            if word in self.relevant_words:
                return True
        return False


class InvertedIndexAccumulator(WindowedTextsAnalyzer, InvertedIndexBased):
    """Build an inverted index from a sequence of corpus texts."""

    def analyze_text(self, window, doc_num=None):
        for word_id in window:
            if word_id is not self._none_token:
                self._inverted_index[word_id].add(self._num_docs)


class WordOccurrenceAccumulator(WindowedTextsAnalyzer):
    """Accumulate word occurrences and co-occurrences from a sequence of corpus texts."""

    def __init__(self, *args):
        super(WordOccurrenceAccumulator, self).__init__(*args)
        self._occurrences = np.zeros(self._vocab_size, dtype='uint32')
        self._co_occurrences = sps.lil_matrix((self._vocab_size, self._vocab_size), dtype='uint32')

        self._uniq_words = np.zeros((self._vocab_size + 1,), dtype=bool)  # add 1 for none token
        self._counter = Counter()

    def __str__(self):
        return self.__class__.__name__

    def accumulate(self, texts, window_size):
        self._co_occurrences = self._co_occurrences.tolil()
        self.partial_accumulate(texts, window_size)
        self._symmetrize()
        return self

    def partial_accumulate(self, texts, window_size):
        """Meant to be called several times to accumulate partial results. The final
        accumulation should be performed with the `accumulate` method as opposed to this one.
        This method does not ensure the co-occurrence matrix is in lil format and does not
        symmetrize it after accumulation.
        """
        self._current_doc_num = -1
        self._token_at_edge = None
        self._counter.clear()

        super(WordOccurrenceAccumulator, self).accumulate(texts, window_size)
        for combo, count in iteritems(self._counter):
            self._co_occurrences[combo] += count

        return self

    def analyze_text(self, window, doc_num=None):
        self._slide_window(window, doc_num)
        mask = self._uniq_words[:-1]  # to exclude none token
        if mask.any():
            self._occurrences[mask] += 1
            self._counter.update(itertools.combinations(np.nonzero(mask)[0], 2))

    def _slide_window(self, window, doc_num):
        if doc_num != self._current_doc_num:
            self._uniq_words[:] = False
            self._uniq_words[np.unique(window)] = True
            self._current_doc_num = doc_num
        else:
            self._uniq_words[self._token_at_edge] = False
            self._uniq_words[window[-1]] = True

        self._token_at_edge = window[0]

    def _symmetrize(self):
        """Word pairs may have been encountered in (i, j) and (j, i) order.
        Rather than enforcing a particular ordering during the update process,
        we choose to symmetrize the co-occurrence matrix after accumulation has completed.
        """
        co_occ = self._co_occurrences
        co_occ.setdiag(self._occurrences)  # diagonal should be equal to occurrence counts
        self._co_occurrences = \
            co_occ + co_occ.T - sps.diags(co_occ.diagonal(), offsets=0, dtype='uint32')

    def _get_occurrences(self, word_id):
        return self._occurrences[word_id]

    def _get_co_occurrences(self, word_id1, word_id2):
        return self._co_occurrences[word_id1, word_id2]

    def merge(self, other):
        self._occurrences += other._occurrences
        self._co_occurrences += other._co_occurrences
        self._num_docs += other._num_docs


class PatchedWordOccurrenceAccumulator(WordOccurrenceAccumulator):
    """Monkey patched for multiprocessing worker usage,
    to move some of the logic to the master process.
    """
    def _iter_texts(self, texts):
        return texts  # master process will handle this


class ParallelWordOccurrenceAccumulator(WindowedTextsAnalyzer):
    """Accumulate word occurrences in parallel."""

    def __init__(self, processes, *args, **kwargs):
        """
        Args:
            processes : number of processes to use; must be at least two.
            args : should include `relevant_ids` and `dictionary` (see `UsesDictionary.__init__`).
            kwargs : can include `batch_size`, which is the number of docs to send to a worker at a
                time. If not included, it defaults to 64.
        """
        super(ParallelWordOccurrenceAccumulator, self).__init__(*args)
        if processes < 2:
            raise ValueError(
                "Must have at least 2 processes to run in parallel; got %d" % processes)
        self.processes = processes
        self.batch_size = kwargs.get('batch_size', 64)

    def __str__(self):
        return "%s(processes=%s, batch_size=%s)" % (
            self.__class__.__name__, self.processes, self.batch_size)

    def accumulate(self, texts, window_size):
        workers, input_q, output_q = self.start_workers(window_size)
        try:
            self.queue_all_texts(input_q, texts, window_size)
            interrupted = False
        except KeyboardInterrupt:
            logger.warn("stats accumulation interrupted; <= %d documents processed", self._num_docs)
            interrupted = True

        accumulators = self.terminate_workers(input_q, output_q, workers, interrupted)
        return self.merge_accumulators(accumulators)

    def start_workers(self, window_size):
        """Set up an input and output queue and start processes for each worker.

        The input queue is used to transmit batches of documents to the workers.
        The output queue is used by workers to transmit the WordOccurrenceAccumulator instances.
        Returns: tuple of (list of workers, input queue, output queue).
        """
        input_q = mp.Queue(maxsize=self.processes)
        output_q = mp.Queue()
        workers = []
        for _ in range(self.processes):
            accumulator = PatchedWordOccurrenceAccumulator(self.relevant_ids, self.dictionary)
            worker = AccumulatingWorker(input_q, output_q, accumulator, window_size)
            worker.start()
            workers.append(worker)

        return workers, input_q, output_q

    def yield_batches(self, texts):
        """Return a generator over the given texts that yields batches of
        `batch_size` texts at a time.
        """
        batch = []
        for text in self._iter_texts(texts):
            batch.append(text)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def queue_all_texts(self, q, texts, window_size):
        """Sequentially place batches of texts on the given queue until `texts` is consumed.
        The texts are filtered so that only those with at least one relevant token are queued.
        """
        for batch_num, batch in enumerate(self.yield_batches(texts)):
            q.put(batch, block=True)
            before = self._num_docs / self.log_every
            self._num_docs += sum(len(doc) - window_size + 1 for doc in batch)
            if before < (self._num_docs / self.log_every):
                logger.info(
                    "%d batches submitted to accumulate stats from %d documents (%d virtual)",
                    (batch_num + 1), (batch_num + 1) * self.batch_size, self._num_docs)

    def terminate_workers(self, input_q, output_q, workers, interrupted=False):
        """Wait until all workers have transmitted their WordOccurrenceAccumulator instances,
        then terminate each. We do not use join here because it has been shown to have some issues
        in Python 2.7 (and even in later versions). This method also closes both the input and output
        queue.

        If `interrupted` is False (normal execution), a None value is placed on the input queue for
        each worker. The workers are looking for this sentinel value and interpret it as a signal to
        terminate themselves. If `interrupted` is True, a KeyboardInterrupt occurred. The workers are
        programmed to recover from this and continue on to transmit their results before terminating.
        So in this instance, the sentinel values are not queued, but the rest of the execution
        continues as usual.
        """
        if not interrupted:
            for _ in workers:
                input_q.put(None, block=True)

        accumulators = []
        while len(accumulators) != len(workers):
            accumulators.append(output_q.get())
        logger.info("%d accumulators retrieved from output queue", len(accumulators))

        for worker in workers:
            if worker.is_alive():
                worker.terminate()

        input_q.close()
        output_q.close()
        return accumulators

    def merge_accumulators(self, accumulators):
        """Merge the list of accumulators into a single `WordOccurrenceAccumulator` with all
        occurrence and co-occurrence counts, and a `num_docs` that reflects the total observed
        by all the individual accumulators.
        """
        accumulator = WordOccurrenceAccumulator(self.relevant_ids, self.dictionary)
        for other_accumulator in accumulators:
            accumulator.merge(other_accumulator)
        # Workers do partial accumulation, so none of the co-occurrence matrices are symmetrized.
        # This is by design, to avoid unnecessary matrix additions/conversions during accumulation.
        accumulator._symmetrize()
        logger.info("accumulated word occurrence stats for %d virtual documents", accumulator.num_docs)
        return accumulator


class AccumulatingWorker(mp.Process):
    """Accumulate stats from texts fed in from queue."""

    def __init__(self, input_q, output_q, accumulator, window_size):
        super(AccumulatingWorker, self).__init__()
        self.input_q = input_q
        self.output_q = output_q
        self.accumulator = accumulator
        self.accumulator.log_every = sys.maxsize  # avoid logging in workers
        self.window_size = window_size

    def run(self):
        try:
            self._run()
        except KeyboardInterrupt:
            logger.info(
                "%s interrupted after processing %d documents",
                self.__class__.__name__, self.accumulator.num_docs)
        except Exception:
            logger.exception("worker encountered unexpected exception")
        finally:
            self.reply_to_master()

    def _run(self):
        batch_num = -1
        n_docs = 0
        while True:
            batch_num += 1
            docs = self.input_q.get(block=True)
            if docs is None:  # sentinel value
                logger.debug("observed sentinel value; terminating")
                break

            self.accumulator.partial_accumulate(docs, self.window_size)
            n_docs += len(docs)
            logger.debug(
                "completed batch %d; %d documents processed (%d virtual)",
                batch_num, n_docs, self.accumulator.num_docs)

        logger.debug(
            "finished all batches; %d documents processed (%d virtual)",
            n_docs, self.accumulator.num_docs)

    def reply_to_master(self):
        logger.info("serializing accumulator to return to master...")
        self.output_q.put(self.accumulator, block=False)
        logger.info("accumulator serialized")


class WordVectorsAccumulator(UsesDictionary):
    """Accumulate context vectors for words using word vector embeddings."""

    def __init__(self, relevant_ids, dictionary, model=None, **model_kwargs):
        """
        Args:
            model: if None, a new Word2Vec model is trained on the given text corpus.
                If not None, it should be a pre-trained Word2Vec context vectors
                (gensim.models.keyedvectors.KeyedVectors instance).
            model_kwargs: if model is None, these keyword arguments will be passed
                through to the Word2Vec constructor.
        """
        super(WordVectorsAccumulator, self).__init__(relevant_ids, dictionary)
        self.model = model
        self.model_kwargs = model_kwargs

    def not_in_vocab(self, words):
        uniq_words = set(flatten(words))
        return set(word for word in uniq_words if word not in self.model.vocab)

    def get_occurrences(self, word):
        """Return number of docs the word occurs in, once `accumulate` has been called."""
        try:
            self.token2id[word]  # is this a token or an id?
        except KeyError:
            word = self.dictionary.id2token[word]
        return self.model.vocab[word].count

    def get_co_occurrences(self, word1, word2):
        """Return number of docs the words co-occur in, once `accumulate` has been called."""
        raise NotImplementedError("Word2Vec model does not support co-occurrence counting")

    def accumulate(self, texts, window_size):
        if self.model is not None:
            logger.debug("model is already trained; no accumulation necessary")
            return self

        kwargs = self.model_kwargs.copy()
        if window_size is not None:
            kwargs['window'] = window_size
        kwargs['min_count'] = kwargs.get('min_count', 1)
        kwargs['sg'] = kwargs.get('sg', 1)
        kwargs['hs'] = kwargs.get('hw', 0)

        self.model = Word2Vec(**kwargs)
        self.model.build_vocab(texts)
        self.model.train(texts, total_examples=self.model.corpus_count, epochs=self.model.iter)
        self.model = self.model.wv  # retain KeyedVectors
        return self

    def ids_similarity(self, ids1, ids2):
        words1 = self._words_with_embeddings(ids1)
        words2 = self._words_with_embeddings(ids2)
        return self.model.n_similarity(words1, words2)

    def _words_with_embeddings(self, ids):
        if not hasattr(ids, '__iter__'):
            ids = [ids]

        words = [self.dictionary.id2token[word_id] for word_id in ids]
        return [word for word in words if word in self.model.vocab]