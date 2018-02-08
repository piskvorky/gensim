#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains functions to perform segmentation on a list of topics."""

import itertools
import logging

from gensim.topic_coherence.text_analysis import (
    CorpusAccumulator, WordOccurrenceAccumulator, ParallelWordOccurrenceAccumulator,
    WordVectorsAccumulator)

logger = logging.getLogger(__name__)


def p_boolean_document(corpus, segmented_topics):
    """Perform the boolean document probability estimation. Boolean document estimates the probability of a single word
    as the number of documents in which the word occurs divided by the total number of documents.

    Parameters
    ----------
    corpus : iterable of list of (int, int)
        The corpus of documents.
    segmented_topics: list of (int, int).
        Each tuple (word_id_set1, word_id_set2) is either a single integer, or a `numpy.ndarray` of integers.

    Returns
    -------
    :class:`~gensim.topic_coherence.text_analysis.CorpusAccumulator`
        Word occurrence accumulator instance that can be used to lookup token frequencies and co-occurrence frequencies.

    Examples
    ---------
    >>> from gensim.topic_coherence import probability_estimation
    >>> from gensim.corpora.hashdictionary import HashDictionary
    >>>
    >>>
    >>> texts = [
    ...     ['human', 'interface', 'computer'],
    ...     ['eps', 'user', 'interface', 'system'],
    ...     ['system', 'human', 'system', 'eps'],
    ...     ['user', 'response', 'time'],
    ...     ['trees'],
    ...     ['graph', 'trees']
    ... ]
    >>> dictionary = HashDictionary(texts)
    >>> w2id = dictionary.token2id
    >>>
    >>> # create segmented_topics
    >>> segmented_topics = [
    ...     [(w2id['system'], w2id['graph']),(w2id['computer'], w2id['graph']),(w2id['computer'], w2id['system'])],
    ...     [(w2id['computer'], w2id['graph']),(w2id['user'], w2id['graph']),(w2id['user'], w2id['computer'])]
    ... ]
    >>>
    >>> # create corpus
    >>> corpus = [dictionary.doc2bow(text) for text in texts]
    >>>
    >>> result = probability_estimation.p_boolean_document(corpus, segmented_topics)
    >>> result.index_to_dict()
    {10608: set([0]), 12736: set([1, 3]), 18451: set([5]), 5798: set([1, 2])}

    """
    top_ids = unique_ids_from_segments(segmented_topics)
    return CorpusAccumulator(top_ids).accumulate(corpus)


def p_boolean_sliding_window(texts, segmented_topics, dictionary, window_size, processes=1):
    """Perform the boolean sliding window probability estimation.

    Parameters
    ----------
    texts : iterable of iterable of str
        Input text
    segmented_topics: list of (int, int)
        Each tuple (word_id_set1, word_id_set2) is either a single integer, or a `numpy.ndarray` of integers.
    dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
        Gensim dictionary mapping of the tokens and ids.
    window_size : int
        Size of the sliding window, 110 found out to be the ideal size for large corpora.
    processes : int, optional
        Number of process that will be used for
        :class:`~gensim.topic_coherence.text_analysis.ParallelWordOccurrenceAccumulator`

    Notes
    -----
    Boolean sliding window determines word counts using a sliding window. The window
    moves over  the documents one word token per step. Each step defines a new virtual
    document  by copying the window content. Boolean document is applied to these virtual
    documents to compute word probabilities.

    Returns
    -------
    :class:`~gensim.topic_coherence.text_analysis.WordOccurrenceAccumulator`
        if `processes` = 1 OR
    :class:`~gensim.topic_coherence.text_analysis.ParallelWordOccurrenceAccumulator`
        otherwise. This is word occurrence accumulator instance that can be used to lookup
        token frequencies and co-occurrence frequencies.

    Examples
    ---------
    >>> from gensim.topic_coherence import probability_estimation
    >>> from gensim.corpora.hashdictionary import HashDictionary
    >>>
    >>>
    >>> texts = [
    ...     ['human', 'interface', 'computer'],
    ...     ['eps', 'user', 'interface', 'system'],
    ...     ['system', 'human', 'system', 'eps'],
    ...     ['user', 'response', 'time'],
    ...     ['trees'],
    ...     ['graph', 'trees']
    ... ]
    >>> dictionary = HashDictionary(texts)
    >>> w2id = dictionary.token2id

    >>>
    >>> # create segmented_topics
    >>> segmented_topics = [
    ...     [(w2id['system'], w2id['graph']),(w2id['computer'], w2id['graph']),(w2id['computer'], w2id['system'])],
    ...     [(w2id['computer'], w2id['graph']),(w2id['user'], w2id['graph']),(w2id['user'], w2id['computer'])]
    ... ]
    >>>
    >>> # create corpus
    >>> corpus = [dictionary.doc2bow(text) for text in texts]
    >>> accumulator = probability_estimation.p_boolean_sliding_window(texts, segmented_topics, dictionary, 2)
    >>>
    >>> (accumulator[w2id['computer']], accumulator[w2id['user']], accumulator[w2id['system']])
    (1, 3, 4)

    """
    top_ids = unique_ids_from_segments(segmented_topics)
    if processes <= 1:
        accumulator = WordOccurrenceAccumulator(top_ids, dictionary)
    else:
        accumulator = ParallelWordOccurrenceAccumulator(processes, top_ids, dictionary)
    logger.info("using %s to estimate probabilities from sliding windows", accumulator)
    return accumulator.accumulate(texts, window_size)


def p_word2vec(texts, segmented_topics, dictionary, window_size=None, processes=1, model=None):
    """Train word2vec model on `texts` if `model` is not None.

    Parameters
    ----------
    texts : iterable of iterable of str
        Input text
    segmented_topics : iterable of iterable of str
        Output from the segmentation of topics. Could be simply topics too.
    dictionary : :class:`~gensim.corpora.dictionary`
        Gensim dictionary mapping of the tokens and ids.
    window_size : int, optional
        Size of the sliding window.
    processes : int, optional
        Number of processes to use.
    model : :class:`~gensim.models.word2vec.Word2Vec` or :class:`~gensim.models.keyedvectors.KeyedVectors`, optional
        If None, a new Word2Vec model is trained on the given text corpus. Otherwise,
        it should be a pre-trained Word2Vec context vectors.

    Returns
    -------
    :class:`~gensim.topic_coherence.text_analysis.WordVectorsAccumulator`
        Text accumulator with trained context vectors.

    Examples
    --------
    >>> from gensim.topic_coherence import probability_estimation
    >>> from gensim.corpora.hashdictionary import HashDictionary
    >>> from gensim.models import word2vec
    >>>
    >>> texts = [
    ...     ['human', 'interface', 'computer'],
    ...     ['eps', 'user', 'interface', 'system'],
    ...     ['system', 'human', 'system', 'eps'],
    ...     ['user', 'response', 'time'],
    ...     ['trees'],
    ...     ['graph', 'trees']
    ... ]
    >>> dictionary = HashDictionary(texts)
    >>> w2id = dictionary.token2id

    >>>
    >>> # create segmented_topics
    >>> segmented_topics = [
    ...     [(w2id['system'], w2id['graph']),(w2id['computer'], w2id['graph']),(w2id['computer'], w2id['system'])],
    ...     [(w2id['computer'], w2id['graph']),(w2id['user'], w2id['graph']),(w2id['user'], w2id['computer'])]
    ... ]
    >>>
    >>> # create corpus
    >>> corpus = [dictionary.doc2bow(text) for text in texts]
    >>> sentences = [['human', 'interface', 'computer'],['survey', 'user', 'computer', 'system', 'response', 'time']]
    >>> model = word2vec.Word2Vec(sentences, size=100,min_count=1)
    >>> accumulator = probability_estimation.p_word2vec(texts, segmented_topics, dictionary, 2, 1, model)

    """
    top_ids = unique_ids_from_segments(segmented_topics)
    accumulator = WordVectorsAccumulator(
        top_ids, dictionary, model, window=window_size, workers=processes)
    return accumulator.accumulate(texts, window_size)


def unique_ids_from_segments(segmented_topics):
    """Return the set of all unique ids in a list of segmented topics.

    Parameters
    ----------
    segmented_topics: list of (int, int).
        Each tuple (word_id_set1, word_id_set2) is either a single integer, or a `numpy.ndarray` of integers.

    Returns
    -------
    set
        Set of unique ids across all topic segments.

    Example
    -------
    >>> from gensim.topic_coherence import probability_estimation
    >>>
    >>> segmentation = [[(1, 2)]]
    >>> probability_estimation.unique_ids_from_segments(segmentation)
    set([1, 2])

    """
    unique_ids = set()  # is a set of all the unique ids contained in topics.
    for s_i in segmented_topics:
        for word_id in itertools.chain.from_iterable(s_i):
            if hasattr(word_id, '__iter__'):
                unique_ids.update(word_id)
            else:
                unique_ids.add(word_id)

    return unique_ids
