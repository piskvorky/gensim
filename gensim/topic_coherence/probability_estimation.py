#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains functions to perform segmentation on a list of topics.
"""

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
    corpus : list
        The corpus of documents.
    segmented_topics : list of list of (str,str)
        Output from the segmentation of topics. Tuples of (word_id_set1, word_id_set2). Could be simply topics too.

    Returns
    -------
    :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`
        Word occurrence accumulator instance that can be used to lookup token frequencies and co-occurrence frequencies.

    Examples
    ---------
    >>> from gensim.topic_coherence import probability_estimation
    >>> from gensim.corpora.hashdictionary import HashDictionary
    >>> from gensim.corpora.dictionary import Dictionary
    >>> texts = [['human', 'interface', 'computer'],['eps', 'user', 'interface', 'system'],
    >>> ['system', 'human', 'system', 'eps'],['user', 'response', 'time'],['trees'],['graph', 'trees']]
    >>> dictionary = HashDictionary(texts)
    >>> token2id = dictionary.token2id
    >>> computer_id = token2id['computer']
    >>> system_id = token2id['system']
    >>> user_id = token2id['user']
    >>> graph_id = token2id['graph']
    >>> segmented_topics = [[(system_id, graph_id),(computer_id, graph_id),(computer_id, system_id)], [
    >>> (computer_id, graph_id),(user_id, graph_id),(user_id, computer_id)]]
    >>> corpus = [dictionary.doc2bow(text) for text in texts]
    >>> probability_estimation.p_boolean_document(corpus, segmented_topics)

    """
    top_ids = unique_ids_from_segments(segmented_topics)
    return CorpusAccumulator(top_ids).accumulate(corpus)


def p_boolean_sliding_window(texts, segmented_topics, dictionary, window_size, processes=1):
    """Perform the boolean sliding window probability estimation.

    Parameters
    ----------
    texts : List of str
    segmented_topics : list of tuples of (word_id_set1, word_id_set2)
        Output from the segmentation of topics. Could be simply topics too.
    dictionary : :class:`~gensim.corpora.dictionary`
        Gensim dictionary mapping of the tokens and ids.
    window_size : int
        Size of the sliding window. 110 found out to be the ideal size for large corpora.

    Notes
    -----
    Boolean sliding window determines word counts using a sliding window. The window
    moves over  the documents one word token per step. Each step defines a new virtual
    document  by copying the window content. Boolean document is applied to these virtual
    documents to compute word probabilities.

    Returns
    -------
    :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`
        Word occurrence accumulator instance that can be used to lookup token frequencies and co-occurrence frequencies.

    Examples
    ---------
    >>> from gensim.topic_coherence import probability_estimation
    >>> from gensim.corpora.hashdictionary import HashDictionary
    >>> from gensim.corpora.dictionary import Dictionary
    >>> texts = [['human', 'interface', 'computer'],['eps', 'user', 'interface', 'system'],
    >>> ['system', 'human', 'system', 'eps'],['user', 'response', 'time'],['trees'],['graph', 'trees']]
    >>> dictionary = HashDictionary(texts)
    >>> token2id = dictionary.token2id
    >>> computer_id = token2id['computer']
    >>> system_id = token2id['system']
    >>> user_id = token2id['user']
    >>> graph_id = token2id['graph']
    >>> segmented_topics = [[(system_id, graph_id),(computer_id, graph_id),(computer_id, system_id)], [
    >>> (computer_id, graph_id),(user_id, graph_id),(user_id, computer_id)]]
    >>> corpus = [dictionary.doc2bow(text) for text in texts]
    >>> accumulator = probability_estimation.p_boolean_sliding_window(texts, segmented_topics, dictionary, 2)
    >>> print accumulator[computer_id], accumulator[user_id], accumulator[graph_id], accumulator[system_id]
    1 3 1 4

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

    Parameters
    ----------
    texts : List of str
    segmented_topics : list of tuples of (word_id_set1, word_id_set2)
        Output from the segmentation of topics. Could be simply topics too.
    dictionary : :class:`~gensim.corpora.dictionary`
        Gensim dictionary mapping of the tokens and ids.
    window_size : int
        Size of the sliding window.
    processes: int
        Number of processes to use.
    model: Word2Vec (:class:`~gensim.models.keyedvectors.KeyedVectors`)
        If None, a new Word2Vec model is trained on the given text corpus. Otherwise,
        it should be a pre-trained Word2Vec context vectors.

    Returns
    -------
    :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`
        Text accumulator with trained context vectors.

    Examples
    --------
    >>> from gensim.topic_coherence import probability_estimation
    >>> from gensim.corpora.hashdictionary import HashDictionary
    >>> from gensim.models import word2vec
    >>> from gensim.corpora.dictionary import Dictionary
    >>> texts = [['human', 'interface', 'computer'],['eps', 'user', 'interface', 'system'],
    >>> ['system', 'human', 'system', 'eps'],['user', 'response', 'time'],['trees'],['graph', 'trees']]
    >>> dictionary = HashDictionary(texts)
    >>> token2id = dictionary.token2id
    >>> computer_id = token2id['computer']
    >>> system_id = token2id['system']
    >>> user_id = token2id['user']
    >>> graph_id = token2id['graph']
    >>> segmented_topics = [[(system_id, graph_id),(computer_id, graph_id),(computer_id, system_id)], [
    >>> (computer_id, graph_id),(user_id, graph_id),(user_id, computer_id)]]
    >>> corpus = [dictionary.doc2bow(text) for text in texts]
    >>> sentences = [['human', 'interface', 'computer'],['survey', 'user', 'computer', 'system', 'response', 'time']]
    >>> model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4) #TODO Ivan fix this holy shield
    >>> accumulator = probability_estimation.p_word2vec(texts, segmented_topics, dictionary, 2, 1, model)
    >>> print accumulator[computer_id], accumulator[user_id], accumulator[graph_id], accumulator[system_id]
    1 3 1 4 # example for model = None
    """
    top_ids = unique_ids_from_segments(segmented_topics)
    accumulator = WordVectorsAccumulator(
        top_ids, dictionary, model, window=window_size, workers=processes)
    return accumulator.accumulate(texts, window_size)


def unique_ids_from_segments(segmented_topics):
    """Return the set of all unique ids in a list of segmented topics.

    Parameters
    ----------
    segmented_topics: list of tuples of (word_id_set1, word_id_set2).
        Each word_id_setis either a single integer, or a `numpy.ndarray` of integers.

    Returns
    -------
    set
        Set of unique ids across all topic segments.

    Example
    -------
    >>> from gensim.topic_coherence import probability_estimation
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
