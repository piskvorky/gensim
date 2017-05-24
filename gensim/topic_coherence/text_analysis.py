#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains classes for analyzing the texts of a corpus to accumulate
statistical information about word occurrences.
"""

import itertools

import numpy as np
import scipy.sparse as sps

from gensim import utils


def _ids_to_words(ids, dictionary):
    """Convert an iterable of ids to their corresponding words using a dictionary.
    This function abstracts away the differences between the HashDictionary and the standard one.
    """
    if not dictionary.id2token:
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
        self.id2contiguous = {word_id: n for n, word_id in enumerate(self.relevant_ids)}
        self._num_docs = 0

    @property
    def num_docs(self):
        return self._num_docs

    def analyze_text(self, text):
        raise NotImplementedError("Base classes should implement analyze_text.")

    def __getitem__(self, word_or_words):
        if hasattr(word_or_words, '__iter__'):
            return self.get_co_occurrences(*word_or_words)
        else:
            return self.get_occurrences(word_or_words)

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
    """Base class for corpus and text analyzers."""

    def __init__(self, relevant_ids, dictionary):
        super(UsesDictionary, self).__init__(relevant_ids)
        self.relevant_words = _ids_to_words(self.relevant_ids, dictionary)
        self.token2id = dictionary.token2id

    def analyze_text(self, text):
        raise NotImplementedError("Base classes should implement analyze_text.")

    def get_occurrences(self, word):
        """Return number of docs the word occurs in, once `accumulate` has been called."""
        try:
            word_id = self.token2id[word]
        except KeyError:
            word_id = word
        return self._get_occurrences(self.id2contiguous[word_id])

    def get_co_occurrences(self, word1, word2):
        """Return number of docs the words co-occur in, once `accumulate` has been called."""
        try:
            word_id1 = self.token2id[word1]
        except KeyError:
            word_id1 = word1
        try:
            word_id2 = self.token2id[word2]
        except KeyError:
            word_id2 = word2
        return self._get_co_occurrences(self.id2contiguous[word_id1], self.id2contiguous[word_id2])


class InvertedIndexBased(BaseAnalyzer):
    """Analyzer that builds up an inverted index to accumulate stats."""

    def __init__(self, *args):
        super(InvertedIndexBased, self).__init__(*args)
        vocab_size = len(self.relevant_ids)
        self._inverted_index = np.array([set() for _ in range(vocab_size)])

    def _get_occurrences(self, word_id):
        return len(self._inverted_index[word_id])

    def _get_co_occurrences(self, word_id1, word_id2):
        s1 = self._inverted_index[word_id1]
        s2 = self._inverted_index[word_id2]
        return len(s1.intersection(s2))

    def index_to_dict(self):
        contiguous2id = {n: word_id for word_id, n in self.id2contiguous.iteritems()}
        return {contiguous2id[n]: doc_id_list for n, doc_id_list in enumerate(self._inverted_index)}


class CorpusAnalyzer(InvertedIndexBased):
    """Gather word occurrence stats from a corpus by iterating over its BoW representation."""

    def analyze_text(self, text):
        doc_words = frozenset(x[0] for x in text)
        top_ids_in_doc = self.relevant_ids.intersection(doc_words)
        if len(top_ids_in_doc) > 0:
            for word_id in top_ids_in_doc:
                self._inverted_index[self.id2contiguous[word_id]].add(self._num_docs)

    def accumulate(self, corpus):
        for document in corpus:
            self.analyze_text(document)
            self._num_docs += 1
        return self


class TextsAnalyzer(UsesDictionary):
    """Gather some statistics about relevant terms a corpus by iterating over texts."""

    def __init__(self, relevant_ids, dictionary):
        """
        Args:
        ----
        relevant_words: the set of words that occurrences should be accumulated for.
        """
        super(TextsAnalyzer, self).__init__(relevant_ids, dictionary)

    def filter_to_relevant_words(self, text):
        """Lazily filter the text to only those words which are relevant."""
        relevant_words = (word for word in text if word in self.relevant_words)
        relevant_ids = (self.token2id[word] for word in relevant_words)
        return (self.id2contiguous[word_id] for word_id in relevant_ids)

    def text_is_relevant(self, text):
        """Return True if the text has any relevant words, else False."""
        for word in text:
            if word in self.relevant_words:
                return True
        return False

    def accumulate(self, texts, window_size):
        relevant_texts = (text for text in texts if self.text_is_relevant(text))
        for virtual_document in utils.iter_windows(relevant_texts, window_size, ignore_below_size=False):
            self.analyze_text(virtual_document)
            self._num_docs += 1
        return self


class InvertedIndexAccumulator(TextsAnalyzer, InvertedIndexBased):
    """Build an inverted index from a sequence of corpus texts."""

    def analyze_text(self, window):
        for word_id in self.filter_to_relevant_words(window):
            self._inverted_index[word_id].add(self._num_docs)


class WordOccurrenceAccumulator(TextsAnalyzer):
    """Accumulate word occurrences and co-occurrences from a corpus of texts."""

    def __init__(self, *args):
        super(WordOccurrenceAccumulator, self).__init__(*args)
        vocab_size = len(self.relevant_words)
        self._occurrences = np.zeros(vocab_size, dtype='uint32')
        self._co_occurrences = sps.lil_matrix((vocab_size, vocab_size), dtype='uint32')

    def analyze_text(self, window):
        relevant_words = list(self.filter_to_relevant_words(window))
        uniq_words = np.array(relevant_words)
        self._occurrences[uniq_words] += 1

        for combo in itertools.combinations(relevant_words, 2):
            self._co_occurrences[combo] += 1

    def _symmetrize(self):
        co_occ = self._co_occurrences
        return co_occ + co_occ.T - np.diag(co_occ.diagonal())

    def accumulate(self, texts, window_size):
        super(WordOccurrenceAccumulator, self).accumulate(texts, window_size)
        self._symmetrize()
        return self

    def _get_occurrences(self, word_id):
        return self._occurrences[word_id]

    def _get_co_occurrences(self, word_id1, word_id2):
        return self._co_occurrences[word_id1, word_id2]
