#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for probability estimation algorithms in the probability_estimation module.
"""

import logging
import unittest

from gensim.topic_coherence import probability_estimation
from gensim.corpora.hashdictionary import HashDictionary
from gensim.corpora.dictionary import Dictionary


class ProbabilityEstimationBase(unittest.TestCase):
    texts = [['human', 'interface', 'computer'],
             ['eps', 'user', 'interface', 'system'],
             ['system', 'human', 'system', 'eps'],
             ['user', 'response', 'time'],
             ['trees'],
             ['graph', 'trees']]


class TestProbabilityEstimation(ProbabilityEstimationBase):
    def setUp(self):
        self.dictionary = HashDictionary(self.texts)
        # Following is the mapping:
        # {'computer': 10608,
        #  'eps': 31049,
        #  'graph': 18451,
        #  'human': 31002,
        #  'interface': 12466,
        #  'response': 5232,
        #  'system': 5798,
        #  'time': 29104,
        #  'trees': 23844,
        #  'user': 12736}
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        # Suppose the segmented topics from s_one_pre are:
        self.segmented_topics = [
            [
                (5798, 18451),
                (10608, 18451),
                (10608, 5798)
            ], [
                (10608, 18451),
                (12736, 18451),
                (12736, 10608)
            ]
        ]

    def testPBooleanDocument(self):
        """Test p_boolean_document()"""
        # Unique topic ids are 5798, 10608, 12736 and 18451
        obtained, _ = probability_estimation.p_boolean_document(self.corpus, self.segmented_topics)
        expected = {18451: {5}, 12736: {1, 3}, 5798: {1, 2}, 10608: {0}}
        self.assertEqual(expected, obtained)

    def testPBooleanSlidingWindow(self):
        """Test p_boolean_sliding_window()"""
        # Test with window size as 2. window_id is zero indexed.
        obtained, _ = probability_estimation.p_boolean_sliding_window(
            self.texts, self.segmented_topics, self.dictionary, 2)
        expected = {10608: {1}, 12736: {8, 2, 3}, 18451: {11}, 5798: {4, 5, 6, 7}}
        self.assertEqual(expected, obtained)


class TestProbabilityEstimationWithNormalDictionary(ProbabilityEstimationBase):
    def setUp(self):
        self.dictionary = Dictionary(self.texts)
        self.dictionary.id2token = {v: k for k, v in self.dictionary.token2id.items()}
        # Following is the mapping:
        # {u'computer': 1,
        #  u'eps': 5,
        #  u'graph': 9,
        #  u'human': 2,
        #  u'interface': 0,
        #  u'response': 6,
        #  u'system': 4,
        #  u'time': 7,
        #  u'trees': 8,
        #  u'user': 3}
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        # Suppose the segmented topics from s_one_pre are:
        self.segmented_topics = [
            [
                (4, 9),
                (1, 9),
                (1, 4)
            ], [
                (1, 9),
                (3, 9),
                (3, 1)
            ]
        ]

    def testPBooleanDocument(self):
        """Test p_boolean_document()"""
        obtained, _ = probability_estimation.p_boolean_document(self.corpus, self.segmented_topics)
        expected = {9: {5}, 3: {1, 3}, 4: {1, 2}, 1: {0}}
        self.assertEqual(expected, obtained)

    def testPBooleanSlidingWindow(self):
        """Test p_boolean_sliding_window()"""
        # Test with window size as 2. window_id is zero indexed.
        obtained, _ = probability_estimation.p_boolean_sliding_window(
            self.texts, self.segmented_topics, self.dictionary, 2)
        expected = {1: {1}, 3: {8, 2, 3}, 9: {11}, 4: {4, 5, 6, 7}}
        self.assertEqual(expected, obtained)


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
