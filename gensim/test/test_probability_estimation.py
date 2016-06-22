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

class TestProbabilityEstimation(unittest.TestCase):
    def setUp(self):
        self.texts = [['human', 'interface', 'computer'],
                      ['eps', 'user', 'interface', 'system'],
                      ['system', 'human', 'system', 'eps'],
                      ['user', 'response', 'time'],
                      ['trees'],
                      ['graph', 'trees']]
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
        self.segmented_topics = [[(5798, 18451), (10608, 18451), (10608, 5798)], [(10608, 18451), (12736, 18451), (12736, 10608)]]

    def testPBooleanDocument(self):
        """Test p_boolean_document()"""
        # Unique topic ids are 5798, 10608, 12736 and 18451
        obtained, _ = probability_estimation.p_boolean_document(self.corpus, self.segmented_topics)
        expected = {18451: set([5]), 12736: set([1, 3]), 5798: set([1, 2]), 10608: set([0])}
        self.assertTrue(obtained == expected)

    def testPBooleanSlidingWindow(self):
        """Test p_boolean_sliding_window()"""
        # Test with window size as 2. window_id is zero indexed.
        obtained, _ = probability_estimation.p_boolean_sliding_window(self.texts, self.segmented_topics, self.dictionary, 2)
        expected = {10608: set([1]), 12736: set([8, 2, 3]), 18451: set([11]), 5798: set([4, 5, 6, 7])}
        self.assertTrue(obtained == expected)

if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
