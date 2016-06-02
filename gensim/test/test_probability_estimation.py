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

from gensim import probability_estimation
from gensim.corpora.dictionary import Dictionary

class TestProbabilityEstimation(unittest.TestCase):
    def setUp(self):
        self.texts = [['human', 'interface', 'computer'],
                      ['eps', 'user', 'interface', 'system'],
                      ['system', 'human', 'system', 'eps'],
                      ['user', 'response', 'time'],
                      ['trees'],
                      ['graph', 'trees']]
        self.dictionary = Dictionary(self.texts)
        # Following is the mapping:
        # {u'graph': 9, u'eps': 5, u'trees': 8, u'system': 4, u'computer': 1, u'user': 3, u'human': 2, u'time': 7, u'interface': 0, u'response': 6}
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        # Suppose the segmented topics from s_one_pre are:
        self.segmented_topics = [[(4, 9), (1, 9), (1, 4)], [(1, 9), (3, 9), (3, 1)]]

    def testPBooleanDocument(self):
        """Test p_boolean_document()"""
        # Unique topic ids are 1, 3, 4 and 9
        obtained = probability_estimation.p_boolean_document(self.corpus, self.segmented_topics)
        expected = {9: set([5]), 3: set([1, 3]), 4: set([1, 2]), 1: set([0])}
        self.assertTrue(obtained == expected)

    def testPBooleanSlidingWindow(self):
        """Test p_boolean_sliding_window()"""
        # Test with window size as 2. window_id is zero indexed.
        obtained = probability_estimation.p_boolean_sliding_window(self.texts, self.segmented_topics, self.dictionary, 2)
        expected = {1: set([1]), 3: set([8, 2, 3]), 4: set([4, 5, 6, 7]), 9: set([11])}
        self.assertTrue(obtained == expected)

if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
