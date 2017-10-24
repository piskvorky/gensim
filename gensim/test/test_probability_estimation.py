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

from gensim.corpora.dictionary import Dictionary
from gensim.corpora.hashdictionary import HashDictionary
from gensim.models._coherence import probability_estimation


class BaseTestCases(object):

    class ProbabilityEstimationBase(unittest.TestCase):
        texts = [
            ['human', 'interface', 'computer'],
            ['eps', 'user', 'interface', 'system'],
            ['system', 'human', 'system', 'eps'],
            ['user', 'response', 'time'],
            ['trees'],
            ['graph', 'trees']
        ]
        dictionary = None

        def build_segmented_topics(self):
            # Suppose the segmented topics from s_one_pre are:
            token2id = self.dictionary.token2id
            computer_id = token2id['computer']
            system_id = token2id['system']
            user_id = token2id['user']
            graph_id = token2id['graph']
            self.segmented_topics = [
                [
                    (system_id, graph_id),
                    (computer_id, graph_id),
                    (computer_id, system_id)
                ], [
                    (computer_id, graph_id),
                    (user_id, graph_id),
                    (user_id, computer_id)
                ]
            ]

            self.computer_id = computer_id
            self.system_id = system_id
            self.user_id = user_id
            self.graph_id = graph_id

        def setup_dictionary(self):
            raise NotImplementedError

        def setUp(self):
            self.setup_dictionary()
            self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
            self.build_segmented_topics()

        def testPBooleanDocument(self):
            """Test p_boolean_document()"""
            accumulator = probability_estimation.p_boolean_document(
                self.corpus, self.segmented_topics)
            obtained = accumulator.index_to_dict()
            expected = {
                self.graph_id: {5},
                self.user_id: {1, 3},
                self.system_id: {1, 2},
                self.computer_id: {0}
            }
            self.assertEqual(expected, obtained)

        def testPBooleanSlidingWindow(self):
            """Test p_boolean_sliding_window()"""
            # Test with window size as 2. window_id is zero indexed.
            accumulator = probability_estimation.p_boolean_sliding_window(
                self.texts, self.segmented_topics, self.dictionary, 2)
            self.assertEqual(1, accumulator[self.computer_id])
            self.assertEqual(3, accumulator[self.user_id])
            self.assertEqual(1, accumulator[self.graph_id])
            self.assertEqual(4, accumulator[self.system_id])


class TestProbabilityEstimation(BaseTestCases.ProbabilityEstimationBase):
    def setup_dictionary(self):
        self.dictionary = HashDictionary(self.texts)


class TestProbabilityEstimationWithNormalDictionary(BaseTestCases.ProbabilityEstimationBase):
    def setup_dictionary(self):
        self.dictionary = Dictionary(self.texts)
        self.dictionary.id2token = {v: k for k, v in self.dictionary.token2id.items()}


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
