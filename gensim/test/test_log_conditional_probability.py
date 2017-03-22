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
from gensim.topic_coherence import direct_confirmation_measure, indirect_confirmation_measure

from gensim.corpora.hashdictionary import HashDictionary

class TestLogConditionalProbablity(unittest.TestCase):
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

    def testAggregationTruePBooleanDocument(self):
        """Test p_boolean_document()"""
        # Unique topic ids are 5798, 10608, 12736 and 18451
        per_topic_postings, _ = probability_estimation.p_boolean_document(self.corpus, self.segmented_topics)
        num_docs=_
        aggregated = True
        New_obtain = direct_confirmation_measure.log_conditional_probability(self.segmented_topics, per_topic_postings, num_docs, aggregated=True)
        expected =[-25.839261646700493,-25.839261646700493,-26.53240882726044,-25.839261646700493,-25.839261646700493,-25.839261646700493]
        self.assertTrue(New_obtain== expected)

    def testAggregationFalsePBooleanDocument(self):
        per_topic_postings, _ = probability_estimation.p_boolean_document(self.corpus, self.segmented_topics)
        num_docs = _
        aggregated = False
        New_obtain = direct_confirmation_measure.log_conditional_probability(self.segmented_topics, per_topic_postings, num_docs, aggregated=False)
        expected = [[-25.839261646700493, -25.839261646700493, -26.53240882726044], [-25.839261646700493, -25.839261646700493, -25.839261646700493]]
        print(New_obtain)
        self.assertTrue(New_obtain == expected)


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
