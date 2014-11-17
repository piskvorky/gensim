#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""


import logging
import unittest
import os
import sys

from gensim import utils
from gensim.models.phrases import Phrases

if sys.version_info[0] >= 3:
    unicode = str

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


sentences = [
    ['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey']
]


class TestPhrasesModel(unittest.TestCase):
    def testSentenceGeneration(self):
        """Test basic bigram using a dummy corpus."""
        bigram = Phrases(sentences)
        # test that we generate the same amount of sentences as the input
        self.assertEqual(len(sentences), len(list(bigram[sentences])))

    def testBigramConstruction(self):
        """Test Phrases bigram construction building."""
        bigram = Phrases(sentences, min_count=1, threshold=1)

        # with this setting we should get response_time and graph_minors
        bigram1_seen = False
        bigram2_seen = False

        for s in bigram[sentences]:
            if u'response_time' in s:
                bigram1_seen = True
            if u'graph_minors' in s:
                bigram2_seen = True
        self.assertTrue(bigram1_seen and bigram2_seen)

        # check the same thing, this time using single doc transformation
        self.assertTrue(u'response_time' in bigram[sentences[1]])
        self.assertTrue(u'response_time' in bigram[sentences[4]])
        self.assertTrue(u'graph_minors' in bigram[sentences[-2]])
        self.assertTrue(u'graph_minors' in bigram[sentences[-1]])

    def testBadParameters(self):
        """Test the phrases module with bad parameters."""
        # should fail with something less or equal than 0
        self.assertRaises(ValueError, Phrases, sentences, min_count=0)

        # threshold should be positive
        self.assertRaises(ValueError, Phrases, sentences, threshold=-1)

    def testEncoding(self):
        """Test that both utf8 and unicode input work; output must be unicode."""
        expected = [u'survey', u'user', u'computer', u'system', u'response_time']

        bigram_utf8 = Phrases(sentences, min_count=1, threshold=1)
        self.assertEquals(bigram_utf8[sentences[1]], expected)

        unicode_sentences = [[utils.to_unicode(w) for w in sentence] for sentence in sentences]
        bigram_unicode = Phrases(unicode_sentences, min_count=1, threshold=1)
        self.assertEquals(bigram_unicode[sentences[1]], expected)

        transformed = ' '.join(bigram_utf8[sentences[1]])
        self.assertTrue(isinstance(transformed, unicode))

    def testPruning(self):
        """Test that max_vocab_size parameter is respected."""
        bigram = Phrases(sentences, max_vocab_size=5)
        self.assertTrue(len(bigram.vocab) <= 5)
#endclass TestPhrasesModel


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
