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
import tempfile

from gensim.models.phrases import Phrases

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


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_word2vec.tst')


class TestPhrasesModel(unittest.TestCase):
    def testSentenceGeneration(self):
        """
        Test basic bigram using a dummy corpus
        """
        bigram = Phrases(sentences)
        # Test that we generate the same amount of senteces as the inpuyt
        self.assertEqual(len(sentences), len(list(bigram[sentences])))

    def testBigramConstruction(self):
        """ Test Phrases bigram construction building   """

        bigram = Phrases(sentences, min_count=1, threshold=1)

        # with this setting we should get response_time and graph_minors
        bigram1_seen = False
        bigram2_seen = False

        for s in bigram[sentences]:
            if 'response_time' in s:
                bigram1_seen = True
            if 'graph_minors' in s:
                bigram2_seen = True

        self.assertTrue(bigram1_seen and bigram2_seen)

    def testBadParameters(self):
        """ Test the phrases module with bad parameters    """

        # should fail with something less or equal than 0
        self.assertRaises(ValueError, Phrases, sentences, min_count=0)

        # threshold should be positive
        self.assertRaises(ValueError, Phrases, sentences, threshold=-1)
#endclass TestPhrasesModel

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
