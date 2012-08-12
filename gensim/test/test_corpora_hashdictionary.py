#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Unit tests for the `corpora.HashDictionary` class.
"""


import logging
import tempfile
import unittest
import os
import os.path

from gensim.corpora.hashdictionary import HashDictionary


# sample data files are located in the same folder
module_path = os.path.dirname(__file__)


def get_tmpfile(suffix):
    return os.path.join(tempfile.gettempdir(), suffix)


class TestHashDictionary(unittest.TestCase):
    def setUp(self):
        self.texts = [
                ['human', 'interface', 'computer'],
                ['survey', 'user', 'computer', 'system', 'response', 'time'],
                ['eps', 'user', 'interface', 'system'],
                ['system', 'human', 'system', 'eps'],
                ['user', 'response', 'time'],
                ['trees'],
                ['graph', 'trees'],
                ['graph', 'minors', 'trees'],
                ['graph', 'minors', 'survey']]

    def testDocFreqOneDoc(self):
        texts = [['human', 'interface', 'computer']]
        d = HashDictionary(texts)
        expected = {10608: 1, 12466: 1, 31002: 1}
        self.assertEqual(d.dfs, expected)

    def testDocFreqAndToken2IdForSeveralDocsWithOneWord(self):
        # two docs
        texts = [['human'], ['human']]
        d = HashDictionary(texts)
        expected = {31002: 2}
        self.assertEqual(d.dfs, expected)
        # only one token (human) should exist
        expected = {'human': 31002}
        self.assertEqual(d.token2id['human'], expected['human'])
        self.assertEqual(d.token2id.keys(), expected.keys())

        # three docs
        texts = [['human'], ['human'], ['human']]
        d = HashDictionary(texts)
        expected = {31002: 3}
        self.assertEqual(d.dfs, expected)
        # only one token (human) should exist
        expected = {'human': 31002}
        self.assertEqual(d.token2id['human'], expected['human'])
        self.assertEqual(d.token2id.keys(), expected.keys())

        # four docs
        texts = [['human'], ['human'], ['human'], ['human']]
        d = HashDictionary(texts)
        expected = {31002: 4}
        self.assertEqual(d.dfs, expected)
        # only one token (human) should exist
        expected = {'human': 31002}
        self.assertEqual(d.token2id['human'], expected['human'])
        self.assertEqual(d.token2id.keys(), expected.keys())

    def testDocFreqForOneDocWithSeveralWord(self):
        # two words
        texts = [['human', 'cat']]
        d = HashDictionary(texts)
        expected = {9273: 1, 31002: 1}
        self.assertEqual(d.dfs, expected)

        # three words
        texts = [['human', 'cat', 'minors']]
        d = HashDictionary(texts)
        expected = {9273: 1, 15001: 1, 31002: 1}
        self.assertEqual(d.dfs, expected)

    def testDebugMode(self):
        # two words
        texts = [['human', 'cat']]
        d = HashDictionary(texts, debug=True)
        expected = {9273: set(['cat']), 31002: set(['human'])}
        self.assertEqual(d.token2id.debug_reverse, expected)

    def testBuild(self):
        d = HashDictionary(self.texts)
        expected =  {5232: 2,
                     5798: 3,
                     10608: 2,
                     12466: 2,
                     12736: 3,
                     15001: 2,
                     18451: 3,
                     23844: 3,
                     28591: 2,
                     29104: 2,
                     31002: 2,
                     31049: 2}

        self.assertEqual(d.dfs, expected)
        expected = {'minors': 15001, 'graph': 18451, 'system': 5798, 'trees': 23844, 'eps': 31049, 'computer': 10608, 'survey': 28591, 'user': 12736, 'human': 31002, 'time': 29104, 'interface': 12466, 'response': 5232}

        for ex in expected:         
            self.assertEqual(d.token2id[ex], expected[ex])

    def testFilter(self):
        d = HashDictionary(self.texts)
        d.filter_extremes(no_below=2, no_above=1.0, keep_n=4)
        expected = {5798: 3, 12736: 3, 18451: 3, 23844: 3}
        self.assertEqual(d.dfs, expected)

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
