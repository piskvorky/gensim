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
        expected = {15832: 1, 22675: 1, 24598: 1}
        self.assertEqual(d.dfs, expected)

    def testDocFreqAndToken2IdForSeveralDocsWithOneWord(self):
        # two docs
        texts = [['human'], ['human']]
        d = HashDictionary(texts)
        expected = {24598: 2}
        self.assertEqual(d.dfs, expected)
        # only one token (human) should exist
        expected = {'human': 24598}
        self.assertEqual(d.token2id['human'], expected['human'])
        self.assertEqual(d.token2id.keys(), expected.keys())

        # three docs
        texts = [['human'], ['human'], ['human']]
        d = HashDictionary(texts)
        expected = {24598: 3}
        self.assertEqual(d.dfs, expected)
        # only one token (human) should exist
        expected = {'human': 24598}
        self.assertEqual(d.token2id['human'], expected['human'])
        self.assertEqual(d.token2id.keys(), expected.keys())

        # four docs
        texts = [['human'], ['human'], ['human'], ['human']]
        d = HashDictionary(texts)
        expected = {24598: 4}
        self.assertEqual(d.dfs, expected)
        # only one token (human) should exist
        expected = {'human': 24598}
        self.assertEqual(d.token2id['human'], expected['human'])
        self.assertEqual(d.token2id.keys(), expected.keys())

    def testDocFreqForOneDocWithSeveralWord(self):
        # two words
        texts = [['human', 'cat']]
        d = HashDictionary(texts)
        expected = {19055: 1, 24598: 1}
        self.assertEqual(d.dfs, expected)

        # three words
        texts = [['human', 'cat', 'minors']]
        d = HashDictionary(texts)
        expected = {19055: 1, 24598: 1, 27396: 1}
        self.assertEqual(d.dfs, expected)

    def testDebugMode(self):
        # two words
        texts = [['human', 'cat']]
        d = HashDictionary(texts, debug=True)
        expected = {19055: set(['cat']), 24598: set(['human'])}
        self.assertEqual(d.token2id.debug_reverse, expected)

    def testBuild(self):
        d = HashDictionary(self.texts)
        expected =   {12269: 2,
                      15832: 2,
                      19925: 3,
                      22675: 2,
                      24564: 2,
                      24598: 2,
                      25678: 3,
                      27396: 2,
                      27639: 2,
                      28125: 2,
                      28973: 3,
                      29993: 3}
        
        self.assertEqual(d.dfs, expected)
        expected = {'minors': 27396, 'graph': 29993, 'eps': 12269, 'trees': 25678, 'system': 28973, 'computer': 22675, 'survey': 24564, 'user': 19925, 'human': 24598, 'time': 27639, 'interface': 15832, 'response': 28125}

        for ex in expected:         
            self.assertEqual(d.token2id[ex], expected[ex])

    def testFilter(self):
        d = HashDictionary(self.texts)
        d.filter_extremes(no_below=2, no_above=1.0, keep_n=4)
        expected = {19925: 3, 25678: 3, 28973: 3, 29993: 3}
        self.assertEqual(d.dfs, expected)

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
