#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Unit tests for the `corpora.Dictionary` class.
"""


import logging
import tempfile
import unittest
import os
import os.path

from gensim.corpora import Dictionary


# sample data files are located in the same folder
module_path = os.path.dirname(__file__)


def get_tmpfile(suffix):
    return os.path.join(tempfile.gettempdir(), suffix)


class TestDictionary(unittest.TestCase):
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
        d = Dictionary(texts)
        expected = {0: 1, 1: 1, 2: 1}
        self.assertEqual(d.dfs, expected)

    def testDocFreqAndToken2IdForSeveralDocsWithOneWord(self):
        # two docs
        texts = [['human'], ['human']]
        d = Dictionary(texts)
        expected = {0: 2}
        self.assertEqual(d.dfs, expected)
        # only one token (human) should exist
        expected = {'human': 0}
        self.assertEqual(d.token2id, expected)

        # three docs
        texts = [['human'], ['human'], ['human']]
        d = Dictionary(texts)
        expected = {0: 3}
        self.assertEqual(d.dfs, expected)
        # only one token (human) should exist
        expected = {'human': 0}
        self.assertEqual(d.token2id, expected)

        # four docs
        texts = [['human'], ['human'], ['human'], ['human']]
        d = Dictionary(texts)
        expected = {0: 4}
        self.assertEqual(d.dfs, expected)
        # only one token (human) should exist
        expected = {'human': 0}
        self.assertEqual(d.token2id, expected)

    def testDocFreqForOneDocWithSeveralWord(self):
        # two words
        texts = [['human', 'cat']]
        d = Dictionary(texts)
        expected = {0: 1, 1: 1}
        self.assertEqual(d.dfs, expected)

        # three words
        texts = [['human', 'cat', 'minors']]
        d = Dictionary(texts)
        expected = {0: 1, 1: 1, 2: 1}
        self.assertEqual(d.dfs, expected)

    def testBuild(self):
        d = Dictionary(self.texts)
        expected = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 2, 7: 3, 8: 2,
                9: 3, 10: 3, 11: 2}
        self.assertEqual(d.dfs, expected)

        expected = {'computer': 0, 'eps': 8, 'graph': 10, 'human': 1,
                'interface': 2, 'minors': 11, 'response': 3, 'survey': 4,
                'system': 5, 'time': 6, 'trees': 9, 'user': 7}
        self.assertEqual(d.token2id, expected)

    def testFilter(self):
        d = Dictionary(self.texts)
        d.filter_extremes(no_below=2, no_above=1.0, keep_n=4)
        expected = {0: 3, 1: 3, 2: 3, 3: 3}
        self.assertEqual(d.dfs, expected)

    def test_saveAsText_and_loadFromText(self):
        """ `Dictionary` can be saved as textfile and loaded again from textfile. """
        tmpf = get_tmpfile('dict_test.txt')
        d = Dictionary(self.texts)
        d.save_as_text(tmpf)
        # does the file exists
        self.assertTrue(os.path.exists(tmpf))

        d_loaded = Dictionary.load_from_text(get_tmpfile('dict_test.txt'))
        self.assertNotEqual(d_loaded, None)
        self.assertEqual(d_loaded.token2id, d.token2id)
#endclass TestDictionary


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
