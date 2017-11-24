#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Unit tests for the `corpora.HashDictionary` class.
"""


import logging
import unittest
import os
import zlib

from gensim.corpora.hashdictionary import HashDictionary
from gensim.test.utils import get_tmpfile, common_texts


class TestHashDictionary(unittest.TestCase):
    def setUp(self):
        self.texts = common_texts

    def testDocFreqOneDoc(self):
        texts = [['human', 'interface', 'computer']]
        d = HashDictionary(texts, myhash=zlib.adler32)
        expected = {10608: 1, 12466: 1, 31002: 1}
        self.assertEqual(d.dfs, expected)

    def testDocFreqAndToken2IdForSeveralDocsWithOneWord(self):
        # two docs
        texts = [['human'], ['human']]
        d = HashDictionary(texts, myhash=zlib.adler32)
        expected = {31002: 2}
        self.assertEqual(d.dfs, expected)
        # only one token (human) should exist
        expected = {'human': 31002}
        self.assertEqual(d.token2id['human'], expected['human'])
        self.assertEqual(d.token2id.keys(), expected.keys())

        # three docs
        texts = [['human'], ['human'], ['human']]
        d = HashDictionary(texts, myhash=zlib.adler32)
        expected = {31002: 3}
        self.assertEqual(d.dfs, expected)
        # only one token (human) should exist
        expected = {'human': 31002}
        self.assertEqual(d.token2id['human'], expected['human'])
        self.assertEqual(d.token2id.keys(), expected.keys())

        # four docs
        texts = [['human'], ['human'], ['human'], ['human']]
        d = HashDictionary(texts, myhash=zlib.adler32)
        expected = {31002: 4}
        self.assertEqual(d.dfs, expected)
        # only one token (human) should exist
        expected = {'human': 31002}
        self.assertEqual(d.token2id['human'], expected['human'])
        self.assertEqual(d.token2id.keys(), expected.keys())

    def testDocFreqForOneDocWithSeveralWord(self):
        # two words
        texts = [['human', 'cat']]
        d = HashDictionary(texts, myhash=zlib.adler32)
        expected = {9273: 1, 31002: 1}
        self.assertEqual(d.dfs, expected)

        # three words
        texts = [['human', 'cat', 'minors']]
        d = HashDictionary(texts, myhash=zlib.adler32)
        expected = {9273: 1, 15001: 1, 31002: 1}
        self.assertEqual(d.dfs, expected)

    def testDebugMode(self):
        # two words
        texts = [['human', 'cat']]
        d = HashDictionary(texts, debug=True, myhash=zlib.adler32)
        expected = {9273: {'cat'}, 31002: {'human'}}
        self.assertEqual(d.id2token, expected)

        # now the same thing, with debug off
        texts = [['human', 'cat']]
        d = HashDictionary(texts, debug=False, myhash=zlib.adler32)
        expected = {}
        self.assertEqual(d.id2token, expected)

    def testRange(self):
        # all words map to the same id
        d = HashDictionary(self.texts, id_range=1, debug=True)
        dfs = {0: 9}
        id2token = {
            0: {
                'minors', 'graph', 'system', 'trees', 'eps', 'computer',
                'survey', 'user', 'human', 'time', 'interface', 'response'
            }
        }
        token2id = {
            'minors': 0, 'graph': 0, 'system': 0, 'trees': 0,
            'eps': 0, 'computer': 0, 'survey': 0, 'user': 0,
            'human': 0, 'time': 0, 'interface': 0, 'response': 0
        }
        self.assertEqual(d.dfs, dfs)
        self.assertEqual(d.id2token, id2token)
        self.assertEqual(d.token2id, token2id)

        # 2 ids: 0/1 for even/odd number of bytes in the word
        d = HashDictionary(self.texts, id_range=2, myhash=lambda key: len(key))
        dfs = {0: 7, 1: 7}
        id2token = {
            0: {'minors', 'system', 'computer', 'survey', 'user', 'time', 'response'},
            1: {'interface', 'graph', 'trees', 'eps', 'human'}
        }
        token2id = {
            'minors': 0, 'graph': 1, 'system': 0, 'trees': 1, 'eps': 1, 'computer': 0,
            'survey': 0, 'user': 0, 'human': 1, 'time': 0, 'interface': 1, 'response': 0
        }
        self.assertEqual(d.dfs, dfs)
        self.assertEqual(d.id2token, id2token)
        self.assertEqual(d.token2id, token2id)

    def testBuild(self):
        d = HashDictionary(self.texts, myhash=zlib.adler32)
        expected = {
            5232: 2, 5798: 3, 10608: 2, 12466: 2, 12736: 3, 15001: 2,
            18451: 3, 23844: 3, 28591: 2, 29104: 2, 31002: 2, 31049: 2
        }

        self.assertEqual(d.dfs, expected)
        expected = {
            'minors': 15001, 'graph': 18451, 'system': 5798, 'trees': 23844,
            'eps': 31049, 'computer': 10608, 'survey': 28591, 'user': 12736,
            'human': 31002, 'time': 29104, 'interface': 12466, 'response': 5232
        }

        for ex in expected:
            self.assertEqual(d.token2id[ex], expected[ex])

    def testFilter(self):
        d = HashDictionary(self.texts, myhash=zlib.adler32)
        d.filter_extremes()
        expected = {}
        self.assertEqual(d.dfs, expected)

        d = HashDictionary(self.texts, myhash=zlib.adler32)
        d.filter_extremes(no_below=0, no_above=0.3)
        expected = {
            29104: 2, 31049: 2, 28591: 2, 5232: 2,
            10608: 2, 12466: 2, 15001: 2, 31002: 2
        }
        self.assertEqual(d.dfs, expected)

        d = HashDictionary(self.texts, myhash=zlib.adler32)
        d.filter_extremes(no_below=3, no_above=1.0, keep_n=4)
        expected = {5798: 3, 12736: 3, 18451: 3, 23844: 3}
        self.assertEqual(d.dfs, expected)

    def test_saveAsText(self):
        """ `HashDictionary` can be saved as textfile. """
        tmpf = get_tmpfile('dict_test.txt')
        # use some utf8 strings, to test encoding serialization
        d = HashDictionary(['žloťoučký koníček'.split(), 'Малйж обльйквюэ ат эжт'.split()])
        d.save_as_text(tmpf)
        self.assertTrue(os.path.exists(tmpf))

    def test_saveAsTextBz2(self):
        """ `HashDictionary` can be saved & loaded as compressed pickle. """
        tmpf = get_tmpfile('dict_test.txt.bz2')
        # use some utf8 strings, to test encoding serialization
        d = HashDictionary(['žloťoučký koníček'.split(), 'Малйж обльйквюэ ат эжт'.split()])
        d.save(tmpf)
        self.assertTrue(os.path.exists(tmpf))
        d2 = d.load(tmpf)
        self.assertEqual(len(d), len(d2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
