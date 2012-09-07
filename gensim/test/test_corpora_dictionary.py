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
        """`Dictionary` can be saved as textfile and loaded again from textfile. """
        tmpf = get_tmpfile('dict_test.txt')
        d = Dictionary(self.texts)
        d.save_as_text(tmpf)
        # does the file exists
        self.assertTrue(os.path.exists(tmpf))

        d_loaded = Dictionary.load_from_text(get_tmpfile('dict_test.txt'))
        self.assertNotEqual(d_loaded, None)
        self.assertEqual(d_loaded.token2id, d.token2id)

    def test_from_corpus(self):
        """build `Dictionary` from an existing corpus"""

        documents = ["Human machine interface for lab abc computer applications",
                "A survey of user opinion of computer system response time",
                "The EPS user interface management system",
                "System and human system engineering testing of EPS",
                "Relation of user perceived response time to error measurement",
                "The generation of random binary unordered trees",
                "The intersection graph of paths in trees",
                "Graph minors IV Widths of trees and well quasi ordering",
                "Graph minors A survey"]
        stoplist = set('for a of the and to in'.split())
        texts = [[word for word in document.lower().split() if word not in stoplist]
                for document in documents]

        # remove words that appear only once
        all_tokens = sum(texts, [])
        tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
        texts = [[word for word in text if word not in tokens_once]
                for text in texts]
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        dictionary_from_corpus = Dictionary.from_corpus(corpus)

        #we have to compare values, because in creating dictionary from corpus
        #informations about words are lost
        dict_token2id_vals = sorted(dictionary.token2id.values())
        dict_from_corpus_vals = sorted(dictionary_from_corpus.token2id.values())
        self.assertEqual(dict_token2id_vals, dict_from_corpus_vals)
        self.assertEqual(dictionary.dfs, dictionary_from_corpus.dfs)
        self.assertEqual(dictionary.num_docs, dictionary_from_corpus.num_docs)
        self.assertEqual(dictionary.num_pos, dictionary_from_corpus.num_pos)
        self.assertEqual(dictionary.num_nnz, dictionary_from_corpus.num_nnz)
#endclass TestDictionary


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
