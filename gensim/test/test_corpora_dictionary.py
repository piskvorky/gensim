#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Unit tests for the `corpora.Dictionary` class.
"""


from collections import Mapping
import logging
import tempfile
import unittest
import os
import os.path

import scipy
import gensim
from gensim.corpora import Dictionary
from six import PY3
from six.moves import zip


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

        # Since we don't specify the order in which dictionaries are built,
        # we cannot reliably test for the mapping; only the keys and values.
        expected_keys = list(range(12))
        expected_values = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]
        self.assertEqual(sorted(d.dfs.keys()), expected_keys)
        self.assertEqual(sorted(d.dfs.values()), expected_values)

        expected_keys = sorted(['computer', 'eps', 'graph', 'human',
                                'interface', 'minors', 'response', 'survey',
                                'system', 'time', 'trees', 'user'])
        expected_values = list(range(12))
        self.assertEqual(sorted(d.token2id.keys()), expected_keys)
        self.assertEqual(sorted(d.token2id.values()), expected_values)

    def testMerge(self):
        d = Dictionary(self.texts)
        f = Dictionary(self.texts[:3])
        g = Dictionary(self.texts[3:])

        f.merge_with(g)
        self.assertEqual(sorted(d.token2id.keys()), sorted(f.token2id.keys()))

    def testFilter(self):
        d = Dictionary(self.texts)
        d.filter_extremes(no_below=2, no_above=1.0, keep_n=4)
        expected = {0: 3, 1: 3, 2: 3, 3: 3}
        self.assertEqual(d.dfs, expected)

    def testFilterKeepTokens_keepTokens(self):
        # provide keep_tokens argument, keep the tokens given
        d = Dictionary(self.texts)
        d.filter_extremes(no_below=3, no_above=1.0, keep_tokens=['human', 'survey'])
        expected = set(['graph', 'trees', 'human', 'system', 'user', 'survey'])
        self.assertEqual(set(d.token2id.keys()), expected)

    def testFilterKeepTokens_unchangedFunctionality(self):
        # do not provide keep_tokens argument, filter_extremes functionality is unchanged
        d = Dictionary(self.texts)
        d.filter_extremes(no_below=3, no_above=1.0)
        expected = set(['graph', 'trees', 'system', 'user'])
        self.assertEqual(set(d.token2id.keys()), expected)

    def testFilterKeepTokens_unseenToken(self):
        # do provide keep_tokens argument with unseen tokens, filter_extremes functionality is unchanged
        d = Dictionary(self.texts)
        d.filter_extremes(no_below=3, no_above=1.0, keep_tokens=['unknown_token'])
        expected = set(['graph', 'trees', 'system', 'user'])
        self.assertEqual(set(d.token2id.keys()), expected)

    def testFilterMostFrequent(self):
        d = Dictionary(self.texts)
        d.filter_n_most_frequent(4)
        expected = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2}
        self.assertEqual(d.dfs, expected)

    def testFilterTokens(self):
        self.maxDiff = 10000
        d = Dictionary(self.texts)

        removed_word = d[0]
        d.filter_tokens([0])

        expected = {'computer': 0, 'eps': 8, 'graph': 10, 'human': 1,
                    'interface': 2, 'minors': 11, 'response': 3, 'survey': 4,
                    'system': 5, 'time': 6, 'trees': 9, 'user': 7}
        del expected[removed_word]
        self.assertEqual(sorted(d.token2id.keys()), sorted(expected.keys()))

        expected[removed_word] = len(expected)
        d.add_documents([[removed_word]])
        self.assertEqual(sorted(d.token2id.keys()), sorted(expected.keys()))

    def test_doc2bow(self):
        d = Dictionary([["žluťoučký"], ["žluťoučký"]])

        # pass a utf8 string
        self.assertEqual(d.doc2bow(["žluťoučký"]), [(0, 1)])

        # doc2bow must raise a TypeError if passed a string instead of array of strings by accident
        self.assertRaises(TypeError, d.doc2bow, "žluťoučký")

        # unicode must be converted to utf8
        self.assertEqual(d.doc2bow([u'\u017elu\u0165ou\u010dk\xfd']), [(0, 1)])

    def test_saveAsText(self):
        """`Dictionary` can be saved as textfile. """
        tmpf = get_tmpfile('save_dict_test.txt')
        small_text = [["prvé", "slovo"],
                      ["slovo", "druhé"],
                      ["druhé", "slovo"]]

        d = Dictionary(small_text)

        d.save_as_text(tmpf)
        with open(tmpf) as file:
            serialized_lines = file.readlines()
            self.assertEqual(serialized_lines[0], "3\n")
            self.assertEqual(len(serialized_lines), 4)
            # We do not know, which word will have which index
            self.assertEqual(serialized_lines[1][1:], "\tdruhé\t2\n")
            self.assertEqual(serialized_lines[2][1:], "\tprvé\t1\n")
            self.assertEqual(serialized_lines[3][1:], "\tslovo\t3\n")

        d.save_as_text(tmpf, sort_by_word=False)
        with open(tmpf) as file:
            serialized_lines = file.readlines()
            self.assertEqual(serialized_lines[0], "3\n")
            self.assertEqual(len(serialized_lines), 4)
            self.assertEqual(serialized_lines[1][1:], "\tslovo\t3\n")
            self.assertEqual(serialized_lines[2][1:], "\tdruhé\t2\n")
            self.assertEqual(serialized_lines[3][1:], "\tprvé\t1\n")

    def test_loadFromText(self):
        tmpf = get_tmpfile('load_dict_test.txt')
        no_num_docs_serialization = "1\tprvé\t1\n2\tslovo\t2\n"
        with open(tmpf, "w") as file:
            file.write(no_num_docs_serialization)

        d = Dictionary.load_from_text(tmpf)
        self.assertEqual(d.token2id["prvé"], 1)
        self.assertEqual(d.token2id["slovo"], 2)
        self.assertEqual(d.dfs[1], 1)
        self.assertEqual(d.dfs[2], 2)
        self.assertEqual(d.num_docs, 0)

        no_num_docs_serialization = "2\n1\tprvé\t1\n2\tslovo\t2\n"
        with open(tmpf, "w") as file:
            file.write(no_num_docs_serialization)

        d = Dictionary.load_from_text(tmpf)
        self.assertEqual(d.token2id["prvé"], 1)
        self.assertEqual(d.token2id["slovo"], 2)
        self.assertEqual(d.dfs[1], 1)
        self.assertEqual(d.dfs[2], 2)
        self.assertEqual(d.num_docs, 2)

    def test_saveAsText_and_loadFromText(self):
        """`Dictionary` can be saved as textfile and loaded again from textfile. """
        tmpf = get_tmpfile('dict_test.txt')
        for sort_by_word in [True, False]:
            d = Dictionary(self.texts)
            d.save_as_text(tmpf, sort_by_word=sort_by_word)
            self.assertTrue(os.path.exists(tmpf))

            d_loaded = Dictionary.load_from_text(tmpf)
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

        # Create dictionary from corpus without a token map
        dictionary_from_corpus = Dictionary.from_corpus(corpus)

        dict_token2id_vals = sorted(dictionary.token2id.values())
        dict_from_corpus_vals = sorted(dictionary_from_corpus.token2id.values())
        self.assertEqual(dict_token2id_vals, dict_from_corpus_vals)
        self.assertEqual(dictionary.dfs, dictionary_from_corpus.dfs)
        self.assertEqual(dictionary.num_docs, dictionary_from_corpus.num_docs)
        self.assertEqual(dictionary.num_pos, dictionary_from_corpus.num_pos)
        self.assertEqual(dictionary.num_nnz, dictionary_from_corpus.num_nnz)

        # Create dictionary from corpus with an id=>token map
        dictionary_from_corpus_2 = Dictionary.from_corpus(corpus, id2word=dictionary)

        self.assertEqual(dictionary.token2id, dictionary_from_corpus_2.token2id)
        self.assertEqual(dictionary.dfs, dictionary_from_corpus_2.dfs)
        self.assertEqual(dictionary.num_docs, dictionary_from_corpus_2.num_docs)
        self.assertEqual(dictionary.num_pos, dictionary_from_corpus_2.num_pos)
        self.assertEqual(dictionary.num_nnz, dictionary_from_corpus_2.num_nnz)

        # Ensure Sparse2Corpus is compatible with from_corpus
        bow = gensim.matutils.Sparse2Corpus(scipy.sparse.rand(10, 100))
        dictionary = Dictionary.from_corpus(bow)
        self.assertEqual(dictionary.num_docs, 100)

    def test_dict_interface(self):
        """Test Python 2 dict-like interface in both Python 2 and 3."""
        d = Dictionary(self.texts)

        self.assertTrue(isinstance(d, Mapping))

        self.assertEqual(list(zip(d.keys(), d.values())), list(d.items()))

        # Even in Py3, we want the iter* members.
        self.assertEqual(list(d.items()), list(d.iteritems()))
        self.assertEqual(list(d.keys()), list(d.iterkeys()))
        self.assertEqual(list(d.values()), list(d.itervalues()))

        # XXX Do we want list results from the dict members in Py3 too?
        if not PY3:
            self.assertTrue(isinstance(d.items(), list))
            self.assertTrue(isinstance(d.keys(), list))
            self.assertTrue(isinstance(d.values(), list))

# endclass TestDictionary


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
