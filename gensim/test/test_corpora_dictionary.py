#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Unit tests for the `corpora.Dictionary` class.
"""

from collections.abc import Mapping
from itertools import chain
import logging
import unittest
import codecs
import os
import os.path

import scipy
import gensim
from gensim.corpora import Dictionary
from gensim.utils import to_utf8
from gensim.test.utils import get_tmpfile, common_texts


class TestDictionary(unittest.TestCase):
    def setUp(self):
        self.texts = common_texts

    def test_doc_freq_one_doc(self):
        texts = [['human', 'interface', 'computer']]
        d = Dictionary(texts)
        expected = {0: 1, 1: 1, 2: 1}
        self.assertEqual(d.dfs, expected)

    def test_doc_freq_and_token2id_for_several_docs_with_one_word(self):
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

    def test_doc_freq_for_one_doc_with_several_word(self):
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

    def test_doc_freq_and_collection_freq(self):
        # one doc
        texts = [['human', 'human', 'human']]
        d = Dictionary(texts)
        self.assertEqual(d.cfs, {0: 3})
        self.assertEqual(d.dfs, {0: 1})

        # two docs
        texts = [['human', 'human'], ['human']]
        d = Dictionary(texts)
        self.assertEqual(d.cfs, {0: 3})
        self.assertEqual(d.dfs, {0: 2})

        # three docs
        texts = [['human'], ['human'], ['human']]
        d = Dictionary(texts)
        self.assertEqual(d.cfs, {0: 3})
        self.assertEqual(d.dfs, {0: 3})

    def test_build(self):
        d = Dictionary(self.texts)

        # Since we don't specify the order in which dictionaries are built,
        # we cannot reliably test for the mapping; only the keys and values.
        expected_keys = list(range(12))
        expected_values = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]
        self.assertEqual(sorted(d.dfs.keys()), expected_keys)
        self.assertEqual(sorted(d.dfs.values()), expected_values)

        expected_keys = sorted([
            'computer', 'eps', 'graph', 'human', 'interface',
            'minors', 'response', 'survey', 'system', 'time', 'trees', 'user'
        ])
        expected_values = list(range(12))
        self.assertEqual(sorted(d.token2id.keys()), expected_keys)
        self.assertEqual(sorted(d.token2id.values()), expected_values)

    def test_merge(self):
        d = Dictionary(self.texts)
        f = Dictionary(self.texts[:3])
        g = Dictionary(self.texts[3:])

        f.merge_with(g)
        self.assertEqual(sorted(d.token2id.keys()), sorted(f.token2id.keys()))

    def test_filter(self):
        d = Dictionary(self.texts)
        d.filter_extremes(no_below=2, no_above=1.0, keep_n=4)
        dfs_expected = {0: 3, 1: 3, 2: 3, 3: 3}
        cfs_expected = {0: 4, 1: 3, 2: 3, 3: 3}
        self.assertEqual(d.dfs, dfs_expected)
        self.assertEqual(d.cfs, cfs_expected)

    def testFilterKeepTokens_keepTokens(self):
        # provide keep_tokens argument, keep the tokens given
        d = Dictionary(self.texts)
        d.filter_extremes(no_below=3, no_above=1.0, keep_tokens=['human', 'survey'])
        expected = {'graph', 'trees', 'human', 'system', 'user', 'survey'}
        self.assertEqual(set(d.token2id.keys()), expected)

    def testFilterKeepTokens_unchangedFunctionality(self):
        # do not provide keep_tokens argument, filter_extremes functionality is unchanged
        d = Dictionary(self.texts)
        d.filter_extremes(no_below=3, no_above=1.0)
        expected = {'graph', 'trees', 'system', 'user'}
        self.assertEqual(set(d.token2id.keys()), expected)

    def testFilterKeepTokens_unseenToken(self):
        # do provide keep_tokens argument with unseen tokens, filter_extremes functionality is unchanged
        d = Dictionary(self.texts)
        d.filter_extremes(no_below=3, no_above=1.0, keep_tokens=['unknown_token'])
        expected = {'graph', 'trees', 'system', 'user'}
        self.assertEqual(set(d.token2id.keys()), expected)

    def testFilterKeepTokens_keepn(self):
        # keep_tokens should also work if the keep_n parameter is used, but only
        # to keep a maximum of n (so if keep_n < len(keep_n) the tokens to keep are
        # still getting removed to reduce the size to keep_n!)
        d = Dictionary(self.texts)
        # Note: there are four tokens with freq 3, all the others have frequence 2
        # in self.texts. In order to make the test result deterministic, we add
        # 2 tokens of frequency one
        d.add_documents([['worda'], ['wordb']])
        # this should keep the 3 tokens with freq 3 and the one we want to keep
        d.filter_extremes(keep_n=5, no_below=0, no_above=1.0, keep_tokens=['worda'])
        expected = {'graph', 'trees', 'system', 'user', 'worda'}
        self.assertEqual(set(d.token2id.keys()), expected)

    def test_filter_most_frequent(self):
        d = Dictionary(self.texts)
        d.filter_n_most_frequent(4)
        expected = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2}
        self.assertEqual(d.dfs, expected)

    def test_filter_tokens(self):
        self.maxDiff = 10000
        d = Dictionary(self.texts)

        removed_word = d[0]
        d.filter_tokens([0])

        expected = {
            'computer': 0, 'eps': 8, 'graph': 10, 'human': 1,
            'interface': 2, 'minors': 11, 'response': 3, 'survey': 4,
            'system': 5, 'time': 6, 'trees': 9, 'user': 7
        }
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
        small_text = [
            ["prvé", "slovo"],
            ["slovo", "druhé"],
            ["druhé", "slovo"]
        ]

        d = Dictionary(small_text)

        d.save_as_text(tmpf)
        with codecs.open(tmpf, 'r', encoding='utf-8') as file:
            serialized_lines = file.readlines()
            self.assertEqual(serialized_lines[0], u"3\n")
            self.assertEqual(len(serialized_lines), 4)
            # We do not know, which word will have which index
            self.assertEqual(serialized_lines[1][1:], u"\tdruhé\t2\n")
            self.assertEqual(serialized_lines[2][1:], u"\tprvé\t1\n")
            self.assertEqual(serialized_lines[3][1:], u"\tslovo\t3\n")

        d.save_as_text(tmpf, sort_by_word=False)
        with codecs.open(tmpf, 'r', encoding='utf-8') as file:
            serialized_lines = file.readlines()
            self.assertEqual(serialized_lines[0], u"3\n")
            self.assertEqual(len(serialized_lines), 4)
            self.assertEqual(serialized_lines[1][1:], u"\tslovo\t3\n")
            self.assertEqual(serialized_lines[2][1:], u"\tdruhé\t2\n")
            self.assertEqual(serialized_lines[3][1:], u"\tprvé\t1\n")

    def test_loadFromText_legacy(self):
        """
        `Dictionary` can be loaded from textfile in legacy format.
        Legacy format does not have num_docs on the first line.
        """
        tmpf = get_tmpfile('load_dict_test_legacy.txt')
        no_num_docs_serialization = to_utf8("1\tprvé\t1\n2\tslovo\t2\n")
        with open(tmpf, "wb") as file:
            file.write(no_num_docs_serialization)

        d = Dictionary.load_from_text(tmpf)
        self.assertEqual(d.token2id[u"prvé"], 1)
        self.assertEqual(d.token2id[u"slovo"], 2)
        self.assertEqual(d.dfs[1], 1)
        self.assertEqual(d.dfs[2], 2)
        self.assertEqual(d.num_docs, 0)

    def test_loadFromText(self):
        """`Dictionary` can be loaded from textfile."""
        tmpf = get_tmpfile('load_dict_test.txt')
        no_num_docs_serialization = to_utf8("2\n1\tprvé\t1\n2\tslovo\t2\n")
        with open(tmpf, "wb") as file:
            file.write(no_num_docs_serialization)

        d = Dictionary.load_from_text(tmpf)
        self.assertEqual(d.token2id[u"prvé"], 1)
        self.assertEqual(d.token2id[u"slovo"], 2)
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

        documents = [
            "Human machine interface for lab abc computer applications",
            "A survey of user opinion of computer system response time",
            "The EPS user interface management system",
            "System and human system engineering testing of EPS",
            "Relation of user perceived response time to error measurement",
            "The generation of random binary unordered trees",
            "The intersection graph of paths in trees",
            "Graph minors IV Widths of trees and well quasi ordering",
            "Graph minors A survey"
        ]
        stoplist = set('for a of the and to in'.split())
        texts = [
            [word for word in document.lower().split() if word not in stoplist]
            for document in documents]

        # remove words that appear only once
        all_tokens = list(chain.from_iterable(texts))
        tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
        texts = [[word for word in text if word not in tokens_once] for text in texts]

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

    def test_patch_with_special_tokens(self):
        special_tokens = {'pad': 0, 'space': 1, 'quake': 3}
        corpus = [["máma", "mele", "maso"], ["ema", "má", "máma"]]
        d = Dictionary(corpus)
        self.assertEqual(len(d.token2id), 5)
        d.patch_with_special_tokens(special_tokens)
        self.assertEqual(d.token2id['pad'], 0)
        self.assertEqual(d.token2id['space'], 1)
        self.assertEqual(d.token2id['quake'], 3)
        self.assertEqual(len(d.token2id), 8)
        self.assertNotIn((0, 1), d.doc2bow(corpus[0]))
        self.assertIn((0, 1), d.doc2bow(['pad'] + corpus[0]))
        corpus_with_special_tokens = [["máma", "mele", "maso"], ["ema", "má", "máma", "space"]]
        d = Dictionary(corpus_with_special_tokens)
        self.assertEqual(len(d.token2id), 6)
        self.assertNotEqual(d.token2id['space'], 1)
        d.patch_with_special_tokens(special_tokens)
        self.assertEqual(len(d.token2id), 8)
        self.assertEqual(max(d.token2id.values()), 7)
        self.assertEqual(d.token2id['space'], 1)
        self.assertNotIn((1, 1), d.doc2bow(corpus_with_special_tokens[0]))
        self.assertIn((1, 1), d.doc2bow(corpus_with_special_tokens[1]))

    def test_most_common_with_n(self):
        texts = [['human', 'human', 'human', 'computer', 'computer', 'interface', 'interface']]
        d = Dictionary(texts)
        expected = [('human', 3), ('computer', 2)]
        assert d.most_common(n=2) == expected

    def test_most_common_without_n(self):
        texts = [['human', 'human', 'human', 'computer', 'computer', 'interface', 'interface']]
        d = Dictionary(texts)
        expected = [('human', 3), ('computer', 2), ('interface', 2)]
        assert d.most_common(n=None) == expected


# endclass TestDictionary


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
