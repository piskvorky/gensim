#!/usr/bin/env python
# encoding: utf-8
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated test to reproduce the results of Mihalcea and Tarau (2004).

Mihalcea and Tarau (2004) introduces the TextRank summarization algorithm.
As a validation of the gensim implementation we reproduced its results
in this test.

"""

import os.path
import logging
import unittest

from gensim import utils
from gensim.corpora import Dictionary
from gensim.summarization import summarize, summarize_corpus, keywords, mz_keywords
from gensim.summarization.commons import remove_unreachable_nodes, build_graph
from gensim.summarization.graph import Graph


class TestGraph(unittest.TestCase):

    def _build_graph(self):
        graph = build_graph(['a', 'b', 'c', 'd'])
        graph.add_edge(('a', 'b'))
        graph.add_edge(('b', 'c'))
        graph.add_edge(('c', 'a'))
        return graph

    def test_build_graph(self):
        graph = self._build_graph()

        self.assertEqual(sorted(graph.nodes()), ['a', 'b', 'c', 'd'])
        self.assertTrue(graph.has_edge(('a', 'b')))
        self.assertTrue(graph.has_edge(('b', 'c')))
        self.assertTrue(graph.has_edge(('c', 'a')))

        graph = build_graph([])
        self.assertEqual(graph.nodes(), [])

    def test_remove_unreachable_nodes(self):
        graph = self._build_graph()
        self.assertTrue(graph.has_node('d'))
        remove_unreachable_nodes(graph)
        self.assertFalse(graph.has_node('d'))

        graph = self._build_graph()
        graph.add_edge(('d', 'a'), wt=0.0)
        graph.add_edge(('b', 'd'), wt=0)
        self.assertTrue(graph.has_node('d'))
        remove_unreachable_nodes(graph)
        self.assertFalse(graph.has_node('d'))

    def test_graph_nodes(self):
        graph = Graph()

        graph.add_node('a')
        graph.add_node(1)
        graph.add_node('b')
        graph.add_node('qwe')

        self.assertTrue(graph.has_node('a'))
        self.assertTrue(graph.has_node('b'))
        self.assertTrue(graph.has_node('qwe'))
        self.assertTrue(graph.has_node(1))
        self.assertFalse(graph.has_node(2))

        graph.del_node(1)
        self.assertEqual(sorted(graph.nodes()), ['a', 'b', 'qwe'])

    def test_graph_edges(self):
        graph = Graph()
        for node in ('a', 'b', 'c', 'd', 'e', 'foo', 'baz', 'qwe', 'rtyu'):
            graph.add_node(node)

        edges = [
            (('a', 'b'), 3.0),
            (('c', 'b'), 5.0),
            (('d', 'e'), 0.5),
            (('a', 'c'), 0.1),
            (('foo', 'baz'), 0.11),
            (('qwe', 'rtyu'), 0.0),
        ]
        for edge, weight in edges:
            graph.add_edge(edge, weight)

        # check on edge weight first to exclude situation when touching will create an edge
        self.assertEqual(graph.edge_weight(('qwe', 'rtyu')), 0.0)
        self.assertEqual(graph.edge_weight(('rtyu', 'qwe')), 0.0)
        self.assertFalse(graph.has_edge(('qwe', 'rtyu')))
        self.assertFalse(graph.has_edge(('rtyu', 'qwe')))

        for (u, v), weight in edges:
            if weight == 0:
                continue
            self.assertTrue(graph.has_edge((u, v)))
            self.assertTrue(graph.has_edge((v, u)))

        edges_list = [(u, v) for (u, v), w in edges if w]
        edges_list.extend((v, u) for (u, v), w in edges if w)
        edges_list.sort()

        self.assertEqual(sorted(graph.iter_edges()), edges_list)

        ret_edges = graph.edges()
        ret_edges.sort()
        self.assertEqual(ret_edges, edges_list)

        for (u, v), weight in edges:
            self.assertEqual(graph.edge_weight((u, v)), weight)
            self.assertEqual(graph.edge_weight((v, u)), weight)

        self.assertEqual(sorted(graph.neighbors('a')), ['b', 'c'])
        self.assertEqual(sorted(graph.neighbors('b')), ['a', 'c'])
        self.assertEqual(graph.neighbors('d'), ['e'])
        self.assertEqual(graph.neighbors('e'), ['d'])
        self.assertEqual(graph.neighbors('foo'), ['baz'])
        self.assertEqual(graph.neighbors('baz'), ['foo'])
        self.assertEqual(graph.neighbors('foo'), ['baz'])
        self.assertEqual(graph.neighbors('qwe'), [])
        self.assertEqual(graph.neighbors('rtyu'), [])

        graph.del_edge(('a', 'b'))
        self.assertFalse(graph.has_edge(('a', 'b')))
        self.assertFalse(graph.has_edge(('b', 'a')))

        graph.add_edge(('baz', 'foo'), 0)
        self.assertFalse(graph.has_edge(('foo', 'baz')))
        self.assertFalse(graph.has_edge(('baz', 'foo')))

        graph.del_node('b')
        self.assertFalse(graph.has_edge(('b', 'c')))
        self.assertFalse(graph.has_edge(('c', 'b')))


class TestSummarizationTest(unittest.TestCase):

    def _get_text_from_test_data(self, file):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')
        with utils.open(os.path.join(pre_path, file), mode="r") as f:
            return f.read()

    def test_text_summarization(self):
        text = self._get_text_from_test_data("mihalcea_tarau.txt")

        # Makes a summary of the text.
        generated_summary = summarize(text)

        # To be compared to the method reference.
        summary = self._get_text_from_test_data("mihalcea_tarau.summ.txt")

        self.assertEqual(generated_summary, summary)

    def test_corpus_summarization(self):
        text = self._get_text_from_test_data("mihalcea_tarau.txt")

        # Generate the corpus.
        sentences = text.split("\n")
        tokens = [sentence.split() for sentence in sentences]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(sentence_tokens) for sentence_tokens in tokens]

        # Extract the most important documents.
        selected_documents = summarize_corpus(corpus)

        # They are compared to the method reference.
        summary = self._get_text_from_test_data("mihalcea_tarau.summ.txt")
        summary = summary.split('\n')

        # Each sentence in the document selection has to be in the model summary.
        for doc_number, document in enumerate(selected_documents):
            # Retrieves all words from the document.
            words = [dictionary[token_id] for (token_id, count) in document]

            # Asserts that all of them are in a sentence from the model reference.
            self.assertTrue(any(all(word in sentence for word in words)) for sentence in summary)

    def test_summary_from_unrelated_sentences(self):
        # Tests that the summarization of a text with unrelated sentences is not empty string.
        text = self._get_text_from_test_data("testsummarization_unrelated.txt")
        generated_summary = summarize(text)
        self.assertNotEqual(generated_summary, u"")

    def test_text_summarization_on_short_input_text_is_empty_string(self):
        text = self._get_text_from_test_data("testsummarization_unrelated.txt")

        # Keeps the first 8 sentences to make the text shorter.
        text = "\n".join(text.split('\n')[:8])

        self.assertNotEqual(summarize(text), u"")

    def test_text_summarization_raises_exception_on_single_input_sentence(self):
        text = self._get_text_from_test_data("testsummarization_unrelated.txt")

        # Keeps the first sentence only.
        text = text.split('\n')[0]

        self.assertRaises(ValueError, summarize, text)

    def test_corpus_summarization_is_not_empty_list_on_short_input_text(self):
        text = self._get_text_from_test_data("testsummarization_unrelated.txt")

        # Keeps the first 8 sentences to make the text shorter.
        sentences = text.split('\n')[:8]

        # Generate the corpus.
        tokens = [sentence.split() for sentence in sentences]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(sentence_tokens) for sentence_tokens in tokens]

        self.assertNotEqual(summarize_corpus(corpus), [])

    def test_empty_text_summarization_is_empty_string(self):
        self.assertEqual(summarize(""), u"")

    def test_empty_text_summarization_with_split_is_empty_list(self):
        self.assertEqual(summarize("", split=True), [])

    def test_empty_corpus_summarization_is_empty_list(self):
        self.assertEqual(summarize_corpus([]), [])

    def test_corpus_summarization_ratio(self):
        text = self._get_text_from_test_data("mihalcea_tarau.txt")

        # Generate the corpus.
        sentences = text.split('\n')
        tokens = [sentence.split() for sentence in sentences]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(sentence_tokens) for sentence_tokens in tokens]

        # Makes summaries of the text using different ratio parameters.
        for x in range(1, 10):
            ratio = x / float(10)
            selected_docs = summarize_corpus(corpus, ratio=ratio)
            expected_summary_length = int(len(corpus) * ratio)

            self.assertEqual(len(selected_docs), expected_summary_length)

    def test_repeated_keywords(self):
        text = self._get_text_from_test_data("testrepeatedkeywords.txt")

        kwds = keywords(text)
        self.assertTrue(len(kwds.splitlines()))

        kwds_u = keywords(utils.to_unicode(text))
        self.assertTrue(len(kwds_u.splitlines()))

        kwds_lst = keywords(text, split=True)
        self.assertTrue(len(kwds_lst))

    def test_keywords_runs(self):
        text = self._get_text_from_test_data("mihalcea_tarau.txt")

        kwds = keywords(text)
        self.assertTrue(len(kwds.splitlines()))

        kwds_u = keywords(utils.to_unicode(text))
        self.assertTrue(len(kwds_u.splitlines()))

        kwds_lst = keywords(text, split=True)
        self.assertTrue(len(kwds_lst))

    def test_mz_keywords(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.open(os.path.join(pre_path, "head500.noblanks.cor"), 'rb') as f:
            text = utils.to_unicode(f.read())
        text = u' '.join(text.split()[:10240])
        kwds = mz_keywords(text)
        self.assertTrue(kwds.startswith('autism'))
        self.assertTrue(kwds.endswith('uk'))
        self.assertTrue(len(kwds.splitlines()))

        kwds_lst = mz_keywords(text, split=True)
        self.assertTrue(len(kwds_lst))
        # Automatic thresholding selects words with n_blocks / n_blocks+1
        # bits of entropy. For this text, n_blocks=10
        n_blocks = 10.
        kwds_auto = mz_keywords(text, scores=True, weighted=False, threshold='auto')
        self.assertTrue(kwds_auto[-1][1] > (n_blocks / (n_blocks + 1.)))

    def test_low_distinct_words_corpus_summarization_is_empty_list(self):
        text = self._get_text_from_test_data("testlowdistinctwords.txt")

        # Generate the corpus.
        sentences = text.split("\n")
        tokens = [sentence.split() for sentence in sentences]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(sentence_tokens) for sentence_tokens in tokens]

        self.assertEqual(summarize_corpus(corpus), [])

    def test_low_distinct_words_summarization_is_empty_string(self):
        text = self._get_text_from_test_data("testlowdistinctwords.txt")
        self.assertEqual(summarize(text), u"")

    def test_low_distinct_words_summarization_with_split_is_empty_list(self):
        text = self._get_text_from_test_data("testlowdistinctwords.txt")
        self.assertEqual(summarize(text, split=True), [])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
