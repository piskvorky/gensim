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
from gensim.summarization import keywords


class TestKeywordsTest(unittest.TestCase):

    def test_text_keywords(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.open(os.path.join(pre_path, "mihalcea_tarau.txt"), mode="r") as f:
            text = f.read()

        # calculate keywords
        generated_keywords = keywords(text, split=True)

        # To be compared to the reference.
        with utils.open(os.path.join(pre_path, "mihalcea_tarau.kw.txt"), mode="r") as f:
            kw = f.read().strip().split("\n")

        self.assertEqual({str(x) for x in generated_keywords}, {str(x) for x in kw})

    def test_text_keywords_words(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.open(os.path.join(pre_path, "mihalcea_tarau.txt"), mode="r") as f:
            text = f.read()

        # calculate exactly 13 keywords
        generated_keywords = keywords(text, words=15, split=True)

        self.assertEqual(len(generated_keywords), 16)

    def test_text_keywords_pos(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.open(os.path.join(pre_path, "mihalcea_tarau.txt"), mode="r") as f:
            text = f.read()

        # calculate keywords using only certain parts of speech
        generated_keywords_nnvbjj = keywords(text, pos_filter=['NN', 'VB', 'JJ'], ratio=0.3, split=True)

        # To be compared to the reference.
        with utils.open(os.path.join(pre_path, "mihalcea_tarau.kwpos.txt"), mode="r") as f:
            kw = f.read().strip().split("\n")

        self.assertEqual({str(x) for x in generated_keywords_nnvbjj}, {str(x) for x in kw})

    def test_text_summarization_raises_exception_on_short_input_text(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.open(os.path.join(pre_path, "testsummarization_unrelated.txt"), mode="r") as f:
            text = f.read()

        # Keeps the first 8 sentences to make the text shorter.
        text = "\n".join(text.split('\n')[:8])

        self.assertTrue(keywords(text) is not None)

    def test_keywords_ratio(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.open(os.path.join(pre_path, "mihalcea_tarau.txt"), mode="r") as f:
            text = f.read()

        # Check ratio parameter is well behaved.  Because length is taken on tokenized clean text
        # we just check that ratio 20% is twice as long as ratio 10%
        # Values of 10% and 20% were carefully selected for this test to avoid
        # numerical instabilities when several keywords have almost the same score
        selected_docs_12 = keywords(text, ratio=0.1, split=True)
        selected_docs_21 = keywords(text, ratio=0.2, split=True)

        self.assertAlmostEqual(float(len(selected_docs_21)) / len(selected_docs_12), float(21) / 12, places=1)

    def test_text_keywords_with_small_graph(self):
        # regression test, we get graph 2x2 on this text
        text = 'IT: Utilities A look at five utilities to make your PCs more, efficient, effective, and efficacious'
        kwds = keywords(text, words=1, split=True)
        self.assertTrue(len(kwds))

    def test_text_keywords_without_graph_edges(self):
        # regression test, we get graph with no edges on this text
        text = 'Sitio construcción. Estaremos línea.'
        kwds = keywords(text, deacc=False, scores=True)
        self.assertFalse(len(kwds))

    def test_keywords_with_words_greater_than_lemmas(self):
        # words parameter is greater than number of words in text variable
        text = 'Test string small length'
        kwds = keywords(text, words=5, split=True)
        self.assertIsNotNone(kwds)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
