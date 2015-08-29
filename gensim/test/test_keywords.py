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
from gensim.summarization import keywords


class TestKeywordsTest(unittest.TestCase):

    def test_text_keywords(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "mihalcea_tarau.txt"), mode="r") as f:
            text = f.read()

        # calculate keywords
        generated_keywords = keywords(text)

        # To be compared to the reference.
        with utils.smart_open(os.path.join(pre_path, "mihalcea_tarau.kw.txt"), mode="r") as f:
            kw = f.read()

        self.assertEqual(set(generated_keywords), set(kw))

    def test_text_keywords_words(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "mihalcea_tarau.txt"), mode="r") as f:
            text = f.read()

        # calculate exactly 13 keywords
        generated_keywords = keywords(text, words=13, split=True)

        self.assertEqual(len(generated_keywords), 13)

    def test_text_keywords_pos(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "mihalcea_tarau.txt"), mode="r") as f:
            text = f.read()

        # calculate keywords using only certain parts of speech
        generated_keywords_NNVBJJ = keywords(text, pos_filter=['NN', 'VB', 'JJ'])

        # To be compared to the reference.
        with utils.smart_open(os.path.join(pre_path, "mihalcea_tarau.kwpos.txt"), mode="r") as f:
            kw = f.read()

        self.assertEqual(generated_keywords_NNVBJJ, kw)

    def test_text_summarization_raises_exception_on_short_input_text(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "testsummarization_unrelated.txt"), mode="r") as f:
            text = f.read()

        # Keeps the first 8 sentences to make the text shorter.
        text = "\n".join(text.split('\n')[:8])

        self.assertTrue(keywords(text) is not None)

    def test_empty_text_keywords_none(self):
        self.assertRaises(ZeroDivisionError)

    def test_keywords_ratio(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "mihalcea_tarau.txt"), mode="r") as f:
            text = f.read()

        # Check ratio parameter is well behaved.  Because length is taken on tokenized clean text
        # we just check that ratio 40% is twice as long as ratio 20%
        selected_docs_20 = keywords(text, ratio=0.2, split=True)
        selected_docs_40 = keywords(text, ratio=0.4, split=True)

        self.assertAlmostEqual(float(len(selected_docs_40))/len(selected_docs_20), 1.9, places=1)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
