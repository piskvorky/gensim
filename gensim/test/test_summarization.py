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
import unittest

from gensim import utils
from gensim.summarization import summarize


class TestSummarizationTest(unittest.TestCase):

    def test_summarization(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "mihalcea_tarau.txt"), mode="r") as f:
            text = f.read()

        generated_summary = summarize(text)

        with utils.smart_open(os.path.join(pre_path, "mihalcea_tarau.summ.txt"), mode="r") as f:
            summary = f.read()

        self.assertEquals(generated_summary, summary)

    def test_summary_from_unrelated_sentences(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "testsummarization_unrelated.txt"), mode="r") as f:
            text = f.read()

        generated_summary = summarize(text)

        self.assertIsNotNone(generated_summary)

    def test_raises_exception_on_short_input_text(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "testsummarization_unrelated.txt"), mode="r") as f:
            text = f.read()

        # Keeps the first 8 sentences to make the text shorter.
        text = "\n".join(text.split('\n')[:8])

        with self.assertRaises(RuntimeError):
            summarize(text)
