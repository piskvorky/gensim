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

from gensim import utils, summarization

TEXT_FILENAME = "mihalcea_tarau.txt"
SUMMARY_FILENAME = "mihalcea_tarau.summ.txt"


class TestSummarizationTest(unittest.TestCase):

    def test_summarization(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, TEXT_FILENAME)) as f:
            text = f.read()

        generated_summary = summarization.summarize(text)

        with utils.smart_open(os.path.join(pre_path, SUMMARY_FILENAME)) as f:
            summary = f.read()

        self.assertEquals(generated_summary, summary)

