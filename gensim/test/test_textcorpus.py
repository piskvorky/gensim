#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking the WikiCorpus
"""


import logging
import unittest

from gensim.corpora.textcorpus import TextCorpus


logger = logging.getLogger(__name__)


class TestTextCorpus(unittest.TestCase):
    # TODO add tests for other methods

    class DummyTextCorpus(TextCorpus):
        def __init__(self):
            self.size = 10
            self.data = [["document%s" % i] for i in range(self.size)]

        def get_texts(self):
            for document in self.data:
                yield document

    def test_sample_text(self):
        corpus = self.DummyTextCorpus()

        sample1 = list(corpus.sample_texts(1))
        self.assertEqual(len(sample1), 1)
        self.assertIn(sample1[0], corpus.data)

        sample2 = list(corpus.sample_texts(corpus.size))
        self.assertEqual(len(sample2), corpus.size)
        for i in range(corpus.size):
            self.assertEqual(sample2[i], ["document%s" % i])

        with self.assertRaises(ValueError):
            list(corpus.sample_texts(corpus.size + 1))

        with self.assertRaises(ValueError):
            list(corpus.sample_texts(-1))

    def test_sample_text_length(self):
        corpus = self.DummyTextCorpus()
        sample1 = list(corpus.sample_texts(1, length=1))
        self.assertEqual(sample1[0], ["document0"])

        sample2 = list(corpus.sample_texts(2, length=2))
        self.assertEqual(sample2[0], ["document0"])
        self.assertEqual(sample2[1], ["document1"])

    def test_sample_text_seed(self):
        corpus = self.DummyTextCorpus()
        sample1 = list(corpus.sample_texts(5, seed=42))
        sample2 = list(corpus.sample_texts(5, seed=42))
        self.assertEqual(sample1, sample2)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
