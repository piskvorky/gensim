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

    def test_sample_text(self):
        class TestTextCorpus(TextCorpus):
            def __init__(self):
                self.data = [["document1"], ["document2"]]

            def get_texts(self):
                for document in self.data:
                    yield document

        corpus = TestTextCorpus()

        sample1 = list(corpus.sample_texts(1))
        self.assertEqual(len(sample1), 1)
        document1 = sample1[0] == ["document1"]
        document2 = sample1[0] == ["document2"]
        self.assertTrue(document1 or document2)

        sample2 = list(corpus.sample_texts(2))
        self.assertEqual(len(sample2), 2)
        self.assertEqual(sample2[0], ["document1"])
        self.assertEqual(sample2[1], ["document2"])

        with self.assertRaises(ValueError):
            list(corpus.sample_texts(3))

        with self.assertRaises(ValueError):
            list(corpus.sample_texts(-1))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
