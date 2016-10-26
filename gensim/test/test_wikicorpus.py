#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking the WikiCorpus
"""


import os
import sys
import types
import logging
import unittest

from gensim.corpora.wikicorpus import WikiCorpus


module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)
FILENAME = 'enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2'

logger = logging.getLogger(__name__)

class TestWikiCorpus(unittest.TestCase):

    def setUp(self):
        wc = WikiCorpus(datapath(FILENAME))

    def test_get_texts_returns_generator_of_lists(self):
        logger.debug("Current Python Version is "+str(sys.version_info))
        if sys.version_info < (2, 7, 0):
            return

        wc = WikiCorpus(datapath(FILENAME))
        l = wc.get_texts()
        self.assertEqual(type(l), types.GeneratorType)
        first = next(l)
        self.assertEqual(type(first), list)
        self.assertTrue(isinstance(first[0], bytes) or isinstance(first[0], str))

    def test_first_element(self):
        """
        First two articles in this sample are
        1) anarchism
        2) autism

        """
        if sys.version_info < (2, 7, 0):
            return
        wc = WikiCorpus(datapath(FILENAME))

        l = wc.get_texts()
        self.assertTrue(b"anarchism" in next(l))
        self.assertTrue(b"autism" in next(l))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
