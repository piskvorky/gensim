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
from gensim import utils

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)
FILENAME = 'enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2'
FILENAME_U = 'bgwiki-latest-pages-articles-shortened.xml.bz2'

logger = logging.getLogger(__name__)


def custom_tokeiner(content, token_min_len=2, token_max_len=15, lower=True):
    return [
        utils.to_unicode(token.lower()) if lower else utils.to_unicode(token) for token in content.split()
        if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
    ]


class TestWikiCorpus(unittest.TestCase):

    # #TODO: sporadic failure to be investigated
    # def test_get_texts_returns_generator_of_lists(self):
    #     logger.debug("Current Python Version is " + str(sys.version_info))
    #     if sys.version_info < (2, 7, 0):
    #         return
    #
    #     wc = WikiCorpus(datapath(FILENAME))
    #     l = wc.get_texts()
    #     self.assertEqual(type(l), types.GeneratorType)
    #     first = next(l)
    #     self.assertEqual(type(first), list)
    #     self.assertTrue(isinstance(first[0], bytes) or isinstance(first[0], str))

    def test_first_element(self):
        """
        First two articles in this sample are
        1) anarchism
        2) autism
        """
        wc = WikiCorpus(datapath(FILENAME), processes=1)

        l = wc.get_texts()
        self.assertTrue(u'anarchism' in next(l))
        self.assertTrue(u'autism' in next(l))

    def test_unicode_element(self):
        """
        First unicode article in this sample is
        1) папа
        """
        wc = WikiCorpus(datapath(FILENAME_U), processes=1)

        l = wc.get_texts()
        self.assertTrue(u'папа' in next(l))

    def test_lower_case_set_true(self):
        """
        set the parameter lower to True and check that upper case 'Anarchism' token doesnt exist
        """
        wc = WikiCorpus(datapath(FILENAME), processes=1, lower=True, lemmatize=False)
        l = wc.get_texts()
        list_tokens = next(l)
        self.assertTrue(u'Anarchism' not in list_tokens)
        self.assertTrue(u'anarchism' in list_tokens)

    def test_lower_case_set_false(self):
        """
        set the parameter lower to False and check that upper case Anarchism' token exist
        """
        wc = WikiCorpus(datapath(FILENAME), processes=1, lower=False, lemmatize=False)
        l = wc.get_texts()
        list_tokens = next(l)
        self.assertTrue(u'Anarchism' in list_tokens)
        self.assertTrue(u'anarchism' in list_tokens)

    def test_min_token_len_not_set(self):
        """
        don't set the parameter token_min_len and check that 'a' as a token doesn't exists
        default token_min_len=2
        """
        wc = WikiCorpus(datapath(FILENAME), processes=1, lemmatize=False)
        l = wc.get_texts()
        self.assertTrue(u'a' not in next(l))

    def test_min_token_len_set(self):
        """
        set the parameter token_min_len to 1 and check that 'a' as a token exists
        """
        wc = WikiCorpus(datapath(FILENAME), processes=1, token_min_len=1, lemmatize=False)
        l = wc.get_texts()
        self.assertTrue(u'a' in next(l))

    def test_max_token_len_not_set(self):
        """
        don't set the parameter token_max_len and check that 'collectivisation' as a token doesn't exists
        default token_max_len=15
        """
        wc = WikiCorpus(datapath(FILENAME), processes=1, lemmatize=False)
        l = wc.get_texts()
        self.assertTrue(u'collectivization' not in next(l))

    def test_max_token_len_set(self):
        """
        set the parameter token_max_len to 16 and check that 'collectivisation' as a token exists
        """
        wc = WikiCorpus(datapath(FILENAME), processes=1, token_max_len=16, lemmatize=False)
        l = wc.get_texts()
        self.assertTrue(u'collectivization' in next(l))

    def test_custom_tokenizer(self):
        """
        define a custom tokenizer function and use it
        """
        wc = WikiCorpus(datapath(FILENAME), processes=1, lemmatize=False, tokenizer_func=custom_tokeiner,
                        token_max_len=16, token_min_len=1, lower=False)
        l = wc.get_texts()
        list_tokens = next(l)
        self.assertTrue(u'Anarchism' in list_tokens)
        self.assertTrue(u'collectivization' in list_tokens)
        self.assertTrue(u'a' in list_tokens)
        self.assertTrue(u'i.e.' in list_tokens)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
