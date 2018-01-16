#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Manos Stergiadis <em.stergiadis@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking the output of gensim.scripts.
"""

from __future__ import unicode_literals

import json
import logging
import os.path
import unittest

from gensim.scripts.segment_wiki import segment_all_articles, segment_and_write_all_articles
from smart_open import smart_open
from gensim.test.utils import datapath, get_tmpfile


class TestSegmentWiki(unittest.TestCase):

    def setUp(self):
        self.fname = datapath('enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2')
        self.expected_title = 'Anarchism'
        self.expected_section_titles = [
            'Introduction',
            'Etymology and terminology',
            'History',
            'Anarchist schools of thought',
            'Internal issues and debates',
            'Topics of interest',
            'Criticisms',
            'References',
            'Further reading',
            'External links'
        ]
        self.expected_interlinks = [
            'political philosophy', 'self-governed', 'stateless societies',
            'hierarchical', 'free associations', 'state',
            'anti-statism', 'authority', 'hierarchical organisation',
            'Anarchist schools of thought', 'individualism', 'social',
            'individualist anarchism', 'left-wing', 'anarchist economics',
            'anarchist legal philosophy', 'anti-authoritarian interpretations', 'communism',
            'collectivism', 'syndicalism', 'mutualism', 'participatory economics'
        ]

    def tearDown(self):
        # remove all temporary test files
        fname = get_tmpfile('script.tst')
        extensions = ['', '.json']
        for ext in extensions:
            try:
                os.remove(fname + ext)
            except OSError:
                pass

    def test_segment_all_articles(self):
        title, sections, interlinks = next(segment_all_articles(self.fname))
        section_titles = [s[0] for s in sections]
        first_section_text = sections[0][1]

        self.assertEqual(title, self.expected_title)
        self.assertEqual(section_titles, self.expected_section_titles)
        first_sentence = "'''Anarchism''' is a political philosophy that advocates self-governed societies"
        self.assertTrue(first_sentence in first_section_text)
        for interlink in self.expected_interlinks:
            self.assertIn(interlink, interlinks)

    def test_generator_len(self):
        expected_num_articles = 106
        num_articles = sum(1 for x in segment_all_articles(self.fname))

        self.assertEqual(num_articles, expected_num_articles)

    def test_json_len(self):
        tmpf = get_tmpfile('script.tst.json')
        segment_and_write_all_articles(self.fname, tmpf, workers=1)

        expected_num_articles = 106
        num_articles = sum(1 for line in smart_open(tmpf))
        self.assertEqual(num_articles, expected_num_articles)

    def test_segment_and_write_all_articles(self):
        tmpf = get_tmpfile('script.tst.json')
        segment_and_write_all_articles(self.fname, tmpf, workers=1)

        # Get the first line from the text file we created.
        with open(tmpf) as f:
            first = next(f)

        # decode JSON line into a Python dictionary object
        article = json.loads(first)
        title, section_titles, interlinks = article['title'], article['section_titles'], article['interlinks']

        self.assertEqual(title, self.expected_title)
        self.assertEqual(section_titles, self.expected_section_titles)
        for interlink in self.expected_interlinks:
            self.assertIn(interlink, interlinks)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
