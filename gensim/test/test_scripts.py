#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Vimig Socrates <vimig.socrates@gmail.com> heavily influenced from @AakaashRao
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

from smart_open import smart_open
import numpy as np

from gensim.scripts.segment_wiki import segment_all_articles, segment_and_write_all_articles
from gensim.test.utils import datapath, get_tmpfile

from gensim.scripts.word2vec2tensor import word2vec2tensor
from gensim.models import KeyedVectors


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
        title, sections, interlinks = next(segment_all_articles(self.fname, include_interlinks=True))

        # Check title
        self.assertEqual(title, self.expected_title)

        # Check section titles
        section_titles = [s[0] for s in sections]
        self.assertEqual(section_titles, self.expected_section_titles)

        # Check text
        first_section_text = sections[0][1]
        first_sentence = "'''Anarchism''' is a political philosophy that advocates self-governed societies"
        self.assertTrue(first_sentence in first_section_text)

        # Check interlinks
        self.assertTrue(interlinks['self-governance'] == 'self-governed')
        self.assertTrue(interlinks['Hierarchy'] == 'hierarchical')
        self.assertTrue(interlinks['Pierre-Joseph Proudhon'] == 'Proudhon')

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
        segment_and_write_all_articles(self.fname, tmpf, workers=1, include_interlinks=True)

        # Get the first line from the text file we created.
        with open(tmpf) as f:
            first = next(f)

        # decode JSON line into a Python dictionary object
        article = json.loads(first)
        title, section_titles, interlinks = article['title'], article['section_titles'], article['interlinks']

        self.assertEqual(title, self.expected_title)
        self.assertEqual(section_titles, self.expected_section_titles)

        # Check interlinks
        self.assertTrue(interlinks['self-governance'] == 'self-governed')
        self.assertTrue(interlinks['Hierarchy'] == 'hierarchical')
        self.assertTrue(interlinks['Pierre-Joseph Proudhon'] == 'Proudhon')


class TestWord2Vec2Tensor(unittest.TestCase):
    def setUp(self):
        self.datapath = datapath('word2vec_pre_kv_c')
        self.output_folder = get_tmpfile('w2v2t_test')
        self.metadata_file = self.output_folder + '_metadata.tsv'
        self.tensor_file = self.output_folder + '_tensor.tsv'
        self.vector_file = self.output_folder + '_vector.tsv'

    def testConversion(self):
        word2vec2tensor(word2vec_model_path=self.datapath, tensor_filename=self.output_folder)

        with smart_open(self.metadata_file, 'rb') as f:
            metadata = f.readlines()

        with smart_open(self.tensor_file, 'rb') as f:
            vectors = f.readlines()

        # check if number of words and vector size in tensor file line up with word2vec
        with smart_open(self.datapath, 'rb') as f:
            first_line = f.readline().strip()

        number_words, vector_size = map(int, first_line.split(b' '))
        self.assertTrue(len(metadata) == len(vectors) == number_words,
            ('Metadata file %s and tensor file %s imply different number of rows.'
                % (self.metadata_file, self.tensor_file)))

        # grab metadata and vectors from written file
        metadata = [word.strip() for word in metadata]
        vectors = [vector.replace(b'\t', b' ') for vector in vectors]

        # get the originaly vector KV model
        orig_model = KeyedVectors.load_word2vec_format(self.datapath, binary=False)

        # check that the KV model and tensor files have the same values key-wise
        for word, vector in zip(metadata, vectors):
            word_string = word.decode("utf8")
            vector_string = vector.decode("utf8")
            vector_array = np.array(list(map(float, vector_string.split())))
            np.testing.assert_almost_equal(orig_model[word_string], vector_array, decimal=5)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
