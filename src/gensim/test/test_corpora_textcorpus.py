#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for the test corpus class in corpora
"""


import logging
import unittest
import os
import tempfile

from gensim.utils import SaveLoad
from gensim.corpora.textCorpus import TextCorpus


# sample data files are located in the same folder
module_path = os.path.dirname(__file__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.WARNING)


class TestTextCorpus(unittest.TestCase):
    def setUp(self):
        self.texts = [
                ['human', 'interface', 'computer'],
                ['survey', 'user', 'computer', 'system', 'response', 'time'],
                ['eps', 'user', 'interface', 'system'],
                ['system', 'human', 'system', 'eps'],
                ['user', 'response', 'time'],
                ['trees'],
                ['graph', 'trees'],
                ['graph', 'minors', 'trees'],
                ['graph', 'minors', 'survey']]

        self.tmpdir = tempfile.gettempdir()

        # delete some files to make sure we don't use old files
        try:
            os.remove(os.path.join(self.tmpdir, 'tc_test.cpickle'))
            os.remove(os.path.join(self.tmpdir, 'tc_test_wordids.txt'))
        except OSError:
            pass

    def tearDown(self):
        # delete some files to make sure we don't use old files
        try:
            os.remove(os.path.join(self.tmpdir, 'tc_test.cpickle'))
            os.remove(os.path.join(self.tmpdir, 'tc_test_wordids.txt'))
        except OSError:
            pass

    def test_init(self):
        """ __init__ should create a TextCorpus. """
        self.tc = TextCorpus()
        self.assertNotEqual(self.tc, None)

    def test_build_tc(self):
        """
        Just make sure that TextCorpus woks like the dictionary.
        This test is copied from test_corpora_dictionary.
        """

        tc = TextCorpus(self.texts)
        expected = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 2, 7: 3, 8: 2,
                9: 3, 10: 3, 11: 2}
        self.assertEqual(tc.docFreq, expected)

        expected = {'computer': 0, 'eps': 8, 'graph': 10, 'human': 1,
                'interface': 2, 'minors': 11, 'response': 3, 'survey': 4,
                'system': 5, 'time': 6, 'trees': 9, 'user': 7}
        self.assertEqual(tc.token2id, expected)

        expected = dict((v, k) for k, v in expected.iteritems())
        self.assertEqual(tc.id2token, expected)

    def test_save_load_ability(self):
        """ Make sure we can save and load (un/pickle) the object. """
        tc = TextCorpus(self.texts)
        tmpfile = os.path.join(tempfile.gettempdir(), 'tc_test.cpickle')
        tc.save(tmpfile)

        tc2 = SaveLoad.load(tmpfile)
        tc2.load(tmpfile)

        self.assertEqual(len(tc), len(tc2))
        self.assertEqual(tc.id2word, tc2.id2word)
        self.assertEqual(tc.token2id, tc2.token2id)

    def test_saveAsText_and_loadFromText(self):
        """ TC can be saved as textfile and loaded again from textfile.
        """
        tc = TextCorpus(self.texts)
        tmpfile = os.path.join(tempfile.gettempdir(), 'tc_test')
        tc.saveAsText(tmpfile)
        # does the file exists
        self.assertTrue(os.path.exists(tmpfile + "_wordids.txt"))

        tc_loaded = TextCorpus.loadFromText(os.path.join(self.tmpdir,
                'tc_test_wordids.txt'))
        print tc_loaded
        self.assertNotEqual(tc_loaded, None)
        self.assertEqual(tc_loaded.token2id, tc.token2id)
        self.assertEqual(tc_loaded.id2token, tc.id2token)

    def test_saveAsMM(self):
        """docstring for test_saveAsMM"""
        #self.assertTrue(False)



