#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for the test corpus class in corpora.
"""


import logging
import unittest
import os
import tempfile

from gensim.utils import SaveLoad
from gensim.corpora.textcorpus import TextCorpus


# sample data files are located in the same folder
module_path = os.path.dirname(__file__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.WARNING)


def get_tmpfile(suffix=None):
    """TODO docstring for get_tmpfile"""
    return os.path.join(tempfile.gettempdir(), suffix)


def rm_tmpfiles():
    """rm all tmpfiles to make sure we only use the new files."""
    os.remove(get_tmpfile('tc_test.cpickle'))
    os.remove(get_tmpfile('tc_test_wordids.txt'))
    os.remove(get_tmpfile('tc_test_bow.mm'))


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

        # delete some files to make sure we don't use old files
        try:
            rm_tmpfiles()
        except OSError:
            pass

        self.headfile = os.path.join(module_path, 'head500.noblanks.cor.bz2')

        self.tc = TextCorpus(
                self.headfile,
                self.texts)

    def tearDown(self):
        # delete some files to make sure we don't use old files
        try:
            rm_tmpfiles()
        except OSError:
            pass

    def test_init(self):
        """ __init__ should create something (not `None`)."""
        self.assertNotEqual(self.tc, None)

    def test_build_tc(self):
        """
        Just make sure that TextCorpus works like the dictionary.
        This test is copied from test_corpora_dictionary.
        
        """
        expected = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 2, 7: 3, 8: 2,
                9: 3, 10: 3, 11: 2}
        self.assertEqual(self.tc.dfs, expected)

        expected = {'computer': 0, 'eps': 8, 'graph': 10, 'human': 1,
                'interface': 2, 'minors': 11, 'response': 3, 'survey': 4,
                'system': 5, 'time': 6, 'trees': 9, 'user': 7}
        self.assertEqual(self.tc.token2id, expected)

    def test_save_load_ability(self):
        """ Make sure we can save and load (un/pickle) the object. """
        tmpf = get_tmpfile('tc_test.cpickle')
        self.tc.save(tmpf)

        tc2 = SaveLoad.load(tmpf)
        tc2.load(tmpf)

        self.assertEqual(len(self.tc), len(tc2))
        self.assertEqual(self.tc.token2id, tc2.token2id)

    def test_saveAsText_and_loadFromText(self):
        """ TC can be saved as textfile and loaded again from textfile. """
        tmpf = get_tmpfile('tc_test')
        self.tc.saveAsText(tmpf)
        # does the file exists
        self.assertTrue(os.path.exists(tmpf + "_wordids.txt"))

        tc_loaded = TextCorpus.loadFromText(get_tmpfile('tc_test_wordids.txt'))
        self.assertNotEqual(tc_loaded, None)
        self.assertEqual(tc_loaded.token2id, self.tc.token2id)

    def test_saveAsMatrixMarket(self):
        tmpf = get_tmpfile('tc_test')
        self.tc.saveAsMatrixMarket(tmpf)
        self.assertTrue(os.path.exists(tmpf + '_bow.mm'))



if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
