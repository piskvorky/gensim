#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""


import logging
import unittest
import os
import tempfile

import numpy

from gensim.models.wrappers import wordrank

# needed because sample data files are located in the same folder
module_path = os.path.dirname(__file__)
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


def testfile():
    raise NotImplementedError


class TestVarembed(unittest.TestCase):
    def setUp(self):
        wr_home = os.environ.get('WR_HOME', None)
        self.wr_path = wr_home if wr_home else None
        self.corpus_file = datapath('lee.cor')
        if self.wr_path:
            self.test_model = wordrank.Wordrank.train(
                self.wr_path, self.corpus_file)

    def testPersistence(self):
        """Test storing/loading the entire model."""
        # if not self.wr_path:
        #     return
        # model = wordrank.Wordrank.train(self.wr_path, self.corpus_file)
        # model.save(testfile())
        # loaded = wordrank.Wordrank.load(testfile())
        # self.models_equal(model, loaded)
        raise NotImplementedError

    def testLoadVarembed(self):
        """Test model successfully loaded from wordrank .test files"""
        raise NotImplementedError

    def testSimilarity(self):
        """Test n_similarity for vocab words"""
        # if not self.wr_path:
        #     return
        # In vocab, sanity check
        # self.assertTrue(numpy.allclose(self.test_model.n_similarity(['the', 'and'], ['and', 'the']), 1.0))
        # self.assertEqual(self.test_model.similarity('the', 'and'), self.test_model.similarity('the', 'and'))
        raise NotImplementedError

    def testLookup(self):
        # if not self.wr_path:
        #     return
        # In vocab, sanity check
        # self.assertTrue(numpy.allclose(self.test_model['night'], self.test_model[['night']]))
        raise NotImplementedError

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
