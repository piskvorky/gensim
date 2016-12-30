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
        raise NotImplementedError

    def testPersistence(self):
        """Test storing/loading the entire model."""
        raise NotImplementedError

    def testLoadVarembed(self):
        """Test model successfully loaded from wordrank .test files"""
        raise NotImplementedError

    def testSimilarity(self):
        """Test n_similarity for vocab words"""
        raise NotImplementedError

    def testLookup(self):
        raise NotImplementedError

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
