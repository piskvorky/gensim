#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayantjain1992@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking the poincare module from the models package.
"""

import logging
import unittest
import os

import numpy

from gensim.models.poincare import PoincareData


module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)
logger = logging.getLogger(__name__)


class TestPoincareData(unittest.TestCase):
    def test_encoding_handling(self):
        non_utf8_file = datapath('poincare_cp852.tsv')
        relations = [relation for relation in PoincareData(non_utf8_file, encoding='cp852')]
        self.assertEqual(len(relations), 2)
        self.assertEqual(relations[0], [u'tímto', u'budeš'])

        utf8_file = datapath('poincare_utf8.tsv')
        relations = [relation for relation in PoincareData(utf8_file)]
        self.assertEqual(len(relations), 2)
        self.assertEqual(relations[0], [u'tímto', u'budeš'])

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
