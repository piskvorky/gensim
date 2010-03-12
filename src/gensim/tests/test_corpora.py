#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking corpus I/O formats (the corpora package).
"""


import logging
import unittest

from gensim.corpora import dmlcorpus, bleicorpus, mmcorpus, lowcorpus, svmlightcorpus


#FIXME TODO

class TestMmCorpus(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus('deerwester.mm')
    
    def tearDown(self):
        pass
    
    def testLoad(self):
        pass
    
    def testSave(self):
        pass
#endclass TestMmCorpus


if __name__ == '__main__':
    unittest.main()
