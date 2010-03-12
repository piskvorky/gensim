#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""


import logging
import unittest

from gensim.corpora import mmcorpus
from gensim.models import lsimodel, ldamodel, tfidfmodel

# FIXME TODO

class TestLsiModel(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus('testcorpus.mm')
    
    def tearDown(self):
        pass
    
    def testInference(self):
        model = lsimodel.LsiModel(self.corpus, numTopics = 2)
        
    
    def testPersistence(self):
        pass
#endclass TestLsiModel


class TestLdaModel(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def testInference(self):
        pass
    
    def testPersistence(self):
        pass
#endclass TestLdaModel


class TestRPModel(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def testInference(self):
        pass
    
    def testPersistence(self):
        pass
#endclass TestRPModel


if __name__ == '__main__':
    unittest.main()
