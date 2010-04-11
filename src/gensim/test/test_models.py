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
import os.path
import tempfile

import numpy

from gensim.corpora import mmcorpus
from gensim.models import lsimodel, ldamodel, tfidfmodel
from gensim import matutils


module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_models.tst')


class TestLsiModel(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(os.path.join(module_path, 'testcorpus.mm'))
    
    def testTransform(self):
        # create the transformation model
        model = lsimodel.LsiModel(self.corpus, numTopics = 2)
        
        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, 2) # convert to dense vector, for easier equality tests
        
        expected = numpy.array([-0.6594664, 0.142115444]) # scaled LSI version
        # expected = numpy.array([-0.1973928, 0.05591352]) # non-scaled LSI version
        
        self.assertTrue(numpy.allclose(abs(vec), abs(expected))) # transformed entries must be equal up to sign
        
    
    def testPersistence(self):
        model = lsimodel.LsiModel(self.corpus, numTopics = 2)
        model.save(testfile())
        model2 = lsimodel.LsiModel.load(testfile())
        self.assertEqual(model.numTopics, model2.numTopics)
        self.assertTrue(numpy.allclose(model.projection, model2.projection))
#endclass TestLsiModel


class TestLdaModel(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(os.path.join(module_path, 'testcorpus.mm'))
    
    def testTransform(self):
        # create the transformation model
        model = ldamodel.LdaModel(self.corpus, numTopics = 2)
        
        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]
        
        vec = matutils.sparse2full(transformed, 2) # convert to dense vector, for easier equality tests
        expected = [0.0, 1.0]
        passed = False
        for i in xrange(10): # lda is randomized, so allow 10 iterations to test for equality
            passed = passed or numpy.allclose(sorted(vec), sorted(expected))  # must contain the same values, up to re-ordering
        self.assertTrue(passed, "Error in randomized LDA test")
        
    
    def testPersistence(self):
        model = lsimodel.LsiModel(self.corpus, numTopics = 2)
        model.save(testfile())
        model2 = lsimodel.LsiModel.load(testfile())
        self.assertEqual(model.numTopics, model2.numTopics)
        self.assertTrue(numpy.allclose(model.projection, model2.projection))
#endclass TestLdaModel


class TestTfidfModel(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(os.path.join(module_path, 'testcorpus.mm'))
    
    def testTransform(self):
        # create the transformation model
        model = tfidfmodel.TfidfModel(self.corpus, normalize = True)
        
        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]
        
        expected =  [(0, 0.57735026918962573), (1, 0.57735026918962573), (2, 0.57735026918962573)]
        self.assertTrue(numpy.allclose(transformed, expected))
        
    
    def testPersistence(self):
        model = tfidfmodel.TfidfModel(self.corpus, normalize = True)
        model.save(testfile())
        model2 = tfidfmodel.TfidfModel.load(testfile())
        self.assertTrue(model.idfs == model2.idfs)
#endclass TestTfidfModel


if __name__ == '__main__':
    logging.basicConfig(level = logging.WARNING)
    unittest.main()
