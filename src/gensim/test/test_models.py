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
from gensim.models import lsimodel, ldamodel, tfidfmodel, rpmodel
from gensim import matutils


module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_models.tst')


class TestLsiModel(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(os.path.join(module_path, 'testcorpus.mm'))
    
    def testTransform(self):
        # create the transformation model
        model = lsimodel.LsiModel(self.corpus, numTopics=2)
        
        # make sure the decomposition is enough accurate
        u, s, vt = numpy.linalg.svd(matutils.corpus2dense(self.corpus, self.corpus.numTerms), full_matrices=False)
        self.assertTrue(numpy.allclose(s[:2], model.projection.s)) # singular values must match
        
        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, 2) # convert to dense vector, for easier equality tests
        
        expected = numpy.array([-0.6594664, 0.142115444]) # scaled LSI version
        # expected = numpy.array([-0.1973928, 0.05591352]) # non-scaled LSI version
        self.assertTrue(numpy.allclose(abs(vec), abs(expected))) # transformed entries must be equal up to sign
    
    
    def testOnlineTransform(self):
        corpus = list(self.corpus)
        doc = corpus[0] # use the corpus' first document for testing
        
        # create the transformation model
        model2 = lsimodel.LsiModel(corpus = corpus, numTopics = 5) # compute everything at once
        model = lsimodel.LsiModel(corpus = None, id2word = model2.id2word, numTopics = 5) # start with no documents, we will add then later
        
        # train model on a single document
        model.addDocuments([corpus[0]])
        model.printDebug()
        
        # transform the testing document with this partial transformation
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, model.numTopics) # convert to dense vector, for easier equality tests
        expected = numpy.array([-1.73205078, 0.0, 0.0, 0.0, 0.0]) # scaled LSI version
        self.assertTrue(numpy.allclose(abs(vec), abs(expected), atol = 1e-6)) # transformed entries must be equal up to sign
        
        # train on another 4 documents
        model.addDocuments(corpus[1:5], chunks = 2) # train in chunks of 2 documents, for the lols
        model.printDebug()
        
        # transform a document with this partial transformation
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, model.numTopics) # convert to dense vector, for easier equality tests
        expected = numpy.array([-0.66493785, -0.28314203, -1.56376302,  0.05488682,  0.17123269]) # scaled LSI version
        m2 = lsimodel.LsiModel(corpus = list(corpus)[:5], numTopics = 5)
        self.assertTrue(numpy.allclose(abs(vec), abs(expected), atol = 1e-6)) # transformed entries must be equal up to sign
        
        # train on the rest of documents
        model.addDocuments(corpus[5:])
        model.printDebug()
        
        # make sure the final transformation is the same as if we had decomposed the whole corpus at once
        vec1 = matutils.sparse2full(model[doc], model.numTopics)
        vec2 = matutils.sparse2full(model2[doc], model2.numTopics)
        self.assertTrue(numpy.allclose(abs(vec1), abs(vec2), atol = 1e-6)) # the two LSI representations must equal up to sign

    
    def testPersistence(self):
        model = lsimodel.LsiModel(self.corpus, numTopics = 2)
        model.save(testfile())
        model2 = lsimodel.LsiModel.load(testfile())
        self.assertEqual(model.numTopics, model2.numTopics)
        self.assertTrue(numpy.allclose(model.projection.u, model2.projection.u))
        self.assertTrue(numpy.allclose(model.projection.s, model2.projection.s))
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector
#endclass TestLsiModel


class TestRpModel(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(os.path.join(module_path, 'testcorpus.mm'))
    
    def testTransform(self):
        # create the transformation model
        numpy.random.seed(13) # HACK; set fixed seed so that we always get the same random matrix (and can compare against expected results)
        model = rpmodel.RpModel(self.corpus, numTopics = 2)
        
        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, 2) # convert to dense vector, for easier equality tests
        
        expected = numpy.array([-0.70710677, 0.70710677])
        self.assertTrue(numpy.allclose(vec, expected)) # transformed entries must be equal up to sign
        
    
    def testPersistence(self):
        model = rpmodel.RpModel(self.corpus, numTopics = 2)
        model.save(testfile())
        model2 = rpmodel.RpModel.load(testfile())
        self.assertEqual(model.numTopics, model2.numTopics)
        self.assertTrue(numpy.allclose(model.projection, model2.projection))
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector
#endclass TestRpModel


class TestLdaModel(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(os.path.join(module_path, 'testcorpus.mm'))
    
    def testTransform(self):
        # create the transformation model
        model = ldamodel.LdaModel(self.corpus, numTopics=2, passes=100) # 100 passes ought to be enough to converge
        
        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]
        
        vec = matutils.sparse2full(transformed, 2) # convert to dense vector, for easier equality tests
        expected = [0.87, 0.13]
        assert numpy.allclose(sorted(vec), sorted(expected), atol=0.01)  # must contain the same values, up to re-ordering
        
    
    def testPersistence(self):
        model = ldamodel.LdaModel(self.corpus, numTopics = 2)
        model.save(testfile())
        model2 = ldamodel.LdaModel.load(testfile())
        self.assertEqual(model.numTopics, model2.numTopics)
        self.assertTrue(numpy.allclose(model.expElogbeta, model2.expElogbeta))
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector
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
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector
#endclass TestTfidfModel


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
