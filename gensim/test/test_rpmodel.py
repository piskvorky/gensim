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

import six
import numpy as np
import scipy.linalg

from gensim.corpora import mmcorpus, Dictionary
from gensim.models import rpmodel
from gensim import matutils

module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


# set up vars used in testing ("Deerwester" from the web tutorial)
texts = [['human', 'interface', 'computer'],
 ['survey', 'user', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'system'],
 ['system', 'human', 'system', 'eps'],
 ['user', 'response', 'time'],
 ['trees'],
 ['graph', 'trees'],
 ['graph', 'minors', 'trees'],
 ['graph', 'minors', 'survey']]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_models.tst')



class TestRpModel(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))

    def testTransform(self):
        # create the transformation model
        np.random.seed(13) # HACK; set fixed seed so that we always get the same random matrix (and can compare against expected results)
        model = rpmodel.RpModel(self.corpus, num_topics=2)

        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, 2) # convert to dense vector, for easier equality tests

        expected = np.array([-0.70710677, 0.70710677])
        self.assertTrue(np.allclose(vec, expected)) # transformed entries must be equal up to sign


    def testPersistence(self):
        fname = testfile()
        model = rpmodel.RpModel(self.corpus, num_topics=2)
        model.save(fname)
        model2 = rpmodel.RpModel.load(fname)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.projection, model2.projection))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector

    def testPersistenceCompressed(self):
        fname = testfile() + '.gz'
        model = rpmodel.RpModel(self.corpus, num_topics=2)
        model.save(fname)
        model2 = rpmodel.RpModel.load(fname, mmap=None)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.projection, model2.projection))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector
#endclass TestRpModel


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
