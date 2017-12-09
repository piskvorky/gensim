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

import numpy as np

from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import rpmodel
from gensim import matutils
from gensim.test.utils import datapath, get_tmpfile


class TestRpModel(unittest.TestCase):
    def setUp(self):
        self.corpus = MmCorpus(datapath('testcorpus.mm'))

    def testTransform(self):
        # create the transformation model
        # HACK; set fixed seed so that we always get the same random matrix (and can compare against expected results)
        np.random.seed(13)
        model = rpmodel.RpModel(self.corpus, num_topics=2)

        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, 2)  # convert to dense vector, for easier equality tests

        expected = np.array([-0.70710677, 0.70710677])
        self.assertTrue(np.allclose(vec, expected))  # transformed entries must be equal up to sign

    def testPersistence(self):
        fname = get_tmpfile('gensim_models.tst')
        model = rpmodel.RpModel(self.corpus, num_topics=2)
        model.save(fname)
        model2 = rpmodel.RpModel.load(fname)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.projection, model2.projection))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector

    def testPersistenceCompressed(self):
        fname = get_tmpfile('gensim_models.tst.gz')
        model = rpmodel.RpModel(self.corpus, num_topics=2)
        model.save(fname)
        model2 = rpmodel.RpModel.load(fname, mmap=None)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.projection, model2.projection))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
