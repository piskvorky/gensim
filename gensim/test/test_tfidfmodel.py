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

from gensim.corpora import mmcorpus, Dictionary
from gensim.models import tfidfmodel
from gensim.test.utils import datapath, get_tmpfile

# set up vars used in testing ("Deerwester" from the web tutorial)
texts = [
    ['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey']
]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


class TestTfidfModel(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))

    def testTransform(self):
        # create the transformation model
        model = tfidfmodel.TfidfModel(self.corpus, normalize=True)

        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]

        expected = [(0, 0.57735026918962573), (1, 0.57735026918962573), (2, 0.57735026918962573)]
        self.assertTrue(np.allclose(transformed, expected))

    def testInit(self):
        # create the transformation model by analyzing a corpus
        # uses the global `corpus`!
        model1 = tfidfmodel.TfidfModel(corpus)

        # make sure the dfs<->idfs transformation works
        self.assertEqual(model1.dfs, dictionary.dfs)
        self.assertEqual(model1.idfs, tfidfmodel.precompute_idfs(model1.wglobal, dictionary.dfs, len(corpus)))

        # create the transformation model by directly supplying a term->docfreq
        # mapping from the global var `dictionary`.
        model2 = tfidfmodel.TfidfModel(dictionary=dictionary)
        self.assertEqual(model1.idfs, model2.idfs)

    def testPersistence(self):
        fname = get_tmpfile('gensim_models.tst')
        model = tfidfmodel.TfidfModel(self.corpus, normalize=True)
        model.save(fname)
        model2 = tfidfmodel.TfidfModel.load(fname)
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector

    def testPersistenceCompressed(self):
        fname = get_tmpfile('gensim_models.tst.gz')
        model = tfidfmodel.TfidfModel(self.corpus, normalize=True)
        model.save(fname)
        model2 = tfidfmodel.TfidfModel.load(fname, mmap=None)
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector
# endclass TestTfidfModel


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
