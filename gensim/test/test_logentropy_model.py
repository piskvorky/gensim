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

import numpy as np

from gensim.corpora import mmcorpus, Dictionary
from gensim.models import logentropy_model

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder


def datapath(fname):
    return os.path.join(module_path, 'test_data', fname)

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


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_models.tst')


class TestLogEntropyModel(unittest.TestCase):
    def setUp(self):
        self.corpus_small = mmcorpus.MmCorpus(datapath('test_corpus_small.mm'))
        self.corpus_ok = mmcorpus.MmCorpus(datapath('test_corpus_ok.mm'))

    def testTransform(self):
        # create the transformation model
        model = logentropy_model.LogEntropyModel(self.corpus_ok, normalize=False)

        # transform one document
        doc = list(self.corpus_ok)[0]
        transformed = model[doc]

        expected = [
            (0, 0.3748900964125389),
            (1, 0.30730215324230725),
            (3, 1.20941755462856)
        ]
        self.assertTrue(np.allclose(transformed, expected))

    def testPersistence(self):
        fname = testfile()
        model = logentropy_model.LogEntropyModel(self.corpus_ok, normalize=True)
        model.save(fname)
        model2 = logentropy_model.LogEntropyModel.load(fname)
        self.assertTrue(model.entr == model2.entr)
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))

    def testPersistenceCompressed(self):
        fname = testfile() + '.gz'
        model = logentropy_model.LogEntropyModel(self.corpus_ok, normalize=True)
        model.save(fname)
        model2 = logentropy_model.LogEntropyModel.load(fname, mmap=None)
        self.assertTrue(model.entr == model2.entr)
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
