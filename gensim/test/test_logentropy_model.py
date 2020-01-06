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
from gensim.models import logentropy_model
from gensim.test.utils import datapath, get_tmpfile


class TestLogEntropyModel(unittest.TestCase):
    TEST_CORPUS = [[(1, 1.0)], [], [(0, 0.5), (2, 1.0)], []]

    def setUp(self):
        self.corpus_small = MmCorpus(datapath('test_corpus_small.mm'))
        self.corpus_ok = MmCorpus(datapath('test_corpus_ok.mm'))
        self.corpus_empty = []

    def test_generator_fail(self):
        """Test creating a model using a generator as input; should fail."""
        def get_generator(test_corpus=TestLogEntropyModel.TEST_CORPUS):
            for test_doc in test_corpus:
                yield test_doc
        self.assertRaises(ValueError, logentropy_model.LogEntropyModel, corpus=get_generator())

    def test_empty_fail(self):
        """Test creating a model using an empty input; should fail."""
        self.assertRaises(ValueError, logentropy_model.LogEntropyModel, corpus=self.corpus_empty)

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
        fname = get_tmpfile('gensim_models_logentry.tst')
        model = logentropy_model.LogEntropyModel(self.corpus_ok, normalize=True)
        model.save(fname)
        model2 = logentropy_model.LogEntropyModel.load(fname)
        self.assertTrue(model.entr == model2.entr)
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))

    def testPersistenceCompressed(self):
        fname = get_tmpfile('gensim_models_logentry.tst.gz')
        model = logentropy_model.LogEntropyModel(self.corpus_ok, normalize=True)
        model.save(fname)
        model2 = logentropy_model.LogEntropyModel.load(fname, mmap=None)
        self.assertTrue(model.entr == model2.entr)
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
