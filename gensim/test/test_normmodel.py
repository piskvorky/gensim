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
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from gensim.corpora import mmcorpus
from gensim.models import normmodel
from gensim.test.utils import datapath


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_models.tst')


class TestNormModel(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        # Choose doc to be normalized. [3] chosen to demonstrate different results for l1 and l2 norm.
        # doc is [(1, 1.0), (5, 2.0), (8, 1.0)]
        self.doc = list(self.corpus)[3]
        self.model_l1 = normmodel.NormModel(self.corpus, norm='l1')
        self.model_l2 = normmodel.NormModel(self.corpus, norm='l2')

    def test_tupleInput_l1(self):
        """Test tuple input for l1 transformation"""
        normalized = self.model_l1.normalize(self.doc)
        expected = [(1, 0.25), (5, 0.5), (8, 0.25)]
        self.assertTrue(np.allclose(normalized, expected))

    def test_sparseCSRInput_l1(self):
        """Test sparse csr matrix input for l1 transformation"""
        row = np.array([0, 0, 1, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        sparse_matrix = csr_matrix((data, (row, col)), shape=(3, 3))
        normalized = self.model_l1.normalize(sparse_matrix)

        # Check if output is of same type
        self.assertTrue(issparse(normalized))

        # Check if output is correct
        expected = np.array([[0.04761905, 0., 0.0952381],
                                [0., 0., 0.14285714],
                                [0.19047619, 0.23809524, 0.28571429]])
        self.assertTrue(np.allclose(normalized.toarray(), expected))

    def test_numpyndarrayInput_l1(self):
        """Test for np ndarray input for l1 transformation"""
        ndarray_matrix = np.array([
            [1, 0, 2],
            [0, 0, 3],
            [4, 5, 6]
        ])
        normalized = self.model_l1.normalize(ndarray_matrix)

        # Check if output is of same type
        self.assertTrue(isinstance(normalized, np.ndarray))

        # Check if output is correct
        expected = np.array([
            [0.04761905, 0., 0.0952381],
            [0., 0., 0.14285714],
            [0.19047619, 0.23809524, 0.28571429]
        ])
        self.assertTrue(np.allclose(normalized, expected))

        # Test if error is raised on unsupported input type
        self.assertRaises(ValueError, lambda model, doc: model.normalize(doc), self.model_l1, [1, 2, 3])

    def test_tupleInput_l2(self):
        """Test tuple input for l2 transformation"""
        normalized = self.model_l2.normalize(self.doc)
        expected = [(1, 0.4082482904638631), (5, 0.8164965809277261), (8, 0.4082482904638631)]
        self.assertTrue(np.allclose(normalized, expected))

    def test_sparseCSRInput_l2(self):
        """Test sparse csr matrix input for l2 transformation"""
        row = np.array([0, 0, 1, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        sparse_matrix = csr_matrix((data, (row, col)), shape=(3, 3))

        normalized = self.model_l2.normalize(sparse_matrix)

        # Check if output is of same type
        self.assertTrue(issparse(normalized))

        # Check if output is correct
        expected = np.array([
            [0.10482848, 0., 0.20965697],
            [0., 0., 0.31448545],
            [0.41931393, 0.52414242, 0.6289709]
        ])
        self.assertTrue(np.allclose(normalized.toarray(), expected))

    def test_numpyndarrayInput_l2(self):
        """Test for np ndarray input for l2 transformation"""
        ndarray_matrix = np.array([
            [1, 0, 2],
            [0, 0, 3],
            [4, 5, 6]
        ])
        normalized = self.model_l2.normalize(ndarray_matrix)

        # Check if output is of same type
        self.assertTrue(isinstance(normalized, np.ndarray))

        # Check if output is correct
        expected = np.array([
            [0.10482848, 0., 0.20965697],
            [0., 0., 0.31448545],
            [0.41931393, 0.52414242, 0.6289709]
        ])
        self.assertTrue(np.allclose(normalized, expected))

        # Test if error is raised on unsupported input type
        self.assertRaises(ValueError, lambda model, doc: model.normalize(doc), self.model_l2, [1, 2, 3])

    def testInit(self):
        """Test if error messages raised on unsupported norm"""
        self.assertRaises(ValueError, normmodel.NormModel, self.corpus, 'l0')

    def testPersistence(self):
        fname = testfile()
        model = normmodel.NormModel(self.corpus)
        model.save(fname)
        model2 = normmodel.NormModel.load(fname)
        self.assertTrue(model.norms == model2.norms)
        tstvec = []
        self.assertTrue(np.allclose(model.normalize(tstvec), model2.normalize(tstvec)))  # try projecting an empty vector

    def testPersistenceCompressed(self):
        fname = testfile() + '.gz'
        model = normmodel.NormModel(self.corpus)
        model.save(fname)
        model2 = normmodel.NormModel.load(fname, mmap=None)
        self.assertTrue(model.norms == model2.norms)
        tstvec = []
        self.assertTrue(np.allclose(model.normalize(tstvec), model2.normalize(tstvec)))  # try projecting an empty vector


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
