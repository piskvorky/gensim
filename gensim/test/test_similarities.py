#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for similarity algorithms (the similarities package).
"""


import logging
import unittest
import os
import os.path
import tempfile

import numpy

from gensim.corpora import mmcorpus, Dictionary
from gensim import matutils, utils, similarities


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
    return os.path.join(tempfile.gettempdir(), 'gensim_similarities.tst')


class TestSimilarityABC(object):
    """
    Base class for SparseMatrixSimilarity and MatrixSimilarity unit tests.
    """

    def testFull(self):
        if self.cls == similarities.Similarity:
            index = self.cls(testfile(), corpus, num_features=len(dictionary), shardsize=5)
        else:
            index = self.cls(corpus)
        if isinstance(index, similarities.MatrixSimilarity):
            expected = numpy.array([
                [ 0.57735026, 0.57735026, 0.57735026, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                [ 0.40824831, 0.0, 0.0, 0.40824831, 0.40824831, 0.40824831, 0.40824831, 0.40824831, 0.0, 0.0, 0.0, 0.0 ],
                [ 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0 ],
                [ 0.0, 0.40824831, 0.0, 0.0, 0.0, 0.81649661, 0.0, 0.0, 0.40824831, 0.0, 0.0, 0.0 ],
                [ 0.0, 0.0, 0.0, 0.57735026, 0.0, 0.0, 0.57735026, 0.57735026, 0.0, 0.0, 0.0, 0.0 ],
                [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
                [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.70710677, 0.70710677, 0.0 ],
                [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.57735026, 0.57735026, 0.57735026 ],
                [ 0.0, 0.0, 0.0, 0.0, 0.57735026, 0.0, 0.0, 0.0, 0.0, 0.0, 0.57735026, 0.57735026 ]
                ], dtype=numpy.float32)
            self.assertTrue(numpy.allclose(expected, index.index))
        query = corpus[0]
        sims = index[query]
        expected = numpy.array([0.99999994, 0.23570226, 0.28867513, 0.23570226, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=numpy.float32)
        self.assertTrue(numpy.allclose(expected, sims))


    def testNumBest(self):
        if self.cls == similarities.Similarity:
            index = self.cls(testfile(), corpus, num_features=len(dictionary), shardsize=5, num_best=3)
        else:
            index = self.cls(corpus, num_best=3)
        query = corpus[0]
        sims = index[query]
        expected = [(0, 0.99999994), (2, 0.28867513), (3, 0.23570226)]
        self.assertTrue(numpy.allclose(expected, sims))


    def testChunking(self):
        if self.cls == similarities.Similarity:
            index = self.cls(testfile(), corpus, num_features=len(dictionary), shardsize=5)
        else:
            index = self.cls(corpus)
        query = corpus[:3]
        sims = index[query]
        expected = numpy.array([
            [ 0.99999994, 0.23570226, 0.28867513, 0.23570226, 0.0, 0.0, 0.0, 0.0, 0.0 ],
            [ 0.23570226, 1.0, 0.40824831, 0.33333334, 0.70710677, 0.0, 0.0, 0.0, 0.23570226 ],
            [ 0.28867513, 0.40824831, 1.0, 0.61237246, 0.28867513, 0.0, 0.0, 0.0, 0.0 ]
            ], dtype=numpy.float32)
        self.assertTrue(numpy.allclose(expected, sims))

        # test the same thing but with num_best
        index.num_best = 3
        sims = index[query]
        expected = [[(0, 0.99999994), (2, 0.28867513), (3, 0.23570226)],
                    [(1, 1.0), (4, 0.70710677), (2, 0.40824831)],
                    [(2, 1.0), (3, 0.61237246), (1, 0.40824831)]]
        self.assertTrue(numpy.allclose(expected, sims))



    def testIter(self):
        if self.cls == similarities.Similarity:
            index = self.cls(testfile(), corpus, num_features=len(dictionary), shardsize=5)
        else:
            index = self.cls(corpus)
        sims = [sim for sim in index]
        expected = numpy.array([
            [ 0.99999994, 0.23570226, 0.28867513, 0.23570226, 0.0, 0.0, 0.0, 0.0, 0.0 ],
            [ 0.23570226, 1.0, 0.40824831, 0.33333334, 0.70710677, 0.0, 0.0, 0.0, 0.23570226 ],
            [ 0.28867513, 0.40824831, 1.0, 0.61237246, 0.28867513, 0.0, 0.0, 0.0, 0.0 ],
            [ 0.23570226, 0.33333334, 0.61237246, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.70710677, 0.28867513, 0.0, 0.99999994, 0.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.70710677, 0.57735026, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.70710677, 0.99999994, 0.81649655, 0.40824828 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.57735026, 0.81649655, 0.99999994, 0.66666663 ],
            [ 0.0, 0.23570226, 0.0, 0.0, 0.0, 0.0, 0.40824828, 0.66666663, 0.99999994 ]
            ], dtype=numpy.float32)
        self.assertTrue(numpy.allclose(expected, sims))


    def testPersistency(self):
        fname = testfile() + '.pkl'
        if self.cls == similarities.Similarity:
            index = self.cls(testfile(), corpus, num_features=len(dictionary), shardsize=5)
        else:
            index = self.cls(corpus)
        index.save(fname)
        index2 = self.cls.load(fname)
        if self.cls == similarities.Similarity:
            # for Similarity, only do a basic check
            self.assertTrue(len(index.shards) == len(index2.shards))
            return
        if isinstance(index, similarities.SparseMatrixSimilarity):
            # hack SparseMatrixSim indexes so they're easy to compare
            index.index = index.index.todense()
            index2.index = index2.index.todense()
        self.assertTrue(numpy.allclose(index.index, index2.index))
        self.assertEqual(index.num_best, index2.num_best)



class TestMatrixSimilarity(unittest.TestCase, TestSimilarityABC):
    def setUp(self):
        self.cls = similarities.MatrixSimilarity


class TestSparseMatrixSimilarity(unittest.TestCase, TestSimilarityABC):
    def setUp(self):
        self.cls = similarities.SparseMatrixSimilarity


class TestSimilarity(unittest.TestCase, TestSimilarityABC):
    def setUp(self):
        self.cls = similarities.Similarity




if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
