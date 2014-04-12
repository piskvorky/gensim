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
    return os.path.join(tempfile.gettempdir(), 'gensim_similarities.tst.pkl')


class _TestSimilarityABC(object):
    """
    Base class for SparseMatrixSimilarity and MatrixSimilarity unit tests.
    """
    def testFull(self, num_best=None, shardsize=100):
        if self.cls == similarities.Similarity:
            index = self.cls(None, corpus, num_features=len(dictionary), shardsize=shardsize)
        else:
            index = self.cls(corpus, num_features=len(dictionary))
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
        index.num_best = num_best
        query = corpus[0]
        sims = index[query]
        expected = [(0, 0.99999994), (2, 0.28867513), (3, 0.23570226), (1, 0.23570226)][ : num_best]

        # convert sims to full numpy arrays, so we can use allclose() and ignore
        # ordering of items with the same similarity value
        expected = matutils.sparse2full(expected, len(index))
        if num_best is not None: # when num_best is None, sims is already a numpy array
            sims = matutils.sparse2full(sims, len(index))
        self.assertTrue(numpy.allclose(expected, sims))
        if self.cls == similarities.Similarity:
            index.destroy()


    def testNumBest(self):
        for num_best in [None, 0, 1, 9, 1000]:
            self.testFull(num_best=num_best)


    def testChunking(self):
        if self.cls == similarities.Similarity:
            index = self.cls(None, corpus, num_features=len(dictionary), shardsize=5)
        else:
            index = self.cls(corpus, num_features=len(dictionary))
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
        if self.cls == similarities.Similarity:
            index.destroy()


    def testIter(self):
        if self.cls == similarities.Similarity:
            index = self.cls(None, corpus, num_features=len(dictionary), shardsize=5)
        else:
            index = self.cls(corpus, num_features=len(dictionary))
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
        if self.cls == similarities.Similarity:
            index.destroy()


    def testPersistency(self):
        fname = testfile()
        if self.cls == similarities.Similarity:
            index = self.cls(None, corpus, num_features=len(dictionary), shardsize=5)
        else:
            index = self.cls(corpus, num_features=len(dictionary))
        index.save(fname)
        index2 = self.cls.load(fname)
        if self.cls == similarities.Similarity:
            # for Similarity, only do a basic check
            self.assertTrue(len(index.shards) == len(index2.shards))
            index.destroy()
        else:
            if isinstance(index, similarities.SparseMatrixSimilarity):
                # hack SparseMatrixSim indexes so they're easy to compare
                index.index = index.index.todense()
                index2.index = index2.index.todense()
            self.assertTrue(numpy.allclose(index.index, index2.index))
            self.assertEqual(index.num_best, index2.num_best)


    def testLarge(self):
        fname = testfile()
        if self.cls == similarities.Similarity:
            index = self.cls(None, corpus, num_features=len(dictionary), shardsize=5)
        else:
            index = self.cls(corpus, num_features=len(dictionary))
        # store all arrays separately
        index.save(fname, sep_limit=0)

        index2 = self.cls.load(fname)
        if self.cls == similarities.Similarity:
            # for Similarity, only do a basic check
            self.assertTrue(len(index.shards) == len(index2.shards))
            index.destroy()
        else:
            if isinstance(index, similarities.SparseMatrixSimilarity):
                # hack SparseMatrixSim indexes so they're easy to compare
                index.index = index.index.todense()
                index2.index = index2.index.todense()
            self.assertTrue(numpy.allclose(index.index, index2.index))
            self.assertEqual(index.num_best, index2.num_best)


    def testMmap(self):
        fname = testfile()
        if self.cls == similarities.Similarity:
            index = self.cls(None, corpus, num_features=len(dictionary), shardsize=5)
        else:
            index = self.cls(corpus, num_features=len(dictionary))
        # store all arrays separately
        index.save(fname, sep_limit=0)

        # same thing, but use mmap to load arrays
        index2 = self.cls.load(fname, mmap='r')
        if self.cls == similarities.Similarity:
            # for Similarity, only do a basic check
            self.assertTrue(len(index.shards) == len(index2.shards))
            index.destroy()
        else:
            if isinstance(index, similarities.SparseMatrixSimilarity):
                # hack SparseMatrixSim indexes so they're easy to compare
                index.index = index.index.todense()
                index2.index = index2.index.todense()
            self.assertTrue(numpy.allclose(index.index, index2.index))
            self.assertEqual(index.num_best, index2.num_best)


class TestMatrixSimilarity(unittest.TestCase, _TestSimilarityABC):
    def setUp(self):
        self.cls = similarities.MatrixSimilarity


class TestSparseMatrixSimilarity(unittest.TestCase, _TestSimilarityABC):
    def setUp(self):
        self.cls = similarities.SparseMatrixSimilarity


class TestSimilarity(unittest.TestCase, _TestSimilarityABC):
    def setUp(self):
        self.cls = similarities.Similarity

    def testSharding(self):
        for num_best in [None, 0, 1, 9, 1000]:
            for shardsize in [1, 2, 9, 1000]:
                self.testFull(num_best=num_best, shardsize=shardsize)

    def testReopen(self):
        """test re-opening partially full shards"""
        index = similarities.Similarity(None, corpus[:5], num_features=len(dictionary), shardsize=9)
        _ = index[corpus[0]] # forces shard close
        index.add_documents(corpus[5:])
        query = corpus[0]
        sims = index[query]
        expected = [(0, 0.99999994), (2, 0.28867513), (3, 0.23570226), (1, 0.23570226)]
        expected = matutils.sparse2full(expected, len(index))
        self.assertTrue(numpy.allclose(expected, sims))
        index.destroy()



if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
