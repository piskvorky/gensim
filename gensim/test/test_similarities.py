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

import numpy
import scipy

from gensim.models import word2vec
from gensim.models import doc2vec
from gensim.models import fasttext
from gensim.models import KeyedVectors
from gensim import matutils, similarities
from gensim.models import Word2Vec
from gensim.test.utils import (datapath, get_tmpfile,
    common_texts as texts, common_dictionary as dictionary, common_corpus as corpus)

try:
    from pyemd import emd  # noqa:F401
    PYEMD_EXT = True
except ImportError:
    PYEMD_EXT = False

sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(texts)]


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
                [0.57735026, 0.57735026, 0.57735026, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.40824831, 0.0, 0.40824831, 0.40824831, 0.40824831, 0.40824831, 0.40824831, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.40824831, 0.0, 0.0, 0.0, 0.81649661, 0.0, 0.40824831, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.57735026, 0.57735026, 0.0, 0.0, 0.57735026, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1., 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.70710677, 0.70710677, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.57735026, 0.57735026, 0.57735026],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.57735026, 0.0, 0.0, 0.0, 0.0, 0.57735026, 0.57735026],
            ], dtype=numpy.float32)
            # HACK: dictionary can be in different order, so compare in sorted order
            self.assertTrue(numpy.allclose(sorted(expected.flat), sorted(index.index.flat)))
        index.num_best = num_best
        query = corpus[0]
        sims = index[query]
        expected = [(0, 0.99999994), (2, 0.28867513), (3, 0.23570226), (1, 0.23570226)][: num_best]

        # convert sims to full numpy arrays, so we can use allclose() and ignore
        # ordering of items with the same similarity value
        expected = matutils.sparse2full(expected, len(index))
        if num_best is not None:  # when num_best is None, sims is already a numpy array
            sims = matutils.sparse2full(sims, len(index))
        self.assertTrue(numpy.allclose(expected, sims))
        if self.cls == similarities.Similarity:
            index.destroy()

    def testNumBest(self):

        if self.cls == similarities.WmdSimilarity and not PYEMD_EXT:
            return

        for num_best in [None, 0, 1, 9, 1000]:
            self.testFull(num_best=num_best)

    def test_full2sparse_clipped(self):

        vec = [0.8, 0.2, 0.0, 0.0, -0.1, -0.15]
        expected = [(0, 0.80000000000000004), (1, 0.20000000000000001), (5, -0.14999999999999999)]
        self.assertTrue(matutils.full2sparse_clipped(vec, topn=3), expected)

    def test_scipy2scipy_clipped(self):
        # Test for scipy vector/row
        vec = [0.8, 0.2, 0.0, 0.0, -0.1, -0.15]
        expected = [(0, 0.80000000000000004), (1, 0.20000000000000001), (5, -0.14999999999999999)]
        vec_scipy = scipy.sparse.csr_matrix(vec)
        vec_scipy_clipped = matutils.scipy2scipy_clipped(vec_scipy, topn=3)
        self.assertTrue(scipy.sparse.issparse(vec_scipy_clipped))
        self.assertTrue(matutils.scipy2sparse(vec_scipy_clipped), expected)

        # Test for scipy matrix
        vec = [0.8, 0.2, 0.0, 0.0, -0.1, -0.15]
        expected = [(0, 0.80000000000000004), (1, 0.20000000000000001), (5, -0.14999999999999999)]
        matrix_scipy = scipy.sparse.csr_matrix([vec] * 3)
        matrix_scipy_clipped = matutils.scipy2scipy_clipped(matrix_scipy, topn=3)
        self.assertTrue(scipy.sparse.issparse(matrix_scipy_clipped))
        self.assertTrue([matutils.scipy2sparse(x) for x in matrix_scipy_clipped], [expected] * 3)

    def testChunking(self):
        if self.cls == similarities.Similarity:
            index = self.cls(None, corpus, num_features=len(dictionary), shardsize=5)
        else:
            index = self.cls(corpus, num_features=len(dictionary))
        query = corpus[:3]
        sims = index[query]
        expected = numpy.array([
            [0.99999994, 0.23570226, 0.28867513, 0.23570226, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.23570226, 1.0, 0.40824831, 0.33333334, 0.70710677, 0.0, 0.0, 0.0, 0.23570226],
            [0.28867513, 0.40824831, 1.0, 0.61237246, 0.28867513, 0.0, 0.0, 0.0, 0.0]
        ], dtype=numpy.float32)
        self.assertTrue(numpy.allclose(expected, sims))

        # test the same thing but with num_best
        index.num_best = 3
        sims = index[query]
        expected = [
            [(0, 0.99999994), (2, 0.28867513), (1, 0.23570226)],
            [(1, 1.0), (4, 0.70710677), (2, 0.40824831)],
            [(2, 1.0), (3, 0.61237246), (1, 0.40824831)]
        ]
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
            [0.99999994, 0.23570226, 0.28867513, 0.23570226, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.23570226, 1.0, 0.40824831, 0.33333334, 0.70710677, 0.0, 0.0, 0.0, 0.23570226],
            [0.28867513, 0.40824831, 1.0, 0.61237246, 0.28867513, 0.0, 0.0, 0.0, 0.0],
            [0.23570226, 0.33333334, 0.61237246, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.70710677, 0.28867513, 0.0, 0.99999994, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.70710677, 0.57735026, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.70710677, 0.99999994, 0.81649655, 0.40824828],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.57735026, 0.81649655, 0.99999994, 0.66666663],
            [0.0, 0.23570226, 0.0, 0.0, 0.0, 0.0, 0.40824828, 0.66666663, 0.99999994]
        ], dtype=numpy.float32)
        self.assertTrue(numpy.allclose(expected, sims))
        if self.cls == similarities.Similarity:
            index.destroy()

    def testPersistency(self):
        if self.cls == similarities.WmdSimilarity and not PYEMD_EXT:
            return

        fname = get_tmpfile('gensim_similarities.tst.pkl')
        if self.cls == similarities.Similarity:
            index = self.cls(None, corpus, num_features=len(dictionary), shardsize=5)
        elif self.cls == similarities.WmdSimilarity:
            index = self.cls(texts, self.w2v_model)
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

    def testPersistencyCompressed(self):
        if self.cls == similarities.WmdSimilarity and not PYEMD_EXT:
            return

        fname = get_tmpfile('gensim_similarities.tst.pkl.gz')
        if self.cls == similarities.Similarity:
            index = self.cls(None, corpus, num_features=len(dictionary), shardsize=5)
        elif self.cls == similarities.WmdSimilarity:
            index = self.cls(texts, self.w2v_model)
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
        if self.cls == similarities.WmdSimilarity and not PYEMD_EXT:
            return

        fname = get_tmpfile('gensim_similarities.tst.pkl')
        if self.cls == similarities.Similarity:
            index = self.cls(None, corpus, num_features=len(dictionary), shardsize=5)
        elif self.cls == similarities.WmdSimilarity:
            index = self.cls(texts, self.w2v_model)
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

    def testLargeCompressed(self):
        if self.cls == similarities.WmdSimilarity and not PYEMD_EXT:
            return

        fname = get_tmpfile('gensim_similarities.tst.pkl.gz')
        if self.cls == similarities.Similarity:
            index = self.cls(None, corpus, num_features=len(dictionary), shardsize=5)
        elif self.cls == similarities.WmdSimilarity:
            index = self.cls(texts, self.w2v_model)
        else:
            index = self.cls(corpus, num_features=len(dictionary))
        # store all arrays separately
        index.save(fname, sep_limit=0)

        index2 = self.cls.load(fname, mmap=None)
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
        if self.cls == similarities.WmdSimilarity and not PYEMD_EXT:
            return

        fname = get_tmpfile('gensim_similarities.tst.pkl')
        if self.cls == similarities.Similarity:
            index = self.cls(None, corpus, num_features=len(dictionary), shardsize=5)
        elif self.cls == similarities.WmdSimilarity:
            index = self.cls(texts, self.w2v_model)
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

    def testMmapCompressed(self):
        if self.cls == similarities.WmdSimilarity and not PYEMD_EXT:
            return

        fname = get_tmpfile('gensim_similarities.tst.pkl.gz')
        if self.cls == similarities.Similarity:
            index = self.cls(None, corpus, num_features=len(dictionary), shardsize=5)
        elif self.cls == similarities.WmdSimilarity:
            index = self.cls(texts, self.w2v_model)
        else:
            index = self.cls(corpus, num_features=len(dictionary))
        # store all arrays separately
        index.save(fname, sep_limit=0)

        # same thing, but use mmap to load arrays
        self.assertRaises(IOError, self.cls.load, fname, mmap='r')


class TestMatrixSimilarity(unittest.TestCase, _TestSimilarityABC):
    def setUp(self):
        self.cls = similarities.MatrixSimilarity


class TestWmdSimilarity(unittest.TestCase, _TestSimilarityABC):
    def setUp(self):
        self.cls = similarities.WmdSimilarity
        self.w2v_model = Word2Vec(texts, min_count=1)

    def testFull(self, num_best=None):
        # Override testFull.

        if not PYEMD_EXT:
            return

        index = self.cls(texts, self.w2v_model)
        index.num_best = num_best
        query = texts[0]
        sims = index[query]

        if num_best is not None:
            # Sparse array.
            for i, sim in sims:
                # Note that similarities are bigger than zero, as they are the 1/ 1 + distances.
                self.assertTrue(numpy.alltrue(sim > 0.0))
        else:
            self.assertTrue(sims[0] == 1.0)  # Similarity of a document with itself is 0.0.
            self.assertTrue(numpy.alltrue(sims[1:] > 0.0))
            self.assertTrue(numpy.alltrue(sims[1:] < 1.0))

    def testNonIncreasing(self):
        ''' Check that similarities are non-increasing when `num_best` is not
        `None`.'''
        # NOTE: this could be implemented for other similarities as well (i.e.
        # in _TestSimilarityABC).

        if not PYEMD_EXT:
            return

        index = self.cls(texts, self.w2v_model, num_best=3)
        query = texts[0]
        sims = index[query]
        sims2 = numpy.asarray(sims)[:, 1]  # Just the similarities themselves.

        # The difference of adjacent elements should be negative.
        cond = sum(numpy.diff(sims2) < 0) == len(sims2) - 1
        self.assertTrue(cond)

    def testChunking(self):
        # Override testChunking.

        if not PYEMD_EXT:
            return

        index = self.cls(texts, self.w2v_model)
        query = texts[:3]
        sims = index[query]

        for i in range(3):
            self.assertTrue(numpy.alltrue(sims[i, i] == 1.0))  # Similarity of a document with itself is 0.0.

        # test the same thing but with num_best
        index.num_best = 3
        sims = index[query]
        for sims_temp in sims:
            for i, sim in sims_temp:
                self.assertTrue(numpy.alltrue(sim > 0.0))
                self.assertTrue(numpy.alltrue(sim <= 1.0))

    def testIter(self):
        # Override testIter.

        if not PYEMD_EXT:
            return

        index = self.cls(texts, self.w2v_model)
        for sims in index:
            self.assertTrue(numpy.alltrue(sims >= 0.0))
            self.assertTrue(numpy.alltrue(sims <= 1.0))


class TestSparseMatrixSimilarity(unittest.TestCase, _TestSimilarityABC):
    def setUp(self):
        self.cls = similarities.SparseMatrixSimilarity

    def testMaintainSparsity(self):
        """Sparsity is correctly maintained when maintain_sparsity=True"""
        num_features = len(dictionary)

        index = self.cls(corpus, num_features=num_features)
        dense_sims = index[corpus]

        index = self.cls(corpus, num_features=num_features, maintain_sparsity=True)
        sparse_sims = index[corpus]

        self.assertFalse(scipy.sparse.issparse(dense_sims))
        self.assertTrue(scipy.sparse.issparse(sparse_sims))
        numpy.testing.assert_array_equal(dense_sims, sparse_sims.todense())

    def testMaintainSparsityWithNumBest(self):
        """Tests that sparsity is correctly maintained when maintain_sparsity=True and num_best is not None"""
        num_features = len(dictionary)

        index = self.cls(corpus, num_features=num_features, maintain_sparsity=False, num_best=3)
        dense_topn_sims = index[corpus]

        index = self.cls(corpus, num_features=num_features, maintain_sparsity=True, num_best=3)
        scipy_topn_sims = index[corpus]

        self.assertFalse(scipy.sparse.issparse(dense_topn_sims))
        self.assertTrue(scipy.sparse.issparse(scipy_topn_sims))
        self.assertEqual(dense_topn_sims, [matutils.scipy2sparse(v) for v in scipy_topn_sims])


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
        _ = index[corpus[0]]  # noqa:F841 forces shard close
        index.add_documents(corpus[5:])
        query = corpus[0]
        sims = index[query]
        expected = [(0, 0.99999994), (2, 0.28867513), (3, 0.23570226), (1, 0.23570226)]
        expected = matutils.sparse2full(expected, len(index))
        self.assertTrue(numpy.allclose(expected, sims))
        index.destroy()

    def testMmapCompressed(self):
        pass
        # turns out this test doesn't exercise this because there are no arrays
        # to be mmaped!

    def testChunksize(self):
        index = self.cls(None, corpus, num_features=len(dictionary), shardsize=5)
        expected = [sim for sim in index]
        index.chunksize = len(index) - 1
        sims = [sim for sim in index]
        self.assertTrue(numpy.allclose(expected, sims))
        index.destroy()


class TestWord2VecAnnoyIndexer(unittest.TestCase):

    def setUp(self):
        try:
            import annoy  # noqa:F401
        except ImportError:
            raise unittest.SkipTest("Annoy library is not available")

        from gensim.similarities.index import AnnoyIndexer
        self.indexer = AnnoyIndexer

    def testWord2Vec(self):
        model = word2vec.Word2Vec(texts, min_count=1)
        model.init_sims()
        index = self.indexer(model, 10)

        self.assertVectorIsSimilarToItself(model.wv, index)
        self.assertApproxNeighborsMatchExact(model, model.wv, index)
        self.assertIndexSaved(index)
        self.assertLoadedIndexEqual(index, model)

    def testFastText(self):
        corpus_file = datapath('lee.cor')
        model = fasttext.FastText(corpus_file)
        model.init_sims()
        index = self.indexer(model, 10)

        self.assertVectorIsSimilarToItself(model.wv, index)
        self.assertApproxNeighborsMatchExact(model, model.wv, index)
        self.assertIndexSaved(index)
        self.assertLoadedIndexEqual(index, model)

    def testAnnoyIndexingOfKeyedVectors(self):
        from gensim.similarities.index import AnnoyIndexer
        keyVectors_file = datapath('lee_fasttext.vec')
        model = KeyedVectors.load_word2vec_format(keyVectors_file)
        index = AnnoyIndexer(model, 10)

        self.assertEqual(index.num_trees, 10)
        self.assertVectorIsSimilarToItself(model, index)
        self.assertApproxNeighborsMatchExact(model, model, index)

    def testLoadMissingRaisesError(self):
        from gensim.similarities.index import AnnoyIndexer
        test_index = AnnoyIndexer()

        self.assertRaises(IOError, test_index.load, fname='test-index')

    def assertVectorIsSimilarToItself(self, wv, index):
        vector = wv.syn0norm[0]
        label = wv.index2word[0]
        approx_neighbors = index.most_similar(vector, 1)
        word, similarity = approx_neighbors[0]

        self.assertEqual(word, label)
        self.assertEqual(similarity, 1.0)

    def assertApproxNeighborsMatchExact(self, model, wv, index):
        vector = wv.syn0norm[0]
        approx_neighbors = model.most_similar([vector], topn=5, indexer=index)
        exact_neighbors = model.most_similar(positive=[vector], topn=5)

        approx_words = [neighbor[0] for neighbor in approx_neighbors]
        exact_words = [neighbor[0] for neighbor in exact_neighbors]

        self.assertEqual(approx_words, exact_words)

    def assertIndexSaved(self, index):
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index.save(fname)
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.exists(fname + '.d'))

    def assertLoadedIndexEqual(self, index, model):
        from gensim.similarities.index import AnnoyIndexer

        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index.save(fname)

        index2 = AnnoyIndexer()
        index2.load(fname)
        index2.model = model

        self.assertEqual(index.index.f, index2.index.f)
        self.assertEqual(index.labels, index2.labels)
        self.assertEqual(index.num_trees, index2.num_trees)


class TestDoc2VecAnnoyIndexer(unittest.TestCase):

    def setUp(self):
        try:
            import annoy  # noqa:F401
        except ImportError:
            raise unittest.SkipTest("Annoy library is not available")

        from gensim.similarities.index import AnnoyIndexer

        self.model = doc2vec.Doc2Vec(sentences, min_count=1)
        self.model.init_sims()
        self.index = AnnoyIndexer(self.model, 300)
        self.vector = self.model.docvecs.doctag_syn0norm[0]

    def testDocumentIsSimilarToItself(self):
        approx_neighbors = self.index.most_similar(self.vector, 1)
        doc, similarity = approx_neighbors[0]

        self.assertEqual(doc, 0)
        self.assertEqual(similarity, 1.0)

    def testApproxNeighborsMatchExact(self):
        approx_neighbors = self.model.docvecs.most_similar([self.vector], topn=5, indexer=self.index)
        exact_neighbors = self.model.docvecs.most_similar(
            positive=[self.vector], topn=5)

        approx_words = [neighbor[0] for neighbor in approx_neighbors]
        exact_words = [neighbor[0] for neighbor in exact_neighbors]

        self.assertEqual(approx_words, exact_words)

    def testSave(self):
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        self.index.save(fname)
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.exists(fname + '.d'))

    def testLoadNotExist(self):
        from gensim.similarities.index import AnnoyIndexer
        self.test_index = AnnoyIndexer()

        self.assertRaises(IOError, self.test_index.load, fname='test-index')

    def testSaveLoad(self):
        from gensim.similarities.index import AnnoyIndexer

        fname = get_tmpfile('gensim_similarities.tst.pkl')
        self.index.save(fname)

        self.index2 = AnnoyIndexer()
        self.index2.load(fname)
        self.index2.model = self.model

        self.assertEqual(self.index.index.f, self.index2.index.f)
        self.assertEqual(self.index.labels, self.index2.labels)
        self.assertEqual(self.index.num_trees, self.index2.num_trees)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
