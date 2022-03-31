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
import math
import os

import numpy
import scipy

from gensim import utils
from gensim.corpora import Dictionary
from gensim.models import word2vec
from gensim.models import doc2vec
from gensim.models import KeyedVectors
from gensim.models import TfidfModel
from gensim import matutils, similarities
from gensim.models import Word2Vec, FastText
from gensim.test.utils import (
    datapath, get_tmpfile,
    common_texts as TEXTS, common_dictionary as DICTIONARY, common_corpus as CORPUS,
)
from gensim.similarities import UniformTermSimilarityIndex
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import LevenshteinSimilarityIndex
from gensim.similarities.docsim import _nlargest
from gensim.similarities.fastss import editdist

try:
    from pyemd import emd  # noqa:F401
    PYEMD_EXT = True
except (ImportError, ValueError):
    PYEMD_EXT = False

SENTENCES = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(TEXTS)]


@unittest.skip("skipping abstract base class")
class _TestSimilarityABC(unittest.TestCase):
    """
    Base class for SparseMatrixSimilarity and MatrixSimilarity unit tests.
    """

    def factoryMethod(self):
        """Creates a SimilarityABC instance."""
        return self.cls(CORPUS, num_features=len(DICTIONARY))

    def test_full(self, num_best=None, shardsize=100):
        if self.cls == similarities.Similarity:
            index = self.cls(None, CORPUS, num_features=len(DICTIONARY), shardsize=shardsize)
        else:
            index = self.cls(CORPUS, num_features=len(DICTIONARY))
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
        query = CORPUS[0]
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

    def test_num_best(self):
        if self.cls == similarities.WmdSimilarity and not PYEMD_EXT:
            self.skipTest("pyemd not installed")

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

    def test_empty_query(self):
        index = self.factoryMethod()
        if isinstance(index, similarities.WmdSimilarity) and not PYEMD_EXT:
            self.skipTest("pyemd not installed")

        query = []
        try:
            sims = index[query]
            self.assertTrue(sims is not None)
        except IndexError:
            self.assertTrue(False)

    def test_chunking(self):
        if self.cls == similarities.Similarity:
            index = self.cls(None, CORPUS, num_features=len(DICTIONARY), shardsize=5)
        else:
            index = self.cls(CORPUS, num_features=len(DICTIONARY))
        query = CORPUS[:3]
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

    def test_iter(self):
        if self.cls == similarities.Similarity:
            index = self.cls(None, CORPUS, num_features=len(DICTIONARY), shardsize=5)
        else:
            index = self.cls(CORPUS, num_features=len(DICTIONARY))
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

    def test_persistency(self):
        if self.cls == similarities.WmdSimilarity and not PYEMD_EXT:
            self.skipTest("pyemd not installed")

        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index = self.factoryMethod()
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

    def test_persistency_compressed(self):
        if self.cls == similarities.WmdSimilarity and not PYEMD_EXT:
            self.skipTest("pyemd not installed")

        fname = get_tmpfile('gensim_similarities.tst.pkl.gz')
        index = self.factoryMethod()
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

    def test_large(self):
        if self.cls == similarities.WmdSimilarity and not PYEMD_EXT:
            self.skipTest("pyemd not installed")

        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index = self.factoryMethod()
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

    def test_large_compressed(self):
        if self.cls == similarities.WmdSimilarity and not PYEMD_EXT:
            self.skipTest("pyemd not installed")

        fname = get_tmpfile('gensim_similarities.tst.pkl.gz')
        index = self.factoryMethod()
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

    def test_mmap(self):
        if self.cls == similarities.WmdSimilarity and not PYEMD_EXT:
            self.skipTest("pyemd not installed")

        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index = self.factoryMethod()
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

    def test_mmap_compressed(self):
        if self.cls == similarities.WmdSimilarity and not PYEMD_EXT:
            self.skipTest("pyemd not installed")

        fname = get_tmpfile('gensim_similarities.tst.pkl.gz')
        index = self.factoryMethod()
        # store all arrays separately
        index.save(fname, sep_limit=0)

        # same thing, but use mmap to load arrays
        self.assertRaises(IOError, self.cls.load, fname, mmap='r')


class TestMatrixSimilarity(_TestSimilarityABC):
    def setUp(self):
        self.cls = similarities.MatrixSimilarity


class TestWmdSimilarity(_TestSimilarityABC):
    def setUp(self):
        self.cls = similarities.WmdSimilarity
        self.w2v_model = Word2Vec(TEXTS, min_count=1).wv

    def factoryMethod(self):
        # Override factoryMethod.
        return self.cls(TEXTS, self.w2v_model)

    @unittest.skipIf(PYEMD_EXT is False, "pyemd not installed")
    def test_full(self, num_best=None):
        # Override testFull.

        index = self.cls(TEXTS, self.w2v_model)
        index.num_best = num_best
        query = TEXTS[0]
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

    @unittest.skipIf(PYEMD_EXT is False, "pyemd not installed")
    def test_non_increasing(self):
        ''' Check that similarities are non-increasing when `num_best` is not
        `None`.'''
        # NOTE: this could be implemented for other similarities as well (i.e.
        # in _TestSimilarityABC).

        index = self.cls(TEXTS, self.w2v_model, num_best=3)
        query = TEXTS[0]
        sims = index[query]
        sims2 = numpy.asarray(sims)[:, 1]  # Just the similarities themselves.

        # The difference of adjacent elements should be negative.
        cond = sum(numpy.diff(sims2) < 0) == len(sims2) - 1
        self.assertTrue(cond)

    @unittest.skipIf(PYEMD_EXT is False, "pyemd not installed")
    def test_chunking(self):
        # Override testChunking.

        index = self.cls(TEXTS, self.w2v_model)
        query = TEXTS[:3]
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

    @unittest.skipIf(PYEMD_EXT is False, "pyemd not installed")
    def test_iter(self):
        # Override testIter.

        index = self.cls(TEXTS, self.w2v_model)
        for sims in index:
            self.assertTrue(numpy.alltrue(sims >= 0.0))
            self.assertTrue(numpy.alltrue(sims <= 1.0))

    @unittest.skipIf(PYEMD_EXT is False, "pyemd not installed")
    def test_str(self):
        index = self.cls(TEXTS, self.w2v_model)
        self.assertTrue(str(index))


class TestSoftCosineSimilarity(_TestSimilarityABC):
    def setUp(self):
        self.cls = similarities.SoftCosineSimilarity
        self.tfidf = TfidfModel(dictionary=DICTIONARY)
        similarity_matrix = scipy.sparse.identity(12, format="lil")
        similarity_matrix[DICTIONARY.token2id["user"], DICTIONARY.token2id["human"]] = 0.5
        similarity_matrix[DICTIONARY.token2id["human"], DICTIONARY.token2id["user"]] = 0.5
        self.similarity_matrix = SparseTermSimilarityMatrix(similarity_matrix)

    def factoryMethod(self):
        return self.cls(CORPUS, self.similarity_matrix)

    def test_full(self, num_best=None):
        # Single query
        index = self.cls(CORPUS, self.similarity_matrix, num_best=num_best)
        query = DICTIONARY.doc2bow(TEXTS[0])
        sims = index[query]
        if num_best is not None:
            # Sparse array.
            for i, sim in sims:
                self.assertTrue(numpy.alltrue(sim <= 1.0))
                self.assertTrue(numpy.alltrue(sim >= 0.0))
        else:
            self.assertAlmostEqual(1.0, sims[0])  # Similarity of a document with itself is 1.0.
            self.assertTrue(numpy.alltrue(sims[1:] >= 0.0))
            self.assertTrue(numpy.alltrue(sims[1:] < 1.0))

        # Corpora
        for query in (
                CORPUS,  # Basic text corpus.
                self.tfidf[CORPUS]):  # Transformed corpus without slicing support.
            index = self.cls(query, self.similarity_matrix, num_best=num_best)
            sims = index[query]
            if num_best is not None:
                # Sparse array.
                for result in sims:
                    for i, sim in result:
                        self.assertTrue(numpy.alltrue(sim <= 1.0))
                        self.assertTrue(numpy.alltrue(sim >= 0.0))
            else:
                for i, result in enumerate(sims):
                    self.assertAlmostEqual(1.0, result[i])  # Similarity of a document with itself is 1.0.
                    self.assertTrue(numpy.alltrue(result[:i] >= 0.0))
                    self.assertTrue(numpy.alltrue(result[:i] < 1.0))
                    self.assertTrue(numpy.alltrue(result[i + 1:] >= 0.0))
                    self.assertTrue(numpy.alltrue(result[i + 1:] < 1.0))

    def test_non_increasing(self):
        """ Check that similarities are non-increasing when `num_best` is not `None`."""
        # NOTE: this could be implemented for other similarities as well (i.e. in _TestSimilarityABC).

        index = self.cls(CORPUS, self.similarity_matrix, num_best=5)
        query = DICTIONARY.doc2bow(TEXTS[0])
        sims = index[query]
        sims2 = numpy.asarray(sims)[:, 1]  # Just the similarities themselves.

        # The difference of adjacent elements should be less than or equal to zero.
        cond = sum(numpy.diff(sims2) <= 0) == len(sims2) - 1
        self.assertTrue(cond)

    def test_chunking(self):
        index = self.cls(CORPUS, self.similarity_matrix)
        query = [DICTIONARY.doc2bow(document) for document in TEXTS[:3]]
        sims = index[query]

        for i in range(3):
            self.assertTrue(numpy.alltrue(sims[i, i] == 1.0))  # Similarity of a document with itself is 1.0.

        # test the same thing but with num_best
        index.num_best = 5
        sims = index[query]
        for i, chunk in enumerate(sims):
            expected = i
            self.assertAlmostEqual(expected, chunk[0][0], places=2)
            expected = 1.0
            self.assertAlmostEqual(expected, chunk[0][1], places=2)

    def test_iter(self):
        index = self.cls(CORPUS, self.similarity_matrix)
        for sims in index:
            self.assertTrue(numpy.alltrue(sims >= 0.0))
            self.assertTrue(numpy.alltrue(sims <= 1.0))


class TestSparseMatrixSimilarity(_TestSimilarityABC):
    def setUp(self):
        self.cls = similarities.SparseMatrixSimilarity

    def test_maintain_sparsity(self):
        """Sparsity is correctly maintained when maintain_sparsity=True"""
        num_features = len(DICTIONARY)

        index = self.cls(CORPUS, num_features=num_features)
        dense_sims = index[CORPUS]

        index = self.cls(CORPUS, num_features=num_features, maintain_sparsity=True)
        sparse_sims = index[CORPUS]

        self.assertFalse(scipy.sparse.issparse(dense_sims))
        self.assertTrue(scipy.sparse.issparse(sparse_sims))
        numpy.testing.assert_array_equal(dense_sims, sparse_sims.todense())

    def test_maintain_sparsity_with_num_best(self):
        """Tests that sparsity is correctly maintained when maintain_sparsity=True and num_best is not None"""
        num_features = len(DICTIONARY)

        index = self.cls(CORPUS, num_features=num_features, maintain_sparsity=False, num_best=3)
        dense_topn_sims = index[CORPUS]

        index = self.cls(CORPUS, num_features=num_features, maintain_sparsity=True, num_best=3)
        scipy_topn_sims = index[CORPUS]

        self.assertFalse(scipy.sparse.issparse(dense_topn_sims))
        self.assertTrue(scipy.sparse.issparse(scipy_topn_sims))
        self.assertEqual(dense_topn_sims, [matutils.scipy2sparse(v) for v in scipy_topn_sims])


class TestSimilarity(_TestSimilarityABC):
    def setUp(self):
        self.cls = similarities.Similarity

    def factoryMethod(self):
        # Override factoryMethod.
        return self.cls(None, CORPUS, num_features=len(DICTIONARY), shardsize=5)

    def test_sharding(self):
        for num_best in [None, 0, 1, 9, 1000]:
            for shardsize in [1, 2, 9, 1000]:
                self.testFull(num_best=num_best, shardsize=shardsize)

    def test_reopen(self):
        """test re-opening partially full shards"""
        index = similarities.Similarity(None, CORPUS[:5], num_features=len(DICTIONARY), shardsize=9)
        _ = index[CORPUS[0]]  # noqa:F841 forces shard close
        index.add_documents(CORPUS[5:])
        query = CORPUS[0]
        sims = index[query]
        expected = [(0, 0.99999994), (2, 0.28867513), (3, 0.23570226), (1, 0.23570226)]
        expected = matutils.sparse2full(expected, len(index))
        self.assertTrue(numpy.allclose(expected, sims))
        index.destroy()

    def test_mmap_compressed(self):
        pass
        # turns out this test doesn't exercise this because there are no arrays
        # to be mmaped!

    def test_chunksize(self):
        index = self.cls(None, CORPUS, num_features=len(DICTIONARY), shardsize=5)
        expected = [sim for sim in index]
        index.chunksize = len(index) - 1
        sims = [sim for sim in index]
        self.assertTrue(numpy.allclose(expected, sims))
        index.destroy()

    def test_nlargest(self):
        sims = ([(0, 0.8), (1, 0.2), (2, 0.0), (3, 0.0), (4, -0.1), (5, -0.15)],)
        expected = [(0, 0.8), (1, 0.2), (5, -0.15)]
        self.assertTrue(_nlargest(3, sims), expected)


class TestWord2VecAnnoyIndexer(unittest.TestCase):

    def setUp(self):
        try:
            import annoy  # noqa:F401
        except ImportError as e:
            raise unittest.SkipTest("Annoy library is not available: %s" % e)

        from gensim.similarities.annoy import AnnoyIndexer
        self.indexer = AnnoyIndexer

    def test_word2vec(self):
        model = word2vec.Word2Vec(TEXTS, min_count=1)
        index = self.indexer(model, 10)

        self.assertVectorIsSimilarToItself(model.wv, index)
        self.assertApproxNeighborsMatchExact(model.wv, model.wv, index)
        self.assertIndexSaved(index)
        self.assertLoadedIndexEqual(index, model)

    def test_fast_text(self):
        class LeeReader:
            def __init__(self, fn):
                self.fn = fn

            def __iter__(self):
                with utils.open(self.fn, 'r', encoding="latin_1") as infile:
                    for line in infile:
                        yield line.lower().strip().split()

        model = FastText(LeeReader(datapath('lee.cor')), bucket=5000)
        index = self.indexer(model, 10)

        self.assertVectorIsSimilarToItself(model.wv, index)
        self.assertApproxNeighborsMatchExact(model.wv, model.wv, index)
        self.assertIndexSaved(index)
        self.assertLoadedIndexEqual(index, model)

    def test_annoy_indexing_of_keyed_vectors(self):
        from gensim.similarities.annoy import AnnoyIndexer
        keyVectors_file = datapath('lee_fasttext.vec')
        model = KeyedVectors.load_word2vec_format(keyVectors_file)
        index = AnnoyIndexer(model, 10)

        self.assertEqual(index.num_trees, 10)
        self.assertVectorIsSimilarToItself(model, index)
        self.assertApproxNeighborsMatchExact(model, model, index)

    def test_load_missing_raises_error(self):
        from gensim.similarities.annoy import AnnoyIndexer
        test_index = AnnoyIndexer()

        self.assertRaises(IOError, test_index.load, fname='test-index')

    def assertVectorIsSimilarToItself(self, wv, index):
        vector = wv.get_normed_vectors()[0]
        label = wv.index_to_key[0]
        approx_neighbors = index.most_similar(vector, 1)
        word, similarity = approx_neighbors[0]

        self.assertEqual(word, label)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def assertApproxNeighborsMatchExact(self, model, wv, index):
        vector = wv.get_normed_vectors()[0]
        approx_neighbors = model.most_similar([vector], topn=5, indexer=index)
        exact_neighbors = model.most_similar(positive=[vector], topn=5)

        approx_words = [neighbor[0] for neighbor in approx_neighbors]
        exact_words = [neighbor[0] for neighbor in exact_neighbors]

        self.assertEqual(approx_words, exact_words)

    def assertAllSimilaritiesDisableIndexer(self, model, wv, index):
        vector = wv.get_normed_vectors()[0]
        approx_similarities = model.most_similar([vector], topn=None, indexer=index)
        exact_similarities = model.most_similar(positive=[vector], topn=None)

        self.assertEqual(approx_similarities, exact_similarities)
        self.assertEqual(len(approx_similarities), len(wv.vectors))

    def assertIndexSaved(self, index):
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index.save(fname)
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.exists(fname + '.d'))

    def assertLoadedIndexEqual(self, index, model):
        from gensim.similarities.annoy import AnnoyIndexer

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
        except ImportError as e:
            raise unittest.SkipTest("Annoy library is not available: %s" % e)

        from gensim.similarities.annoy import AnnoyIndexer

        self.model = doc2vec.Doc2Vec(SENTENCES, min_count=1)
        self.index = AnnoyIndexer(self.model, 300)
        self.vector = self.model.dv.get_normed_vectors()[0]

    def test_document_is_similar_to_itself(self):
        approx_neighbors = self.index.most_similar(self.vector, 1)
        doc, similarity = approx_neighbors[0]

        self.assertEqual(doc, 0)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def test_approx_neighbors_match_exact(self):
        approx_neighbors = self.model.dv.most_similar([self.vector], topn=5, indexer=self.index)
        exact_neighbors = self.model.dv.most_similar([self.vector], topn=5)

        approx_words = [neighbor[0] for neighbor in approx_neighbors]
        exact_words = [neighbor[0] for neighbor in exact_neighbors]

        self.assertEqual(approx_words, exact_words)

    def test_save(self):
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        self.index.save(fname)
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.exists(fname + '.d'))

    def test_load_not_exist(self):
        from gensim.similarities.annoy import AnnoyIndexer
        self.test_index = AnnoyIndexer()

        self.assertRaises(IOError, self.test_index.load, fname='test-index')

    def test_save_load(self):
        from gensim.similarities.annoy import AnnoyIndexer

        fname = get_tmpfile('gensim_similarities.tst.pkl')
        self.index.save(fname)

        self.index2 = AnnoyIndexer()
        self.index2.load(fname)
        self.index2.model = self.model

        self.assertEqual(self.index.index.f, self.index2.index.f)
        self.assertEqual(self.index.labels, self.index2.labels)
        self.assertEqual(self.index.num_trees, self.index2.num_trees)


class TestWord2VecNmslibIndexer(unittest.TestCase):

    def setUp(self):
        try:
            import nmslib  # noqa:F401
        except ImportError as e:
            raise unittest.SkipTest("NMSLIB library is not available: %s" % e)

        from gensim.similarities.nmslib import NmslibIndexer
        self.indexer = NmslibIndexer

    def test_word2vec(self):
        model = word2vec.Word2Vec(TEXTS, min_count=1)
        index = self.indexer(model)

        self.assertVectorIsSimilarToItself(model.wv, index)
        self.assertApproxNeighborsMatchExact(model.wv, model.wv, index)
        self.assertIndexSaved(index)
        self.assertLoadedIndexEqual(index, model)

    def test_fasttext(self):
        class LeeReader:
            def __init__(self, fn):
                self.fn = fn

            def __iter__(self):
                with utils.open(self.fn, 'r', encoding="latin_1") as infile:
                    for line in infile:
                        yield line.lower().strip().split()

        model = FastText(LeeReader(datapath('lee.cor')), bucket=5000)
        index = self.indexer(model)

        self.assertVectorIsSimilarToItself(model.wv, index)
        self.assertApproxNeighborsMatchExact(model.wv, model.wv, index)
        self.assertIndexSaved(index)
        self.assertLoadedIndexEqual(index, model)

    def test_indexing_keyedvectors(self):
        from gensim.similarities.nmslib import NmslibIndexer
        keyVectors_file = datapath('lee_fasttext.vec')
        model = KeyedVectors.load_word2vec_format(keyVectors_file)
        index = NmslibIndexer(model)

        self.assertVectorIsSimilarToItself(model, index)
        self.assertApproxNeighborsMatchExact(model, model, index)

    def test_load_missing_raises_error(self):
        from gensim.similarities.nmslib import NmslibIndexer

        self.assertRaises(IOError, NmslibIndexer.load, fname='test-index')

    def assertVectorIsSimilarToItself(self, wv, index):
        vector = wv.get_normed_vectors()[0]
        label = wv.index_to_key[0]
        approx_neighbors = index.most_similar(vector, 1)
        word, similarity = approx_neighbors[0]

        self.assertEqual(word, label)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def assertApproxNeighborsMatchExact(self, model, wv, index):
        vector = wv.get_normed_vectors()[0]
        approx_neighbors = model.most_similar([vector], topn=5, indexer=index)
        exact_neighbors = model.most_similar([vector], topn=5)

        approx_words = [word_id for word_id, similarity in approx_neighbors]
        exact_words = [word_id for word_id, similarity in exact_neighbors]

        self.assertEqual(approx_words, exact_words)

    def assertIndexSaved(self, index):
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index.save(fname)
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.exists(fname + '.d'))

    def assertLoadedIndexEqual(self, index, model):
        from gensim.similarities.nmslib import NmslibIndexer

        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index.save(fname)

        index2 = NmslibIndexer.load(fname)
        index2.model = model

        self.assertEqual(index.labels, index2.labels)
        self.assertEqual(index.index_params, index2.index_params)
        self.assertEqual(index.query_time_params, index2.query_time_params)


class TestDoc2VecNmslibIndexer(unittest.TestCase):

    def setUp(self):
        try:
            import nmslib  # noqa:F401
        except ImportError as e:
            raise unittest.SkipTest("NMSLIB library is not available: %s" % e)

        from gensim.similarities.nmslib import NmslibIndexer

        self.model = doc2vec.Doc2Vec(SENTENCES, min_count=1)
        self.index = NmslibIndexer(self.model)
        self.vector = self.model.dv.get_normed_vectors()[0]

    def test_document_is_similar_to_itself(self):
        approx_neighbors = self.index.most_similar(self.vector, 1)
        doc, similarity = approx_neighbors[0]

        self.assertEqual(doc, 0)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def test_approx_neighbors_match_exact(self):
        approx_neighbors = self.model.dv.most_similar([self.vector], topn=5, indexer=self.index)
        exact_neighbors = self.model.dv.most_similar([self.vector], topn=5)

        approx_tags = [tag for tag, similarity in approx_neighbors]
        exact_tags = [tag for tag, similarity in exact_neighbors]

        self.assertEqual(approx_tags, exact_tags)

    def test_save(self):
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        self.index.save(fname)
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.exists(fname + '.d'))

    def test_load_not_exist(self):
        from gensim.similarities.nmslib import NmslibIndexer

        self.assertRaises(IOError, NmslibIndexer.load, fname='test-index')

    def test_save_load(self):
        from gensim.similarities.nmslib import NmslibIndexer

        fname = get_tmpfile('gensim_similarities.tst.pkl')
        self.index.save(fname)

        self.index2 = NmslibIndexer.load(fname)
        self.index2.model = self.model

        self.assertEqual(self.index.labels, self.index2.labels)
        self.assertEqual(self.index.index_params, self.index2.index_params)
        self.assertEqual(self.index.query_time_params, self.index2.query_time_params)


class TestUniformTermSimilarityIndex(unittest.TestCase):
    def setUp(self):
        self.documents = [[u"government", u"denied", u"holiday"], [u"holiday", u"slowing", u"hollingworth"]]
        self.dictionary = Dictionary(self.documents)

    def test_most_similar(self):
        """Test most_similar returns expected results."""

        # check that the topn works as expected
        index = UniformTermSimilarityIndex(self.dictionary)
        results = list(index.most_similar(u"holiday", topn=1))
        self.assertLess(0, len(results))
        self.assertGreaterEqual(1, len(results))
        results = list(index.most_similar(u"holiday", topn=4))
        self.assertLess(1, len(results))
        self.assertGreaterEqual(4, len(results))

        # check that the term itself is not returned
        index = UniformTermSimilarityIndex(self.dictionary)
        terms = [term for term, similarity in index.most_similar(u"holiday", topn=len(self.dictionary))]
        self.assertFalse(u"holiday" in terms)

        # check that the term_similarity works as expected
        index = UniformTermSimilarityIndex(self.dictionary, term_similarity=0.2)
        similarities = numpy.array([
            similarity for term, similarity in index.most_similar(u"holiday", topn=len(self.dictionary))])
        self.assertTrue(numpy.all(similarities == 0.2))


class TestSparseTermSimilarityMatrix(unittest.TestCase):
    def setUp(self):
        self.documents = [
            [u"government", u"denied", u"holiday"],
            [u"government", u"denied", u"holiday", u"slowing", u"hollingworth"]]
        self.dictionary = Dictionary(self.documents)
        self.tfidf = TfidfModel(dictionary=self.dictionary)
        zero_index = UniformTermSimilarityIndex(self.dictionary, term_similarity=0.0)
        self.index = UniformTermSimilarityIndex(self.dictionary, term_similarity=0.5)
        self.identity_matrix = SparseTermSimilarityMatrix(zero_index, self.dictionary)
        self.uniform_matrix = SparseTermSimilarityMatrix(self.index, self.dictionary)
        self.vec1 = self.dictionary.doc2bow([u"government", u"government", u"denied"])
        self.vec2 = self.dictionary.doc2bow([u"government", u"holiday"])

    def test_empty_dictionary(self):
        with self.assertRaises(ValueError):
            SparseTermSimilarityMatrix(self.index, [])

    def test_type(self):
        """Test the type of the produced matrix."""
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary).matrix
        self.assertTrue(isinstance(matrix, scipy.sparse.csc_matrix))

    def test_diagonal(self):
        """Test the existence of ones on the main diagonal."""
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary).matrix.todense()
        self.assertTrue(numpy.all(numpy.diag(matrix) == numpy.ones(matrix.shape[0])))

    def test_order(self):
        """Test the matrix order."""
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary).matrix.todense()
        self.assertEqual(matrix.shape[0], len(self.dictionary))
        self.assertEqual(matrix.shape[1], len(self.dictionary))

    def test_dtype(self):
        """Test the dtype parameter of the matrix constructor."""
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, dtype=numpy.float32).matrix.todense()
        self.assertEqual(numpy.float32, matrix.dtype)

        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, dtype=numpy.float64).matrix.todense()
        self.assertEqual(numpy.float64, matrix.dtype)

    def test_nonzero_limit(self):
        """Test the nonzero_limit parameter of the matrix constructor."""
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=100).matrix.todense()
        self.assertGreaterEqual(101, numpy.max(numpy.sum(matrix != 0, axis=0)))

        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=4).matrix.todense()
        self.assertGreaterEqual(5, numpy.max(numpy.sum(matrix != 0, axis=0)))

        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=1).matrix.todense()
        self.assertGreaterEqual(2, numpy.max(numpy.sum(matrix != 0, axis=0)))

        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary, nonzero_limit=0).matrix.todense()
        self.assertEqual(1, numpy.max(numpy.sum(matrix != 0, axis=0)))
        self.assertTrue(numpy.all(matrix == numpy.eye(matrix.shape[0])))

    def test_symmetric(self):
        """Test the symmetric parameter of the matrix constructor."""
        matrix = SparseTermSimilarityMatrix(self.index, self.dictionary).matrix.todense()
        self.assertTrue(numpy.all(matrix == matrix.T))

        matrix = SparseTermSimilarityMatrix(
            self.index, self.dictionary, nonzero_limit=1).matrix.todense()
        expected_matrix = numpy.array([
            [1.0, 0.5, 0.0, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(numpy.all(expected_matrix == matrix))

        matrix = SparseTermSimilarityMatrix(
            self.index, self.dictionary, nonzero_limit=1, symmetric=False).matrix.todense()
        expected_matrix = numpy.array([
            [1.0, 0.5, 0.5, 0.5, 0.5],
            [0.5, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(numpy.all(expected_matrix == matrix))

    def test_dominant(self):
        """Test the dominant parameter of the matrix constructor."""
        negative_index = UniformTermSimilarityIndex(self.dictionary, term_similarity=-0.5)
        matrix = SparseTermSimilarityMatrix(
            negative_index, self.dictionary, nonzero_limit=2).matrix.todense()
        expected_matrix = numpy.array([
            [1.0, -.5, -.5, 0.0, 0.0],
            [-.5, 1.0, 0.0, -.5, 0.0],
            [-.5, 0.0, 1.0, 0.0, 0.0],
            [0.0, -.5, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(numpy.all(expected_matrix == matrix))

        matrix = SparseTermSimilarityMatrix(
            negative_index, self.dictionary, nonzero_limit=2, dominant=True).matrix.todense()
        expected_matrix = numpy.array([
            [1.0, -.5, 0.0, 0.0, 0.0],
            [-.5, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(numpy.all(expected_matrix == matrix))

    def test_tfidf(self):
        """Test the tfidf parameter of the matrix constructor."""
        matrix = SparseTermSimilarityMatrix(
            self.index, self.dictionary, nonzero_limit=1).matrix.todense()
        expected_matrix = numpy.array([
            [1.0, 0.5, 0.0, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(numpy.all(expected_matrix == matrix))

        matrix = SparseTermSimilarityMatrix(
            self.index, self.dictionary, nonzero_limit=1, tfidf=self.tfidf).matrix.todense()
        expected_matrix = numpy.array([
            [1.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(numpy.all(expected_matrix == matrix))

    def test_encapsulation(self):
        """Test the matrix encapsulation."""

        # check that a sparse matrix will be converted to a CSC format
        expected_matrix = numpy.array([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [0.0, 0.0, 1.0]])

        matrix = SparseTermSimilarityMatrix(scipy.sparse.csc_matrix(expected_matrix)).matrix
        self.assertTrue(isinstance(matrix, scipy.sparse.csc_matrix))
        self.assertTrue(numpy.all(matrix.todense() == expected_matrix))

        matrix = SparseTermSimilarityMatrix(scipy.sparse.csr_matrix(expected_matrix)).matrix
        self.assertTrue(isinstance(matrix, scipy.sparse.csc_matrix))
        self.assertTrue(numpy.all(matrix.todense() == expected_matrix))

    def test_inner_product_zerovector_zerovector_default(self):
        """Test the inner product between two zero vectors with the default normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], []))

    def test_inner_product_zerovector_zerovector_false_maintain(self):
        """Test the inner product between two zero vectors with the (False, 'maintain') normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=(False, 'maintain')))

    def test_inner_product_zerovector_zerovector_false_true(self):
        """Test the inner product between two zero vectors with the (False, True) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=(False, True)))

    def test_inner_product_zerovector_zerovector_maintain_false(self):
        """Test the inner product between two zero vectors with the ('maintain', False) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=('maintain', False)))

    def test_inner_product_zerovector_zerovector_maintain_maintain(self):
        """Test the inner product between two zero vectors with the ('maintain', 'maintain') normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=('maintain', 'maintain')))

    def test_inner_product_zerovector_zerovector_maintain_true(self):
        """Test the inner product between two zero vectors with the ('maintain', True) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=('maintain', True)))

    def test_inner_product_zerovector_zerovector_true_false(self):
        """Test the inner product between two zero vectors with the (True, False) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=(True, False)))

    def test_inner_product_zerovector_zerovector_true_maintain(self):
        """Test the inner product between two zero vectors with the (True, 'maintain') normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=(True, 'maintain')))

    def test_inner_product_zerovector_zerovector_true_true(self):
        """Test the inner product between two zero vectors with the (True, True) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], [], normalized=(True, True)))

    def test_inner_product_zerovector_vector_default(self):
        """Test the inner product between a zero vector and a vector with the default normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2))

    def test_inner_product_zerovector_vector_false_maintain(self):
        """Test the inner product between a zero vector and a vector with the (False, 'maintain') normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=(False, 'maintain')))

    def test_inner_product_zerovector_vector_false_true(self):
        """Test the inner product between a zero vector and a vector with the (False, True) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=(False, True)))

    def test_inner_product_zerovector_vector_maintain_false(self):
        """Test the inner product between a zero vector and a vector with the ('maintain', False) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=('maintain', False)))

    def test_inner_product_zerovector_vector_maintain_maintain(self):
        """Test the inner product between a zero vector and a vector with the ('maintain', 'maintain') normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=('maintain', 'maintain')))

    def test_inner_product_zerovector_vector_maintain_true(self):
        """Test the inner product between a zero vector and a vector with the ('maintain', True) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=('maintain', True)))

    def test_inner_product_zerovector_vector_true_false(self):
        """Test the inner product between a zero vector and a vector with the (True, False) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=(True, False)))

    def test_inner_product_zerovector_vector_true_maintain(self):
        """Test the inner product between a zero vector and a vector with the (True, 'maintain') normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=(True, 'maintain')))

    def test_inner_product_zerovector_vector_true_true(self):
        """Test the inner product between a zero vector and a vector with the (True, True) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product([], self.vec2, normalized=(True, True)))

    def test_inner_product_vector_zerovector_default(self):
        """Test the inner product between a vector and a zero vector with the default normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, []))

    def test_inner_product_vector_zerovector_false_maintain(self):
        """Test the inner product between a vector and a zero vector with the (False, 'maintain') normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=(False, 'maintain')))

    def test_inner_product_vector_zerovector_false_true(self):
        """Test the inner product between a vector and a zero vector with the (False, True) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=(False, True)))

    def test_inner_product_vector_zerovector_maintain_false(self):
        """Test the inner product between a vector and a zero vector with the ('maintain', False) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=('maintain', False)))

    def test_inner_product_vector_zerovector_maintain_maintain(self):
        """Test the inner product between a vector and a zero vector with the ('maintain', 'maintain') normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=('maintain', 'maintain')))

    def test_inner_product_vector_zerovector_maintain_true(self):
        """Test the inner product between a vector and a zero vector with the ('maintain', True) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=('maintain', True)))

    def test_inner_product_vector_zerovector_true_false(self):
        """Test the inner product between a vector and a zero vector with the (True, False) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=(True, False)))

    def test_inner_product_vector_zerovector_true_maintain(self):
        """Test the inner product between a vector and a zero vector with the (True, 'maintain') normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=(True, 'maintain')))

    def test_inner_product_vector_zerovector_true_true(self):
        """Test the inner product between a vector and a zero vector with the (True, True) normalization."""

        self.assertEqual(0.0, self.uniform_matrix.inner_product(self.vec1, [], normalized=(True, True)))

    def test_inner_product_vector_vector_default(self):
        """Test the inner product between two vectors with the default normalization."""

        expected_result = 0.0
        expected_result += 2 * 1.0 * 1  # government * s_{ij} * government
        expected_result += 2 * 0.5 * 1  # government * s_{ij} * holiday
        expected_result += 1 * 0.5 * 1  # denied * s_{ij} * government
        expected_result += 1 * 0.5 * 1  # denied * s_{ij} * holiday
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_false_maintain(self):
        """Test the inner product between two vectors with the (False, 'maintain') normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=(False, 'maintain'))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_false_true(self):
        """Test the inner product between two vectors with the (False, True) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=(False, True))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_maintain_false(self):
        """Test the inner product between two vectors with the ('maintain', False) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=('maintain', False))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_maintain_maintain(self):
        """Test the inner product between two vectors with the ('maintain', 'maintain') normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=('maintain', 'maintain'))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_maintain_true(self):
        """Test the inner product between two vectors with the ('maintain', True) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=('maintain', True))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_true_false(self):
        """Test the inner product between two vectors with the (True, False) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=(True, False))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_true_maintain(self):
        """Test the inner product between two vectors with the (True, 'maintain') normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=(True, 'maintain'))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_vector_true_true(self):
        """Test the inner product between two vectors with the (True, True) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        result = self.uniform_matrix.inner_product(self.vec1, self.vec2, normalized=(True, True))
        self.assertAlmostEqual(expected_result, result, places=5)

    def test_inner_product_vector_corpus_default(self):
        """Test the inner product between a vector and a corpus with the default normalization."""

        expected_result = 0.0
        expected_result += 2 * 1.0 * 1  # government * s_{ij} * government
        expected_result += 2 * 0.5 * 1  # government * s_{ij} * holiday
        expected_result += 1 * 0.5 * 1  # denied * s_{ij} * government
        expected_result += 1 * 0.5 * 1  # denied * s_{ij} * holiday
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2)
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_false_maintain(self):
        """Test the inner product between a vector and a corpus with the (False, 'maintain') normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=(False, 'maintain'))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_false_true(self):
        """Test the inner product between a vector and a corpus with the (False, True) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=(False, True))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_maintain_false(self):
        """Test the inner product between a vector and a corpus with the ('maintain', False) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=('maintain', False))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_maintain_maintain(self):
        """Test the inner product between a vector and a corpus with the ('maintain', 'maintain') normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=('maintain', 'maintain'))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_maintain_true(self):
        """Test the inner product between a vector and a corpus with the ('maintain', True) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=('maintain', True))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_true_false(self):
        """Test the inner product between a vector and a corpus with the (True, False) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=(True, False))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_true_maintain(self):
        """Test the inner product between a vector and a corpus with the (True, 'maintain') normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=(True, 'maintain'))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_vector_corpus_true_true(self):
        """Test the inner product between a vector and a corpus with the (True, True) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((1, 2), expected_result)
        result = self.uniform_matrix.inner_product(self.vec1, [self.vec2] * 2, normalized=(True, True))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_default(self):
        """Test the inner product between a corpus and a vector with the default normalization."""

        expected_result = 0.0
        expected_result += 2 * 1.0 * 1  # government * s_{ij} * government
        expected_result += 2 * 0.5 * 1  # government * s_{ij} * holiday
        expected_result += 1 * 0.5 * 1  # denied * s_{ij} * government
        expected_result += 1 * 0.5 * 1  # denied * s_{ij} * holiday
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2)
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_false_maintain(self):
        """Test the inner product between a corpus and a vector with the (False, 'maintain') normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=(False, 'maintain'))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_false_true(self):
        """Test the inner product between a corpus and a vector with the (False, True) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=(False, True))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_maintain_false(self):
        """Test the inner product between a corpus and a vector with the ('maintain', False) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=('maintain', False))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_maintain_maintain(self):
        """Test the inner product between a corpus and a vector with the ('maintain', 'maintain') normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=('maintain', 'maintain'))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_maintain_true(self):
        """Test the inner product between a corpus and a vector with the ('maintain', True) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=('maintain', True))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_true_false(self):
        """Test the inner product between a corpus and a vector with the (True, False) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=(True, False))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_true_maintain(self):
        """Test the inner product between a corpus and a vector with the (True, 'maintain') normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=(True, 'maintain'))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_vector_true_true(self):
        """Test the inner product between a corpus and a vector with the (True, True) normalization."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 1), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, self.vec2, normalized=(True, True))
        self.assertTrue(isinstance(result, numpy.ndarray))
        self.assertTrue(numpy.allclose(expected_result, result))

    def test_inner_product_corpus_corpus_default(self):
        """Test the inner product between two corpora with the default normalization."""

        expected_result = 0.0
        expected_result += 2 * 1.0 * 1  # government * s_{ij} * government
        expected_result += 2 * 0.5 * 1  # government * s_{ij} * holiday
        expected_result += 1 * 0.5 * 1  # denied * s_{ij} * government
        expected_result += 1 * 0.5 * 1  # denied * s_{ij} * holiday
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2)
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_false_maintain(self):
        """Test the inner product between two corpora with the (False, 'maintain')."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=(False, 'maintain'))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_false_true(self):
        """Test the inner product between two corpora with the (False, True)."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=(False, True))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_maintain_false(self):
        """Test the inner product between two corpora with the ('maintain', False)."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=('maintain', False))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_maintain_maintain(self):
        """Test the inner product between two corpora with the ('maintain', 'maintain')."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2,
            normalized=('maintain', 'maintain'))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_maintain_true(self):
        """Test the inner product between two corpora with the ('maintain', True)."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=('maintain', True))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_true_false(self):
        """Test the inner product between two corpora with the (True, False)."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=(True, False))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_true_maintain(self):
        """Test the inner product between two corpora with the (True, 'maintain')."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result *= math.sqrt(self.identity_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=(True, 'maintain'))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))

    def test_inner_product_corpus_corpus_true_true(self):
        """Test the inner product between two corpora with the (True, True)."""

        expected_result = self.uniform_matrix.inner_product(self.vec1, self.vec2)
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec1, self.vec1))
        expected_result /= math.sqrt(self.uniform_matrix.inner_product(self.vec2, self.vec2))
        expected_result = numpy.full((3, 2), expected_result)
        result = self.uniform_matrix.inner_product([self.vec1] * 3, [self.vec2] * 2, normalized=(True, True))
        self.assertTrue(isinstance(result, scipy.sparse.csr_matrix))
        self.assertTrue(numpy.allclose(expected_result, result.todense()))


class TestLevenshteinSimilarityIndex(unittest.TestCase):
    def setUp(self):
        self.documents = [[u"government", u"denied", u"holiday"], [u"holiday", u"slowing", u"hollingworth"]]
        self.dictionary = Dictionary(self.documents)
        max_distance = max(len(term) for term in self.dictionary.values())
        self.index = LevenshteinSimilarityIndex(self.dictionary, max_distance=max_distance)

    def test_most_similar_topn(self):
        """Test most_similar returns expected results."""
        results = list(self.index.most_similar(u"holiday", topn=0))
        self.assertEqual(0, len(results))

        results = list(self.index.most_similar(u"holiday", topn=1))
        self.assertEqual(1, len(results))

        results = list(self.index.most_similar(u"holiday", topn=4))
        self.assertEqual(4, len(results))

        results = list(self.index.most_similar(u"holiday", topn=len(self.dictionary)))
        self.assertEqual(len(self.dictionary) - 1, len(results))
        self.assertNotIn(u"holiday", results)

    def test_most_similar_result_order(self):
        results = self.index.most_similar(u"holiday", topn=4)
        terms, _ = zip(*results)
        expected_terms = (u"hollingworth", u"denied", u"slowing", u"government")
        self.assertEqual(expected_terms, terms)

    def test_most_similar_alpha(self):
        index = LevenshteinSimilarityIndex(self.dictionary, alpha=1.0)
        first_similarities = numpy.array([similarity for term, similarity in index.most_similar(u"holiday", topn=10)])
        index = LevenshteinSimilarityIndex(self.dictionary, alpha=2.0)
        second_similarities = numpy.array([similarity for term, similarity in index.most_similar(u"holiday", topn=10)])
        self.assertTrue(numpy.allclose(2.0 * first_similarities, second_similarities))

    def test_most_similar_beta(self):
        index = LevenshteinSimilarityIndex(self.dictionary, alpha=1.0, beta=1.0)
        first_similarities = numpy.array([similarity for term, similarity in index.most_similar(u"holiday", topn=10)])
        index = LevenshteinSimilarityIndex(self.dictionary, alpha=1.0, beta=2.0)
        second_similarities = numpy.array([similarity for term, similarity in index.most_similar(u"holiday", topn=10)])
        self.assertTrue(numpy.allclose(first_similarities ** 2.0, second_similarities))


class TestWordEmbeddingSimilarityIndex(unittest.TestCase):
    def setUp(self):
        self.vectors = KeyedVectors.load_word2vec_format(
            datapath('euclidean_vectors.bin'), binary=True, datatype=numpy.float64)

    def test_most_similar(self):
        """Test most_similar returns expected results."""

        # check the handling of out-of-dictionary terms
        index = WordEmbeddingSimilarityIndex(self.vectors)
        self.assertLess(0, len(list(index.most_similar(u"holiday", topn=10))))
        self.assertEqual(0, len(list(index.most_similar(u"out-of-dictionary term", topn=10))))

        # check that the topn works as expected
        index = WordEmbeddingSimilarityIndex(self.vectors)
        results = list(index.most_similar(u"holiday", topn=10))
        self.assertLess(0, len(results))
        self.assertGreaterEqual(10, len(results))
        results = list(index.most_similar(u"holiday", topn=20))
        self.assertLess(10, len(results))
        self.assertGreaterEqual(20, len(results))

        # check that the term itself is not returned
        index = WordEmbeddingSimilarityIndex(self.vectors)
        terms = [term for term, similarity in index.most_similar(u"holiday", topn=len(self.vectors))]
        self.assertFalse(u"holiday" in terms)

        # check that the threshold works as expected
        index = WordEmbeddingSimilarityIndex(self.vectors, threshold=0.0)
        results = list(index.most_similar(u"holiday", topn=10))
        self.assertLess(0, len(results))
        self.assertGreaterEqual(10, len(results))

        index = WordEmbeddingSimilarityIndex(self.vectors, threshold=1.0)
        results = list(index.most_similar(u"holiday", topn=10))
        self.assertEqual(0, len(results))

        # check that the exponent works as expected
        index = WordEmbeddingSimilarityIndex(self.vectors, exponent=1.0)
        first_similarities = numpy.array([similarity for term, similarity in index.most_similar(u"holiday", topn=10)])
        index = WordEmbeddingSimilarityIndex(self.vectors, exponent=2.0)
        second_similarities = numpy.array([similarity for term, similarity in index.most_similar(u"holiday", topn=10)])
        self.assertTrue(numpy.allclose(first_similarities**2.0, second_similarities))


class TestFastSS(unittest.TestCase):
    def test_editdist_same_unicode_kind_latin1(self):
        """Test editdist returns the expected result with two Latin-1 strings."""
        expected = 2
        actual = editdist('Zizka', 'siska')
        assert expected == actual

    def test_editdist_same_unicode_kind_ucs2(self):
        """Test editdist returns the expected result with two UCS-2 strings."""
        expected = 2
        actual = editdist('Žižka', 'šiška')
        assert expected == actual

    def test_editdist_same_unicode_kind_ucs4(self):
        """Test editdist returns the expected result with two UCS-4 strings."""
        expected = 2
        actual = editdist('Žižka 😀', 'šiška 😀')
        assert expected == actual

    def test_editdist_different_unicode_kinds(self):
        """Test editdist returns the expected result with strings of different Unicode kinds."""
        expected = 2
        actual = editdist('Žižka', 'siska')
        assert expected == actual


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
