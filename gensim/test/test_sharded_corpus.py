"""
Tests for ShardedCorpus.
"""

import os
import unittest
import random
import shutil

import numpy as np
from scipy import sparse

from gensim.utils import is_corpus, mock_data
from gensim.corpora.sharded_corpus import ShardedCorpus

#############################################################################


class TestShardedCorpus(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls):
    #     cls.dim = 1000
    #     cls.data = mock_data(dim=cls.dim)
    #
    #     random_string = ''.join(random.choice('1234567890') for _ in range(8))
    #
    #     cls.tmp_dir = 'test-temp-' + random_string
    #     os.makedirs(cls.tmp_dir)
    #
    #     cls.tmp_fname = os.path.join(cls.tmp_dir,
    #                                  'shcorp.' + random_string + '.tmp')

    # @classmethod
    # def tearDownClass(cls):
    #     shutil.rmtree(cls.tmp_dir)

    def setUp(self):
        self.dim = 1000
        self.random_string = ''.join(random.choice('1234567890') for _ in range(8))
        self.tmp_dir = 'test-temp-' + self.random_string
        os.makedirs(self.tmp_dir)

        self.tmp_fname = os.path.join(self.tmp_dir,
                                      'shcorp.' + self.random_string + '.tmp')
        self.data = mock_data(dim=1000)
        self.corpus = ShardedCorpus(self.tmp_fname, self.data, dim=self.dim,
                                    shardsize=100)

    def tearDown(self):

        shutil.rmtree(self.tmp_dir)

    def test_init(self):

        # Test that the shards were actually created during setUp
        self.assertTrue(os.path.isfile(self.tmp_fname + '.1'))

    def test_load(self):

        # Test that the shards were actually created
        self.assertTrue(os.path.isfile(self.tmp_fname + '.1'))

        self.corpus.save()
        loaded_corpus = ShardedCorpus.load(self.tmp_fname)

        self.assertEqual(loaded_corpus.dim, self.corpus.dim)
        self.assertEqual(loaded_corpus.n_shards, self.corpus.n_shards)

    def test_getitem(self):

        _ = self.corpus[130]  # noqa:F841
        # Does retrieving the item load the correct shard?
        self.assertEqual(self.corpus.current_shard_n, 1)

        item = self.corpus[220:227]

        self.assertEqual((7, self.corpus.dim), item.shape)
        self.assertEqual(self.corpus.current_shard_n, 2)

        for i in range(220, 227):
            self.assertTrue(np.array_equal(self.corpus[i], item[i - 220]))

    def test_sparse_serialization(self):

        no_exception = True
        try:
            ShardedCorpus(self.tmp_fname, self.data, shardsize=100, dim=self.dim, sparse_serialization=True)
        except Exception:
            no_exception = False
            raise
        finally:
            self.assertTrue(no_exception)

    def test_getitem_dense2dense(self):

        corpus = ShardedCorpus(
            self.tmp_fname, self.data, shardsize=100, dim=self.dim,
            sparse_serialization=False, sparse_retrieval=False
        )

        item = corpus[3]
        self.assertTrue(isinstance(item, np.ndarray))
        self.assertEqual(item.shape, (corpus.dim,))

        dslice = corpus[2:6]
        self.assertTrue(isinstance(dslice, np.ndarray))
        self.assertEqual(dslice.shape, (4, corpus.dim))

        ilist = corpus[[2, 3, 4, 5]]
        self.assertTrue(isinstance(ilist, np.ndarray))
        self.assertEqual(ilist.shape, (4, corpus.dim))

        self.assertEqual(ilist.all(), dslice.all())

    def test_getitem_dense2sparse(self):

        corpus = ShardedCorpus(
            self.tmp_fname, self.data, shardsize=100, dim=self.dim,
            sparse_serialization=False, sparse_retrieval=True
        )

        item = corpus[3]
        self.assertTrue(isinstance(item, sparse.csr_matrix))
        self.assertEqual(item.shape, (1, corpus.dim))

        dslice = corpus[2:6]
        self.assertTrue(isinstance(dslice, sparse.csr_matrix))
        self.assertEqual(dslice.shape, (4, corpus.dim))

        ilist = corpus[[2, 3, 4, 5]]
        self.assertTrue(isinstance(ilist, sparse.csr_matrix))
        self.assertEqual(ilist.shape, (4, corpus.dim))

        self.assertEqual((ilist != dslice).getnnz(), 0)

    def test_getitem_sparse2sparse(self):

        sp_tmp_fname = self.tmp_fname + '.sparse'
        corpus = ShardedCorpus(
            sp_tmp_fname, self.data, shardsize=100, dim=self.dim,
            sparse_serialization=True, sparse_retrieval=True
        )

        dense_corpus = ShardedCorpus(
            self.tmp_fname, self.data, shardsize=100, dim=self.dim,
            sparse_serialization=False, sparse_retrieval=True
        )

        item = corpus[3]
        self.assertTrue(isinstance(item, sparse.csr_matrix))
        self.assertEqual(item.shape, (1, corpus.dim))

        dslice = corpus[2:6]
        self.assertTrue(isinstance(dslice, sparse.csr_matrix))
        self.assertEqual(dslice.shape, (4, corpus.dim))

        expected_nnz = sum(len(self.data[i]) for i in range(2, 6))
        self.assertEqual(dslice.getnnz(), expected_nnz)

        ilist = corpus[[2, 3, 4, 5]]
        self.assertTrue(isinstance(ilist, sparse.csr_matrix))
        self.assertEqual(ilist.shape, (4, corpus.dim))

        # Also compare with what the dense dataset is giving us
        d_dslice = dense_corpus[2:6]
        self.assertEqual((d_dslice != dslice).getnnz(), 0)
        self.assertEqual((ilist != dslice).getnnz(), 0)

    def test_getitem_sparse2dense(self):
        sp_tmp_fname = self.tmp_fname + '.sparse'
        corpus = ShardedCorpus(
            sp_tmp_fname, self.data, shardsize=100, dim=self.dim,
            sparse_serialization=True, sparse_retrieval=False
        )

        dense_corpus = ShardedCorpus(
            self.tmp_fname, self.data, shardsize=100, dim=self.dim,
            sparse_serialization=False, sparse_retrieval=False
        )

        item = corpus[3]
        self.assertTrue(isinstance(item, np.ndarray))
        self.assertEqual(item.shape, (1, corpus.dim))

        dslice = corpus[2:6]
        self.assertTrue(isinstance(dslice, np.ndarray))
        self.assertEqual(dslice.shape, (4, corpus.dim))

        ilist = corpus[[2, 3, 4, 5]]
        self.assertTrue(isinstance(ilist, np.ndarray))
        self.assertEqual(ilist.shape, (4, corpus.dim))

        # Also compare with what the dense dataset is giving us
        d_dslice = dense_corpus[2:6]
        self.assertEqual(dslice.all(), d_dslice.all())
        self.assertEqual(ilist.all(), dslice.all())

    def test_getitem_dense2gensim(self):

        corpus = ShardedCorpus(
            self.tmp_fname, self.data, shardsize=100, dim=self.dim,
            sparse_serialization=False, gensim=True
        )

        item = corpus[3]
        self.assertTrue(isinstance(item, list))
        self.assertTrue(isinstance(item[0], tuple))

        dslice = corpus[2:6]
        self.assertTrue(next(dslice) == corpus[2])
        dslice = list(dslice)
        self.assertTrue(isinstance(dslice, list))
        self.assertTrue(isinstance(dslice[0], list))
        self.assertTrue(isinstance(dslice[0][0], tuple))

        iscorp, _ = is_corpus(dslice)
        self.assertTrue(iscorp, "Is the object returned by slice notation a gensim corpus?")

        ilist = corpus[[2, 3, 4, 5]]
        self.assertTrue(next(ilist) == corpus[2])
        ilist = list(ilist)
        self.assertTrue(isinstance(ilist, list))
        self.assertTrue(isinstance(ilist[0], list))
        self.assertTrue(isinstance(ilist[0][0], tuple))

        # From generators to lists

        self.assertEqual(len(ilist), len(dslice))
        for i in range(len(ilist)):
            self.assertEqual(len(ilist[i]), len(dslice[i]),
                             "Row %d: dims %d/%d" % (i, len(ilist[i]),
                                                     len(dslice[i])))
            for j in range(len(ilist[i])):
                self.assertEqual(ilist[i][j], dslice[i][j],
                                 "ilist[%d][%d] = %s ,dslice[%d][%d] = %s" % (
                                     i, j, str(ilist[i][j]), i, j,
                                     str(dslice[i][j])))

        iscorp, _ = is_corpus(ilist)
        self.assertTrue(iscorp, "Is the object returned by list notation a gensim corpus?")

    def test_resize(self):

        dataset = ShardedCorpus(self.tmp_fname, self.data, shardsize=100,
                                dim=self.dim)

        self.assertEqual(10, dataset.n_shards)

        dataset.resize_shards(250)

        self.assertEqual(4, dataset.n_shards)
        for n in range(dataset.n_shards):
            fname = dataset._shard_name(n)
            self.assertTrue(os.path.isfile(fname))

    def test_init_with_generator(self):

        def data_generator():
            yield [(0, 1)]
            yield [(1, 1)]

        gen_tmp_fname = self.tmp_fname + '.generator'
        corpus = ShardedCorpus(gen_tmp_fname, data_generator(), dim=2)

        self.assertEqual(2, len(corpus))
        self.assertEqual(1, corpus[0][0])


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestShardedCorpus)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
