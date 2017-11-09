#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayantjain1992@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking the poincare module from the models package.
"""

import logging
import os
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from gensim.models.poincare import PoincareRelations, PoincareModel
from gensim.test.utils import datapath


logger = logging.getLogger(__name__)


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_word2vec.tst')


class TestPoincareData(unittest.TestCase):
    def test_encoding_handling(self):
        """Tests whether utf8 and non-utf8 data loaded correctly."""
        non_utf8_file = datapath('poincare_cp852.tsv')
        relations = [relation for relation in PoincareRelations(non_utf8_file, encoding='cp852')]
        self.assertEqual(len(relations), 2)
        self.assertEqual(relations[0], (u'tímto', u'budeš'))

        utf8_file = datapath('poincare_utf8.tsv')
        relations = [relation for relation in PoincareRelations(utf8_file)]
        self.assertEqual(len(relations), 2)
        self.assertEqual(relations[0], (u'tímto', u'budeš'))


class TestPoincareModel(unittest.TestCase):
    def setUp(self):
        self.data = PoincareRelations(datapath('poincare_hypernyms.tsv'))
        self.data_large = PoincareRelations(datapath('poincare_hypernyms_large.tsv'))

    def models_equal(self, model_1, model_2):
        self.assertEqual(len(model_1.wv.vocab), len(model_2.wv.vocab))
        self.assertEqual(set(model_1.wv.vocab.keys()), set(model_2.wv.vocab.keys()))
        self.assertTrue(np.allclose(model_1.wv.syn0, model_2.wv.syn0))

    def test_data_counts(self):
        """Tests whether data has been loaded correctly and completely."""
        model = PoincareModel(self.data)
        self.assertEqual(len(model.all_relations), 5)
        self.assertEqual(len(model.node_relations[model.wv.vocab['kangaroo.n.01'].index]), 3)
        self.assertEqual(len(model.wv.vocab), 7)
        self.assertTrue('mammal.n.01' not in model.node_relations)

    def test_persistence(self):
        """Tests whether the model is saved and loaded correctly."""
        model = PoincareModel(self.data, burn_in=0, negative=3)
        model.train(epochs=1)
        model.save(testfile())
        loaded = PoincareModel.load(testfile())
        self.models_equal(model, loaded)

    def test_persistence_separate_file(self):
        """Tests whether the model is saved and loaded correctly when the arrays are stored separately."""
        model = PoincareModel(self.data, burn_in=0, negative=3)
        model.train(epochs=1)
        model.save(testfile(), sep_limit=1)
        loaded = PoincareModel.load(testfile())
        self.models_equal(model, loaded)

    def test_invalid_data_raises_error(self):
        """Tests that error is raised on invalid input data."""
        with self.assertRaises(ValueError):
            model = PoincareModel([("a", "b", "c")])
        with self.assertRaises(ValueError):
            model = PoincareModel(["a", "b", "c"])
        with self.assertRaises(ValueError):
            model = PoincareModel("ab")

    def test_vector_shape(self):
        """Tests whether vectors are initialized with the correct size."""
        model = PoincareModel(self.data, size=20)
        self.assertEqual(model.wv.syn0.shape, (7, 20))

    def test_training(self):
        """Tests that vectors are different before and after training."""
        model = PoincareModel(self.data_large, burn_in=0, negative=3)
        old_vectors = np.copy(model.wv.syn0)
        model.train(epochs=2)
        self.assertFalse(np.allclose(old_vectors, model.wv.syn0))

    def test_gradients_check(self):
        """Tests that the gradients check succeeds during training."""
        model = PoincareModel(self.data, negative=3)
        old_vectors = np.copy(model.wv.syn0)
        model.train(epochs=1, batch_size=1, check_gradients_every=1)
        self.assertFalse(np.allclose(old_vectors, model.wv.syn0))

    def test_wrong_gradients_raises_assertion(self):
        """Tests that discrepancy in gradients raises an error."""
        model = PoincareModel(self.data, negative=3)
        model._loss_grad = Mock(return_value=np.zeros((2 + model.negative, model.size)))
        with self.assertRaises(AssertionError):
            model.train(epochs=1, batch_size=1, check_gradients_every=1)

    def test_reproducible(self):
        """Tests that vectors are same for two independent models trained with the same seed."""
        model_1 = PoincareModel(self.data_large, seed=1, negative=3, burn_in=1)
        model_1.train(epochs=2)

        model_2 = PoincareModel(self.data_large, seed=1, negative=3, burn_in=1)
        model_2.train(epochs=2)
        self.assertTrue(np.allclose(model_1.wv.syn0, model_2.wv.syn0))

    def test_burn_in(self):
        """Tests that vectors are different for models with and without burn-in."""
        model_1 = PoincareModel(self.data, burn_in=0, negative=3)
        model_1.train(epochs=1)

        model_2 = PoincareModel(self.data, burn_in=1, negative=3)
        model_2.train(epochs=1)
        self.assertFalse(np.allclose(model_1.wv.syn0, model_2.wv.syn0))

    def test_negatives(self):
        """Tests that correct number of negatives are sampled."""
        model = PoincareModel(self.data, negative=5)
        self.assertEqual(len(model._get_candidate_negatives()), 5)

    def test_error_if_negative_more_than_population(self):
        """Tests error is rased if number of negatives to sample is more than remaining nodes."""
        model = PoincareModel(self.data, negative=5)
        with self.assertRaises(ValueError):
            model.train(epochs=1)

    def test_no_duplicates_and_positives_in_negative_sample(self):
        """Tests that no duplicates or positively related nodes are present in negative samples."""
        model = PoincareModel(self.data_large, negative=3)
        positive_nodes = model.node_relations[0]  # Positive nodes for node 0
        num_samples = 100  # Repeat experiment multiple times
        for i in range(num_samples):
            negatives = model._sample_negatives(0)
            self.assertFalse(positive_nodes & set(negatives))
            self.assertEqual(len(negatives), len(set(negatives)))

    def test_handle_duplicates(self):
        """Tests that correct number of negatives are used."""
        vector_updates = np.array([[0.5, 0.5], [0.1, 0.2], [0.3, -0.2]])
        node_indices = [0, 1, 0]
        PoincareModel._handle_duplicates(vector_updates, node_indices)
        vector_updates_expected = np.array([[0.0, 0.0], [0.1, 0.2], [0.8, 0.3]])
        self.assertTrue((vector_updates == vector_updates_expected).all())

    @classmethod
    def tearDownClass(cls):
        try:
            os.unlink(testfile())
        except OSError:
            pass


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
