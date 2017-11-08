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
import unittest
from unittest.mock import Mock
import os

import numpy as np

from gensim.models.poincare import PoincareData, PoincareModel


module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)
logger = logging.getLogger(__name__)


class TestPoincareData(unittest.TestCase):
    def test_encoding_handling(self):
        """Tests whether utf8 and non-utf8 data loaded correctly."""
        non_utf8_file = datapath('poincare_cp852.tsv')
        relations = [relation for relation in PoincareData(non_utf8_file, encoding='cp852')]
        self.assertEqual(len(relations), 2)
        self.assertEqual(relations[0], (u'tímto', u'budeš'))

        utf8_file = datapath('poincare_utf8.tsv')
        relations = [relation for relation in PoincareData(utf8_file)]
        self.assertEqual(len(relations), 2)
        self.assertEqual(relations[0], (u'tímto', u'budeš'))


class TestPoincareModel(unittest.TestCase):
    def setUp(self):
        self.data = PoincareData(datapath('poincare_hypernyms.tsv'))

    def test_data_counts(self):
        """Tests whether data has been loaded correctly and completely."""
        model = PoincareModel(self.data)
        self.assertEqual(len(model.all_relations), 5)
        self.assertEqual(len(model.term_relations[model.wv.vocab['kangaroo.n.01'].index]), 3)
        self.assertEqual(len(model.wv.vocab), 7)
        self.assertTrue('mammal.n.01' not in model.term_relations)

    def test_vector_shape(self):
        """Tests whether vectors are initialized with the correct size."""
        model = PoincareModel(self.data, size=20)
        self.assertEqual(model.wv.syn0.shape, (7, 20))

    def test_training(self):
        """Tests that vectors are different before and after training."""
        model = PoincareModel(self.data, iter=2)
        old_vectors = np.copy(model.wv.syn0)
        model.train()
        self.assertFalse(np.allclose(old_vectors, model.wv.syn0))

    def test_gradients_check(self):
        """Tests that the gradients check succeeds during training."""
        model = PoincareModel(self.data, iter=2)
        old_vectors = np.copy(model.wv.syn0)
        model.train(batch_size=1, check_gradients_every=1)
        self.assertFalse(np.allclose(old_vectors, model.wv.syn0))

    def test_wrong_gradients_raises_assertion(self):
        model = PoincareModel(self.data, iter=2)
        model.loss_grad = Mock(return_value=np.zeros((2 + model.negative, model.size)))
        with self.assertRaises(AssertionError):
            model.train(batch_size=1, check_gradients_every=1)

    def test_reproducible(self):
        """Tests that vectors are same for two independent models trained with the same seed."""
        model_1 = PoincareModel(self.data, iter=2, seed=1)
        model_1.train()

        model_2 = PoincareModel(self.data, iter=2, seed=1)
        model_2.train()
        self.assertTrue(np.allclose(model_1.wv.syn0, model_2.wv.syn0))

    def test_burn_in(self):
        """Tests that vectors are different for models with and without burn-in."""
        model_1 = PoincareModel(self.data, iter=2, burn_in=0)
        model_1.train()

        model_2 = PoincareModel(self.data, iter=2, burn_in=1)
        model_2.train()
        self.assertFalse(np.allclose(model_1.wv.syn0, model_2.wv.syn0))

    def test_negatives(self):
        """Tests that correct number of negatives are used."""
        model = PoincareModel(self.data, negative=5)
        self.assertEqual(len(model.get_candidate_negatives()), 5)

    def test_handle_duplicates(self):
        """Tests that correct number of negatives are used."""
        vector_updates = np.array([[0.5, 0.5], [0.1, 0.2], [0.3, -0.2]])
        node_indices = [0, 1, 0]
        PoincareModel.handle_duplicates(vector_updates, node_indices)
        vector_updates_expected = np.array([[0.0, 0.0], [0.1, 0.2], [0.8, 0.3]])
        self.assertTrue((vector_updates == vector_updates_expected).all())


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
