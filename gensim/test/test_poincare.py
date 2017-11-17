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
try:
    from mock import Mock
except ImportError:
    from unittest.mock import Mock

import numpy as np
try:
    import autograd  # noqa:F401
    autograd_installed = True
except ImportError:
    autograd_installed = False

from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
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
        self.assertEqual(len(model_1.kv.vocab), len(model_2.kv.vocab))
        self.assertEqual(set(model_1.kv.vocab.keys()), set(model_2.kv.vocab.keys()))
        self.assertTrue(np.allclose(model_1.kv.syn0, model_2.kv.syn0))

    def test_data_counts(self):
        """Tests whether data has been loaded correctly and completely."""
        model = PoincareModel(self.data)
        self.assertEqual(len(model.all_relations), 5)
        self.assertEqual(len(model.node_relations[model.kv.vocab['kangaroo.n.01'].index]), 3)
        self.assertEqual(len(model.kv.vocab), 7)
        self.assertTrue('mammal.n.01' not in model.node_relations)

    def test_data_counts_with_bytes(self):
        """Tests whether input bytes data is loaded correctly and completely."""
        model = PoincareModel([(b'\x80\x01c', b'\x50\x71a'), (b'node.1', b'node.2')])
        self.assertEqual(len(model.all_relations), 2)
        self.assertEqual(len(model.node_relations[model.kv.vocab[b'\x80\x01c'].index]), 1)
        self.assertEqual(len(model.kv.vocab), 4)
        self.assertTrue(b'\x50\x71a' not in model.node_relations)

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
            PoincareModel([("a", "b", "c")])
        with self.assertRaises(ValueError):
            PoincareModel(["a", "b", "c"])
        with self.assertRaises(ValueError):
            PoincareModel("ab")

    def test_vector_shape(self):
        """Tests whether vectors are initialized with the correct size."""
        model = PoincareModel(self.data, size=20)
        self.assertEqual(model.kv.syn0.shape, (7, 20))

    def test_vector_dtype(self):
        """Tests whether vectors have the correct dtype before and after training."""
        model = PoincareModel(self.data_large, dtype=np.float32, burn_in=0, negative=3)
        self.assertEqual(model.kv.syn0.dtype, np.float32)
        model.train(epochs=1)
        self.assertEqual(model.kv.syn0.dtype, np.float32)

    def test_training(self):
        """Tests that vectors are different before and after training."""
        model = PoincareModel(self.data_large, burn_in=0, negative=3)
        old_vectors = np.copy(model.kv.syn0)
        model.train(epochs=2)
        self.assertFalse(np.allclose(old_vectors, model.kv.syn0))

    def test_training_multiple(self):
        """Tests that calling train multiple times results in different vectors."""
        model = PoincareModel(self.data_large, burn_in=0, negative=3)
        model.train(epochs=2)
        old_vectors = np.copy(model.kv.syn0)

        model.train(epochs=1)
        self.assertFalse(np.allclose(old_vectors, model.kv.syn0))

        old_vectors = np.copy(model.kv.syn0)
        model.train(epochs=0)
        self.assertTrue(np.allclose(old_vectors, model.kv.syn0))

    def test_gradients_check(self):
        """Tests that the model is trained successfully with gradients check enabled."""
        model = PoincareModel(self.data, negative=3)
        try:
            model.train(epochs=1, batch_size=1, check_gradients_every=1)
        except Exception as e:
            self.fail('Exception %s raised unexpectedly while training with gradient checking' % repr(e))

    @unittest.skipIf(not autograd_installed, 'autograd needs to be installed for this test')
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
        self.assertTrue(np.allclose(model_1.kv.syn0, model_2.kv.syn0))

    def test_burn_in(self):
        """Tests that vectors are different after burn-in."""
        model = PoincareModel(self.data, burn_in=1, negative=3)
        original_vectors = np.copy(model.kv.syn0)
        model.train(epochs=0)
        self.assertFalse(np.allclose(model.kv.syn0, original_vectors))

    def test_burn_in_only_done_once(self):
        """Tests that burn-in does not happen when train is called a second time."""
        model = PoincareModel(self.data, negative=3, burn_in=1)
        model.train(epochs=0)
        original_vectors = np.copy(model.kv.syn0)
        model.train(epochs=0)
        self.assertTrue(np.allclose(model.kv.syn0, original_vectors))

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


class TestPoincareKeyedVectors(unittest.TestCase):
    def setUp(self):
        self.vectors = PoincareKeyedVectors.load_word2vec_format(datapath('poincare_vectors.bin'), binary=True)

    def test_most_similar(self):
        """Test most_similar returns expected results."""
        expected = [
            'canine.n.02',
            'hunting_dog.n.01',
            'carnivore.n.01',
            'placental.n.01',
            'mammal.n.01'
        ]
        predicted = [result[0] for result in self.vectors.most_similar('dog.n.01', topn=5)]
        self.assertEqual(expected, predicted)

    def test_most_similar_topn(self):
        """Test most_similar returns correct results when `topn` is specified."""
        self.assertEqual(len(self.vectors.most_similar('dog.n.01', topn=5)), 5)
        self.assertEqual(len(self.vectors.most_similar('dog.n.01', topn=10)), 10)

        predicted = self.vectors.most_similar('dog.n.01', topn=None)
        self.assertEqual(len(predicted), len(self.vectors.vocab) - 1)
        self.assertEqual(predicted[-1][0], 'gallant_fox.n.01')

    def test_most_similar_raises_keyerror(self):
        """Test most_similar raises KeyError when input is out of vocab."""
        with self.assertRaises(KeyError):
            self.vectors.most_similar('not_in_vocab')

    def test_most_similar_restrict_vocab(self):
        """Test most_similar returns handles restrict_vocab correctly."""
        expected = set(self.vectors.index2word[:5])
        predicted = set(result[0] for result in self.vectors.most_similar('dog.n.01', topn=5, restrict_vocab=5))
        self.assertEqual(expected, predicted)

    def test_most_similar_to_given(self):
        """Test most_similar_to_given returns correct results."""
        predicted = self.vectors.most_similar_to_given('dog.n.01', ['carnivore.n.01', 'placental.n.01', 'mammal.n.01'])
        self.assertEqual(predicted, 'carnivore.n.01')

    def test_distance(self):
        """Test that distance returns expected values."""
        self.assertTrue(np.allclose(self.vectors.distance('dog.n.01', 'mammal.n.01'), 4.5278745))
        self.assertEqual(self.vectors.distance('dog.n.01', 'dog.n.01'), 0)

    def test_distances(self):
        """Test that distances between one word and multiple other words have expected values."""
        distances = self.vectors.distances('dog.n.01', ['mammal.n.01', 'dog.n.01'])
        self.assertTrue(np.allclose(distances, [4.5278745, 0]))

        distances = self.vectors.distances('dog.n.01')
        self.assertEqual(len(distances), len(self.vectors.vocab))
        self.assertTrue(np.allclose(distances[-1], 10.04756))

    def test_closest_child(self):
        """Test closest_child returns expected value and returns None for lowest node in hierarchy."""
        self.assertEqual(self.vectors.closest_child('dog.n.01'), 'terrier.n.01')
        self.assertEqual(self.vectors.closest_child('harbor_porpoise.n.01'), None)

    def test_closest_parent(self):
        """Test closest_parent returns expected value and returns None for highest node in hierarchy."""
        self.assertEqual(self.vectors.closest_parent('dog.n.01'), 'canine.n.02')
        self.assertEqual(self.vectors.closest_parent('mammal.n.01'), None)

    def test_ancestors(self):
        """Test ancestors returns expected list and returns empty list for highest node in hierarchy."""
        expected = ['canine.n.02', 'carnivore.n.01', 'placental.n.01', 'mammal.n.01']
        self.assertEqual(self.vectors.ancestors('dog.n.01'), expected)
        expected = []
        self.assertEqual(self.vectors.ancestors('mammal.n.01'), expected)

    def test_descendants(self):
        """Test descendants returns expected list and returns empty list for lowest node in hierarchy."""
        expected = [
            'terrier.n.01', 'sporting_dog.n.01', 'spaniel.n.01', 'water_spaniel.n.01', 'irish_water_spaniel.n.01'
        ]
        self.assertEqual(self.vectors.descendants('dog.n.01'), expected)
        self.assertEqual(self.vectors.descendants('dog.n.01', max_depth=3), expected[:3])

    def test_similarity(self):
        """Test similarity returns expected value for two nodes, and for identical nodes."""
        self.assertTrue(np.allclose(self.vectors.similarity('dog.n.01', 'dog.n.01'), 1))
        self.assertTrue(np.allclose(self.vectors.similarity('dog.n.01', 'mammal.n.01'), 0.728260))

    def test_similarities(self):
        """Test similarities returns expected values for multiple nodes."""
        similarities = self.vectors.similarities('dog.n.01', ['mammal.n.01', 'dog.n.01'])
        self.assertTrue(np.allclose(similarities, [0.7282602, 1]))

        similarities = self.vectors.similarities('dog.n.01')
        self.assertEqual(len(similarities), len(self.vectors.vocab))
        self.assertTrue(np.allclose(similarities[-1], 0.39699667))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
