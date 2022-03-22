#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayantjain1992@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking the poincare module from the models package.
"""

import functools
import logging
import unittest

import numpy as np

from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors

logger = logging.getLogger(__name__)


class TestKeyedVectors(unittest.TestCase):
    def setUp(self):
        self.vectors = KeyedVectors.load_word2vec_format(datapath('euclidean_vectors.bin'), binary=True)
        self.model_path = datapath("w2v_keyedvectors_load_test.modeldata")
        self.vocab_path = datapath("w2v_keyedvectors_load_test.vocab")

    def test_most_similar(self):
        """Test most_similar returns expected results."""
        expected = [
            'conflict',
            'administration',
            'terrorism',
            'call',
            'israel'
        ]
        predicted = [result[0] for result in self.vectors.most_similar('war', topn=5)]
        self.assertEqual(expected, predicted)

    def test_most_similar_vector(self):
        """Can we pass vectors to most_similar directly?"""
        positive = self.vectors.vectors[0:5]
        most_similar = self.vectors.most_similar(positive=positive)
        assert most_similar is not None

    def test_most_similar_parameter_types(self):
        """Are the positive/negative parameter types are getting interpreted correctly?"""
        partial = functools.partial(self.vectors.most_similar, topn=5)

        position = partial('war', 'peace')
        position_list = partial(['war'], ['peace'])
        keyword = partial(positive='war', negative='peace')
        keyword_list = partial(positive=['war'], negative=['peace'])

        #
        # The above calls should all yield identical results.
        #
        assert position == position_list
        assert position == keyword
        assert position == keyword_list

    def test_most_similar_cosmul_parameter_types(self):
        """Are the positive/negative parameter types are getting interpreted correctly?"""
        partial = functools.partial(self.vectors.most_similar_cosmul, topn=5)

        position = partial('war', 'peace')
        position_list = partial(['war'], ['peace'])
        keyword = partial(positive='war', negative='peace')
        keyword_list = partial(positive=['war'], negative=['peace'])

        #
        # The above calls should all yield identical results.
        #
        assert position == position_list
        assert position == keyword
        assert position == keyword_list

    def test_vectors_for_all_list(self):
        """Test vectors_for_all returns expected results with a list of keys."""
        words = [
            'conflict',
            'administration',
            'terrorism',
            'an out-of-vocabulary word',
            'another out-of-vocabulary word',
        ]
        vectors_for_all = self.vectors.vectors_for_all(words)

        expected = 3
        predicted = len(vectors_for_all)
        assert expected == predicted

        expected = self.vectors['conflict']
        predicted = vectors_for_all['conflict']
        assert np.allclose(expected, predicted)

    def test_vectors_for_all_with_copy_vecattrs(self):
        """Test vectors_for_all returns can copy vector attributes."""
        words = ['conflict']
        vectors_for_all = self.vectors.vectors_for_all(words, copy_vecattrs=True)

        expected = self.vectors.get_vecattr('conflict', 'count')
        predicted = vectors_for_all.get_vecattr('conflict', 'count')
        assert expected == predicted

    def test_vectors_for_all_without_copy_vecattrs(self):
        """Test vectors_for_all returns can copy vector attributes."""
        words = ['conflict']
        vectors_for_all = self.vectors.vectors_for_all(words, copy_vecattrs=False)

        not_expected = self.vectors.get_vecattr('conflict', 'count')
        predicted = vectors_for_all.get_vecattr('conflict', 'count')
        assert not_expected != predicted

    def test_most_similar_topn(self):
        """Test most_similar returns correct results when `topn` is specified."""
        self.assertEqual(len(self.vectors.most_similar('war', topn=5)), 5)
        self.assertEqual(len(self.vectors.most_similar('war', topn=10)), 10)

        predicted = self.vectors.most_similar('war', topn=None)
        self.assertEqual(len(predicted), len(self.vectors))

        predicted = self.vectors.most_similar('war', topn=0)
        self.assertEqual(len(predicted), 0)

        predicted = self.vectors.most_similar('war', topn=np.uint8(0))
        self.assertEqual(len(predicted), 0)

    def test_relative_cosine_similarity(self):
        """Test relative_cosine_similarity returns expected results with an input of a word pair and topn"""
        wordnet_syn = [
            'good', 'goodness', 'commodity', 'trade_good', 'full', 'estimable', 'honorable',
            'respectable', 'beneficial', 'just', 'upright', 'adept', 'expert', 'practiced', 'proficient',
            'skillful', 'skilful', 'dear', 'near', 'dependable', 'safe', 'secure', 'right', 'ripe', 'well',
            'effective', 'in_effect', 'in_force', 'serious', 'sound', 'salutary', 'honest', 'undecomposed',
            'unspoiled', 'unspoilt', 'thoroughly', 'soundly',
        ]  # synonyms for "good" as per wordnet
        cos_sim = [self.vectors.similarity("good", syn) for syn in wordnet_syn if syn in self.vectors]
        cos_sim = sorted(cos_sim, reverse=True)  # cosine_similarity of "good" with wordnet_syn in decreasing order
        # computing relative_cosine_similarity of two similar words
        rcs_wordnet = self.vectors.similarity("good", "nice") / sum(cos_sim[i] for i in range(10))
        rcs = self.vectors.relative_cosine_similarity("good", "nice", 10)
        self.assertTrue(rcs_wordnet >= rcs)
        self.assertTrue(np.allclose(rcs_wordnet, rcs, 0, 0.125))
        # computing relative_cosine_similarity for two non-similar words
        rcs = self.vectors.relative_cosine_similarity("good", "worst", 10)
        self.assertTrue(rcs < 0.10)

    def test_most_similar_raises_keyerror(self):
        """Test most_similar raises KeyError when input is out of vocab."""
        with self.assertRaises(KeyError):
            self.vectors.most_similar('not_in_vocab')

    def test_most_similar_restrict_vocab(self):
        """Test most_similar returns handles restrict_vocab correctly."""
        expected = set(self.vectors.index_to_key[:5])
        predicted = set(result[0] for result in self.vectors.most_similar('war', topn=5, restrict_vocab=5))
        self.assertEqual(expected, predicted)

    def test_most_similar_with_vector_input(self):
        """Test most_similar returns expected results with an input vector instead of an input word."""
        expected = [
            'war',
            'conflict',
            'administration',
            'terrorism',
            'call',
        ]
        input_vector = self.vectors['war']
        predicted = [result[0] for result in self.vectors.most_similar([input_vector], topn=5)]
        self.assertEqual(expected, predicted)

    def test_most_similar_to_given(self):
        """Test most_similar_to_given returns correct results."""
        predicted = self.vectors.most_similar_to_given('war', ['terrorism', 'call', 'waging'])
        self.assertEqual(predicted, 'terrorism')

    def test_similar_by_word(self):
        """Test similar_by_word returns expected results."""
        expected = [
            'conflict',
            'administration',
            'terrorism',
            'call',
            'israel',
        ]
        predicted = [result[0] for result in self.vectors.similar_by_word('war', topn=5)]
        self.assertEqual(expected, predicted)

    def test_similar_by_vector(self):
        """Test similar_by_word returns expected results."""
        expected = [
            'war',
            'conflict',
            'administration',
            'terrorism',
            'call',
        ]
        input_vector = self.vectors['war']
        predicted = [result[0] for result in self.vectors.similar_by_vector(input_vector, topn=5)]
        self.assertEqual(expected, predicted)

    def test_distance(self):
        """Test that distance returns expected values."""
        self.assertTrue(np.allclose(self.vectors.distance('war', 'conflict'), 0.06694602))
        self.assertEqual(self.vectors.distance('war', 'war'), 0)

    def test_similarity(self):
        """Test similarity returns expected value for two words, and for identical words."""
        self.assertTrue(np.allclose(self.vectors.similarity('war', 'war'), 1))
        self.assertTrue(np.allclose(self.vectors.similarity('war', 'conflict'), 0.93305397))

    def test_closer_than(self):
        """Test words_closer_than returns expected value for distinct and identical nodes."""
        self.assertEqual(self.vectors.closer_than('war', 'war'), [])
        expected = set(['conflict', 'administration'])
        self.assertEqual(set(self.vectors.closer_than('war', 'terrorism')), expected)

    def test_rank(self):
        """Test rank returns expected value for distinct and identical nodes."""
        self.assertEqual(self.vectors.rank('war', 'war'), 1)
        self.assertEqual(self.vectors.rank('war', 'terrorism'), 3)

    def test_add_single(self):
        """Test that adding entity in a manual way works correctly."""
        entities = [f'___some_entity{i}_not_present_in_keyed_vectors___' for i in range(5)]
        vectors = [np.random.randn(self.vectors.vector_size) for _ in range(5)]

        # Test `add` on already filled kv.
        for ent, vector in zip(entities, vectors):
            self.vectors.add_vectors(ent, vector)

        for ent, vector in zip(entities, vectors):
            self.assertTrue(np.allclose(self.vectors[ent], vector))

        # Test `add` on empty kv.
        kv = KeyedVectors(self.vectors.vector_size)
        for ent, vector in zip(entities, vectors):
            kv.add_vectors(ent, vector)

        for ent, vector in zip(entities, vectors):
            self.assertTrue(np.allclose(kv[ent], vector))

    def test_add_multiple(self):
        """Test that adding a bulk of entities in a manual way works correctly."""
        entities = ['___some_entity{}_not_present_in_keyed_vectors___'.format(i) for i in range(5)]
        vectors = [np.random.randn(self.vectors.vector_size) for _ in range(5)]

        # Test `add` on already filled kv.
        vocab_size = len(self.vectors)
        self.vectors.add_vectors(entities, vectors, replace=False)
        self.assertEqual(vocab_size + len(entities), len(self.vectors))

        for ent, vector in zip(entities, vectors):
            self.assertTrue(np.allclose(self.vectors[ent], vector))

        # Test `add` on empty kv.
        kv = KeyedVectors(self.vectors.vector_size)
        kv[entities] = vectors
        self.assertEqual(len(kv), len(entities))

        for ent, vector in zip(entities, vectors):
            self.assertTrue(np.allclose(kv[ent], vector))

    def test_add_type(self):
        kv = KeyedVectors(2)
        assert kv.vectors.dtype == REAL

        words, vectors = ["a"], np.array([1., 1.], dtype=np.float64).reshape(1, -1)
        kv.add_vectors(words, vectors)

        assert kv.vectors.dtype == REAL

    def test_set_item(self):
        """Test that __setitem__ works correctly."""
        vocab_size = len(self.vectors)

        # Add new entity.
        entity = '___some_new_entity___'
        vector = np.random.randn(self.vectors.vector_size)
        self.vectors[entity] = vector

        self.assertEqual(len(self.vectors), vocab_size + 1)
        self.assertTrue(np.allclose(self.vectors[entity], vector))

        # Replace vector for entity in vocab.
        vocab_size = len(self.vectors)
        vector = np.random.randn(self.vectors.vector_size)
        self.vectors['war'] = vector

        self.assertEqual(len(self.vectors), vocab_size)
        self.assertTrue(np.allclose(self.vectors['war'], vector))

        # __setitem__ on several entities.
        vocab_size = len(self.vectors)
        entities = ['war', '___some_new_entity1___', '___some_new_entity2___', 'terrorism', 'conflict']
        vectors = [np.random.randn(self.vectors.vector_size) for _ in range(len(entities))]

        self.vectors[entities] = vectors

        self.assertEqual(len(self.vectors), vocab_size + 2)
        for ent, vector in zip(entities, vectors):
            self.assertTrue(np.allclose(self.vectors[ent], vector))

    def test_load_model_and_vocab_file_strict(self):
        """Test loading model and voacab files which have decoding errors: strict mode"""
        with self.assertRaises(UnicodeDecodeError):
            gensim.models.KeyedVectors.load_word2vec_format(
                self.model_path, fvocab=self.vocab_path, binary=False, unicode_errors="strict")

    def test_load_model_and_vocab_file_replace(self):
        """Test loading model and voacab files which have decoding errors: replace mode"""
        model = gensim.models.KeyedVectors.load_word2vec_format(
            self.model_path, fvocab=self.vocab_path, binary=False, unicode_errors="replace")
        self.assertEqual(model.get_vecattr(u'ありがとう�', 'count'), 123)
        self.assertEqual(model.get_vecattr(u'どういたしまして�', 'count'), 789)
        self.assertEqual(model.key_to_index[u'ありがとう�'], 0)
        self.assertEqual(model.key_to_index[u'どういたしまして�'], 1)
        self.assertTrue(np.array_equal(
            model.get_vector(u'ありがとう�'), np.array([.6, .6, .6], dtype=np.float32)))
        self.assertTrue(np.array_equal(
            model.get_vector(u'どういたしまして�'), np.array([.1, .2, .3], dtype=np.float32)))

    def test_load_model_and_vocab_file_ignore(self):
        """Test loading model and voacab files which have decoding errors: ignore mode"""
        model = gensim.models.KeyedVectors.load_word2vec_format(
            self.model_path, fvocab=self.vocab_path, binary=False, unicode_errors="ignore")
        self.assertEqual(model.get_vecattr(u'ありがとう', 'count'), 123)
        self.assertEqual(model.get_vecattr(u'どういたしまして', 'count'), 789)
        self.assertEqual(model.key_to_index[u'ありがとう'], 0)
        self.assertEqual(model.key_to_index[u'どういたしまして'], 1)
        self.assertTrue(np.array_equal(
            model.get_vector(u'ありがとう'), np.array([.6, .6, .6], dtype=np.float32)))
        self.assertTrue(np.array_equal(
            model.get_vector(u'どういたしまして'), np.array([.1, .2, .3], dtype=np.float32)))

    def test_save_reload(self):
        randkv = KeyedVectors(vector_size=100)
        count = 20
        keys = [str(i) for i in range(count)]
        weights = [pseudorandom_weak_vector(randkv.vector_size) for _ in range(count)]
        randkv.add_vectors(keys, weights)
        tmpfiletxt = gensim.test.utils.get_tmpfile("tmp_kv.txt")
        randkv.save_word2vec_format(tmpfiletxt, binary=False)
        reloadtxtkv = KeyedVectors.load_word2vec_format(tmpfiletxt, binary=False)
        self.assertEqual(randkv.index_to_key, reloadtxtkv.index_to_key)
        self.assertTrue((randkv.vectors == reloadtxtkv.vectors).all())
        tmpfilebin = gensim.test.utils.get_tmpfile("tmp_kv.bin")
        randkv.save_word2vec_format(tmpfilebin, binary=True)
        reloadbinkv = KeyedVectors.load_word2vec_format(tmpfilebin, binary=True)
        self.assertEqual(randkv.index_to_key, reloadbinkv.index_to_key)
        self.assertTrue((randkv.vectors == reloadbinkv.vectors).all())

    def test_no_header(self):
        randkv = KeyedVectors(vector_size=100)
        count = 20
        keys = [str(i) for i in range(count)]
        weights = [pseudorandom_weak_vector(randkv.vector_size) for _ in range(count)]
        randkv.add_vectors(keys, weights)
        tmpfiletxt = gensim.test.utils.get_tmpfile("tmp_kv.txt")
        randkv.save_word2vec_format(tmpfiletxt, binary=False, write_header=False)
        reloadtxtkv = KeyedVectors.load_word2vec_format(tmpfiletxt, binary=False, no_header=True)
        self.assertEqual(randkv.index_to_key, reloadtxtkv.index_to_key)
        self.assertTrue((randkv.vectors == reloadtxtkv.vectors).all())

    def test_get_mean_vector(self):
        """Test get_mean_vector returns expected results."""
        keys = [
            'conflict',
            'administration',
            'terrorism',
            'call',
            'an out-of-vocabulary word',
        ]
        weights = [1, 2, 3, 1, 2]
        expected_result_1 = np.array([
            0.02000151, -0.12685453, 0.09196121, 0.25514853, 0.25740655,
            -0.11134843, -0.0502661, -0.19278568, -0.83346179, -0.12068878,
            ], dtype=np.float32)
        expected_result_2 = np.array([
            -0.0145228, -0.11530358, 0.1169825, 0.22537769, 0.29353586,
            -0.10458107, -0.05272481, -0.17547795, -0.84245106, -0.10356515,
            ], dtype=np.float32)
        expected_result_3 = np.array([
            0.01343237, -0.47651053, 0.45645328, 0.98304356, 1.1840123,
            -0.51647933, -0.25308795, -0.77931081, -3.55954733, -0.55429711,
            ], dtype=np.float32)

        self.assertTrue(np.allclose(self.vectors.get_mean_vector(keys), expected_result_1))
        self.assertTrue(np.allclose(self.vectors.get_mean_vector(keys, weights), expected_result_2))
        self.assertTrue(np.allclose(
            self.vectors.get_mean_vector(keys, pre_normalize=False), expected_result_3)
        )


class Gensim320Test(unittest.TestCase):
    def test(self):
        path = datapath('old_keyedvectors_320.dat')
        vectors = gensim.models.keyedvectors.KeyedVectors.load(path)
        self.assertTrue(vectors.get_vector('computer') is not None)


def save_dict_to_word2vec_formated_file(fname, word2vec_dict):
    with gensim.utils.open(fname, "wb") as f:
        num_words = len(word2vec_dict)
        vector_length = len(list(word2vec_dict.values())[0])

        header = "%d %d\n" % (num_words, vector_length)
        f.write(header.encode(encoding="ascii"))

        for word, vector in word2vec_dict.items():
            f.write(word.encode())
            f.write(' '.encode())
            f.write(np.array(vector).astype(np.float32).tobytes())


class LoadWord2VecFormatTest(unittest.TestCase):

    def assert_dict_equal_to_model(self, d, m):
        self.assertEqual(len(d), len(m))

        for word in d.keys():
            self.assertSequenceEqual(list(d[word]), list(m[word]))

    def verify_load2vec_binary_result(self, w2v_dict, binary_chunk_size, limit):
        tmpfile = gensim.test.utils.get_tmpfile("tmp_w2v")
        save_dict_to_word2vec_formated_file(tmpfile, w2v_dict)
        w2v_model = \
            gensim.models.keyedvectors._load_word2vec_format(
                cls=gensim.models.KeyedVectors,
                fname=tmpfile,
                binary=True,
                limit=limit,
                binary_chunk_size=binary_chunk_size)
        if limit is None:
            limit = len(w2v_dict)

        w2v_keys_postprocessed = list(w2v_dict.keys())[:limit]
        w2v_dict_postprocessed = {k.lstrip(): w2v_dict[k] for k in w2v_keys_postprocessed}

        self.assert_dict_equal_to_model(w2v_dict_postprocessed, w2v_model)

    def test_load_word2vec_format_basic(self):
        w2v_dict = {"abc": [1, 2, 3],
                    "cde": [4, 5, 6],
                    "def": [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=None)

        w2v_dict = {"abc": [1, 2, 3],
                    "cdefg": [4, 5, 6],
                    "d": [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=None)

    def test_load_word2vec_format_limit(self):
        w2v_dict = {"abc": [1, 2, 3],
                    "cde": [4, 5, 6],
                    "def": [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=1)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=1)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=1)

        w2v_dict = {"abc": [1, 2, 3],
                    "cde": [4, 5, 6],
                    "def": [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=2)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=2)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=2)

        w2v_dict = {"abc": [1, 2, 3],
                    "cdefg": [4, 5, 6],
                    "d": [7, 8, 9]}

        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=1)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=1)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=1)

        w2v_dict = {"abc": [1, 2, 3],
                    "cdefg": [4, 5, 6],
                    "d": [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=2)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=2)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=2)

    def test_load_word2vec_format_space_stripping(self):
        w2v_dict = {"\nabc": [1, 2, 3],
                    "cdefdg": [4, 5, 6],
                    "\n\ndef": [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=1)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
