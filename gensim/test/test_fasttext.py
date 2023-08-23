#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import gzip
import io
import logging
import unittest
import os
import shutil
import subprocess
import struct
import sys

import numpy as np
import pytest

from gensim import utils
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim, FastTextKeyedVectors, _unpack
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import (
    datapath, get_tmpfile, temporary_file, common_texts as sentences, lee_corpus_list as list_corpus,
)
from gensim.test.test_word2vec import TestWord2VecModel
import gensim.models._fasttext_bin
from gensim.models.fasttext_inner import compute_ngrams, compute_ngrams_bytes, ft_hash_bytes

import gensim.models.fasttext

try:
    from ot import emd2  # noqa:F401
    POT_EXT = True
except (ImportError, ValueError):
    POT_EXT = False

logger = logging.getLogger(__name__)

IS_WIN32 = (os.name == "nt") and (struct.calcsize('P') * 8 == 32)
MAX_WORDVEC_COMPONENT_DIFFERENCE = 1.0e-10

# Limit the size of FastText ngram buckets, for RAM reasons.
# See https://github.com/RaRe-Technologies/gensim/issues/2790
BUCKET = 10000

FT_HOME = os.environ.get("FT_HOME")
FT_CMD = shutil.which("fasttext", path=FT_HOME) or shutil.which("fasttext")


new_sentences = [
    ['computer', 'artificial', 'intelligence'],
    ['artificial', 'trees'],
    ['human', 'intelligence'],
    ['artificial', 'graph'],
    ['intelligence'],
    ['artificial', 'intelligence', 'system']
]


class TestFastTextModel(unittest.TestCase):

    def setUp(self):
        self.test_model_file = datapath('lee_fasttext.bin')
        self.test_model = gensim.models.fasttext.load_facebook_model(self.test_model_file)
        self.test_new_model_file = datapath('lee_fasttext_new.bin')

    def test_training(self):
        model = FT_gensim(vector_size=12, min_count=1, hs=1, negative=0, seed=42, workers=1, bucket=BUCKET)
        model.build_vocab(sentences)
        self.model_sanity(model)

        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        sims = model.wv.most_similar('graph', topn=10)

        self.assertEqual(model.wv.vectors.shape, (12, 12))
        self.assertEqual(len(model.wv), 12)
        self.assertEqual(model.wv.vectors_vocab.shape[1], 12)
        self.assertEqual(model.wv.vectors_ngrams.shape[1], 12)
        self.model_sanity(model)

        # test querying for "most similar" by vector
        graph_vector = model.wv.get_vector('graph', norm=True)
        sims2 = model.wv.most_similar(positive=[graph_vector], topn=11)
        sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
        self.assertEqual(sims, sims2)

        # build vocab and train in one step; must be the same as above
        model2 = FT_gensim(sentences, vector_size=12, min_count=1, hs=1, negative=0, seed=42, workers=1, bucket=BUCKET)
        self.models_equal(model, model2)

        # verify oov-word vector retrieval
        invocab_vec = model.wv['minors']  # invocab word
        self.assertEqual(len(invocab_vec), 12)

        oov_vec = model.wv['minor']  # oov word
        self.assertEqual(len(oov_vec), 12)

    def test_fast_text_train_parameters(self):

        model = FT_gensim(vector_size=12, min_count=1, hs=1, negative=0, seed=42, workers=1, bucket=BUCKET)
        model.build_vocab(corpus_iterable=sentences)

        self.assertRaises(TypeError, model.train, corpus_file=11111, total_examples=1, epochs=1)
        self.assertRaises(TypeError, model.train, corpus_iterable=11111, total_examples=1, epochs=1)
        self.assertRaises(
            TypeError, model.train, corpus_iterable=sentences, corpus_file='test', total_examples=1, epochs=1)
        self.assertRaises(TypeError, model.train, corpus_iterable=None, corpus_file=None, total_examples=1, epochs=1)
        self.assertRaises(TypeError, model.train, corpus_file=sentences, total_examples=1, epochs=1)

    def test_training_fromfile(self):
        with temporary_file('gensim_fasttext.tst') as corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)

            model = FT_gensim(vector_size=12, min_count=1, hs=1, negative=0, seed=42, workers=1, bucket=BUCKET)
            model.build_vocab(corpus_file=corpus_file)
            self.model_sanity(model)

            model.train(corpus_file=corpus_file, total_words=model.corpus_total_words, epochs=model.epochs)
            sims = model.wv.most_similar('graph', topn=10)

            self.assertEqual(model.wv.vectors.shape, (12, 12))
            self.assertEqual(len(model.wv), 12)
            self.assertEqual(model.wv.vectors_vocab.shape[1], 12)
            self.assertEqual(model.wv.vectors_ngrams.shape[1], 12)
            self.model_sanity(model)

            # test querying for "most similar" by vector
            graph_vector = model.wv.get_vector('graph', norm=True)
            sims2 = model.wv.most_similar(positive=[graph_vector], topn=11)
            sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
            self.assertEqual(sims, sims2)

            # verify oov-word vector retrieval
            invocab_vec = model.wv['minors']  # invocab word
            self.assertEqual(len(invocab_vec), 12)

            oov_vec = model.wv['minor']  # oov word
            self.assertEqual(len(oov_vec), 12)

    def models_equal(self, model, model2):
        self.assertEqual(len(model.wv), len(model2.wv))
        self.assertEqual(model.wv.bucket, model2.wv.bucket)
        self.assertTrue(np.allclose(model.wv.vectors_vocab, model2.wv.vectors_vocab))
        self.assertTrue(np.allclose(model.wv.vectors_ngrams, model2.wv.vectors_ngrams))
        self.assertTrue(np.allclose(model.wv.vectors, model2.wv.vectors))
        if model.hs:
            self.assertTrue(np.allclose(model.syn1, model2.syn1))
        if model.negative:
            self.assertTrue(np.allclose(model.syn1neg, model2.syn1neg))
        most_common_word = max(model.wv.key_to_index, key=lambda word: model.wv.get_vecattr(word, 'count'))[0]
        self.assertTrue(np.allclose(model.wv[most_common_word], model2.wv[most_common_word]))

    def test_persistence(self):
        tmpf = get_tmpfile('gensim_fasttext.tst')
        model = FT_gensim(sentences, min_count=1, bucket=BUCKET)
        model.save(tmpf)
        self.models_equal(model, FT_gensim.load(tmpf))
        #  test persistence of the KeyedVectors of a model
        wv = model.wv
        wv.save(tmpf)
        loaded_wv = FastTextKeyedVectors.load(tmpf)
        self.assertTrue(np.allclose(wv.vectors_ngrams, loaded_wv.vectors_ngrams))
        self.assertEqual(len(wv), len(loaded_wv))

    def test_persistence_fromfile(self):
        with temporary_file('gensim_fasttext1.tst') as corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)

            tmpf = get_tmpfile('gensim_fasttext.tst')
            model = FT_gensim(corpus_file=corpus_file, min_count=1, bucket=BUCKET)
            model.save(tmpf)
            self.models_equal(model, FT_gensim.load(tmpf))
            #  test persistence of the KeyedVectors of a model
            wv = model.wv
            wv.save(tmpf)
            loaded_wv = FastTextKeyedVectors.load(tmpf)
            self.assertTrue(np.allclose(wv.vectors_ngrams, loaded_wv.vectors_ngrams))
            self.assertEqual(len(wv), len(loaded_wv))

    def model_sanity(self, model):
        self.model_structural_sanity(model)
        # TODO: add semantic tests, where appropriate

    def model_structural_sanity(self, model):
        """Check a model for basic self-consistency, necessary properties & property
        correspondences, but no semantic tests."""
        self.assertEqual(model.wv.vectors.shape, (len(model.wv), model.vector_size))
        self.assertEqual(model.wv.vectors_vocab.shape, (len(model.wv), model.vector_size))
        self.assertEqual(model.wv.vectors_ngrams.shape, (model.wv.bucket, model.vector_size))
        self.assertLessEqual(len(model.wv.vectors_ngrams_lockf), len(model.wv.vectors_ngrams))
        self.assertLessEqual(len(model.wv.vectors_vocab_lockf), len(model.wv.index_to_key))
        self.assertTrue(np.isfinite(model.wv.vectors_ngrams).all(), "NaN in ngrams")
        self.assertTrue(np.isfinite(model.wv.vectors_vocab).all(), "NaN in vectors_vocab")
        if model.negative:
            self.assertTrue(np.isfinite(model.syn1neg).all(), "NaN in syn1neg")
        if model.hs:
            self.assertTrue(np.isfinite(model.syn1).all(), "NaN in syn1neg")

    def test_load_fasttext_format(self):
        try:
            model = gensim.models.fasttext.load_facebook_model(self.test_model_file)
        except Exception as exc:
            self.fail('Unable to load FastText model from file %s: %s' % (self.test_model_file, exc))
        vocab_size, model_size = 1762, 10
        self.assertEqual(model.wv.vectors.shape, (vocab_size, model_size))
        self.assertEqual(len(model.wv), vocab_size, model_size)
        self.assertEqual(model.wv.vectors_ngrams.shape, (model.wv.bucket, model_size))

        expected_vec = [
            -0.57144,
            -0.0085561,
            0.15748,
            -0.67855,
            -0.25459,
            -0.58077,
            -0.09913,
            1.1447,
            0.23418,
            0.060007
        ]  # obtained using ./fasttext print-word-vectors lee_fasttext_new.bin
        actual_vec = model.wv["hundred"]
        self.assertTrue(np.allclose(actual_vec, expected_vec, atol=1e-4))

        # vector for oov words are slightly different from original FastText due to discarding unused ngrams
        # obtained using a modified version of ./fasttext print-word-vectors lee_fasttext_new.bin
        expected_vec_oov = [
            -0.21929,
            -0.53778,
            -0.22463,
            -0.41735,
            0.71737,
            -1.59758,
            -0.24833,
            0.62028,
            0.53203,
            0.77568
        ]
        actual_vec_oov = model.wv["rejection"]
        self.assertTrue(np.allclose(actual_vec_oov, expected_vec_oov, atol=1e-4))

        self.assertEqual(model.min_count, 5)
        self.assertEqual(model.window, 5)
        self.assertEqual(model.epochs, 5)
        self.assertEqual(model.negative, 5)
        self.assertEqual(model.sample, 0.0001)
        self.assertEqual(model.wv.bucket, 1000)
        self.assertEqual(model.wv.max_n, 6)
        self.assertEqual(model.wv.min_n, 3)
        self.assertEqual(model.wv.vectors.shape, (len(model.wv), model.vector_size))
        self.assertEqual(model.wv.vectors_ngrams.shape, (model.wv.bucket, model.vector_size))

    def test_load_fasttext_new_format(self):
        try:
            new_model = gensim.models.fasttext.load_facebook_model(self.test_new_model_file)
        except Exception as exc:
            self.fail('Unable to load FastText model from file %s: %s' % (self.test_new_model_file, exc))
        vocab_size, model_size = 1763, 10
        self.assertEqual(new_model.wv.vectors.shape, (vocab_size, model_size))
        self.assertEqual(len(new_model.wv), vocab_size, model_size)
        self.assertEqual(new_model.wv.vectors_ngrams.shape, (new_model.wv.bucket, model_size))

        expected_vec = [
            -0.025627,
            -0.11448,
            0.18116,
            -0.96779,
            0.2532,
            -0.93224,
            0.3929,
            0.12679,
            -0.19685,
            -0.13179
        ]  # obtained using ./fasttext print-word-vectors lee_fasttext_new.bin
        actual_vec = new_model.wv["hundred"]
        self.assertTrue(np.allclose(actual_vec, expected_vec, atol=1e-4))

        # vector for oov words are slightly different from original FastText due to discarding unused ngrams
        # obtained using a modified version of ./fasttext print-word-vectors lee_fasttext_new.bin
        expected_vec_oov = [
            -0.49111,
            -0.13122,
            -0.02109,
            -0.88769,
            -0.20105,
            -0.91732,
            0.47243,
            0.19708,
            -0.17856,
            0.19815
        ]
        actual_vec_oov = new_model.wv["rejection"]
        self.assertTrue(np.allclose(actual_vec_oov, expected_vec_oov, atol=1e-4))

        self.assertEqual(new_model.min_count, 5)
        self.assertEqual(new_model.window, 5)
        self.assertEqual(new_model.epochs, 5)
        self.assertEqual(new_model.negative, 5)
        self.assertEqual(new_model.sample, 0.0001)
        self.assertEqual(new_model.wv.bucket, 1000)
        self.assertEqual(new_model.wv.max_n, 6)
        self.assertEqual(new_model.wv.min_n, 3)
        self.assertEqual(new_model.wv.vectors.shape, (len(new_model.wv), new_model.vector_size))
        self.assertEqual(new_model.wv.vectors_ngrams.shape, (new_model.wv.bucket, new_model.vector_size))

    def test_load_model_supervised(self):
        with self.assertRaises(NotImplementedError):
            gensim.models.fasttext.load_facebook_model(datapath('pang_lee_polarity_fasttext.bin'))

    def test_load_model_with_non_ascii_vocab(self):
        model = gensim.models.fasttext.load_facebook_model(datapath('non_ascii_fasttext.bin'))
        self.assertTrue(u'který' in model.wv)
        try:
            model.wv[u'který']
        except UnicodeDecodeError:
            self.fail('Unable to access vector for utf8 encoded non-ascii word')

    def test_load_model_non_utf8_encoding(self):
        model = gensim.models.fasttext.load_facebook_model(datapath('cp852_fasttext.bin'), encoding='cp852')
        self.assertTrue(u'který' in model.wv)
        try:
            model.wv[u'který']
        except KeyError:
            self.fail('Unable to access vector for cp-852 word')

    def test_oov_similarity(self):
        word = 'someoovword'
        most_similar = self.test_model.wv.most_similar(word)
        top_neighbor, top_similarity = most_similar[0]
        v1 = self.test_model.wv[word]
        v2 = self.test_model.wv[top_neighbor]
        top_similarity_direct = self.test_model.wv.cosine_similarities(v1, v2.reshape(1, -1))[0]
        self.assertAlmostEqual(top_similarity, top_similarity_direct, places=6)

    def test_n_similarity(self):
        # In vocab, sanity check
        self.assertTrue(np.allclose(self.test_model.wv.n_similarity(['the', 'and'], ['and', 'the']), 1.0))
        self.assertEqual(
            self.test_model.wv.n_similarity(['the'], ['and']), self.test_model.wv.n_similarity(['and'], ['the']))
        # Out of vocab check
        self.assertTrue(np.allclose(self.test_model.wv.n_similarity(['night', 'nights'], ['nights', 'night']), 1.0))
        self.assertEqual(
            self.test_model.wv.n_similarity(['night'], ['nights']),
            self.test_model.wv.n_similarity(['nights'], ['night'])
        )

    def test_similarity(self):
        # In vocab, sanity check
        self.assertTrue(np.allclose(self.test_model.wv.similarity('the', 'the'), 1.0))
        self.assertEqual(self.test_model.wv.similarity('the', 'and'), self.test_model.wv.similarity('and', 'the'))
        # Out of vocab check
        self.assertTrue(np.allclose(self.test_model.wv.similarity('nights', 'nights'), 1.0))
        self.assertEqual(
            self.test_model.wv.similarity('night', 'nights'), self.test_model.wv.similarity('nights', 'night'))

    def test_most_similar(self):
        # In vocab, sanity check
        self.assertEqual(len(self.test_model.wv.most_similar(positive=['the', 'and'], topn=5)), 5)
        self.assertEqual(self.test_model.wv.most_similar('the'), self.test_model.wv.most_similar(positive=['the']))
        # Out of vocab check
        self.assertEqual(len(self.test_model.wv.most_similar(['night', 'nights'], topn=5)), 5)
        self.assertEqual(
            self.test_model.wv.most_similar('nights'), self.test_model.wv.most_similar(positive=['nights']))

    def test_most_similar_cosmul(self):
        # In vocab, sanity check
        self.assertEqual(len(self.test_model.wv.most_similar_cosmul(positive=['the', 'and'], topn=5)), 5)
        self.assertEqual(
            self.test_model.wv.most_similar_cosmul('the'),
            self.test_model.wv.most_similar_cosmul(positive=['the']))
        # Out of vocab check
        self.assertEqual(len(self.test_model.wv.most_similar_cosmul(['night', 'nights'], topn=5)), 5)
        self.assertEqual(
            self.test_model.wv.most_similar_cosmul('nights'),
            self.test_model.wv.most_similar_cosmul(positive=['nights']))
        self.assertEqual(
            self.test_model.wv.most_similar_cosmul('the', 'and'),
            self.test_model.wv.most_similar_cosmul(positive=['the'], negative=['and']))

    def test_lookup(self):
        # In vocab, sanity check
        self.assertTrue('night' in self.test_model.wv.key_to_index)
        self.assertTrue(np.allclose(self.test_model.wv['night'], self.test_model.wv[['night']]))
        # Out of vocab check
        self.assertFalse('nights' in self.test_model.wv.key_to_index)
        self.assertTrue(np.allclose(self.test_model.wv['nights'], self.test_model.wv[['nights']]))

    def test_contains(self):
        # In vocab, sanity check
        self.assertTrue('night' in self.test_model.wv.key_to_index)
        self.assertTrue('night' in self.test_model.wv)
        # Out of vocab check
        self.assertFalse(self.test_model.wv.has_index_for('nights'))
        self.assertFalse('nights' in self.test_model.wv.key_to_index)
        self.assertTrue('nights' in self.test_model.wv)

    @unittest.skipIf(POT_EXT is False, "POT not installed")
    def test_wm_distance(self):
        doc = ['night', 'payment']
        oov_doc = ['nights', 'forests', 'payments']

        dist = self.test_model.wv.wmdistance(doc, oov_doc)
        self.assertNotEqual(float('inf'), dist)

    def test_cbow_neg_training(self):
        model_gensim = FT_gensim(
            vector_size=48, sg=0, cbow_mean=1, alpha=0.05, window=5, hs=0, negative=5,
            min_count=5, epochs=10, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
            sorted_vocab=1, workers=1, min_alpha=0.0, bucket=BUCKET)

        lee_data = LineSentence(datapath('lee_background.cor'))
        model_gensim.build_vocab(lee_data)
        orig0 = np.copy(model_gensim.wv.vectors[0])
        model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.epochs)
        self.assertFalse((orig0 == model_gensim.wv.vectors[0]).all())  # vector should vary after training

        sims_gensim = model_gensim.wv.most_similar('night', topn=10)
        sims_gensim_words = [word for (word, distance) in sims_gensim]  # get similar words
        expected_sims_words = [
            u'night.',
            u'night,',
            u'eight',
            u'fight',
            u'month',
            u'hearings',
            u'Washington',
            u'remains',
            u'overnight',
            u'running']
        overlaps = set(sims_gensim_words).intersection(expected_sims_words)
        overlap_count = len(overlaps)
        self.assertGreaterEqual(
            overlap_count, 2,
            "only %i overlap in expected %s & actual %s" % (overlap_count, expected_sims_words, sims_gensim_words))

    def test_cbow_neg_training_fromfile(self):
        with temporary_file('gensim_fasttext.tst') as corpus_file:
            model_gensim = FT_gensim(
                vector_size=48, sg=0, cbow_mean=1, alpha=0.05, window=5, hs=0, negative=5,
                min_count=5, epochs=10, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
                sorted_vocab=1, workers=1, min_alpha=0.0, bucket=BUCKET)

            lee_data = LineSentence(datapath('lee_background.cor'))
            utils.save_as_line_sentence(lee_data, corpus_file)

            model_gensim.build_vocab(corpus_file=corpus_file)
            orig0 = np.copy(model_gensim.wv.vectors[0])
            model_gensim.train(corpus_file=corpus_file,
                               total_words=model_gensim.corpus_total_words,
                               epochs=model_gensim.epochs)
            self.assertFalse((orig0 == model_gensim.wv.vectors[0]).all())  # vector should vary after training

            sims_gensim = model_gensim.wv.most_similar('night', topn=10)
            sims_gensim_words = [word for (word, distance) in sims_gensim]  # get similar words
            expected_sims_words = [
                u'night.',
                u'night,',
                u'eight',
                u'fight',
                u'month',
                u'hearings',
                u'Washington',
                u'remains',
                u'overnight',
                u'running']
            overlaps = set(sims_gensim_words).intersection(expected_sims_words)
            overlap_count = len(overlaps)
            self.assertGreaterEqual(
                overlap_count, 2,
                "only %i overlap in expected %s & actual %s" % (overlap_count, expected_sims_words, sims_gensim_words))

    def test_sg_neg_training(self):

        model_gensim = FT_gensim(
            vector_size=48, sg=1, cbow_mean=1, alpha=0.025, window=5, hs=0, negative=5,
            min_count=5, epochs=10, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
            sorted_vocab=1, workers=1, min_alpha=0.0, bucket=BUCKET * 4)

        lee_data = LineSentence(datapath('lee_background.cor'))
        model_gensim.build_vocab(lee_data)
        orig0 = np.copy(model_gensim.wv.vectors[0])
        model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.epochs)
        self.assertFalse((orig0 == model_gensim.wv.vectors[0]).all())  # vector should vary after training

        sims_gensim = model_gensim.wv.most_similar('night', topn=10)
        sims_gensim_words = [word for (word, distance) in sims_gensim]  # get similar words
        expected_sims_words = [
            u'night.',
            u'night,',
            u'eight',
            u'overnight',
            u'overnight.',
            u'month',
            u'land',
            u'firm',
            u'singles',
            u'death']
        overlaps = set(sims_gensim_words).intersection(expected_sims_words)
        overlap_count = len(overlaps)
        self.assertGreaterEqual(
            overlap_count, 2,
            "only %i overlap in expected %s & actual %s" % (overlap_count, expected_sims_words, sims_gensim_words))

    def test_sg_neg_training_fromfile(self):
        with temporary_file('gensim_fasttext.tst') as corpus_file:
            model_gensim = FT_gensim(
                vector_size=48, sg=1, cbow_mean=1, alpha=0.025, window=5, hs=0, negative=5,
                min_count=5, epochs=10, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
                sorted_vocab=1, workers=1, min_alpha=0.0, bucket=BUCKET * 4)

            lee_data = LineSentence(datapath('lee_background.cor'))
            utils.save_as_line_sentence(lee_data, corpus_file)

            model_gensim.build_vocab(corpus_file=corpus_file)
            orig0 = np.copy(model_gensim.wv.vectors[0])
            model_gensim.train(corpus_file=corpus_file,
                               total_words=model_gensim.corpus_total_words,
                               epochs=model_gensim.epochs)
            self.assertFalse((orig0 == model_gensim.wv.vectors[0]).all())  # vector should vary after training

            sims_gensim = model_gensim.wv.most_similar('night', topn=10)
            sims_gensim_words = [word for (word, distance) in sims_gensim]  # get similar words
            expected_sims_words = [
                u'night.',
                u'night,',
                u'eight',
                u'overnight',
                u'overnight.',
                u'month',
                u'land',
                u'firm',
                u'singles',
                u'death']
            overlaps = set(sims_gensim_words).intersection(expected_sims_words)
            overlap_count = len(overlaps)
            self.assertGreaterEqual(
                overlap_count, 2,
                "only %i overlap in expected %s & actual %s" % (overlap_count, expected_sims_words, sims_gensim_words))

    def test_online_learning(self):
        model_hs = FT_gensim(sentences, vector_size=12, min_count=1, seed=42, hs=1, negative=0, bucket=BUCKET)
        self.assertEqual(len(model_hs.wv), 12)
        self.assertEqual(model_hs.wv.get_vecattr('graph', 'count'), 3)
        model_hs.build_vocab(new_sentences, update=True)  # update vocab
        self.assertEqual(len(model_hs.wv), 14)
        self.assertEqual(model_hs.wv.get_vecattr('graph', 'count'), 4)
        self.assertEqual(model_hs.wv.get_vecattr('artificial', 'count'), 4)

    def test_online_learning_fromfile(self):
        with temporary_file('gensim_fasttext1.tst') as corpus_file, \
                temporary_file('gensim_fasttext2.tst') as new_corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)
            utils.save_as_line_sentence(new_sentences, new_corpus_file)

            model_hs = FT_gensim(
                corpus_file=corpus_file, vector_size=12, min_count=1, seed=42, hs=1, negative=0, bucket=BUCKET)
            self.assertTrue(len(model_hs.wv), 12)
            self.assertTrue(model_hs.wv.get_vecattr('graph', 'count'), 3)
            model_hs.build_vocab(corpus_file=new_corpus_file, update=True)  # update vocab
            self.assertEqual(len(model_hs.wv), 14)
            self.assertTrue(model_hs.wv.get_vecattr('graph', 'count'), 4)
            self.assertTrue(model_hs.wv.get_vecattr('artificial', 'count'), 4)

    def test_online_learning_after_save(self):
        tmpf = get_tmpfile('gensim_fasttext.tst')
        model_neg = FT_gensim(sentences, vector_size=12, min_count=0, seed=42, hs=0, negative=5, bucket=BUCKET)
        model_neg.save(tmpf)
        model_neg = FT_gensim.load(tmpf)
        self.assertTrue(len(model_neg.wv), 12)
        model_neg.build_vocab(new_sentences, update=True)  # update vocab
        model_neg.train(new_sentences, total_examples=model_neg.corpus_count, epochs=model_neg.epochs)
        self.assertEqual(len(model_neg.wv), 14)

    def test_online_learning_through_ft_format_saves(self):
        tmpf = get_tmpfile('gensim_ft_format.tst')
        model = FT_gensim(sentences, vector_size=12, min_count=0, seed=42, hs=0, negative=5, bucket=BUCKET)
        gensim.models.fasttext.save_facebook_model(model, tmpf)
        model_reload = gensim.models.fasttext.load_facebook_model(tmpf)
        self.assertTrue(len(model_reload.wv), 12)
        self.assertEqual(len(model_reload.wv), len(model_reload.wv.vectors))
        self.assertEqual(len(model_reload.wv), len(model_reload.wv.vectors_vocab))
        model_reload.build_vocab(new_sentences, update=True)  # update vocab
        model_reload.train(new_sentences, total_examples=model_reload.corpus_count, epochs=model_reload.epochs)
        self.assertEqual(len(model_reload.wv), 14)
        self.assertEqual(len(model_reload.wv), len(model_reload.wv.vectors))
        self.assertEqual(len(model_reload.wv), len(model_reload.wv.vectors_vocab))
        tmpf2 = get_tmpfile('gensim_ft_format2.tst')
        gensim.models.fasttext.save_facebook_model(model_reload, tmpf2)

    def test_online_learning_after_save_fromfile(self):
        with temporary_file('gensim_fasttext1.tst') as corpus_file, \
                temporary_file('gensim_fasttext2.tst') as new_corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)
            utils.save_as_line_sentence(new_sentences, new_corpus_file)

            tmpf = get_tmpfile('gensim_fasttext.tst')
            model_neg = FT_gensim(
                corpus_file=corpus_file, vector_size=12, min_count=0, seed=42, hs=0, negative=5, bucket=BUCKET)
            model_neg.save(tmpf)
            model_neg = FT_gensim.load(tmpf)
            self.assertTrue(len(model_neg.wv), 12)
            model_neg.build_vocab(corpus_file=new_corpus_file, update=True)  # update vocab
            model_neg.train(corpus_file=new_corpus_file, total_words=model_neg.corpus_total_words,
                            epochs=model_neg.epochs)
            self.assertEqual(len(model_neg.wv), 14)

    def online_sanity(self, model):
        terro, others = [], []
        for line in list_corpus:
            if 'terrorism' in line:
                terro.append(line)
            else:
                others.append(line)
        self.assertTrue(all('terrorism' not in line for line in others))
        model.build_vocab(others)
        start_vecs = model.wv.vectors_vocab.copy()
        model.train(others, total_examples=model.corpus_count, epochs=model.epochs)
        # checks that `vectors_vocab` has been changed by training
        self.assertFalse(np.all(np.equal(start_vecs, model.wv.vectors_vocab)))
        # checks that `vectors` is different from `vectors_vocab`
        self.assertFalse(np.all(np.equal(model.wv.vectors, model.wv.vectors_vocab)))
        self.assertFalse('terrorism' in model.wv.key_to_index)
        model.build_vocab(terro, update=True)  # update vocab
        self.assertTrue(model.wv.vectors_ngrams.dtype == 'float32')
        self.assertTrue('terrorism' in model.wv.key_to_index)
        orig0_all = np.copy(model.wv.vectors_ngrams)
        model.train(terro, total_examples=len(terro), epochs=model.epochs)
        self.assertFalse(np.allclose(model.wv.vectors_ngrams, orig0_all))
        sim = model.wv.n_similarity(['war'], ['terrorism'])
        assert abs(sim) > 0.6

    def test_sg_hs_online(self):
        model = FT_gensim(sg=1, window=2, hs=1, negative=0, min_count=3, epochs=1, seed=42, workers=1, bucket=BUCKET)
        self.online_sanity(model)

    def test_sg_neg_online(self):
        model = FT_gensim(sg=1, window=2, hs=0, negative=5, min_count=3, epochs=1, seed=42, workers=1, bucket=BUCKET)
        self.online_sanity(model)

    def test_cbow_hs_online(self):
        model = FT_gensim(
            sg=0, cbow_mean=1, alpha=0.05, window=2, hs=1, negative=0, min_count=3, epochs=1, seed=42, workers=1,
            bucket=BUCKET,
        )
        self.online_sanity(model)

    def test_cbow_neg_online(self):
        model = FT_gensim(
            sg=0, cbow_mean=1, alpha=0.05, window=2, hs=0, negative=5,
            min_count=5, epochs=1, seed=42, workers=1, sample=0, bucket=BUCKET
        )
        self.online_sanity(model)

    def test_get_vocab_word_vecs(self):
        model = FT_gensim(vector_size=12, min_count=1, seed=42, bucket=BUCKET)
        model.build_vocab(sentences)
        original_syn0_vocab = np.copy(model.wv.vectors_vocab)
        model.wv.adjust_vectors()
        self.assertTrue(np.all(np.equal(model.wv.vectors_vocab, original_syn0_vocab)))

    def test_persistence_word2vec_format(self):
        """Test storing/loading the model in word2vec format."""
        tmpf = get_tmpfile('gensim_fasttext_w2v_format.tst')
        model = FT_gensim(sentences, min_count=1, vector_size=12, bucket=BUCKET)
        model.wv.save_word2vec_format(tmpf, binary=True)
        loaded_model_kv = KeyedVectors.load_word2vec_format(tmpf, binary=True)
        self.assertEqual(len(model.wv), len(loaded_model_kv))
        self.assertTrue(np.allclose(model.wv['human'], loaded_model_kv['human']))

    def test_bucket_ngrams(self):
        model = FT_gensim(vector_size=12, min_count=1, bucket=20)
        model.build_vocab(sentences)
        self.assertEqual(model.wv.vectors_ngrams.shape, (20, 12))
        model.build_vocab(new_sentences, update=True)
        self.assertEqual(model.wv.vectors_ngrams.shape, (20, 12))

    def test_estimate_memory(self):
        model = FT_gensim(sg=1, hs=1, vector_size=12, negative=5, min_count=3, bucket=BUCKET)
        model.build_vocab(sentences)
        report = model.estimate_memory()
        self.assertEqual(report['vocab'], 2800)
        self.assertEqual(report['syn0_vocab'], 192)
        self.assertEqual(report['syn1'], 192)
        self.assertEqual(report['syn1neg'], 192)
        # TODO: these fixed numbers for particular implementation generations encumber changes without real QA
        # perhaps instead verify reports' total is within some close factor of a deep-audit of actual memory used?
        self.assertEqual(report['syn0_ngrams'], model.vector_size * np.dtype(np.float32).itemsize * BUCKET)
        self.assertEqual(report['buckets_word'], 688)
        self.assertEqual(report['total'], 484064)

    def obsolete_testLoadOldModel(self):
        """Test loading fasttext models from previous version"""

        model_file = 'fasttext_old'
        model = FT_gensim.load(datapath(model_file))
        self.assertTrue(model.wv.vectors.shape == (12, 100))
        self.assertTrue(len(model.wv) == 12)
        self.assertTrue(len(model.wv.index_to_key) == 12)
        self.assertIsNone(model.corpus_total_words)
        self.assertTrue(model.syn1neg.shape == (len(model.wv), model.vector_size))
        self.assertTrue(model.wv.vectors_lockf.shape == (12, ))
        self.assertTrue(model.cum_table.shape == (12, ))

        self.assertEqual(model.wv.vectors_vocab.shape, (12, 100))
        self.assertEqual(model.wv.vectors_ngrams.shape, (2000000, 100))

        # Model stored in multiple files
        model_file = 'fasttext_old_sep'
        model = FT_gensim.load(datapath(model_file))
        self.assertTrue(model.wv.vectors.shape == (12, 100))
        self.assertTrue(len(model.wv) == 12)
        self.assertTrue(len(model.wv.index_to_key) == 12)
        self.assertIsNone(model.corpus_total_words)
        self.assertTrue(model.syn1neg.shape == (len(model.wv), model.vector_size))
        self.assertTrue(model.wv.vectors_lockf.shape == (12, ))
        self.assertTrue(model.cum_table.shape == (12, ))

        self.assertEqual(model.wv.vectors_vocab.shape, (12, 100))
        self.assertEqual(model.wv.vectors_ngrams.shape, (2000000, 100))

    def test_vectors_for_all_with_inference(self):
        """Test vectors_for_all can infer new vectors."""
        words = [
            'responding',
            'approached',
            'chairman',
            'an out-of-vocabulary word',
            'another out-of-vocabulary word',
        ]
        vectors_for_all = self.test_model.wv.vectors_for_all(words)

        expected = 5
        predicted = len(vectors_for_all)
        assert expected == predicted

        expected = self.test_model.wv['responding']
        predicted = vectors_for_all['responding']
        assert np.allclose(expected, predicted)

        smaller_distance = np.linalg.norm(
            vectors_for_all['an out-of-vocabulary word']
            - vectors_for_all['another out-of-vocabulary word']
        )
        greater_distance = np.linalg.norm(
            vectors_for_all['an out-of-vocabulary word']
            - vectors_for_all['responding']
        )
        assert greater_distance > smaller_distance

    def test_vectors_for_all_without_inference(self):
        """Test vectors_for_all does not infer new vectors when prohibited."""
        words = [
            'responding',
            'approached',
            'chairman',
            'an out-of-vocabulary word',
            'another out-of-vocabulary word',
        ]
        vectors_for_all = self.test_model.wv.vectors_for_all(words, allow_inference=False)

        expected = 3
        predicted = len(vectors_for_all)
        assert expected == predicted

        expected = self.test_model.wv['responding']
        predicted = vectors_for_all['responding']
        assert np.allclose(expected, predicted)

    def test_negative_ns_exp(self):
        """The model should accept a negative ns_exponent as a valid value."""
        model = FT_gensim(sentences, ns_exponent=-1, min_count=1, workers=1)
        tmpf = get_tmpfile('fasttext_negative_exp.tst')
        model.save(tmpf)
        loaded_model = FT_gensim.load(tmpf)
        loaded_model.train(sentences, total_examples=model.corpus_count, epochs=1)
        assert loaded_model.ns_exponent == -1, loaded_model.ns_exponent


@pytest.mark.parametrize('shrink_windows', [True, False])
def test_cbow_hs_training(shrink_windows):
    model_gensim = FT_gensim(
        vector_size=48, sg=0, cbow_mean=1, alpha=0.05, window=5, hs=1, negative=0,
        min_count=5, epochs=10, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
        sorted_vocab=1, workers=1, min_alpha=0.0, bucket=BUCKET, shrink_windows=shrink_windows)

    lee_data = LineSentence(datapath('lee_background.cor'))
    model_gensim.build_vocab(lee_data)
    orig0 = np.copy(model_gensim.wv.vectors[0])
    model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.epochs)
    assert not (orig0 == model_gensim.wv.vectors[0]).all()  # vector should vary after training

    sims_gensim = model_gensim.wv.most_similar('night', topn=10)
    sims_gensim_words = [word for (word, distance) in sims_gensim]  # get similar words
    expected_sims_words = [
        u'night,',
        u'night.',
        u'rights',
        u'kilometres',
        u'in',
        u'eight',
        u'according',
        u'flights',
        u'during',
        u'comes']
    overlaps = set(sims_gensim_words).intersection(expected_sims_words)
    overlap_count = len(overlaps)

    message = f"only {overlap_count} overlap in expected {expected_sims_words} & actual {sims_gensim_words}"
    assert overlap_count >= 2, message


@pytest.mark.parametrize('shrink_windows', [True, False])
def test_cbow_hs_training_fromfile(shrink_windows):
    with temporary_file('gensim_fasttext.tst') as corpus_file:
        model_gensim = FT_gensim(
            vector_size=48, sg=0, cbow_mean=1, alpha=0.05, window=5, hs=1, negative=0,
            min_count=5, epochs=10, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
            sorted_vocab=1, workers=1, min_alpha=0.0, bucket=BUCKET * 4, shrink_windows=shrink_windows)

        lee_data = LineSentence(datapath('lee_background.cor'))
        utils.save_as_line_sentence(lee_data, corpus_file)

        model_gensim.build_vocab(corpus_file=corpus_file)
        orig0 = np.copy(model_gensim.wv.vectors[0])
        model_gensim.train(corpus_file=corpus_file,
                           total_words=model_gensim.corpus_total_words,
                           epochs=model_gensim.epochs)
        assert not (orig0 == model_gensim.wv.vectors[0]).all()  # vector should vary after training

        sims_gensim = model_gensim.wv.most_similar('night', topn=10)
        sims_gensim_words = [word for (word, distance) in sims_gensim]  # get similar words
        expected_sims_words = [
            u'night,',
            u'night.',
            u'rights',
            u'kilometres',
            u'in',
            u'eight',
            u'according',
            u'flights',
            u'during',
            u'comes']
        overlaps = set(sims_gensim_words).intersection(expected_sims_words)
        overlap_count = len(overlaps)
        message = f"only {overlap_count} overlap in expected {expected_sims_words} & actual {sims_gensim_words}"
        assert overlap_count >= 2, message


@pytest.mark.parametrize('shrink_windows', [True, False])
def test_sg_hs_training(shrink_windows):
    model_gensim = FT_gensim(
        vector_size=48, sg=1, cbow_mean=1, alpha=0.025, window=5, hs=1, negative=0,
        min_count=5, epochs=10, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
        sorted_vocab=1, workers=1, min_alpha=0.0, bucket=BUCKET, shrink_windows=shrink_windows)

    lee_data = LineSentence(datapath('lee_background.cor'))
    model_gensim.build_vocab(lee_data)
    orig0 = np.copy(model_gensim.wv.vectors[0])
    model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.epochs)
    assert not (orig0 == model_gensim.wv.vectors[0]).all()  # vector should vary after training

    sims_gensim = model_gensim.wv.most_similar('night', topn=10)
    sims_gensim_words = [word for (word, distance) in sims_gensim]  # get similar words
    expected_sims_words = [
        u'night,',
        u'night.',
        u'eight',
        u'nine',
        u'overnight',
        u'crew',
        u'overnight.',
        u'manslaughter',
        u'north',
        u'flight']
    overlaps = set(sims_gensim_words).intersection(expected_sims_words)
    overlap_count = len(overlaps)

    message = f"only {overlap_count} overlap in expected {expected_sims_words} & actual {sims_gensim_words}"
    assert overlap_count >= 2, message


@pytest.mark.parametrize('shrink_windows', [True, False])
def test_sg_hs_training_fromfile(shrink_windows):
    with temporary_file('gensim_fasttext.tst') as corpus_file:
        model_gensim = FT_gensim(
            vector_size=48, sg=1, cbow_mean=1, alpha=0.025, window=5, hs=1, negative=0,
            min_count=5, epochs=10, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
            sorted_vocab=1, workers=1, min_alpha=0.0, bucket=BUCKET, shrink_windows=shrink_windows)

        lee_data = LineSentence(datapath('lee_background.cor'))
        utils.save_as_line_sentence(lee_data, corpus_file)

        model_gensim.build_vocab(corpus_file=corpus_file)
        orig0 = np.copy(model_gensim.wv.vectors[0])
        model_gensim.train(corpus_file=corpus_file,
                           total_words=model_gensim.corpus_total_words,
                           epochs=model_gensim.epochs)
        assert not (orig0 == model_gensim.wv.vectors[0]).all()  # vector should vary after training

        sims_gensim = model_gensim.wv.most_similar('night', topn=10)
        sims_gensim_words = [word for (word, distance) in sims_gensim]  # get similar words
        expected_sims_words = [
            u'night,',
            u'night.',
            u'eight',
            u'nine',
            u'overnight',
            u'crew',
            u'overnight.',
            u'manslaughter',
            u'north',
            u'flight']
        overlaps = set(sims_gensim_words).intersection(expected_sims_words)
        overlap_count = len(overlaps)
        message = f"only {overlap_count} overlap in expected {expected_sims_words} & actual {sims_gensim_words}"
        assert overlap_count >= 2, message


with open(datapath('toy-data.txt')) as fin:
    TOY_SENTENCES = [fin.read().strip().split(' ')]


def train_gensim(bucket=100, min_count=5):
    #
    # Set parameters to match those in the load_native function
    #
    model = FT_gensim(bucket=bucket, vector_size=5, alpha=0.05, workers=1, sample=0.0001, min_count=min_count)
    model.build_vocab(TOY_SENTENCES)
    model.train(TOY_SENTENCES, total_examples=len(TOY_SENTENCES), epochs=model.epochs)
    return model


def load_native():
    #
    # trained using:
    #
    # ./fasttext cbow -input toy-data.txt -output toy-model -bucket 100 -dim 5
    #
    path = datapath('toy-model.bin')
    model = gensim.models.fasttext.load_facebook_model(path)
    return model


def load_vec(fin):
    fin.readline()  # array shape
    for line in fin:
        columns = line.strip().split(u' ')
        word = columns.pop(0)
        vector = [float(c) for c in columns]
        yield word, np.array(vector, dtype=np.float32)


def compare_wv(a, b, t):
    a_count = {key: a.get_vecattr(key, 'count') for key in a.key_to_index}
    b_count = {key: b.get_vecattr(key, 'count') for key in b.key_to_index}
    t.assertEqual(a_count, b_count)

    #
    # We do not compare most matrices directly, because they will never
    # be equal unless many conditions are strictly controlled.
    #
    t.assertEqual(a.vectors.shape, b.vectors.shape)
    # t.assertTrue(np.allclose(a.vectors, b.vectors))

    t.assertEqual(a.vectors_vocab.shape, b.vectors_vocab.shape)
    # t.assertTrue(np.allclose(a.vectors_vocab, b.vectors_vocab))


def compare_nn(a, b, t):
    #
    # Ensure the neural networks are identical for both cases.
    #
    t.assertEqual(a.syn1neg.shape, b.syn1neg.shape)

    #
    # Only if match_gensim=True in init_post_load
    #
    # t.assertEqual(a.vectors_ngrams_lockf.shape, b.vectors_ngrams_lockf.shape)
    # t.assertTrue(np.allclose(a.vectors_ngrams_lockf, b.vectors_ngrams_lockf))

    # t.assertEqual(a.vectors_vocab_lockf.shape, b.vectors_vocab_lockf.shape)
    # t.assertTrue(np.allclose(a.vectors_vocab_lockf, b.vectors_vocab_lockf))


def compare_vocabulary(a, b, t):
    t.assertEqual(a.max_vocab_size, b.max_vocab_size)
    t.assertEqual(a.min_count, b.min_count)
    t.assertEqual(a.sample, b.sample)
    t.assertEqual(a.sorted_vocab, b.sorted_vocab)
    t.assertEqual(a.null_word, b.null_word)
    t.assertTrue(np.allclose(a.cum_table, b.cum_table))
    t.assertEqual(a.raw_vocab, b.raw_vocab)
    t.assertEqual(a.max_final_vocab, b.max_final_vocab)
    t.assertEqual(a.ns_exponent, b.ns_exponent)


class NativeTrainingContinuationTest(unittest.TestCase):
    maxDiff = None
    model_structural_sanity = TestFastTextModel.model_structural_sanity

    def setUp(self):
        #
        # $ echo "quick brown fox jumps over lazy dog" | ./fasttext print-word-vectors gensim/test/test_data/toy-model.bin  # noqa: E501
        #
        expected = {
            u"quick": [0.023393, 0.11499, 0.11684, -0.13349, 0.022543],
            u"brown": [0.015288, 0.050404, -0.041395, -0.090371, 0.06441],
            u"fox": [0.061692, 0.082914, 0.020081, -0.039159, 0.03296],
            u"jumps": [0.070107, 0.081465, 0.051763, 0.012084, 0.0050402],
            u"over": [0.055023, 0.03465, 0.01648, -0.11129, 0.094555],
            u"lazy": [-0.022103, -0.020126, -0.033612, -0.049473, 0.0054174],
            u"dog": [0.084983, 0.09216, 0.020204, -0.13616, 0.01118],
        }
        self.oov_expected = {
            word: np.array(arr, dtype=np.float32)
            for word, arr in expected.items()
        }

    def test_in_vocab(self):
        """Test for correct representation of in-vocab words."""
        native = load_native()
        with utils.open(datapath('toy-model.vec'), 'r', encoding='utf-8') as fin:
            expected = dict(load_vec(fin))

        for word, expected_vector in expected.items():
            actual_vector = native.wv.get_vector(word)
            self.assertTrue(np.allclose(expected_vector, actual_vector, atol=1e-5))

        self.model_structural_sanity(native)

    def test_out_of_vocab(self):
        """Test for correct representation of out-of-vocab words."""
        native = load_native()

        for word, expected_vector in self.oov_expected.items():
            actual_vector = native.wv.get_vector(word)
            self.assertTrue(np.allclose(expected_vector, actual_vector, atol=1e-5))

        self.model_structural_sanity(native)

    def test_sanity(self):
        """Compare models trained on toy data.  They should be equal."""
        trained = train_gensim()
        native = load_native()

        self.assertEqual(trained.wv.bucket, native.wv.bucket)
        #
        # Only if match_gensim=True in init_post_load
        #
        # self.assertEqual(trained.bucket, native.bucket)

        compare_wv(trained.wv, native.wv, self)
        compare_vocabulary(trained, native, self)
        compare_nn(trained, native, self)

        self.model_structural_sanity(trained)
        self.model_structural_sanity(native)

    def test_continuation_native(self):
        """Ensure that training has had a measurable effect."""
        native = load_native()
        self.model_structural_sanity(native)

        #
        # Pick a word that is in both corpuses.
        # Its vectors should be different between training runs.
        #
        word = 'society'
        old_vector = native.wv.get_vector(word).tolist()

        native.train(list_corpus, total_examples=len(list_corpus), epochs=native.epochs)

        new_vector = native.wv.get_vector(word).tolist()
        self.assertNotEqual(old_vector, new_vector)
        self.model_structural_sanity(native)

    def test_continuation_gensim(self):
        """Ensure that continued training has had a measurable effect."""
        model = train_gensim(min_count=0)
        self.model_structural_sanity(model)
        vectors_ngrams_before = np.copy(model.wv.vectors_ngrams)

        word = 'human'
        old_vector = model.wv.get_vector(word).tolist()

        model.train(list_corpus, total_examples=len(list_corpus), epochs=model.epochs)

        vectors_ngrams_after = np.copy(model.wv.vectors_ngrams)
        self.assertFalse(np.allclose(vectors_ngrams_before, vectors_ngrams_after))
        new_vector = model.wv.get_vector(word).tolist()

        self.assertNotEqual(old_vector, new_vector)
        self.model_structural_sanity(model)

    def test_save_load_gensim(self):
        """Test that serialization works end-to-end.  Not crashing is a success."""
        #
        # This is a workaround for a problem with temporary files on AppVeyor:
        #
        # - https://bugs.python.org/issue14243 (problem discussion)
        # - https://github.com/dropbox/pyannotate/pull/48/files (workaround source code)
        #
        model_name = 'test_ft_saveload_native.model'

        with temporary_file(model_name):
            train_gensim().save(model_name)

            model = FT_gensim.load(model_name)
            self.model_structural_sanity(model)
            model.train(list_corpus, total_examples=len(list_corpus), epochs=model.epochs)

            model.save(model_name)
            self.model_structural_sanity(model)

    def test_save_load_native(self):
        """Test that serialization works end-to-end.  Not crashing is a success."""

        model_name = 'test_ft_saveload_fb.model'

        with temporary_file(model_name):
            load_native().save(model_name)

            model = FT_gensim.load(model_name)
            self.model_structural_sanity(model)
            model.train(list_corpus, total_examples=len(list_corpus), epochs=model.epochs)

            model.save(model_name)
            self.model_structural_sanity(model)

    def test_load_native_pretrained(self):
        model = gensim.models.fasttext.load_facebook_model(datapath('toy-model-pretrained.bin'))
        actual = model.wv['monarchist']
        expected = np.array([0.76222, 1.0669, 0.7055, -0.090969, -0.53508])
        self.assertTrue(np.allclose(expected, actual, atol=10e-4))
        self.model_structural_sanity(model)

    def test_load_native_vectors(self):
        cap_path = datapath("crime-and-punishment.bin")
        fbkv = gensim.models.fasttext.load_facebook_vectors(cap_path)
        self.assertFalse('landlord' in fbkv.key_to_index)
        self.assertTrue('landlady' in fbkv.key_to_index)
        oov_vector = fbkv['landlord']
        iv_vector = fbkv['landlady']
        self.assertFalse(np.allclose(oov_vector, iv_vector))

    def test_no_ngrams(self):
        model = gensim.models.fasttext.load_facebook_model(datapath('crime-and-punishment.bin'))

        v1 = model.wv['']
        origin = np.zeros(v1.shape, v1.dtype)
        self.assertTrue(np.allclose(v1, origin))
        self.model_structural_sanity(model)


def _train_model_with_pretrained_vectors():
    """Generate toy-model-pretrained.bin for use in test_load_native_pretrained.

    Requires https://github.com/facebookresearch/fastText/tree/master/python to be installed.

    """
    import fastText

    training_text = datapath('toy-data.txt')
    pretrained_file = datapath('pretrained.vec')
    model = fastText.train_unsupervised(
        training_text,
        bucket=100, model='skipgram', dim=5, pretrainedVectors=pretrained_file
    )
    model.save_model(datapath('toy-model-pretrained.bin'))


class HashCompatibilityTest(unittest.TestCase):
    def test_compatibility_true(self):
        m = FT_gensim.load(datapath('compatible-hash-true.model'))
        self.assertTrue(m.wv.compatible_hash)

    def test_hash_native(self):
        m = load_native()
        self.assertTrue(m.wv.compatible_hash)


class FTHashResultsTest(unittest.TestCase):
    """Loosely based on the test described here:

    https://github.com/RaRe-Technologies/gensim/issues/2059#issuecomment-432300777

    With a broken hash, vectors for non-ASCII keywords don't match when loaded
    from a native model.
    """
    def setUp(self):
        #
        # ./fasttext skipgram -minCount 0 -bucket 100 -input crime-and-punishment.txt -output crime-and-punishment -dim 5  # noqa: E501
        #
        self.model = gensim.models.fasttext.load_facebook_model(datapath('crime-and-punishment.bin'))
        with utils.open(datapath('crime-and-punishment.vec'), 'r', encoding='utf-8') as fin:
            self.expected = dict(load_vec(fin))

    def test_ascii(self):
        word = u'landlady'
        expected = self.expected[word]
        actual = self.model.wv[word]
        self.assertTrue(np.allclose(expected, actual, atol=1e-5))

    def test_unicode(self):
        word = u'хозяйка'
        expected = self.expected[word]
        actual = self.model.wv[word]
        self.assertTrue(np.allclose(expected, actual, atol=1e-5))

    def test_out_of_vocab(self):
        longword = u'rechtsschutzversicherungsgesellschaften'  # many ngrams
        expected = {
            u'steamtrain': np.array([0.031988, 0.022966, 0.059483, 0.094547, 0.062693]),
            u'паровоз': np.array([-0.0033987, 0.056236, 0.036073, 0.094008, 0.00085222]),
            longword: np.array([-0.012889, 0.029756, 0.018020, 0.099077, 0.041939]),
        }
        actual = {w: self.model.wv[w] for w in expected}
        self.assertTrue(np.allclose(expected[u'steamtrain'], actual[u'steamtrain'], atol=1e-5))
        self.assertTrue(np.allclose(expected[u'паровоз'], actual[u'паровоз'], atol=1e-5))
        self.assertTrue(np.allclose(expected[longword], actual[longword], atol=1e-5))


def hash_main(alg):
    """Generate hash values for test from standard input."""
    hashmap = {
        'cy_bytes': ft_hash_bytes,
    }
    try:
        fun = hashmap[alg]
    except KeyError:
        raise KeyError('invalid alg: %r expected one of %r' % (alg, sorted(hashmap)))

    for line in sys.stdin:
        if 'bytes' in alg:
            words = line.encode('utf-8').rstrip().split(b' ')
        else:
            words = line.rstrip().split(' ')
        for word in words:
            print('u%r: %r,' % (word, fun(word)))


class FTHashFunctionsTest(unittest.TestCase):
    def setUp(self):
        #
        # I obtained these expected values using:
        #
        # $ echo word1 ... wordN | python -c 'from gensim.test.test_fasttext import hash_main;hash_main("alg")'  # noqa: E501
        #
        # where alg is cy_bytes (previous options had included: py_bytes, py_broken, cy_bytes, cy_broken.)

        #
        self.expected = {
            u'команда': 1725507386,
            u'маленьких': 3011324125,
            u'друзей': 737001801,
            u'возит': 4225261911,
            u'грузы': 1301826944,
            u'всех': 706328732,
            u'быстрей': 1379730754,
            u'mysterious': 1903186891,
            u'asteroid': 1988297200,
            u'odyssey': 310195777,
            u'introduction': 2848265721,
            u'北海道': 4096045468,
            u'札幌': 3909947444,
            u'西区': 3653372632,
        }

    def test_cython(self):
        actual = {k: ft_hash_bytes(k.encode('utf-8')) for k in self.expected}
        self.assertEqual(self.expected, actual)


#
# Run with:
#
#   python -c 'import gensim.test.test_fasttext as t;t.ngram_main()' py_text 3 5
#
def ngram_main():
    """Generate ngrams for tests from standard input."""

    alg = sys.argv[1]
    minn = int(sys.argv[2])
    maxn = int(sys.argv[3])

    assert minn <= maxn, 'expected sane command-line parameters'

    hashmap = {
        'cy_text': compute_ngrams,
        'cy_bytes': compute_ngrams_bytes,
    }
    try:
        fun = hashmap[alg]
    except KeyError:
        raise KeyError('invalid alg: %r expected one of %r' % (alg, sorted(hashmap)))

    for line in sys.stdin:
        word = line.rstrip('\n')
        ngrams = fun(word, minn, maxn)
        print("%r: %r," % (word, ngrams))


class NgramsTest(unittest.TestCase):
    def setUp(self):
        self.expected_text = {
            'test': ['<te', 'tes', 'est', 'st>', '<tes', 'test', 'est>', '<test', 'test>'],
            'at the': [
                '<at', 'at ', 't t', ' th', 'the', 'he>',
                '<at ', 'at t', 't th', ' the', 'the>', '<at t', 'at th', 't the', ' the>'
            ],
            'at\nthe': [
                '<at', 'at\n', 't\nt', '\nth', 'the', 'he>',
                '<at\n', 'at\nt', 't\nth', '\nthe', 'the>', '<at\nt', 'at\nth', 't\nthe', '\nthe>'
            ],
            'тест': ['<те', 'тес', 'ест', 'ст>', '<тес', 'тест', 'ест>', '<тест', 'тест>'],
            'テスト': ['<テス', 'テスト', 'スト>', '<テスト', 'テスト>', '<テスト>'],
            '試し': ['<試し', '試し>', '<試し>'],
        }
        self.expected_bytes = {
            'test': [b'<te', b'<tes', b'<test', b'tes', b'test', b'test>', b'est', b'est>', b'st>'],
            'at the': [
                b'<at', b'<at ', b'<at t', b'at ', b'at t', b'at th', b't t',
                b't th', b't the', b' th', b' the', b' the>', b'the', b'the>', b'he>'
            ],
            'тест': [
                b'<\xd1\x82\xd0\xb5', b'<\xd1\x82\xd0\xb5\xd1\x81', b'<\xd1\x82\xd0\xb5\xd1\x81\xd1\x82',
                b'\xd1\x82\xd0\xb5\xd1\x81', b'\xd1\x82\xd0\xb5\xd1\x81\xd1\x82', b'\xd1\x82\xd0\xb5\xd1\x81\xd1\x82>',
                b'\xd0\xb5\xd1\x81\xd1\x82', b'\xd0\xb5\xd1\x81\xd1\x82>', b'\xd1\x81\xd1\x82>'
            ],
            'テスト': [
                b'<\xe3\x83\x86\xe3\x82\xb9', b'<\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88',
                b'<\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88>', b'\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88',
                b'\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88>', b'\xe3\x82\xb9\xe3\x83\x88>'
            ],
            '試し': [b'<\xe8\xa9\xa6\xe3\x81\x97', b'<\xe8\xa9\xa6\xe3\x81\x97>', b'\xe8\xa9\xa6\xe3\x81\x97>'],
        }

        self.expected_text_wide_unicode = {
            '🚑🚒🚓🚕': [
                '<🚑🚒', '🚑🚒🚓', '🚒🚓🚕', '🚓🚕>',
                '<🚑🚒🚓', '🚑🚒🚓🚕', '🚒🚓🚕>', '<🚑🚒🚓🚕', '🚑🚒🚓🚕>'
             ],
        }
        self.expected_bytes_wide_unicode = {
            '🚑🚒🚓🚕': [
                b'<\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92',
                b'<\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93',
                b'<\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95',
                b'\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93',
                b'\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95',
                b'\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95>',
                b'\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95',
                b'\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95>',
                b'\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95>'
            ],
        }

    def test_text_cy(self):
        for word in self.expected_text:
            expected = self.expected_text[word]
            actual = compute_ngrams(word, 3, 5)
            self.assertEqual(expected, actual)

    @unittest.skipIf(sys.maxunicode == 0xffff, "Python interpreter doesn't support UCS-4 (wide unicode)")
    def test_text_cy_wide_unicode(self):
        for word in self.expected_text_wide_unicode:
            expected = self.expected_text_wide_unicode[word]
            actual = compute_ngrams(word, 3, 5)
            self.assertEqual(expected, actual)

    def test_bytes_cy(self):
        for word in self.expected_bytes:
            expected = self.expected_bytes[word]
            actual = compute_ngrams_bytes(word, 3, 5)
            self.assertEqual(expected, actual)

            expected_text = self.expected_text[word]
            actual_text = [n.decode('utf-8') for n in actual]
            self.assertEqual(sorted(expected_text), sorted(actual_text))

        for word in self.expected_bytes_wide_unicode:
            expected = self.expected_bytes_wide_unicode[word]
            actual = compute_ngrams_bytes(word, 3, 5)
            self.assertEqual(expected, actual)

            expected_text = self.expected_text_wide_unicode[word]
            actual_text = [n.decode('utf-8') for n in actual]
            self.assertEqual(sorted(expected_text), sorted(actual_text))

    def test_fb(self):
        """Test against results from Facebook's implementation."""
        with utils.open(datapath('fb-ngrams.txt'), 'r', encoding='utf-8') as fin:
            fb = dict(_read_fb(fin))

        for word, expected in fb.items():
            #
            # The model was trained with minn=3, maxn=6
            #
            actual = compute_ngrams(word, 3, 6)
            self.assertEqual(sorted(expected), sorted(actual))


def _read_fb(fin):
    """Read ngrams from output of the FB utility."""
    #
    # $ cat words.txt
    # test
    # at the
    # at\nthe
    # тест
    # テスト
    # 試し
    # 🚑🚒🚓🚕
    # $ while read w;
    # do
    #   echo "<start>";
    #   echo $w;
    #   ./fasttext print-ngrams gensim/test/test_data/crime-and-punishment.bin "$w";
    #   echo "<end>";
    # done < words.txt > gensim/test/test_data/fb-ngrams.txt
    #
    while fin:
        line = fin.readline().rstrip()
        if not line:
            break

        assert line == '<start>'
        word = fin.readline().rstrip()

        fin.readline()  # ignore this line, it contains an origin vector for the full term

        ngrams = []
        while True:
            line = fin.readline().rstrip()
            if line == '<end>':
                break

            columns = line.split(' ')
            term = ' '.join(columns[:-5])
            ngrams.append(term)

        yield word, ngrams


class ZeroBucketTest(unittest.TestCase):
    """Test FastText with no buckets / no-ngrams: essentially FastText-as-Word2Vec."""
    def test_in_vocab(self):
        model = train_gensim(bucket=0)
        self.assertIsNotNone(model.wv['anarchist'])

    def test_out_of_vocab(self):
        model = train_gensim(bucket=0)
        with self.assertRaises(KeyError):
            model.wv.get_vector('streamtrain')

    def test_cbow_neg(self):
        """See `gensim.test.test_word2vec.TestWord2VecModel.test_cbow_neg`."""
        model = FT_gensim(
            sg=0, cbow_mean=1, alpha=0.05, window=5, hs=0, negative=15,
            min_count=5, epochs=10, workers=2, sample=0,
            max_n=0  # force no char-ngram buckets
        )
        TestWord2VecModel.model_sanity(self, model)


class UnicodeVocabTest(unittest.TestCase):
    def test_ascii(self):
        buf = io.BytesIO()
        buf.name = 'dummy name to keep fasttext happy'
        buf.write(struct.pack('@3i', 2, -1, -1))  # vocab_size, nwords, nlabels
        buf.write(struct.pack('@1q', 10))  # ntokens
        buf.write(b'hello')
        buf.write(b'\x00')
        buf.write(struct.pack('@qb', 1, -1))
        buf.write(b'world')
        buf.write(b'\x00')
        buf.write(struct.pack('@qb', 2, -1))
        buf.seek(0)

        raw_vocab, vocab_size, nlabels, ntokens = gensim.models._fasttext_bin._load_vocab(buf, False)
        expected = {'hello': 1, 'world': 2}
        self.assertEqual(expected, dict(raw_vocab))

        self.assertEqual(vocab_size, 2)
        self.assertEqual(nlabels, -1)

        self.assertEqual(ntokens, 10)

    def test_bad_unicode(self):
        buf = io.BytesIO()
        buf.name = 'dummy name to keep fasttext happy'
        buf.write(struct.pack('@3i', 2, -1, -1))  # vocab_size, nwords, nlabels
        buf.write(struct.pack('@1q', 10))  # ntokens
        #
        # encountered in https://github.com/RaRe-Technologies/gensim/issues/2378
        # The model from downloaded from
        # https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.bin.zip
        # suffers from bad characters in a few of the vocab terms.  The native
        # fastText utility loads the model fine, but we trip up over the bad
        # characters.
        #
        buf.write(
            b'\xe8\x8b\xb1\xe8\xaa\x9e\xe7\x89\x88\xe3\x82\xa6\xe3\x82\xa3\xe3'
            b'\x82\xad\xe3\x83\x9a\xe3\x83\x87\xe3\x82\xa3\xe3\x82\xa2\xe3\x81'
            b'\xb8\xe3\x81\xae\xe6\x8a\x95\xe7\xa8\xbf\xe3\x81\xaf\xe3\x81\x84'
            b'\xe3\x81\xa4\xe3\x81\xa7\xe3\x82\x82\xe6'
        )
        buf.write(b'\x00')
        buf.write(struct.pack('@qb', 1, -1))
        buf.write(
            b'\xd0\xb0\xd0\xb4\xd0\xbc\xd0\xb8\xd0\xbd\xd0\xb8\xd1\x81\xd1\x82'
            b'\xd1\x80\xd0\xb0\xd1\x82\xd0\xb8\xd0\xb2\xd0\xbd\xd0\xbe-\xd1\x82'
            b'\xd0\xb5\xd1\x80\xd1\x80\xd0\xb8\xd1\x82\xd0\xbe\xd1\x80\xd0\xb8'
            b'\xd0\xb0\xd0\xbb\xd1\x8c\xd0\xbd\xd1'
        )
        buf.write(b'\x00')
        buf.write(struct.pack('@qb', 2, -1))
        buf.seek(0)

        raw_vocab, vocab_size, nlabels, ntokens = gensim.models._fasttext_bin._load_vocab(buf, False)

        expected = {
            u'英語版ウィキペディアへの投稿はいつでも\\xe6': 1,
            u'административно-территориальн\\xd1': 2,
        }

        self.assertEqual(expected, dict(raw_vocab))

        self.assertEqual(vocab_size, 2)
        self.assertEqual(nlabels, -1)
        self.assertEqual(ntokens, 10)


_BYTES = b'the quick brown fox jumps over the lazy dog'
_ARRAY = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.], dtype=np.dtype('float32'))


class TestFromfile(unittest.TestCase):
    def test_decompressed(self):
        with open(datapath('reproduce.dat'), 'rb') as fin:
            self._run(fin)

    def test_compressed(self):
        with gzip.GzipFile(datapath('reproduce.dat.gz'), 'rb') as fin:
            self._run(fin)

    def _run(self, fin):
        actual = fin.read(len(_BYTES))
        self.assertEqual(_BYTES, actual)

        array = gensim.models._fasttext_bin._fromfile(fin, _ARRAY.dtype, _ARRAY.shape[0])
        logger.error('array: %r', array)
        self.assertTrue(np.allclose(_ARRAY, array))


def _create_and_save_fb_model(fname, model_params):
    model = FT_gensim(**model_params)
    lee_data = LineSentence(datapath('lee_background.cor'))
    model.build_vocab(lee_data)
    model.train(lee_data, total_examples=model.corpus_count, epochs=model.epochs)
    gensim.models.fasttext.save_facebook_model(model, fname)
    return model


def calc_max_diff(v1, v2):
    return np.max(np.abs(v1 - v2))


class SaveFacebookFormatModelTest(unittest.TestCase):

    def _check_roundtrip(self, sg):
        model_params = {
            "sg": sg,
            "vector_size": 10,
            "min_count": 1,
            "hs": 1,
            "negative": 5,
            "seed": 42,
            "bucket": BUCKET,
            "workers": 1}

        with temporary_file("roundtrip_model_to_model.bin") as fpath:
            model_trained = _create_and_save_fb_model(fpath, model_params)
            model_loaded = gensim.models.fasttext.load_facebook_model(fpath)

        self.assertEqual(model_trained.vector_size, model_loaded.vector_size)
        self.assertEqual(model_trained.window, model_loaded.window)
        self.assertEqual(model_trained.epochs, model_loaded.epochs)
        self.assertEqual(model_trained.negative, model_loaded.negative)
        self.assertEqual(model_trained.hs, model_loaded.hs)
        self.assertEqual(model_trained.sg, model_loaded.sg)
        self.assertEqual(model_trained.wv.bucket, model_loaded.wv.bucket)
        self.assertEqual(model_trained.wv.min_n, model_loaded.wv.min_n)
        self.assertEqual(model_trained.wv.max_n, model_loaded.wv.max_n)
        self.assertEqual(model_trained.sample, model_loaded.sample)
        self.assertEqual(set(model_trained.wv.index_to_key), set(model_loaded.wv.index_to_key))

        for w in model_trained.wv.index_to_key:
            v_orig = model_trained.wv[w]
            v_loaded = model_loaded.wv[w]
            self.assertLess(calc_max_diff(v_orig, v_loaded), MAX_WORDVEC_COMPONENT_DIFFERENCE)

    def test_skipgram(self):
        self._check_roundtrip(sg=1)

    def test_cbow(self):
        self._check_roundtrip(sg=0)


def _read_binary_file(fname):
    with open(fname, "rb") as f:
        data = f.read()
    return data


class SaveGensimByteIdentityTest(unittest.TestCase):
    """
    This class containts tests that check the following scenario:

    + create binary fastText file model1.bin using gensim
    + load file model1.bin to variable `model`
    + save `model` to model2.bin
    + check if files model1.bin and model2.bin are byte identical
    """

    def _check_roundtrip_file_file(self, sg):
        model_params = {
            "sg": sg,
            "vector_size": 10,
            "min_count": 1,
            "hs": 1,
            "negative": 0,
            "bucket": BUCKET,
            "seed": 42,
            "workers": 1}

        with temporary_file("roundtrip_file_to_file1.bin") as fpath1, \
            temporary_file("roundtrip_file_to_file2.bin") as fpath2:
            _create_and_save_fb_model(fpath1, model_params)
            model = gensim.models.fasttext.load_facebook_model(fpath1)
            gensim.models.fasttext.save_facebook_model(model, fpath2)
            bin1 = _read_binary_file(fpath1)
            bin2 = _read_binary_file(fpath2)

        self.assertEqual(bin1, bin2)

    def test_skipgram(self):
        self._check_roundtrip_file_file(sg=1)

    def test_cbow(self):
        self._check_roundtrip_file_file(sg=0)


def _save_test_model(out_base_fname, model_params):
    inp_fname = datapath('lee_background.cor')

    model_type = "cbow" if model_params["sg"] == 0 else "skipgram"
    size = str(model_params["vector_size"])
    seed = str(model_params["seed"])

    cmd = [
        FT_CMD, model_type, "-input", inp_fname, "-output",
        out_base_fname, "-dim", size, "-seed", seed]

    subprocess.check_call(cmd)


@unittest.skipIf(not FT_CMD, "fasttext not in FT_HOME or PATH, skipping test")
class SaveFacebookByteIdentityTest(unittest.TestCase):
    """
    This class containts tests that check the following scenario:

    + create binary fastText file model1.bin using facebook_binary (FT)
    + load file model1.bin to variable `model`
    + save `model` to model2.bin using gensim
    + check if files model1.bin and model2.bin are byte-identical
    """

    def _check_roundtrip_file_file(self, sg):
        model_params = {"vector_size": 10, "sg": sg, "seed": 42}

        # fasttext tool creates both *vec and *bin files, so we have to remove both, even thought *vec is unused

        with temporary_file("m1.bin") as m1, temporary_file("m2.bin") as m2, temporary_file("m1.vec"):

            m1_basename = m1[:-4]
            _save_test_model(m1_basename, model_params)
            model = gensim.models.fasttext.load_facebook_model(m1)
            gensim.models.fasttext.save_facebook_model(model, m2)
            bin1 = _read_binary_file(m1)
            bin2 = _read_binary_file(m2)

        self.assertEqual(bin1, bin2)

    def test_skipgram(self):
        self._check_roundtrip_file_file(sg=1)

    def test_cbow(self):
        self._check_roundtrip_file_file(sg=0)


def _read_wordvectors_using_fasttext(fasttext_fname, words):
    def line_to_array(line):
        return np.array([float(s) for s in line.split()[1:]], dtype=np.float32)

    cmd = [FT_CMD, "print-word-vectors", fasttext_fname]
    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE)
    words_str = '\n'.join(words)
    out, _ = process.communicate(input=words_str.encode("utf-8"))
    return np.array([line_to_array(line) for line in out.splitlines()], dtype=np.float32)


@unittest.skipIf(not FT_CMD, "fasttext not in FT_HOME or PATH, skipping test")
class SaveFacebookFormatReadingTest(unittest.TestCase):
    """
    This class containts tests that check the following scenario:

    + create fastText model using gensim
    + save file to model.bin
    + retrieve word vectors from model.bin using fasttext Facebook utility
    + compare vectors retrieved by Facebook utility with those obtained directly from gensim model
    """

    def _check_load_fasttext_format(self, sg):
        model_params = {
            "sg": sg,
            "vector_size": 10,
            "min_count": 1,
            "hs": 1,
            "negative": 5,
            "bucket": BUCKET,
            "seed": 42,
            "workers": 1}

        with temporary_file("load_fasttext.bin") as fpath:
            model = _create_and_save_fb_model(fpath, model_params)
            wv = _read_wordvectors_using_fasttext(fpath, model.wv.index_to_key)

        for i, w in enumerate(model.wv.index_to_key):
            diff = calc_max_diff(wv[i, :], model.wv[w])
            # Because fasttext command line prints vectors with limited accuracy
            self.assertLess(diff, 1.0e-4)

    def test_skipgram(self):
        self._check_load_fasttext_format(sg=1)

    def test_cbow(self):
        self._check_load_fasttext_format(sg=0)


class UnpackTest(unittest.TestCase):
    def test_sanity(self):
        m = np.array(range(9))
        m.shape = (3, 3)
        hash2index = {10: 0, 11: 1, 12: 2}

        n = _unpack(m, 25, hash2index)
        self.assertTrue(np.all(np.array([0, 1, 2]) == n[10]))
        self.assertTrue(np.all(np.array([3, 4, 5]) == n[11]))
        self.assertTrue(np.all(np.array([6, 7, 8]) == n[12]))

    def test_tricky(self):
        m = np.array(range(9))
        m.shape = (3, 3)
        hash2index = {1: 0, 0: 1, 12: 2}

        n = _unpack(m, 25, hash2index)
        self.assertTrue(np.all(np.array([3, 4, 5]) == n[0]))
        self.assertTrue(np.all(np.array([0, 1, 2]) == n[1]))
        self.assertTrue(np.all(np.array([6, 7, 8]) == n[12]))

    def test_identity(self):
        m = np.array(range(9))
        m.shape = (3, 3)
        hash2index = {0: 0, 1: 1, 2: 2}

        n = _unpack(m, 25, hash2index)
        self.assertTrue(np.all(np.array([0, 1, 2]) == n[0]))
        self.assertTrue(np.all(np.array([3, 4, 5]) == n[1]))
        self.assertTrue(np.all(np.array([6, 7, 8]) == n[2]))


class FastTextKeyedVectorsTest(unittest.TestCase):
    def test_add_vector(self):
        wv = FastTextKeyedVectors(vector_size=2, min_n=3, max_n=6, bucket=2000000)
        wv.add_vector("test_key", np.array([0, 0]))

        self.assertEqual(wv.key_to_index["test_key"], 0)
        self.assertEqual(wv.index_to_key[0], "test_key")
        self.assertTrue(np.all(wv.vectors[0] == np.array([0, 0])))

    def test_add_vectors(self):
        wv = FastTextKeyedVectors(vector_size=2, min_n=3, max_n=6, bucket=2000000)
        wv.add_vectors(["test_key1", "test_key2"], np.array([[0, 0], [1, 1]]))

        self.assertEqual(wv.key_to_index["test_key1"], 0)
        self.assertEqual(wv.index_to_key[0], "test_key1")
        self.assertTrue(np.all(wv.vectors[0] == np.array([0, 0])))

        self.assertEqual(wv.key_to_index["test_key2"], 1)
        self.assertEqual(wv.index_to_key[1], "test_key2")
        self.assertTrue(np.all(wv.vectors[1] == np.array([1, 1])))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
