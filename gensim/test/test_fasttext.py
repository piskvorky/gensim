#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import gzip
import io
import logging
import unittest
import os
import struct
import six

import numpy as np

from gensim import utils
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim
from gensim.models.wrappers.fasttext import FastTextKeyedVectors
from gensim.models.wrappers.fasttext import FastText as FT_wrapper
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as sentences
import gensim.models._fasttext_bin


import gensim.models.fasttext

try:
    from pyemd import emd  # noqa:F401
    PYEMD_EXT = True
except (ImportError, ValueError):
    PYEMD_EXT = False

logger = logging.getLogger(__name__)

IS_WIN32 = (os.name == "nt") and (struct.calcsize('P') * 8 == 32)


class LeeCorpus(object):
    def __iter__(self):
        with open(datapath('lee_background.cor')) as f:
            for line in f:
                yield utils.simple_preprocess(line)


list_corpus = list(LeeCorpus())

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
        ft_home = os.environ.get('FT_HOME', None)
        self.ft_path = os.path.join(ft_home, 'fasttext') if ft_home else None
        self.test_model_file = datapath('lee_fasttext.bin')
        self.test_model = gensim.models.fasttext.load_facebook_model(self.test_model_file)
        self.test_new_model_file = datapath('lee_fasttext_new.bin')

    def test_training(self):
        model = FT_gensim(size=10, min_count=1, hs=1, negative=0, seed=42, workers=1)
        model.build_vocab(sentences)
        self.model_sanity(model)

        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        sims = model.wv.most_similar('graph', topn=10)

        self.assertEqual(model.wv.vectors.shape, (12, 10))
        self.assertEqual(len(model.wv.vocab), 12)
        self.assertEqual(model.wv.vectors_vocab.shape[1], 10)
        self.assertEqual(model.wv.vectors_ngrams.shape[1], 10)
        self.model_sanity(model)

        # test querying for "most similar" by vector
        graph_vector = model.wv.vectors_norm[model.wv.vocab['graph'].index]
        sims2 = model.wv.most_similar(positive=[graph_vector], topn=11)
        sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
        self.assertEqual(sims, sims2)

        # build vocab and train in one step; must be the same as above
        model2 = FT_gensim(sentences, size=10, min_count=1, hs=1, negative=0, seed=42, workers=1)
        self.models_equal(model, model2)

        # verify oov-word vector retrieval
        invocab_vec = model.wv['minors']  # invocab word
        self.assertEqual(len(invocab_vec), 10)

        oov_vec = model.wv['minor']  # oov word
        self.assertEqual(len(oov_vec), 10)

    def testFastTextTrainParameters(self):

        model = FT_gensim(size=10, min_count=1, hs=1, negative=0, seed=42, workers=1)
        model.build_vocab(sentences=sentences)

        self.assertRaises(TypeError, model.train, corpus_file=11111)
        self.assertRaises(TypeError, model.train, sentences=11111)
        self.assertRaises(TypeError, model.train, sentences=sentences, corpus_file='test')
        self.assertRaises(TypeError, model.train, sentences=None, corpus_file=None)
        self.assertRaises(TypeError, model.train, corpus_file=sentences)

    @unittest.skipIf(os.name == 'nt' and six.PY2, "corpus_file training is not supported on Windows + Py27")
    def test_training_fromfile(self):
        with temporary_file(get_tmpfile('gensim_fasttext.tst')) as corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)

            model = FT_gensim(size=10, min_count=1, hs=1, negative=0, seed=42, workers=1)
            model.build_vocab(corpus_file=corpus_file)
            self.model_sanity(model)

            model.train(corpus_file=corpus_file, total_words=model.corpus_total_words, epochs=model.epochs)
            sims = model.wv.most_similar('graph', topn=10)

            self.assertEqual(model.wv.vectors.shape, (12, 10))
            self.assertEqual(len(model.wv.vocab), 12)
            self.assertEqual(model.wv.vectors_vocab.shape[1], 10)
            self.assertEqual(model.wv.vectors_ngrams.shape[1], 10)
            self.model_sanity(model)

            # test querying for "most similar" by vector
            graph_vector = model.wv.vectors_norm[model.wv.vocab['graph'].index]
            sims2 = model.wv.most_similar(positive=[graph_vector], topn=11)
            sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
            self.assertEqual(sims, sims2)

            # verify oov-word vector retrieval
            invocab_vec = model.wv['minors']  # invocab word
            self.assertEqual(len(invocab_vec), 10)

            oov_vec = model.wv['minor']  # oov word
            self.assertEqual(len(oov_vec), 10)

    def models_equal(self, model, model2):
        self.assertEqual(len(model.wv.vocab), len(model2.wv.vocab))
        self.assertEqual(model.wv.num_ngram_vectors, model2.wv.num_ngram_vectors)
        self.assertTrue(np.allclose(model.wv.vectors_vocab, model2.wv.vectors_vocab))
        self.assertTrue(np.allclose(model.wv.vectors_ngrams, model2.wv.vectors_ngrams))
        self.assertTrue(np.allclose(model.wv.vectors, model2.wv.vectors))
        if model.hs:
            self.assertTrue(np.allclose(model.trainables.syn1, model2.trainables.syn1))
        if model.negative:
            self.assertTrue(np.allclose(model.trainables.syn1neg, model2.trainables.syn1neg))
        most_common_word = max(model.wv.vocab.items(), key=lambda item: item[1].count)[0]
        self.assertTrue(np.allclose(model.wv[most_common_word], model2.wv[most_common_word]))

    @unittest.skipIf(IS_WIN32, "avoid memory error with Appveyor x32")
    def test_persistence(self):
        tmpf = get_tmpfile('gensim_fasttext.tst')
        model = FT_gensim(sentences, min_count=1)
        model.save(tmpf)
        self.models_equal(model, FT_gensim.load(tmpf))
        #  test persistence of the KeyedVectors of a model
        wv = model.wv
        wv.save(tmpf)
        loaded_wv = FastTextKeyedVectors.load(tmpf)
        self.assertTrue(np.allclose(wv.vectors_ngrams, loaded_wv.vectors_ngrams))
        self.assertEqual(len(wv.vocab), len(loaded_wv.vocab))

    @unittest.skipIf(os.name == 'nt',
        "corpus_file is not supported for Windows + Py2 and avoid memory error with Appveyor x32")
    def test_persistence_fromfile(self):
        with temporary_file(get_tmpfile('gensim_fasttext1.tst')) as corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)

            tmpf = get_tmpfile('gensim_fasttext.tst')
            model = FT_gensim(corpus_file=corpus_file, min_count=1)
            model.save(tmpf)
            self.models_equal(model, FT_gensim.load(tmpf))
            #  test persistence of the KeyedVectors of a model
            wv = model.wv
            wv.save(tmpf)
            loaded_wv = FastTextKeyedVectors.load(tmpf)
            self.assertTrue(np.allclose(wv.vectors_ngrams, loaded_wv.vectors_ngrams))
            self.assertEqual(len(wv.vocab), len(loaded_wv.vocab))

    @unittest.skipIf(IS_WIN32, "avoid memory error with Appveyor x32")
    def test_norm_vectors_not_saved(self):
        tmpf = get_tmpfile('gensim_fasttext.tst')
        model = FT_gensim(sentences, min_count=1)
        model.init_sims()
        model.save(tmpf)
        loaded_model = FT_gensim.load(tmpf)
        self.assertTrue(loaded_model.wv.vectors_norm is None)
        self.assertTrue(loaded_model.wv.vectors_ngrams_norm is None)

        wv = model.wv
        wv.save(tmpf)
        loaded_kv = FastTextKeyedVectors.load(tmpf)
        self.assertTrue(loaded_kv.vectors_norm is None)
        self.assertTrue(loaded_kv.vectors_ngrams_norm is None)

    def model_sanity(self, model):
        self.assertEqual(model.wv.vectors.shape, (len(model.wv.vocab), model.vector_size))
        self.assertEqual(model.wv.vectors_vocab.shape, (len(model.wv.vocab), model.vector_size))
        self.assertEqual(model.wv.vectors_ngrams.shape, (model.wv.num_ngram_vectors, model.vector_size))

    def test_load_fasttext_format(self):
        try:
            model = gensim.models.fasttext.load_facebook_model(self.test_model_file)
        except Exception as exc:
            self.fail('Unable to load FastText model from file %s: %s' % (self.test_model_file, exc))
        vocab_size, model_size = 1762, 10
        self.assertEqual(model.wv.vectors.shape, (vocab_size, model_size))
        self.assertEqual(len(model.wv.vocab), vocab_size, model_size)
        self.assertEqual(model.wv.vectors_ngrams.shape, (model.wv.num_ngram_vectors, model_size))

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

        self.assertEqual(model.vocabulary.min_count, 5)
        self.assertEqual(model.window, 5)
        self.assertEqual(model.epochs, 5)
        self.assertEqual(model.negative, 5)
        self.assertEqual(model.vocabulary.sample, 0.0001)
        self.assertEqual(model.trainables.bucket, 1000)
        self.assertEqual(model.wv.max_n, 6)
        self.assertEqual(model.wv.min_n, 3)
        self.assertEqual(model.wv.vectors.shape, (len(model.wv.vocab), model.vector_size))
        self.assertEqual(model.wv.vectors_ngrams.shape, (model.wv.num_ngram_vectors, model.vector_size))

    def test_load_fasttext_new_format(self):
        try:
            new_model = gensim.models.fasttext.load_facebook_model(self.test_new_model_file)
        except Exception as exc:
            self.fail('Unable to load FastText model from file %s: %s' % (self.test_new_model_file, exc))
        vocab_size, model_size = 1763, 10
        self.assertEqual(new_model.wv.vectors.shape, (vocab_size, model_size))
        self.assertEqual(len(new_model.wv.vocab), vocab_size, model_size)
        self.assertEqual(new_model.wv.vectors_ngrams.shape, (new_model.wv.num_ngram_vectors, model_size))

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

        self.assertEqual(new_model.vocabulary.min_count, 5)
        self.assertEqual(new_model.window, 5)
        self.assertEqual(new_model.epochs, 5)
        self.assertEqual(new_model.negative, 5)
        self.assertEqual(new_model.vocabulary.sample, 0.0001)
        self.assertEqual(new_model.trainables.bucket, 1000)
        self.assertEqual(new_model.wv.max_n, 6)
        self.assertEqual(new_model.wv.min_n, 3)
        self.assertEqual(new_model.wv.vectors.shape, (len(new_model.wv.vocab), new_model.vector_size))
        self.assertEqual(new_model.wv.vectors_ngrams.shape, (new_model.wv.num_ngram_vectors, new_model.vector_size))

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

    def test_lookup(self):
        # In vocab, sanity check
        self.assertTrue('night' in self.test_model.wv.vocab)
        self.assertTrue(np.allclose(self.test_model.wv['night'], self.test_model.wv[['night']]))
        # Out of vocab check
        self.assertFalse('nights' in self.test_model.wv.vocab)
        self.assertTrue(np.allclose(self.test_model.wv['nights'], self.test_model.wv[['nights']]))

    def test_contains(self):
        # In vocab, sanity check
        self.assertTrue('night' in self.test_model.wv.vocab)
        self.assertTrue('night' in self.test_model.wv)
        # Out of vocab check
        self.assertFalse('nights' in self.test_model.wv.vocab)
        self.assertTrue('nights' in self.test_model.wv)

    @unittest.skipIf(PYEMD_EXT is False, "pyemd not installed or have some issues")
    def test_wm_distance(self):
        doc = ['night', 'payment']
        oov_doc = ['nights', 'forests', 'payments']

        dist = self.test_model.wv.wmdistance(doc, oov_doc)
        self.assertNotEqual(float('inf'), dist)

    def test_cbow_hs_training(self):

        model_gensim = FT_gensim(
            size=50, sg=0, cbow_mean=1, alpha=0.05, window=5, hs=1, negative=0,
            min_count=5, iter=5, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
            sorted_vocab=1, workers=1, min_alpha=0.0)

        lee_data = LineSentence(datapath('lee_background.cor'))
        model_gensim.build_vocab(lee_data)
        orig0 = np.copy(model_gensim.wv.vectors[0])
        model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.epochs)
        self.assertFalse((orig0 == model_gensim.wv.vectors[0]).all())  # vector should vary after training

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
        overlap_count = len(set(sims_gensim_words).intersection(expected_sims_words))
        self.assertGreaterEqual(overlap_count, 2)

    @unittest.skipIf(os.name == 'nt' and six.PY2, "corpus_file training is not supported on Windows + Py27")
    def test_cbow_hs_training_fromfile(self):
        with temporary_file(get_tmpfile('gensim_fasttext.tst')) as corpus_file:
            model_gensim = FT_gensim(
                size=50, sg=0, cbow_mean=1, alpha=0.05, window=5, hs=1, negative=0,
                min_count=5, iter=5, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
                sorted_vocab=1, workers=1, min_alpha=0.0)

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
            overlap_count = len(set(sims_gensim_words).intersection(expected_sims_words))
            self.assertGreaterEqual(overlap_count, 2)

    def test_sg_hs_training(self):

        model_gensim = FT_gensim(
            size=50, sg=1, cbow_mean=1, alpha=0.025, window=5, hs=1, negative=0,
            min_count=5, iter=5, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
            sorted_vocab=1, workers=1, min_alpha=0.0)

        lee_data = LineSentence(datapath('lee_background.cor'))
        model_gensim.build_vocab(lee_data)
        orig0 = np.copy(model_gensim.wv.vectors[0])
        model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.epochs)
        self.assertFalse((orig0 == model_gensim.wv.vectors[0]).all())  # vector should vary after training

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
        overlap_count = len(set(sims_gensim_words).intersection(expected_sims_words))
        self.assertGreaterEqual(overlap_count, 2)

    @unittest.skipIf(os.name == 'nt' and six.PY2, "corpus_file training is not supported on Windows + Py27")
    def test_sg_hs_training_fromfile(self):
        with temporary_file(get_tmpfile('gensim_fasttext.tst')) as corpus_file:
            model_gensim = FT_gensim(
                size=50, sg=1, cbow_mean=1, alpha=0.025, window=5, hs=1, negative=0,
                min_count=5, iter=5, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
                sorted_vocab=1, workers=1, min_alpha=0.0)

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
            overlap_count = len(set(sims_gensim_words).intersection(expected_sims_words))
            self.assertGreaterEqual(overlap_count, 2)

    def test_cbow_neg_training(self):

        model_gensim = FT_gensim(
            size=50, sg=0, cbow_mean=1, alpha=0.05, window=5, hs=0, negative=5,
            min_count=5, iter=5, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
            sorted_vocab=1, workers=1, min_alpha=0.0)

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
        overlap_count = len(set(sims_gensim_words).intersection(expected_sims_words))
        self.assertGreaterEqual(overlap_count, 2)

    @unittest.skipIf(os.name == 'nt' and six.PY2, "corpus_file training is not supported on Windows + Py27")
    def test_cbow_neg_training_fromfile(self):
        with temporary_file(get_tmpfile('gensim_fasttext.tst')) as corpus_file:
            model_gensim = FT_gensim(
                size=50, sg=0, cbow_mean=1, alpha=0.05, window=5, hs=0, negative=5,
                min_count=5, iter=5, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
                sorted_vocab=1, workers=1, min_alpha=0.0)

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
            overlap_count = len(set(sims_gensim_words).intersection(expected_sims_words))
            self.assertGreaterEqual(overlap_count, 2)

    def test_sg_neg_training(self):

        model_gensim = FT_gensim(
            size=50, sg=1, cbow_mean=1, alpha=0.025, window=5, hs=0, negative=5,
            min_count=5, iter=5, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
            sorted_vocab=1, workers=1, min_alpha=0.0)

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
        overlap_count = len(set(sims_gensim_words).intersection(expected_sims_words))
        self.assertGreaterEqual(overlap_count, 2)

    @unittest.skipIf(os.name == 'nt' and six.PY2, "corpus_file training is not supported on Windows + Py27")
    def test_sg_neg_training_fromfile(self):
        with temporary_file(get_tmpfile('gensim_fasttext.tst')) as corpus_file:
            model_gensim = FT_gensim(
                size=50, sg=1, cbow_mean=1, alpha=0.025, window=5, hs=0, negative=5,
                min_count=5, iter=5, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
                sorted_vocab=1, workers=1, min_alpha=0.0)

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
            overlap_count = len(set(sims_gensim_words).intersection(expected_sims_words))
            self.assertGreaterEqual(overlap_count, 2)

    def test_online_learning(self):
        model_hs = FT_gensim(sentences, size=10, min_count=1, seed=42, hs=1, negative=0)
        self.assertTrue(len(model_hs.wv.vocab), 12)
        self.assertTrue(model_hs.wv.vocab['graph'].count, 3)
        model_hs.build_vocab(new_sentences, update=True)  # update vocab
        self.assertEqual(len(model_hs.wv.vocab), 14)
        self.assertTrue(model_hs.wv.vocab['graph'].count, 4)
        self.assertTrue(model_hs.wv.vocab['artificial'].count, 4)

    @unittest.skipIf(os.name == 'nt' and six.PY2, "corpus_file training is not supported on Windows + Py27")
    def test_online_learning_fromfile(self):
        with temporary_file(get_tmpfile('gensim_fasttext1.tst')) as corpus_file, \
                temporary_file(get_tmpfile('gensim_fasttext2.tst')) as new_corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)
            utils.save_as_line_sentence(new_sentences, new_corpus_file)

            model_hs = FT_gensim(corpus_file=corpus_file, size=10, min_count=1, seed=42, hs=1, negative=0)
            self.assertTrue(len(model_hs.wv.vocab), 12)
            self.assertTrue(model_hs.wv.vocab['graph'].count, 3)
            model_hs.build_vocab(corpus_file=new_corpus_file, update=True)  # update vocab
            self.assertEqual(len(model_hs.wv.vocab), 14)
            self.assertTrue(model_hs.wv.vocab['graph'].count, 4)
            self.assertTrue(model_hs.wv.vocab['artificial'].count, 4)

    def test_online_learning_after_save(self):
        tmpf = get_tmpfile('gensim_fasttext.tst')
        model_neg = FT_gensim(sentences, size=10, min_count=0, seed=42, hs=0, negative=5)
        model_neg.save(tmpf)
        model_neg = FT_gensim.load(tmpf)
        self.assertTrue(len(model_neg.wv.vocab), 12)
        model_neg.build_vocab(new_sentences, update=True)  # update vocab
        model_neg.train(new_sentences, total_examples=model_neg.corpus_count, epochs=model_neg.epochs)
        self.assertEqual(len(model_neg.wv.vocab), 14)

    @unittest.skipIf(os.name == 'nt' and six.PY2, "corpus_file training is not supported on Windows + Py27")
    def test_online_learning_after_save_fromfile(self):
        with temporary_file(get_tmpfile('gensim_fasttext1.tst')) as corpus_file, \
                temporary_file(get_tmpfile('gensim_fasttext2.tst')) as new_corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)
            utils.save_as_line_sentence(new_sentences, new_corpus_file)

            tmpf = get_tmpfile('gensim_fasttext.tst')
            model_neg = FT_gensim(corpus_file=corpus_file, size=10, min_count=0, seed=42, hs=0, negative=5)
            model_neg.save(tmpf)
            model_neg = FT_gensim.load(tmpf)
            self.assertTrue(len(model_neg.wv.vocab), 12)
            model_neg.build_vocab(corpus_file=new_corpus_file, update=True)  # update vocab
            model_neg.train(corpus_file=new_corpus_file, total_words=model_neg.corpus_total_words,
                            epochs=model_neg.epochs)
            self.assertEqual(len(model_neg.wv.vocab), 14)

    def online_sanity(self, model):
        terro, others = [], []
        for l in list_corpus:
            if 'terrorism' in l:
                terro.append(l)
            else:
                others.append(l)
        self.assertTrue(all('terrorism' not in l for l in others))
        model.build_vocab(others)
        model.train(others, total_examples=model.corpus_count, epochs=model.epochs)
        # checks that `vectors` is different from `vectors_vocab`
        self.assertFalse(np.all(np.equal(model.wv.vectors, model.wv.vectors_vocab)))
        self.assertFalse('terrorism' in model.wv.vocab)
        model.build_vocab(terro, update=True)  # update vocab
        self.assertTrue(model.wv.vectors_ngrams.dtype == 'float32')
        self.assertTrue('terrorism' in model.wv.vocab)
        orig0_all = np.copy(model.wv.vectors_ngrams)
        model.train(terro, total_examples=len(terro), epochs=model.epochs)
        self.assertFalse(np.allclose(model.wv.vectors_ngrams, orig0_all))
        sim = model.wv.n_similarity(['war'], ['terrorism'])
        self.assertLess(0., sim)

    @unittest.skipIf(IS_WIN32, "avoid memory error with Appveyor x32")
    def test_sg_hs_online(self):
        model = FT_gensim(sg=1, window=2, hs=1, negative=0, min_count=3, iter=1, seed=42, workers=1)
        self.online_sanity(model)

    @unittest.skipIf(IS_WIN32, "avoid memory error with Appveyor x32")
    def test_sg_neg_online(self):
        model = FT_gensim(sg=1, window=2, hs=0, negative=5, min_count=3, iter=1, seed=42, workers=1)
        self.online_sanity(model)

    @unittest.skipIf(IS_WIN32, "avoid memory error with Appveyor x32")
    def test_cbow_hs_online(self):
        model = FT_gensim(
            sg=0, cbow_mean=1, alpha=0.05, window=2, hs=1, negative=0, min_count=3, iter=1, seed=42, workers=1
        )
        self.online_sanity(model)

    @unittest.skipIf(IS_WIN32, "avoid memory error with Appveyor x32")
    def test_cbow_neg_online(self):
        model = FT_gensim(
            sg=0, cbow_mean=1, alpha=0.05, window=2, hs=0, negative=5,
            min_count=5, iter=1, seed=42, workers=1, sample=0
        )
        self.online_sanity(model)

    def test_get_vocab_word_vecs(self):
        model = FT_gensim(size=10, min_count=1, seed=42)
        model.build_vocab(sentences)
        original_syn0_vocab = np.copy(model.wv.vectors_vocab)
        model.wv.adjust_vectors()
        self.assertTrue(np.all(np.equal(model.wv.vectors_vocab, original_syn0_vocab)))

    def test_persistence_word2vec_format(self):
        """Test storing/loading the model in word2vec format."""
        tmpf = get_tmpfile('gensim_fasttext_w2v_format.tst')
        model = FT_gensim(sentences, min_count=1, size=10)
        model.wv.save_word2vec_format(tmpf, binary=True)
        loaded_model_kv = Word2VecKeyedVectors.load_word2vec_format(tmpf, binary=True)
        self.assertEqual(len(model.wv.vocab), len(loaded_model_kv.vocab))
        self.assertTrue(np.allclose(model.wv['human'], loaded_model_kv['human']))

    def test_bucket_ngrams(self):
        model = FT_gensim(size=10, min_count=1, bucket=20)
        model.build_vocab(sentences)
        self.assertEqual(model.wv.vectors_ngrams.shape, (20, 10))
        model.build_vocab(new_sentences, update=True)
        self.assertEqual(model.wv.vectors_ngrams.shape, (20, 10))

    def test_estimate_memory(self):
        model = FT_gensim(sg=1, hs=1, size=10, negative=5, min_count=3)
        model.build_vocab(sentences)
        report = model.estimate_memory()
        self.assertEqual(report['vocab'], 2800)
        self.assertEqual(report['syn0_vocab'], 160)
        self.assertEqual(report['syn1'], 160)
        self.assertEqual(report['syn1neg'], 160)
        self.assertEqual(report['syn0_ngrams'], 2240)
        self.assertEqual(report['buckets_word'], 640)
        self.assertEqual(report['total'], 6160)

    def testLoadOldModel(self):
        """Test loading fasttext models from previous version"""

        model_file = 'fasttext_old'
        model = FT_gensim.load(datapath(model_file))
        self.assertTrue(model.wv.vectors.shape == (12, 100))
        self.assertTrue(len(model.wv.vocab) == 12)
        self.assertTrue(len(model.wv.index2word) == 12)
        self.assertIsNone(model.corpus_total_words)
        self.assertTrue(model.trainables.syn1neg.shape == (len(model.wv.vocab), model.vector_size))
        self.assertTrue(model.trainables.vectors_lockf.shape == (12, ))
        self.assertTrue(model.vocabulary.cum_table.shape == (12, ))

        self.assertEqual(model.wv.vectors_vocab.shape, (12, 100))
        self.assertEqual(model.wv.vectors_ngrams.shape, (2000000, 100))

        # Model stored in multiple files
        model_file = 'fasttext_old_sep'
        model = FT_gensim.load(datapath(model_file))
        self.assertTrue(model.wv.vectors.shape == (12, 100))
        self.assertTrue(len(model.wv.vocab) == 12)
        self.assertTrue(len(model.wv.index2word) == 12)
        self.assertIsNone(model.corpus_total_words)
        self.assertTrue(model.trainables.syn1neg.shape == (len(model.wv.vocab), model.vector_size))
        self.assertTrue(model.trainables.vectors_lockf.shape == (12, ))
        self.assertTrue(model.vocabulary.cum_table.shape == (12, ))

        self.assertEqual(model.wv.vectors_vocab.shape, (12, 100))
        self.assertEqual(model.wv.vectors_ngrams.shape, (2000000, 100))

    def compare_with_wrapper(self, model_gensim, model_wrapper):
        # make sure we get >=2 overlapping words for top-10 similar words suggested for `night`
        sims_gensim = model_gensim.wv.most_similar('night', topn=10)
        sims_gensim_words = (list(map(lambda x: x[0], sims_gensim)))  # get similar words

        sims_wrapper = model_wrapper.most_similar('night', topn=10)
        sims_wrapper_words = (list(map(lambda x: x[0], sims_wrapper)))  # get similar words

        overlap_count = len(set(sims_gensim_words).intersection(sims_wrapper_words))

        # overlap increases as we increase `iter` value, min overlap set to 2 to avoid unit-tests taking too long
        # this limit can be increased when using Cython code
        self.assertGreaterEqual(overlap_count, 2)

    def test_cbow_hs_against_wrapper(self):
        if self.ft_path is None:
            logger.info("FT_HOME env variable not set, skipping test")
            return

        tmpf = get_tmpfile('gensim_fasttext.tst')
        model_wrapper = FT_wrapper.train(ft_path=self.ft_path, corpus_file=datapath('lee_background.cor'),
                                         output_file=tmpf, model='cbow', size=50, alpha=0.05, window=5, min_count=5,
                                         word_ngrams=1,
                                         loss='hs', sample=1e-3, negative=0, iter=5, min_n=3, max_n=6, sorted_vocab=1,
                                         threads=12)

        model_gensim = FT_gensim(size=50, sg=0, cbow_mean=1, alpha=0.05, window=5, hs=1, negative=0,
                                 min_count=5, iter=5, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
                                 sorted_vocab=1, workers=1, min_alpha=0.0)

        lee_data = LineSentence(datapath('lee_background.cor'))
        model_gensim.build_vocab(lee_data)
        orig0 = np.copy(model_gensim.wv.vectors[0])
        model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.epochs)
        self.assertFalse((orig0 == model_gensim.wv.vectors[0]).all())  # vector should vary after training
        self.compare_with_wrapper(model_gensim, model_wrapper)

    def test_sg_hs_against_wrapper(self):
        if self.ft_path is None:
            logger.info("FT_HOME env variable not set, skipping test")
            return

        tmpf = get_tmpfile('gensim_fasttext.tst')
        model_wrapper = FT_wrapper.train(ft_path=self.ft_path, corpus_file=datapath('lee_background.cor'),
                                         output_file=tmpf, model='skipgram', size=50, alpha=0.025, window=5,
                                         min_count=5, word_ngrams=1,
                                         loss='hs', sample=1e-3, negative=0, iter=5, min_n=3, max_n=6, sorted_vocab=1,
                                         threads=12)

        model_gensim = FT_gensim(size=50, sg=1, cbow_mean=1, alpha=0.025, window=5, hs=1, negative=0,
                                 min_count=5, iter=5, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
                                 sorted_vocab=1, workers=1, min_alpha=0.0)

        lee_data = LineSentence(datapath('lee_background.cor'))
        model_gensim.build_vocab(lee_data)
        orig0 = np.copy(model_gensim.wv.vectors[0])
        model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.epochs)
        self.assertFalse((orig0 == model_gensim.wv.vectors[0]).all())  # vector should vary after training
        self.compare_with_wrapper(model_gensim, model_wrapper)


with open(datapath('toy-data.txt')) as fin:
    TOY_SENTENCES = [fin.read().strip().split(' ')]


def train_gensim(bucket=100, min_count=5):
    #
    # Set parameters to match those in the load_native function
    #
    model = FT_gensim(bucket=bucket, size=5, alpha=0.05, workers=1, sample=0.0001, min_count=min_count)
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
    a_count = {key: value.count for (key, value) in a.vocab.items()}
    b_count = {key: value.count for (key, value) in b.vocab.items()}
    t.assertEqual(a_count, b_count)

    #
    # We don't compare indices because they depend on several things we
    # cannot control during testing:
    #
    # 1. The order in which ties are broken when sorting the vocabulary
    #    in prepare_vocab
    # 2. The order in which vocab terms are added to vocab_raw
    #
    if False:
        a_indices = {key: value.index for (key, value) in a.vocab.items()}
        b_indices = {key: value.index for (key, value) in b.vocab.items()}
        a_words = [k for k in sorted(a_indices, key=lambda x: a_indices[x])]
        b_words = [k for k in sorted(b_indices, key=lambda x: b_indices[x])]
        t.assertEqual(a_words, b_words)

        t.assertEqual(a.index2word, b.index2word)
        t.assertEqual(a.hash2index, b.hash2index)

    #
    # We do not compare most matrices directly, because they will never
    # be equal unless many conditions are strictly controlled.
    #
    t.assertEqual(a.vectors.shape, b.vectors.shape)
    # t.assertTrue(np.allclose(a.vectors, b.vectors))

    t.assertEqual(a.vectors_vocab.shape, b.vectors_vocab.shape)
    # t.assertTrue(np.allclose(a.vectors_vocab, b.vectors_vocab))

    #
    # Only if match_gensim=True in init_post_load
    #
    # t.assertEqual(a.vectors_ngrams.shape, b.vectors_ngrams.shape)


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
            actual_vector = native.wv.word_vec(word)
            self.assertTrue(np.allclose(expected_vector, actual_vector, atol=1e-5))

    def test_out_of_vocab(self):
        """Test for correct representation of out-of-vocab words."""
        native = load_native()

        for word, expected_vector in self.oov_expected.items():
            actual_vector = native.wv.word_vec(word)
            self.assertTrue(np.allclose(expected_vector, actual_vector, atol=1e-5))

    @unittest.skip('this test does not pass currently, I suspect a bug in our FT implementation')
    def test_out_of_vocab_gensim(self):
        """Test whether gensim gives similar results to FB for OOV words.

        Seems to be broken for our toy model.
        """
        model = train_gensim()

        for word, expected_vector in self.oov_expected.items():
            actual_vector = model.wv.word_vec(word)
            self.assertTrue(np.allclose(expected_vector, actual_vector, atol=1e-5))

    def test_sanity(self):
        """Compare models trained on toy data.  They should be equal."""
        trained = train_gensim()
        native = load_native()

        self.assertEqual(trained.bucket, native.bucket)
        #
        # Only if match_gensim=True in init_post_load
        #
        # self.assertEqual(trained.num_ngram_vectors, native.num_ngram_vectors)

        compare_wv(trained.wv, native.wv, self)
        compare_vocabulary(trained.vocabulary, native.vocabulary, self)
        compare_nn(trained.trainables, native.trainables, self)

    def test_continuation_native(self):
        """Ensure that training has had a measurable effect."""
        native = load_native()

        #
        # Pick a word that's is in both corpuses.
        # Its vectors should be different between training runs.
        #
        word = 'human'
        old_vector = native.wv.word_vec(word).tolist()

        native.train(list_corpus, total_examples=len(list_corpus), epochs=native.epochs)

        new_vector = native.wv.word_vec(word).tolist()
        self.assertNotEqual(old_vector, new_vector)

    def test_continuation_gensim(self):
        """Ensure that continued training has had a measurable effect."""
        model = train_gensim(min_count=0)
        vectors_ngrams_before = np.copy(model.wv.vectors_ngrams)

        word = 'human'
        old_vector = model.wv.word_vec(word).tolist()

        model.train(list_corpus, total_examples=len(list_corpus), epochs=model.epochs)

        vectors_ngrams_after = np.copy(model.wv.vectors_ngrams)
        self.assertFalse(np.allclose(vectors_ngrams_before, vectors_ngrams_after))
        new_vector = model.wv.word_vec(word).tolist()

        self.assertNotEqual(old_vector, new_vector)

    def test_continuation_load_gensim(self):
        #
        # This is a model from 3.6.0
        #
        model = FT_gensim.load(datapath('compatible-hash-false.model'))
        vectors_ngrams_before = np.copy(model.wv.vectors_ngrams)
        old_vector = model.wv.word_vec('human').tolist()

        model.train(list_corpus, total_examples=len(list_corpus), epochs=model.epochs)
        new_vector = model.wv.word_vec('human').tolist()

        self.assertFalse(np.allclose(vectors_ngrams_before, model.wv.vectors_ngrams))
        self.assertNotEqual(old_vector, new_vector)

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
            model.train(list_corpus, total_examples=len(list_corpus), epochs=model.epochs)

            model.save(model_name)

    def test_save_load_native(self):
        """Test that serialization works end-to-end.  Not crashing is a success."""

        model_name = 'test_ft_saveload_fb.model'

        with temporary_file(model_name):
            load_native().save(model_name)

            model = FT_gensim.load(model_name)
            model.train(list_corpus, total_examples=len(list_corpus), epochs=model.epochs)

            model.save(model_name)

    def test_load_native_pretrained(self):
        model = gensim.models.fasttext.load_facebook_model(datapath('toy-model-pretrained.bin'))
        actual = model['monarchist']
        expected = np.array([0.76222, 1.0669, 0.7055, -0.090969, -0.53508])
        self.assertTrue(np.allclose(expected, actual, atol=10e-4))

    def test_load_native_vectors(self):
        cap_path = datapath("crime-and-punishment.bin")
        fbkv = gensim.models.fasttext.load_facebook_vectors(cap_path)
        self.assertFalse('landlord' in fbkv.vocab)
        self.assertTrue('landlady' in fbkv.vocab)
        oov_vector = fbkv['landlord']
        iv_vector = fbkv['landlady']
        self.assertFalse(np.allclose(oov_vector, iv_vector))

    def test_no_ngrams(self):
        model = gensim.models.fasttext.load_facebook_model(datapath('crime-and-punishment.bin'))

        v1 = model.wv['']
        origin = np.zeros(v1.shape, v1.dtype)
        self.assertTrue(np.allclose(v1, origin))


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
        self.assertEqual(m.trainables.bucket, m.wv.bucket)

    def test_compatibility_false(self):
        #
        # Originally obtained using and older version of gensim (e.g. 3.6.0).
        #
        m = FT_gensim.load(datapath('compatible-hash-false.model'))
        self.assertFalse(m.wv.compatible_hash)
        self.assertEqual(m.trainables.bucket, m.wv.bucket)

    def test_hash_native(self):
        m = load_native()
        self.assertTrue(m.wv.compatible_hash)
        self.assertEqual(m.trainables.bucket, m.wv.bucket)


class HashTest(unittest.TestCase):
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


class ZeroBucketTest(unittest.TestCase):
    def test_in_vocab(self):
        model = train_gensim(bucket=0)
        self.assertIsNotNone(model.wv['anarchist'])

    def test_out_of_vocab(self):
        model = train_gensim(bucket=0)
        self.assertRaises(KeyError, model.wv.word_vec, 'streamtrain')


class UnicodeVocabTest(unittest.TestCase):
    def test_ascii(self):
        buf = io.BytesIO()
        buf.name = 'dummy name to keep fasttext happy'
        buf.write(struct.pack('@3i', 2, -1, -1))  # vocab_size, nwords, nlabels
        buf.write(struct.pack('@1q', -1))
        buf.write(b'hello')
        buf.write(b'\x00')
        buf.write(struct.pack('@qb', 1, -1))
        buf.write(b'world')
        buf.write(b'\x00')
        buf.write(struct.pack('@qb', 2, -1))
        buf.seek(0)

        raw_vocab, vocab_size, nlabels = gensim.models._fasttext_bin._load_vocab(buf, False)
        expected = {'hello': 1, 'world': 2}
        self.assertEqual(expected, dict(raw_vocab))

        self.assertEqual(vocab_size, 2)
        self.assertEqual(nlabels, -1)

    def test_bad_unicode(self):
        buf = io.BytesIO()
        buf.name = 'dummy name to keep fasttext happy'
        buf.write(struct.pack('@3i', 2, -1, -1))  # vocab_size, nwords, nlabels
        buf.write(struct.pack('@1q', -1))
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

        raw_vocab, vocab_size, nlabels = gensim.models._fasttext_bin._load_vocab(buf, False)

        expected = {
            u'英語版ウィキペディアへの投稿はいつでも\\xe6': 1,
            u'административно-территориальн\\xd1': 2,
        }

        self.assertEqual(expected, dict(raw_vocab))

        self.assertEqual(vocab_size, 2)
        self.assertEqual(nlabels, -1)


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


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
