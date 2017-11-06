#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import unittest
import tempfile
import os
import struct

import numpy as np

from gensim import utils
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim
from gensim.models.wrappers.fasttext import FastTextKeyedVectors
from gensim.models.wrappers.fasttext import FastText as FT_wrapper

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)
logger = logging.getLogger(__name__)

IS_WIN32 = (os.name == "nt") and (struct.calcsize('P') * 8 == 32)


class LeeCorpus(object):
    def __iter__(self):
        with open(datapath('lee_background.cor')) as f:
            for line in f:
                yield utils.simple_preprocess(line)


list_corpus = list(LeeCorpus())

sentences = [
    ['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey']
]

new_sentences = [
    ['computer', 'artificial', 'intelligence'],
    ['artificial', 'trees'],
    ['human', 'intelligence'],
    ['artificial', 'graph'],
    ['intelligence'],
    ['artificial', 'intelligence', 'system']
]


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_fasttext.tst')


class TestFastTextModel(unittest.TestCase):

    def setUp(self):
        ft_home = os.environ.get('FT_HOME', None)
        self.ft_path = os.path.join(ft_home, 'fasttext') if ft_home else None
        self.test_model_file = datapath('lee_fasttext')
        self.test_model = FT_gensim.load_fasttext_format(self.test_model_file)
        self.test_new_model_file = datapath('lee_fasttext_new')

    def test_training(self):
        model = FT_gensim(size=10, min_count=1, hs=1, negative=0, seed=42, workers=1)
        model.build_vocab(sentences)
        self.model_sanity(model)

        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        sims = model.most_similar('graph', topn=10)

        self.assertEqual(model.wv.syn0.shape, (12, 10))
        self.assertEqual(len(model.wv.vocab), 12)
        self.assertEqual(model.wv.syn0_vocab.shape[1], 10)
        self.assertEqual(model.wv.syn0_ngrams.shape[1], 10)
        self.model_sanity(model)

        # test querying for "most similar" by vector
        graph_vector = model.wv.syn0norm[model.wv.vocab['graph'].index]
        sims2 = model.most_similar(positive=[graph_vector], topn=11)
        sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
        self.assertEqual(sims, sims2)

        # build vocab and train in one step; must be the same as above
        model2 = FT_gensim(sentences, size=10, min_count=1, hs=1, negative=0, seed=42, workers=1)
        self.models_equal(model, model2)

        # verify oov-word vector retrieval
        invocab_vec = model['minors']  # invocab word
        self.assertEqual(len(invocab_vec), 10)

        oov_vec = model['minor']  # oov word
        self.assertEqual(len(oov_vec), 10)

    def models_equal(self, model, model2):
        self.assertEqual(len(model.wv.vocab), len(model2.wv.vocab))
        self.assertEqual(model.num_ngram_vectors, model2.num_ngram_vectors)
        self.assertTrue(np.allclose(model.wv.syn0_vocab, model2.wv.syn0_vocab))
        self.assertTrue(np.allclose(model.wv.syn0_ngrams, model2.wv.syn0_ngrams))
        self.assertTrue(np.allclose(model.wv.syn0, model2.wv.syn0))
        if model.hs:
            self.assertTrue(np.allclose(model.syn1, model2.syn1))
        if model.negative:
            self.assertTrue(np.allclose(model.syn1neg, model2.syn1neg))
        most_common_word = max(model.wv.vocab.items(), key=lambda item: item[1].count)[0]
        self.assertTrue(np.allclose(model[most_common_word], model2[most_common_word]))

    @unittest.skipIf(IS_WIN32, "avoid memory error with Appveyor x32")
    def test_persistence(self):
        model = FT_gensim(sentences, min_count=1)
        model.save(testfile())
        self.models_equal(model, FT_gensim.load(testfile()))
        #  test persistence of the KeyedVectors of a model
        wv = model.wv
        wv.save(testfile())
        loaded_wv = FastTextKeyedVectors.load(testfile())
        self.assertTrue(np.allclose(wv.syn0_ngrams, loaded_wv.syn0_ngrams))
        self.assertEqual(len(wv.vocab), len(loaded_wv.vocab))
        self.assertEqual(len(wv.ngrams), len(loaded_wv.ngrams))

    @unittest.skipIf(IS_WIN32, "avoid memory error with Appveyor x32")
    def test_norm_vectors_not_saved(self):
        model = FT_gensim(sentences, min_count=1)
        model.init_sims()
        model.save(testfile())
        loaded_model = FT_gensim.load(testfile())
        self.assertTrue(loaded_model.wv.syn0norm is None)
        self.assertTrue(loaded_model.wv.syn0_ngrams_norm is None)

        wv = model.wv
        wv.save(testfile())
        loaded_kv = FastTextKeyedVectors.load(testfile())
        self.assertTrue(loaded_kv.syn0norm is None)
        self.assertTrue(loaded_kv.syn0_ngrams_norm is None)

    def model_sanity(self, model):
        self.assertEqual(model.wv.syn0.shape, (len(model.wv.vocab), model.vector_size))
        self.assertEqual(model.wv.syn0_vocab.shape, (len(model.wv.vocab), model.vector_size))
        self.assertEqual(model.wv.syn0_ngrams.shape, (model.num_ngram_vectors, model.vector_size))

    def test_load_fasttext_format(self):
        try:
            model = FT_gensim.load_fasttext_format(self.test_model_file)
        except Exception as exc:
            self.fail('Unable to load FastText model from file %s: %s' % (self.test_model_file, exc))
        vocab_size, model_size = 1762, 10
        self.assertEqual(model.wv.syn0.shape, (vocab_size, model_size))
        self.assertEqual(len(model.wv.vocab), vocab_size, model_size)
        self.assertEqual(model.wv.syn0_ngrams.shape, (model.num_ngram_vectors, model_size))

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
        self.assertTrue(np.allclose(model["hundred"], expected_vec, atol=1e-4))

        # vector for oov words are slightly different from original FastText due to discarding unused ngrams
        # obtained using a modified version of ./fasttext print-word-vectors lee_fasttext_new.bin
        expected_vec_oov = [
            -0.23825,
            -0.58482,
            -0.22276,
            -0.41215,
            0.91015,
            -1.6786,
            -0.26724,
            0.58818,
            0.57828,
            0.75801
        ]
        self.assertTrue(np.allclose(model["rejection"], expected_vec_oov, atol=1e-4))

        self.assertEqual(model.min_count, 5)
        self.assertEqual(model.window, 5)
        self.assertEqual(model.iter, 5)
        self.assertEqual(model.negative, 5)
        self.assertEqual(model.sample, 0.0001)
        self.assertEqual(model.bucket, 1000)
        self.assertEqual(model.wv.max_n, 6)
        self.assertEqual(model.wv.min_n, 3)
        self.assertEqual(model.wv.syn0.shape, (len(model.wv.vocab), model.vector_size))
        self.assertEqual(model.wv.syn0_ngrams.shape, (model.num_ngram_vectors, model.vector_size))

    def test_load_fasttext_new_format(self):
        try:
            new_model = FT_gensim.load_fasttext_format(self.test_new_model_file)
        except Exception as exc:
            self.fail('Unable to load FastText model from file %s: %s' % (self.test_new_model_file, exc))
        vocab_size, model_size = 1763, 10
        self.assertEqual(new_model.wv.syn0.shape, (vocab_size, model_size))
        self.assertEqual(len(new_model.wv.vocab), vocab_size, model_size)
        self.assertEqual(new_model.wv.syn0_ngrams.shape, (new_model.num_ngram_vectors, model_size))

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
        self.assertTrue(np.allclose(new_model["hundred"], expected_vec, atol=1e-4))

        # vector for oov words are slightly different from original FastText due to discarding unused ngrams
        # obtained using a modified version of ./fasttext print-word-vectors lee_fasttext_new.bin
        expected_vec_oov = [
            -0.53378,
            -0.19,
            0.013482,
            -0.86767,
            -0.21684,
            -0.89928,
            0.45124,
            0.18025,
            -0.14128,
            0.22508
        ]
        self.assertTrue(np.allclose(new_model["rejection"], expected_vec_oov, atol=1e-4))

        self.assertEqual(new_model.min_count, 5)
        self.assertEqual(new_model.window, 5)
        self.assertEqual(new_model.iter, 5)
        self.assertEqual(new_model.negative, 5)
        self.assertEqual(new_model.sample, 0.0001)
        self.assertEqual(new_model.bucket, 1000)
        self.assertEqual(new_model.wv.max_n, 6)
        self.assertEqual(new_model.wv.min_n, 3)
        self.assertEqual(new_model.wv.syn0.shape, (len(new_model.wv.vocab), new_model.vector_size))
        self.assertEqual(new_model.wv.syn0_ngrams.shape, (new_model.num_ngram_vectors, new_model.vector_size))

    def test_load_model_supervised(self):
        with self.assertRaises(NotImplementedError):
            FT_gensim.load_fasttext_format(datapath('pang_lee_polarity_fasttext'))

    def test_load_model_with_non_ascii_vocab(self):
        model = FT_gensim.load_fasttext_format(datapath('non_ascii_fasttext'))
        self.assertTrue(u'který' in model)
        try:
            model[u'který']
        except UnicodeDecodeError:
            self.fail('Unable to access vector for utf8 encoded non-ascii word')

    def test_load_model_non_utf8_encoding(self):
        model = FT_gensim.load_fasttext_format(datapath('cp852_fasttext'), encoding='cp852')
        self.assertTrue(u'který' in model)
        try:
            model[u'který']
        except KeyError:
            self.fail('Unable to access vector for cp-852 word')

    def test_n_similarity(self):
        # In vocab, sanity check
        self.assertTrue(np.allclose(self.test_model.n_similarity(['the', 'and'], ['and', 'the']), 1.0))
        self.assertEqual(self.test_model.n_similarity(['the'], ['and']), self.test_model.n_similarity(['and'], ['the']))
        # Out of vocab check
        self.assertTrue(np.allclose(self.test_model.n_similarity(['night', 'nights'], ['nights', 'night']), 1.0))
        self.assertEqual(self.test_model.n_similarity(['night'], ['nights']), self.test_model.n_similarity(['nights'], ['night']))

    def test_similarity(self):
        # In vocab, sanity check
        self.assertTrue(np.allclose(self.test_model.similarity('the', 'the'), 1.0))
        self.assertEqual(self.test_model.similarity('the', 'and'), self.test_model.similarity('and', 'the'))
        # Out of vocab check
        self.assertTrue(np.allclose(self.test_model.similarity('nights', 'nights'), 1.0))
        self.assertEqual(self.test_model.similarity('night', 'nights'), self.test_model.similarity('nights', 'night'))

    def test_most_similar(self):
        # In vocab, sanity check
        self.assertEqual(len(self.test_model.most_similar(positive=['the', 'and'], topn=5)), 5)
        self.assertEqual(self.test_model.most_similar('the'), self.test_model.most_similar(positive=['the']))
        # Out of vocab check
        self.assertEqual(len(self.test_model.most_similar(['night', 'nights'], topn=5)), 5)
        self.assertEqual(self.test_model.most_similar('nights'), self.test_model.most_similar(positive=['nights']))

    def test_most_similar_cosmul(self):
        # In vocab, sanity check
        self.assertEqual(len(self.test_model.most_similar_cosmul(positive=['the', 'and'], topn=5)), 5)
        self.assertEqual(
            self.test_model.most_similar_cosmul('the'),
            self.test_model.most_similar_cosmul(positive=['the']))
        # Out of vocab check
        self.assertEqual(len(self.test_model.most_similar_cosmul(['night', 'nights'], topn=5)), 5)
        self.assertEqual(
            self.test_model.most_similar_cosmul('nights'),
            self.test_model.most_similar_cosmul(positive=['nights']))

    def test_lookup(self):
        # In vocab, sanity check
        self.assertTrue('night' in self.test_model.wv.vocab)
        self.assertTrue(np.allclose(self.test_model['night'], self.test_model[['night']]))
        # Out of vocab check
        self.assertFalse('nights' in self.test_model.wv.vocab)
        self.assertTrue(np.allclose(self.test_model['nights'], self.test_model[['nights']]))
        # Word with no ngrams in model
        self.assertRaises(KeyError, lambda: self.test_model['a!@'])

    def test_contains(self):
        # In vocab, sanity check
        self.assertTrue('night' in self.test_model.wv.vocab)
        self.assertTrue('night' in self.test_model)
        # Out of vocab check
        self.assertFalse('nights' in self.test_model.wv.vocab)
        self.assertTrue('nights' in self.test_model)
        # Word with no ngrams in model
        self.assertFalse('a!@' in self.test_model.wv.vocab)
        self.assertFalse('a!@' in self.test_model)

    def test_wm_distance(self):
        doc = ['night', 'payment']
        oov_doc = ['nights', 'forests', 'payments']
        ngrams_absent_doc = ['a!@', 'b#$']

        dist = self.test_model.wmdistance(doc, oov_doc)
        self.assertNotEqual(float('inf'), dist)
        dist = self.test_model.wmdistance(doc, ngrams_absent_doc)
        self.assertEqual(float('inf'), dist)

    def test_doesnt_match(self):
        oov_words = ['nights', 'forests', 'payments']
        # Out of vocab check
        for word in oov_words:
            self.assertFalse(word in self.test_model.wv.vocab)
        try:
            self.test_model.doesnt_match(oov_words)
        except Exception:
            self.fail('model.doesnt_match raises exception for oov words')

    def compare_with_wrapper(self, model_gensim, model_wrapper):
        # make sure we get >=2 overlapping words for top-10 similar words suggested for `night`
        sims_gensim = model_gensim.most_similar('night', topn=10)
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

        model_wrapper = FT_wrapper.train(ft_path=self.ft_path, corpus_file=datapath('lee_background.cor'),
            output_file=testfile(), model='cbow', size=50, alpha=0.05, window=5, min_count=5, word_ngrams=1,
            loss='hs', sample=1e-3, negative=0, iter=5, min_n=3, max_n=6, sorted_vocab=1, threads=12)

        model_gensim = FT_gensim(size=50, sg=0, cbow_mean=1, alpha=0.05, window=5, hs=1, negative=0,
            min_count=5, iter=5, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
            sorted_vocab=1, workers=1, min_alpha=0.0)

        lee_data = LineSentence(datapath('lee_background.cor'))
        model_gensim.build_vocab(lee_data)
        orig0 = np.copy(model_gensim.wv.syn0[0])
        model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.iter)
        self.assertFalse((orig0 == model_gensim.wv.syn0[0]).all())  # vector should vary after training
        self.compare_with_wrapper(model_gensim, model_wrapper)

    def test_sg_hs_against_wrapper(self):
        if self.ft_path is None:
            logger.info("FT_HOME env variable not set, skipping test")
            return

        model_wrapper = FT_wrapper.train(ft_path=self.ft_path, corpus_file=datapath('lee_background.cor'),
            output_file=testfile(), model='skipgram', size=50, alpha=0.025, window=5, min_count=5, word_ngrams=1,
            loss='hs', sample=1e-3, negative=0, iter=5, min_n=3, max_n=6, sorted_vocab=1, threads=12)

        model_gensim = FT_gensim(size=50, sg=1, cbow_mean=1, alpha=0.025, window=5, hs=1, negative=0,
            min_count=5, iter=5, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
            sorted_vocab=1, workers=1, min_alpha=0.0)

        lee_data = LineSentence(datapath('lee_background.cor'))
        model_gensim.build_vocab(lee_data)
        orig0 = np.copy(model_gensim.wv.syn0[0])
        model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.iter)
        self.assertFalse((orig0 == model_gensim.wv.syn0[0]).all())  # vector should vary after training
        self.compare_with_wrapper(model_gensim, model_wrapper)

    def test_online_learning(self):
        model_hs = FT_gensim(sentences, size=10, min_count=1, seed=42, hs=1, negative=0)
        self.assertTrue(len(model_hs.wv.vocab), 12)
        self.assertTrue(len(model_hs.wv.ngrams), 202)
        self.assertTrue(model_hs.wv.vocab['graph'].count, 3)
        self.assertFalse('tif' in model_hs.wv.ngrams)
        model_hs.build_vocab(new_sentences, update=True)  # update vocab
        self.assertEqual(len(model_hs.wv.vocab), 14)
        self.assertTrue(len(model_hs.wv.ngrams), 271)
        self.assertTrue(model_hs.wv.vocab['graph'].count, 4)
        self.assertTrue(model_hs.wv.vocab['artificial'].count, 4)
        self.assertTrue('tif' in model_hs.wv.ngrams)  # ngram added because of the word `artificial`

    def test_online_learning_after_save(self):
        model_neg = FT_gensim(sentences, size=10, min_count=0, seed=42, hs=0, negative=5)
        model_neg.save(testfile())
        model_neg = FT_gensim.load(testfile())
        self.assertTrue(len(model_neg.wv.vocab), 12)
        self.assertTrue(len(model_neg.wv.ngrams), 202)
        model_neg.build_vocab(new_sentences, update=True)  # update vocab
        model_neg.train(new_sentences, total_examples=model_neg.corpus_count, epochs=model_neg.iter)
        self.assertEqual(len(model_neg.wv.vocab), 14)
        self.assertTrue(len(model_neg.wv.ngrams), 271)

    def online_sanity(self, model):
        terro, others = [], []
        for l in list_corpus:
            if 'terrorism' in l:
                terro.append(l)
            else:
                others.append(l)
        self.assertTrue(all(['terrorism' not in l for l in others]))
        model.build_vocab(others)
        model.train(others, total_examples=model.corpus_count, epochs=model.iter)
        self.assertFalse('terrorism' in model.wv.vocab)
        self.assertFalse('orism>' in model.wv.ngrams)
        model.build_vocab(terro, update=True)  # update vocab
        self.assertTrue('terrorism' in model.wv.vocab)
        self.assertTrue('orism>' in model.wv.ngrams)
        orig0_all = np.copy(model.wv.syn0_ngrams)
        model.train(terro, total_examples=len(terro), epochs=model.iter)
        self.assertFalse(np.allclose(model.wv.syn0_ngrams, orig0_all))
        sim = model.n_similarity(['war'], ['terrorism'])
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
        model = FT_gensim(sg=0, cbow_mean=1, alpha=0.05, window=2, hs=1, negative=0, min_count=3, iter=1, seed=42, workers=1)
        self.online_sanity(model)

    @unittest.skipIf(IS_WIN32, "avoid memory error with Appveyor x32")
    def test_cbow_neg_online(self):
        model = FT_gensim(sg=0, cbow_mean=1, alpha=0.05, window=2, hs=0, negative=5, min_count=5, iter=1, seed=42, workers=1, sample=0)
        self.online_sanity(model)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
