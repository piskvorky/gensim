#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import unittest
import os

import numpy as np

from gensim import utils
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim
from gensim.models.wrappers.fasttext import FastText as FT_wrapper

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)
logger = logging.getLogger(__name__)

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

def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_fasttext.tst')


class TestFastTextModel(unittest.TestCase):

    def setUp(self):
        ft_home = os.environ.get('FT_HOME', None)
        self.ft_exec_path = os.path.join(ft_home, 'fasttext') if ft_home else None

    def models_equal(self, model, model2):
        self.assertEqual(len(model.wv.vocab), len(model2.wv.vocab))
        self.assertTrue(np.allclose(model.wv.syn0, model2.wv.syn0))
        self.assertTrue(np.allclose(model.wv.syn0_all, model2.wv.syn0_all))
        if model.hs:
            self.assertTrue(np.allclose(model.syn1, model2.syn1))
        if model.negative:
            self.assertTrue(np.allclose(model.syn1neg, model2.syn1neg))
        most_common_word = max(model.wv.vocab.items(), key=lambda item: item[1].count)[0]
        self.assertTrue(np.allclose(model[most_common_word], model2[most_common_word]))

    def testTraining(self):
        model = FT_gensim(size=2, min_count=1, hs=1, negative=0)
        model.build_vocab(sentences)

        self.assertTrue(model.wv.syn0_all.shape == (len(model.wv.vocab) + len(model.wv.ngrams), 2))
        self.assertTrue(model.syn1.shape == (len(model.wv.vocab), 2))

        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        sims = model.most_similar('graph', topn=10)

        # test querying for "most similar" by vector
        graph_vector = model.wv.syn0norm[model.wv.vocab['graph'].index]
        sims2 = model.most_similar(positive=[graph_vector], topn=11)
        sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
        self.assertEqual(sims, sims2)

        # build vocab and train in one step; must be the same as above
        model2 = FT_gensim(sentences, size=2, min_count=1, hs=1, negative=0)
        self.models_equal(model, model2)

    def test_against_fasttext_wrapper(model_gensim, model_wrapper):
        sims_gensim = model_gensim.most_similar('night', topn=10)
        sims_gensim_words = (list(map(lambda x:x[0], sims_gensim)))

        sims_wrapper = model_wrapper.most_similar('night', topn=10)
        sims_wrapper_words = (list(map(lambda x:x[0], sims_wrapper)))

        self.assertEqual(sims_gensim, sims_wrapper)

    def test_cbow_hs(self):
        if self.ft_exec_path is None:
            logger.info("FT_HOME env variable not set, skipping test")
            return

        model_wrapper = FT_wrapper.train(ft_path=self.ft_path, corpus_file=datapath('lee_background.cor'),
            output_file=testfile(), model='cbow', size=50, alpha=0.05, window=2, min_count=5, word_ngrams=1,
            loss='hs', sample=1e-3, negative=0, iter=3, min_n=3, max_n=6, sorted_vocab=1, threads=1)

        model_gensim = FT_gensim(size=50, sg=0, cbow_mean=1, alpha=0.05, window=2, hs=1, negative=0,
            min_count=5, iter=3, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
            sorted_vocab=1, workers=12, min_alpha=0.0)
            
        lee_data = LineSentence(datapath('lee_background.cor'))
        model_gensim.build_vocab(lee_data)
        orig0 = np.copy(model_gensim.wv.syn0[0])
        model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.iter)
        self.assertFalse((orig0 == model_gensim.wv.syn0[0]).all())  # vector should vary after training
        
        self.test_against_fasttext_wrapper(model_gensim, model_wrapper)

    def test_cbow_neg(self):
        if self.ft_exec_path is None:
            logger.info("FT_HOME env variable not set, skipping test")
            return

        model_wrapper = FT_wrapper.train(ft_path=self.ft_exec_path, corpus_file=datapath('lee_background.cor'),
            output_file=testfile(), model='cbow', size=50, alpha=0.05, window=2, min_count=5, word_ngrams=1, loss='ns',
            sample=1e-3, negative=15, iter=7, min_n=3, max_n=6, sorted_vocab=1, threads=1)

        model_gensim = FT_gensim(size=50, sg=0, cbow_mean=1, alpha=0.05, window=2, hs=0, negative=15,
            min_count=1, iter=7, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
            sorted_vocab=1, workers=1, min_alpha=0.0)
            
        lee_data = LineSentence(datapath('lee_background.cor'))
        model_gensim.build_vocab(lee_data)
        orig0 = np.copy(model_gensim.wv.syn0[0])
        model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.iter)
        self.assertFalse((orig0 == model_gensim.wv.syn0[0]).all())  # vector should vary after training
        
        self.test_against_fasttext_wrapper(model_gensim, model_wrapper)

    def test_sg_hs(self):
        if self.ft_exec_path is None:
            logger.info("FT_HOME env variable not set, skipping test")
            return

        model_wrapper = FT_wrapper.train(ft_path=self.ft_exec_path, corpus_file=datapath('lee_background.cor'),
            output_file=testfile(), model='skipgram', size=50, alpha=0.05, window=2, min_count=5, word_ngrams=1,
            loss='hs', sample=1e-3, negative=0, iter=3, min_n=3, max_n=6, sorted_vocab=1, threads=1)

        model_gensim = FT_gensim(size=50, sg=1, cbow_mean=1, alpha=0.05, window=2, hs=1, negative=0,
            min_count=5, iter=3, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
            sorted_vocab=1, workers=12, min_alpha=0.0)

        lee_data = LineSentence(datapath('lee_background.cor'))
        model_gensim.build_vocab(lee_data)
        orig0 = np.copy(model_gensim.wv.syn0[0])
        model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.iter)
        self.assertFalse((orig0 == model_gensim.wv.syn0[0]).all())  # vector should vary after training

        self.test_against_fasttext_wrapper(model_gensim, model_wrapper)

    def test_sg_neg(self):
        if self.ft_exec_path is None:
            logger.info("FT_HOME env variable not set, skipping test")
            return

        model_wrapper = FT_wrapper.train(ft_path=self.ft_exec_path, corpus_file=datapath('lee_background.cor'),
            output_file=testfile(), model='skipgram', size=50, alpha=0.05, window=2, min_count=5, word_ngrams=1,
            loss='ns', sample=1e-3, negative=15, iter=1, min_n=3, max_n=6, sorted_vocab=1, threads=1)

        model_gensim = FT_gensim(size=50, sg=1, cbow_mean=1, alpha=0.05, window=2, hs=0, negative=0,
            min_count=5, iter=1, batch_words=1000, word_ngrams=1, sample=1e-3, min_n=3, max_n=6,
            sorted_vocab=1, workers=1, min_alpha=0.0)

        lee_data = LineSentence(datapath('lee_background.cor'))
        model_gensim.build_vocab(lee_data)
        orig0 = np.copy(model_gensim.wv.syn0[0])
        model_gensim.train(lee_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.iter)
        self.assertFalse((orig0 == model_gensim.wv.syn0[0]).all())  # vector should vary after training

        self.test_against_fasttext_wrapper(model_gensim, model_wrapper)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
