#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Prakhar Pratyush (er.prakhar2b@gmail.com)

import logging
import unittest
import os
from gensim import utils

import numpy as np

from gensim.models.fasttext import FastText

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)

# sentences = LeeCorpus()
# self.assertTrue(FastText(sentences, min_count=100, size=100, workers=3))
# self.assertTrue(FastText(sentences, sg=1, min_count=100, size=100, workers=3))

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


class TestUnsupervisedFastText(unittest.TestCase):

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
        """Test FastText training."""
        # build vocabulary, don't train yet
        model = FastText(size=2, min_count=1, hs=1, negative=0)
        model.build_vocab(sentences)

        self.assertTrue(model.wv.syn0_all.shape == (len(model.wv.ngrams), 2))
        # syn0_all is initialized as (model.bucket + len(model.wv.vocab), 2) but after ngrams
        # vocab building, it is re-arranged into a smaller matrix of size (len(model.wv.ngrams), 2)

        self.assertTrue(model.syn1.shape == (len(model.wv.vocab), 2))

        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
        sims = model.most_similar('graph', topn=10)

        # print(model['trees'])
        # print(model.wv.syn0[model.wv.vocab['trees'].index])

        # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # test querying for "most similar" by vector
        graph_vector = model.wv.syn0norm[model.wv.vocab['graph'].index]
        sims2 = model.most_similar(positive=[graph_vector], topn=11)
        sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
        self.assertEqual(sims, sims2)

        # build vocab and train in one step; must be the same as above
        model2 = FastText(sentences, size=2, min_count=1, hs=1, negative=0)
        self.models_equal(model, model2)

    def model_sanity(self, model, train=True):
        """Even tiny models trained on LeeCorpus should pass these sanity checks"""
        # run extra before/after training tests if train=True
        """
        if train:
            model.build_vocab(list_corpus)
            orig0 = np.copy(model.wv.syn0[0])
            model.train(list_corpus, total_examples=model.corpus_count, epochs=model.iter)
            self.assertFalse((orig0 == model.wv.syn0[1]).all())  # vector should vary after training
        sims = model.most_similar('war', topn=len(model.wv.index2word))
        t_rank = [word for word, score in sims].index('terrorism')
        # in >200 calibration runs w/ calling parameters, 'terrorism' in 50-most_sim for 'war'
        self.assertLess(t_rank, 150)
        war_vec = model['war']
        sims2 = model.most_similar([war_vec], topn=151)
        self.assertTrue('war' in [word for word, score in sims2])
        self.assertTrue('terrorism' in [word for word, score in sims2])"""
        pass

    def test_sg_neg(self):
        """Test skipgram w/ negative sampling"""
        model = FastText(sg=1, window=4, hs=0, negative=15, min_count=5, iter=10)
        self.model_sanity(model)

    def test_sg_hs(self):
        """Test skipgram w/ hierarchical softmax"""
        model = FastText(sg=1, window=4, hs=1, negative=0, min_count=5, iter=10)
        self.model_sanity(model)


    def test_cbow_hs(self):
        """Test CBOW w/ hierarchical softmax"""
        model = FastText(sg=0, cbow_mean=1, alpha=0.05, window=8, hs=1, negative=0,
                                  min_count=5, iter=10, batch_words=1000)
        self.model_sanity(model)

    def test_cbow_neg(self):
        """Test CBOW w/ negative sampling"""
        model = FastText(sg=0, cbow_mean=1, alpha=0.05, window=5, hs=0, negative=15,
                                  min_count=5, iter=10, sample=0)
        self.model_sanity(model)

    def testTrainingCbow(self):
        """Test CBOW word2vec training."""
        pass

    def testTrainingSgNegative(self):
        """Test skip-gram (negative sampling) word2vec training."""
        pass

    def testTrainingCbowNegative(self):
        """Test CBOW (negative sampling) FastText training."""
        pass


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
