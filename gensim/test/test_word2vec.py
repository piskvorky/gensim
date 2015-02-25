#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""


import logging
import unittest
import os
import tempfile
import itertools
import bz2

import numpy

from gensim import utils, matutils
from gensim.models import word2vec

module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


class LeeCorpus(object):
    def __iter__(self):
        with open(datapath('lee_background.cor')) as f:
            for line in f:
                yield utils.simple_preprocess(line)


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
    return os.path.join(tempfile.gettempdir(), 'gensim_word2vec.tst')


class TestWord2VecModel(unittest.TestCase):
    def testPersistence(self):
        """Test storing/loading the entire model."""
        model = word2vec.Word2Vec(sentences, min_count=1)
        model.save(testfile())
        self.models_equal(model, word2vec.Word2Vec.load(testfile()))

    def testPersistenceWord2VecFormat(self):
        """Test storing/loading the entire model in word2vec format."""
        model = word2vec.Word2Vec(sentences, min_count=1)
        model.init_sims()
        model.save_word2vec_format(testfile(), binary=True)
        binary_model = word2vec.Word2Vec.load_word2vec_format(testfile(), binary=True, norm_only=False)
        self.assertTrue(numpy.allclose(model['human'], binary_model['human']))
        norm_only_model = word2vec.Word2Vec.load_word2vec_format(testfile(), binary=True, norm_only=True)
        self.assertFalse(numpy.allclose(model['human'], norm_only_model['human']))
        self.assertTrue(numpy.allclose(model.syn0norm[model.vocab['human'].index], norm_only_model['human']))

    def testPersistenceWord2VecFormatNonBinary(self):
        """Test storing/loading the entire model in word2vec non-binary format."""
        model = word2vec.Word2Vec(sentences, min_count=1)
        model.init_sims()
        model.save_word2vec_format(testfile(), binary=False)
        text_model = word2vec.Word2Vec.load_word2vec_format(testfile(), binary=False, norm_only=False)
        self.assertTrue(numpy.allclose(model['human'], text_model['human'], atol=1e-6))
        norm_only_model = word2vec.Word2Vec.load_word2vec_format(testfile(), binary=False, norm_only=True)
        self.assertFalse(numpy.allclose(model['human'], norm_only_model['human'], atol=1e-6))
        
        self.assertTrue(numpy.allclose(model.syn0norm[model.vocab['human'].index], norm_only_model['human'], atol=1e-4))

    def testPersistenceWord2VecFormatWithVocab(self):
        """Test storing/loading the entire model and vocabulary in word2vec format."""
        model = word2vec.Word2Vec(sentences, min_count=1)
        model.init_sims()
        testvocab = os.path.join(tempfile.gettempdir(), 'gensim_word2vec.vocab')
        model.save_word2vec_format(testfile(), testvocab, binary=True)
        binary_model_with_vocab = word2vec.Word2Vec.load_word2vec_format(testfile(), testvocab, binary=True)
        self.assertEqual(model.vocab['human'].count, binary_model_with_vocab.vocab['human'].count)

    def testLargeMmap(self):
        """Test storing/loading the entire model."""
        model = word2vec.Word2Vec(sentences, min_count=1)

        # test storing the internal arrays into separate files
        model.save(testfile(), sep_limit=0)
        self.models_equal(model, word2vec.Word2Vec.load(testfile()))

        # make sure mmaping the arrays back works, too
        self.models_equal(model, word2vec.Word2Vec.load(testfile(), mmap='r'))

    def testVocab(self):
        """Test word2vec vocabulary building."""
        corpus = LeeCorpus()
        total_words = sum(len(sentence) for sentence in corpus)

        # try vocab building explicitly, using all words
        model = word2vec.Word2Vec(min_count=1)
        model.build_vocab(corpus)
        self.assertTrue(len(model.vocab) == 6981)
        # with min_count=1, we're not throwing away anything, so make sure the word counts add up to be the entire corpus
        self.assertEqual(sum(v.count for v in model.vocab.values()),
                         total_words)
        # make sure the binary codes are correct
        numpy.allclose(model.vocab['the'].code, [1, 1, 0, 0])

        # test building vocab with default params
        model = word2vec.Word2Vec()
        model.build_vocab(corpus)
        self.assertTrue(len(model.vocab) == 1750)
        numpy.allclose(model.vocab['the'].code, [1, 1, 1, 0])

        # no input => "RuntimeError: you must first build vocabulary before training the model"
        self.assertRaises(RuntimeError, word2vec.Word2Vec, [])

        # input not empty, but rather completely filtered out
        self.assertRaises(RuntimeError, word2vec.Word2Vec, corpus, min_count=total_words+1)

    def testTraining(self):
        """Test word2vec training."""
        # build vocabulary, don't train yet
        model = word2vec.Word2Vec(size=2, min_count=1)
        model.build_vocab(sentences)

        self.assertTrue(model.syn0.shape == (len(model.vocab), 2))
        self.assertTrue(model.syn1.shape == (len(model.vocab), 2))

        model.train(sentences)
        sims = model.most_similar('graph', topn=10)
        # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # test querying for "most similar" by vector
        graph_vector = model.syn0norm[model.vocab['graph'].index]
        sims2 = model.most_similar(positive=[graph_vector], topn=11)
        self.assertEqual(sims, sims2[1:])  # ignore first element of sims2, which is 'graph' itself

        # build vocab and train in one step; must be the same as above
        model2 = word2vec.Word2Vec(sentences, size=2, min_count=1)
        self.models_equal(model, model2)

    def testTrainingCbow(self):
        """Test CBOW word2vec training."""
        # to test training, make the corpus larger by repeating its sentences over and over
        # build vocabulary, don't train yet
        model = word2vec.Word2Vec(size=2, min_count=1, sg=0)
        model.build_vocab(sentences)
        self.assertTrue(model.syn0.shape == (len(model.vocab), 2))
        self.assertTrue(model.syn1.shape == (len(model.vocab), 2))

        model.train(sentences)
        sims = model.most_similar('graph', topn=10)
        # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # test querying for "most similar" by vector
        graph_vector = model.syn0norm[model.vocab['graph'].index]
        sims2 = model.most_similar(positive=[graph_vector], topn=11)
        self.assertEqual(sims, sims2[1:])  # ignore first element of sims2, which is 'graph' itself

        # build vocab and train in one step; must be the same as above
        model2 = word2vec.Word2Vec(sentences, size=2, min_count=1, sg=0)
        self.models_equal(model, model2)

    def testTrainingSgNegative(self):
        """Test skip-gram (negative sampling) word2vec training."""
        # to test training, make the corpus larger by repeating its sentences over and over
        # build vocabulary, don't train yet
        model = word2vec.Word2Vec(size=2, min_count=1, hs=0, negative=2)
        model.build_vocab(sentences)
        self.assertTrue(model.syn0.shape == (len(model.vocab), 2))
        self.assertTrue(model.syn1neg.shape == (len(model.vocab), 2))

        model.train(sentences)
        sims = model.most_similar('graph', topn=10)
        # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # test querying for "most similar" by vector
        graph_vector = model.syn0norm[model.vocab['graph'].index]
        sims2 = model.most_similar(positive=[graph_vector], topn=11)
        self.assertEqual(sims, sims2[1:])  # ignore first element of sims2, which is 'graph' itself

        # build vocab and train in one step; must be the same as above
        model2 = word2vec.Word2Vec(sentences, size=2, min_count=1, hs=0, negative=2)
        self.models_equal(model, model2)

    def testTrainingCbowNegative(self):
        """Test CBOW (negative sampling) word2vec training."""
        # to test training, make the corpus larger by repeating its sentences over and over
        # build vocabulary, don't train yet
        model = word2vec.Word2Vec(size=2, min_count=1, sg=0, hs=0, negative=2)
        model.build_vocab(sentences)
        self.assertTrue(model.syn0.shape == (len(model.vocab), 2))
        self.assertTrue(model.syn1neg.shape == (len(model.vocab), 2))

        model.train(sentences)
        sims = model.most_similar('graph', topn=10)
        # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # test querying for "most similar" by vector
        graph_vector = model.syn0norm[model.vocab['graph'].index]
        sims2 = model.most_similar(positive=[graph_vector], topn=11)
        self.assertEqual(sims, sims2[1:])  # ignore first element of sims2, which is 'graph' itself

        # build vocab and train in one step; must be the same as above
        model2 = word2vec.Word2Vec(sentences, size=2, min_count=1, sg=0, hs=0, negative=2)
        self.models_equal(model, model2)

    def testSimilarities(self):
        """Test similarity and n_similarity methods."""
        # The model is trained using CBOW
        model = word2vec.Word2Vec(size=2, min_count=1, sg=0, hs=0, negative=2)
        model.build_vocab(sentences)
        model.train(sentences)

        self.assertTrue(model.n_similarity(['graph', 'trees'], ['trees', 'graph']))
        self.assertTrue(model.n_similarity(['graph'], ['trees']) == model.similarity('graph', 'trees'))

    def testParallel(self):
        """Test word2vec parallel training."""
        if word2vec.FAST_VERSION < 0:  # don't test the plain NumPy version for parallelism (too slow)
            return

        corpus = utils.RepeatCorpus(LeeCorpus(), 10000)

        for workers in [2, 4]:
            model = word2vec.Word2Vec(corpus, workers=workers)
            sims = model.most_similar('israeli')
            # the exact vectors and therefore similarities may differ, due to different thread collisions/randomization
            # so let's test only for top3
            # TODO: commented out for now; find a more robust way to compare against "gold standard"
            # self.assertTrue('palestinian' in [sims[i][0] for i in range(3)])

    def testRNG(self):
        """Test word2vec results identical with identical RNG seed."""
        model = word2vec.Word2Vec(sentences, min_count=2, seed=42, workers=1)
        model2 = word2vec.Word2Vec(sentences, min_count=2, seed=42, workers=1)
        self.models_equal(model, model2)

    def models_equal(self, model, model2):
        self.assertEqual(len(model.vocab), len(model2.vocab))
        self.assertTrue(numpy.allclose(model.syn0, model2.syn0))
        if model.hs:
            self.assertTrue(numpy.allclose(model.syn1, model2.syn1))
        if model.negative:
            self.assertTrue(numpy.allclose(model.syn1neg, model2.syn1neg))
        most_common_word = max(model.vocab.items(), key=lambda item: item[1].count)[0]
        self.assertTrue(numpy.allclose(model[most_common_word], model2[most_common_word]))
#endclass TestWord2VecModel


class TestWord2VecSentenceIterators(unittest.TestCase):
    def testLineSentenceWorksWithFilename(self):
        """Does LineSentence work with a filename argument?"""
        with utils.smart_open(datapath('lee_background.cor')) as orig:
            sentences = word2vec.LineSentence(datapath('lee_background.cor'))
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig.readline()).split())

    def testLineSentenceWorksWithCompressedFile(self):
        """Does LineSentence work with a compressed file object argument?"""
        with utils.smart_open(datapath('head500.noblanks.cor')) as orig:
            sentences = word2vec.LineSentence(bz2.BZ2File(datapath('head500.noblanks.cor.bz2')))
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig.readline()).split())

    def testLineSentenceWorksWithNormalFile(self):
        """Does LineSentence work with a file object argument, rather than filename?"""
        with utils.smart_open(datapath('head500.noblanks.cor')) as orig:
            with utils.smart_open(datapath('head500.noblanks.cor')) as fin:
                sentences = word2vec.LineSentence(fin)
                for words in sentences:
                    self.assertEqual(words, utils.to_unicode(orig.readline()).split())
#endclass TestWord2VecSentenceIterators


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    logging.info("using optimization %s" % word2vec.FAST_VERSION)
    unittest.main()
