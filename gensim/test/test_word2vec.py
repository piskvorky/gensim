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

import numpy

from gensim import utils, matutils
from gensim.models import word2vec

module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


class LeeCorpus(object):
    def __iter__(self):
        for line in open(datapath('lee_background.cor')):
            yield utils.simple_preprocess(line)


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_models.tst')


class TestWord2VecModel(unittest.TestCase):
    def testPersistence(self):
        """Test storing/loading the entire model."""
        model = word2vec.Word2Vec(LeeCorpus(), min_count=2)
        model.save(testfile())
        self.models_equal(model, word2vec.Word2Vec.load(testfile()))

    def testVocab(self):
        """Test word2vec vocabulary building."""
        corpus = LeeCorpus()
        total_words = sum(len(sentence) for sentence in corpus)

        # try vocab building explicitly, using all words
        model = word2vec.Word2Vec(min_count=1)
        model.build_vocab(corpus)
        self.assertTrue(len(model.vocab) == 6981)
        # with min_count=1, we're not throwing away anything, so make sure the word counts add up to be the entire corpus
        self.assertTrue(sum(v.count for v in model.vocab.itervalues()) == total_words)
        # make sure the binary codes are correct
        numpy.allclose(model.vocab['the'].code, [1, 1, 0, 0])

        # test building directly from constructor, with default params
        model = word2vec.Word2Vec(corpus)
        self.assertTrue(len(model.vocab) == 1750)
        numpy.allclose(model.vocab['the'].code, [1, 1, 1, 0])

        # no input => "RuntimeError: you must first build vocabulary before training the model"
        self.assertRaises(RuntimeError, word2vec.Word2Vec, [])

        # input not empty, but rather completely filtered out
        self.assertRaises(RuntimeError, word2vec.Word2Vec, corpus, min_count=total_words+1)


    def testTraining(self):
        """Test word2vec training."""
        # for training, make the corpus larger by repeating its sentences 10k times
        corpus = lambda: itertools.islice(itertools.cycle(LeeCorpus()), 10000)

        # build vocabulary, don't train yet
        model = word2vec.Word2Vec(size=50)
        model.build_vocab(corpus())
        self.assertTrue(model.syn0.shape == (len(model.vocab), 50))
        self.assertTrue(model.syn1.shape == (len(model.vocab), 50))

        model.train(corpus())
        sims = model.most_similar('israeli')
        self.assertTrue(sims[0][0] == 'palestinian', sims)  # most similar

        # build vocab and train in one step; must be the same as above
        model2 = word2vec.Word2Vec(utils.RepeatCorpus(LeeCorpus(), 10000), size=50)
        self.models_equal(model, model2)


    def testParallel(self):
        """Test word2vec training."""
        corpus = utils.RepeatCorpus(LeeCorpus(), 20000)

        for workers in [2, 4]:
            model = word2vec.Word2Vec(corpus, workers=workers)
            sims = model.most_similar('israeli')
            # the exact vectors and therefore similarities may differ, due to different thread collisions
            # so let's test only for top3
            self.assertTrue('palestinian' in [sims[i][0] for i in xrange(3)])


    def testRNG(self):
        """Test word2vec results identical with identical RNG seed."""
        model = word2vec.Word2Vec(LeeCorpus(), min_count=2, seed=42, workers=1)
        model2 = word2vec.Word2Vec(LeeCorpus(), min_count=2, seed=42, workers=1)
        self.models_equal(model, model2)


    def models_equal(self, model, model2):
        self.assertEqual(len(model.vocab), len(model2.vocab))
        self.assertTrue(numpy.allclose(model.syn0, model2.syn0))
        self.assertTrue(numpy.allclose(model.syn1, model2.syn1))
        self.assertTrue(numpy.allclose(model['the'], model2['the']))
#endclass TestWord2VecModel


if __name__ == '__main__':
    logging.root.setLevel(logging.DEBUG)
    unittest.main()
