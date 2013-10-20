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

        # filter out *all* words, make sure things still work as expected
        self.assertRaises(RuntimeError, word2vec.Word2Vec, corpus, min_count=total_words+1)


    def testTraining(self):
        """Test word2vec training."""
        pass


    def testResumeTraining(self):
        """Test resuming training from a trained word2vec model."""
        pass


    def testAccuracy(self):
        pass


    def testParallel(self):
        pass


    def testPersistence(self):
        """Test storing/loading the entire model."""
        model = word2vec.Word2Vec(LeeCorpus(), min_count=2)
        model.save(testfile())
        model2 = word2vec.Word2Vec.load(testfile())
        self.assertEqual(model.min_count, model2.min_count)
        self.assertTrue(numpy.allclose(model.syn0, model2.syn0))
        self.assertTrue(numpy.allclose(model.syn1, model2.syn1))
        self.assertTrue(numpy.allclose(model['the'], model2['the']))
#endclass TestWord2VecModel


if __name__ == '__main__':
    logging.root.setLevel(logging.DEBUG)
    unittest.main()
