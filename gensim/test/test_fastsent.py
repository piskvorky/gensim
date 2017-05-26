#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""

import sys
import logging
import unittest
import os
import tempfile
import itertools
import bz2
import numpy



from gensim import utils, matutils
from gensim.utils import check_output
from subprocess import PIPE
from imp import load_source
from gensim.models import fastsent


module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
#local_gensim = os.path.join(os.path.dirname(__file__), os.pardir)
#fastsent_path = os.path.join(local_gensim, 'models')
#sys.path.append(fastsent_path)
#import fastsent, fastsent_inner

datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


class LeeCorpus(object):
    def __iter__(self):
        with open(datapath('lee_background.cor')) as f:
            for line in f:
                yield utils.simple_preprocess(line)

list_corpus = list(LeeCorpus())

sentences = [
    ['the','very', 'interface', 'computer','is','red'],
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
    return os.path.join(tempfile.gettempdir(), 'gensim_fastsent.tst')

class TestFastSentModel(unittest.TestCase):
    def testPersistence(self):
        """Test storing/loading the entire model."""
        model = fastsent.FastSent(sentences, min_count=1)
        model.save(testfile())
        self.models_equal(model, fastsent.FastSent.load(testfile()))

    def testPersistenceWithMinCount(self):
        """Test storing/loading the entire model with min count."""
        model = fastsent.FastSent(sentences, min_count=2)
        model.save(testfile())
        self.models_equal(model, fastsent.FastSent.load(testfile()))

    def testMinCount(self):
        """Test min_count is effective when building vocab."""
        model = fastsent.FastSent(sentences + [["occurs_only_once"]], min_count=2)
        self.assertTrue("human" not in model.vocab)
        self.assertTrue("occurs_only_once" not in model.vocab)
        self.assertTrue("interface" in model.vocab)

    def testPersistenceFastSentFormat(self):
        """Test storing/loading the entire model in fastsent format."""
        model = fastsent.FastSent(sentences, min_count=1)
        model.init_sims()
        model.save_fastsent_format(testfile(), binary=True)
        binary_model = fastsent.FastSent.load_fastsent_format(testfile(), binary=True)
        binary_model.init_sims(replace=False)
        self.assertTrue(numpy.allclose(model['human'], binary_model['human']))
        norm_only_model = fastsent.FastSent.load_fastsent_format(testfile(), binary=True)
        norm_only_model.init_sims(replace=True)
        self.assertFalse(numpy.allclose(model['human'], norm_only_model['human']))
        self.assertTrue(numpy.allclose(model.syn0norm[model.vocab['human'].index], norm_only_model['human']))

    def testPersistenceFastSentFormatNonBinary(self):
        """Test storing/loading the entire model in fastsent non-binary format."""
        model = fastsent.FastSent(sentences, min_count=1)
        model.init_sims()
        model.save_fastsent_format(testfile(), binary=False)
        text_model = fastsent.FastSent.load_fastsent_format(testfile(), binary=False)
        text_model.init_sims(False)
        self.assertTrue(numpy.allclose(model['human'], text_model['human'], atol=1e-6))
        norm_only_model = fastsent.FastSent.load_fastsent_format(testfile(), binary=False)
        norm_only_model.init_sims(True)
        self.assertFalse(numpy.allclose(model['human'], norm_only_model['human'], atol=1e-6))

        self.assertTrue(numpy.allclose(model.syn0norm[model.vocab['human'].index], norm_only_model['human'], atol=1e-4))

    def testPersistenceFastSentFormatWithVocab(self):
        """Test storing/loading the entire model and vocabulary in fastsent format."""
        model = fastsent.FastSent(sentences, min_count=1)
        model.init_sims()
        testvocab = os.path.join(tempfile.gettempdir(), 'gensim_fastsent.vocab')
        model.save_fastsent_format(testfile(), testvocab, binary=True)
        binary_model_with_vocab = fastsent.FastSent.load_fastsent_format(testfile(), testvocab, binary=True)
        self.assertEqual(model.vocab['human'].count, binary_model_with_vocab.vocab['human'].count)

    def testPersistenceFastSentFormatCombinationWithStandardPersistence(self):
        """Test storing/loading the entire model and vocabulary in fastsent format chained with
         saving and loading via `save` and `load` methods`."""
        model = fastsent.FastSent(sentences, min_count=1)
        model.init_sims()
        testvocab = os.path.join(tempfile.gettempdir(), 'gensim_fastsent.vocab')
        model.save_fastsent_format(testfile(), testvocab, binary=True)
        binary_model_with_vocab = fastsent.FastSent.load_fastsent_format(testfile(), testvocab, binary=True)
        binary_model_with_vocab.save(testfile())
        binary_model_with_vocab = fastsent.FastSent.load(testfile())
        self.assertEqual(model.vocab['human'].count, binary_model_with_vocab.vocab['human'].count)

    def testLargeMmap(self):
        """Test storing/loading the entire model."""
        model = fastsent.FastSent(sentences, min_count=1)

        # test storing the internal arrays into separate files
        model.save(testfile(), sep_limit=0)
        self.models_equal(model, fastsent.FastSent.load(testfile()))

        # make sure mmaping the arrays back works, too
        self.models_equal(model, fastsent.FastSent.load(testfile(), mmap='r'))

    def testVocab(self):
        """Test fastsent vocabulary building."""
        corpus = LeeCorpus()
        total_words = sum(len(sentence) for sentence in corpus)

        # try vocab building explicitly, using all words
        model = fastsent.FastSent(min_count=1)
        model.build_vocab(corpus)
        self.assertTrue(len(model.vocab) == 6981)
        # with min_count=1, we're not throwing away anything, so make sure the word counts add up to be the entire corpus
        self.assertEqual(sum(v.count for v in model.vocab.values()), total_words)
        # make sure the binary codes are correct
        numpy.allclose(model.vocab['the'].code, [1, 1, 0, 0])

        # test building vocab with default params
        model = fastsent.FastSent()
        model.build_vocab(corpus)
        self.assertTrue(len(model.vocab) == 1750)
        numpy.allclose(model.vocab['the'].code, [1, 1, 1, 0])

        # no input => "RuntimeError: you must first build vocabulary before training the model"
        self.assertRaises(RuntimeError, fastsent.FastSent, [])

        # input not empty, but rather completely filtered out
        self.assertRaises(RuntimeError, fastsent.FastSent, corpus, min_count=total_words+1)

    def testTraining(self):
        """Test fastsent training."""
        # build vocabulary, don't train yet
        model = fastsent.FastSent(size=2, min_count=1)
        model.build_vocab(sentences)

        self.assertTrue(model.syn0.shape == (len(model.vocab), 2))
        self.assertTrue(model.syn1.shape == (len(model.vocab), 2))

        model.train(sentences)
        sim1 = model.sentence_similarity('the human is red', 'the graph is human')
        # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # test feedforward model works as it should
        word_vecs1 = numpy.array([model[w] for w in 'the human is red'.split()])
        word_vecs2 = numpy.array([model[w] for w in 'the graph is human'.split()])
        if model.fastsent_mean:
            sent1 = numpy.mean(word_vecs1, axis=0)
            sent2 = numpy.mean(word_vecs2, axis=0)
        else:
            sent1 = numpy.sum(word_vecs1, axis=0)
            sent2 = numpy.sum(word_vecs2, axis=0)
        sim2 = numpy.dot(matutils.unitvec(sent1), matutils.unitvec(sent2))
        self.assertEqual(sim1, sim2)

        # build vocab and train in one step; must be the same as above
        model2 = fastsent.FastSent(sentences, size=2, min_count=1)
        self.models_equal(model, model2)

    def testLocking(self):
        """Test fastsent training doesn't change locked vectors."""
        corpus = LeeCorpus()
        # build vocabulary, don't train yet
        model = fastsent.FastSent(size=4, min_count=1, window=5)
        model.build_vocab(corpus)

        # remember two vectors
        locked0 = numpy.copy(model.syn0[0])
        unlocked1 = numpy.copy(model.syn0[1])
        # lock the vector in slot 0 against change
        model.syn0_lockf[0] = 0.0

        model.train(corpus)
        self.assertFalse((unlocked1 == model.syn0[1]).all())  # unlocked vector should vary
        self.assertTrue((locked0 == model.syn0[0]).all())  # locked vector should not vary

    def model_sanity(self, model, train=True):
        """Even tiny models trained on LeeCorpus should pass these sanity checks"""
        # run extra before/after training tests if train=True
        if train:
            model.build_vocab(list_corpus)
            orig0 = numpy.copy(model.syn0[0])
            model.train(list_corpus)
            self.assertFalse((orig0 == model.syn0[1]).all())  # vector should vary after training
        sim1 = model.sentence_similarity('the child is human', 'the road is human')
        sim2 = model.sentence_similarity('the child is red', 'sport closed home')
        self.assertTrue(sim1 > sim2)

    def test_fastsent(self):
        """Test skipgram w/ hierarchical softmax"""
        model = fastsent.FastSent(window=4, min_count=1, iter=10, workers=2)
        self.model_sanity(model)

    def testSimilarities(self):
        """Test similarity and n_similarity methods."""
        # The model is trained using CBOW
        model = fastsent.FastSent(size=2, min_count=1)
        model.build_vocab(sentences)
        model.train(sentences)

        self.assertTrue(model.sentence_similarity('the human is red', 'the graph is human'))

    def testParallel(self):
        """Test fastsent parallel training."""
        if fastsent.FAST_VERSION < 0:  # don't test the plain NumPy version for parallelism (too slow)
            return

        corpus = utils.RepeatCorpus(LeeCorpus(), 10000)

        for workers in [2, 4]:
            model = fastsent.FastSent(corpus, workers=workers, min_count=1)
            model.sentence_similarity('the human is red', 'the graph is human')
            # the exact vectors and therefore similarities may differ, due to different thread collisions/randomization
            # so let's test only for top3
            # TODO: commented out for now; find a more robust way to compare against "gold standard"
            # self.assertTrue('palestinian' in [sims[i][0] for i in range(3)])

    def testRNG(self):
        """Test fastsent results identical with identical RNG seed."""
        model = fastsent.FastSent(sentences, min_count=2, seed=42, workers=1)
        model2 = fastsent.FastSent(sentences, min_count=2, seed=42, workers=1)
        self.models_equal(model, model2)

    def models_equal(self, model, model2):
        self.assertEqual(len(model.vocab), len(model2.vocab))
        self.assertTrue(numpy.allclose(model.syn0, model2.syn0))
        self.assertTrue(numpy.allclose(model.syn1, model2.syn1))
        most_common_word = max(model.vocab.items(), key=lambda item: item[1].count)[0]
        self.assertTrue(numpy.allclose(model[most_common_word], model2[most_common_word]))
#endclass TestFastSentModel

    def test_sentences_should_not_be_a_generator(self):
        """
        Is sentences a generator object?
        """
        gen = (s for s in sentences)
        self.assertRaises(TypeError, fastsent.FastSent, (gen,))


class TestFastSentSentenceIterators(unittest.TestCase):
    def testLineSentenceWorksWithFilename(self):
        """Does LineSentence work with a filename argument?"""
        with utils.smart_open(datapath('lee_background.cor')) as orig:
            sentences = fastsent.LineSentence(datapath('lee_background.cor'))
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig.readline()).split())

    def testLineSentenceWorksWithCompressedFile(self):
        """Does LineSentence work with a compressed file object argument?"""
        with utils.smart_open(datapath('head500.noblanks.cor')) as orig:
            sentences = fastsent.LineSentence(bz2.BZ2File(datapath('head500.noblanks.cor.bz2')))
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig.readline()).split())

    def testLineSentenceWorksWithNormalFile(self):
        """Does LineSentence work with a file object argument, rather than filename?"""
        with utils.smart_open(datapath('head500.noblanks.cor')) as orig:
            with utils.smart_open(datapath('head500.noblanks.cor')) as fin:
                sentences = fastsent.LineSentence(fin)
                for words in sentences:
                    self.assertEqual(words, utils.to_unicode(orig.readline()).split())
#endclass TestFastSentSentenceIterators

# TODO: get correct path to Python binary
# class TestFastSentScripts(unittest.TestCase):
#     def testFastSentStandAloneScript(self):
#         """Does FastSent script launch standalone?"""
#         cmd = 'python -m gensim.scripts.fastsent_standalone -train ' + datapath('testcorpus.txt') + ' -output vec.txt -size 200 -sample 1e-4 -binary 0 -iter 3 -min_count 1'
#         output = check_output(cmd, stderr=PIPE)
#         self.assertEqual(output, '0')
# #endclass TestFastSentScripts


if not hasattr(TestFastSentModel, 'assertLess'):
    # workaround for python 2.6
    def assertLess(self, a, b, msg=None):
        self.assertTrue(a < b, msg="%s is not less than %s" % (a, b))

    setattr(TestFastSentModel, 'assertLess', assertLess)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.DEBUG)
    logging.info("using optimization %s", fastsent.FAST_VERSION)
    unittest.main()
