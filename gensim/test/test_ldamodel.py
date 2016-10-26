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
import os.path
import tempfile
import numbers

import six
import numpy
import scipy.linalg

from gensim.corpora import mmcorpus, Dictionary
from gensim.models import ldamodel, ldamulticore
from gensim import matutils
from gensim.test import basetests


module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


# set up vars used in testing ("Deerwester" from the web tutorial)
texts = [['human', 'interface', 'computer'],
 ['survey', 'user', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'system'],
 ['system', 'human', 'system', 'eps'],
 ['user', 'response', 'time'],
 ['trees'],
 ['graph', 'trees'],
 ['graph', 'minors', 'trees'],
 ['graph', 'minors', 'survey']]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_models.tst')


def testRandomState():
    testcases = [numpy.random.seed(0), None, numpy.random.RandomState(0), 0]
    for testcase in testcases:
        assert(isinstance(ldamodel.get_random_state(testcase), numpy.random.RandomState))


class TestLdaModel(unittest.TestCase, basetests.TestBaseTopicModel):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = ldamodel.LdaModel
        self.model = self.class_(corpus, id2word=dictionary, num_topics=2, passes=100)

    def testTransform(self):
        passed = False
        # sometimes, LDA training gets stuck at a local minimum
        # in that case try re-training the model from scratch, hoping for a
        # better random initialization
        for i in range(25): # restart at most 5 times
            # create the transformation model
            model = self.class_(id2word=dictionary, num_topics=2, passes=100)
            model.update(self.corpus)

            # transform one document
            doc = list(corpus)[0]
            transformed = model[doc]

            vec = matutils.sparse2full(transformed, 2) # convert to dense vector, for easier equality tests
            expected = [0.13, 0.87]
            passed = numpy.allclose(sorted(vec), sorted(expected), atol=1e-1) # must contain the same values, up to re-ordering
            if passed:
                break
            logging.warning("LDA failed to converge on attempt %i (got %s, expected %s)" %
                            (i, sorted(vec), sorted(expected)))
        self.assertTrue(passed)

    def testAlphaAuto(self):
        model1 = self.class_(corpus, id2word=dictionary, alpha='symmetric', passes=10)
        modelauto = self.class_(corpus, id2word=dictionary, alpha='auto', passes=10)

        # did we learn something?
        self.assertFalse(all(numpy.equal(model1.alpha, modelauto.alpha)))

    def testAlpha(self):
        kwargs = dict(
            id2word=dictionary,
            num_topics=2,
            alpha=None
        )
        expected_shape = (2,)

        # should not raise anything
        self.class_(**kwargs)

        kwargs['alpha'] = 'symmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == numpy.array([0.5, 0.5])))

        kwargs['alpha'] = 'asymmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(numpy.allclose(model.alpha, [0.630602, 0.369398]))

        kwargs['alpha'] = 0.3
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == numpy.array([0.3, 0.3])))

        kwargs['alpha'] = 3
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == numpy.array([3, 3])))

        kwargs['alpha'] = [0.3, 0.3]
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == numpy.array([0.3, 0.3])))

        kwargs['alpha'] = numpy.array([0.3, 0.3])
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == numpy.array([0.3, 0.3])))

        # all should raise an exception for being wrong shape
        kwargs['alpha'] = [0.3, 0.3, 0.3]
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['alpha'] = [[0.3], [0.3]]
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['alpha'] = [0.3]
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['alpha'] = "gensim is cool"
        self.assertRaises(ValueError, self.class_, **kwargs)


    def testEtaAuto(self):
        model1 = self.class_(corpus, id2word=dictionary, eta='symmetric', passes=10)
        modelauto = self.class_(corpus, id2word=dictionary, eta='auto', passes=10)

        # did we learn something?
        self.assertFalse(all(numpy.equal(model1.eta, modelauto.eta)))

    def testEta(self):
        kwargs = dict(
            id2word=dictionary,
            num_topics=2,
            eta=None
        )
        expected_shape = (2, 1)

        # should not raise anything
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == numpy.array([[0.5], [0.5]])))

        kwargs['eta'] = 'symmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == numpy.array([[0.5], [0.5]])))

        kwargs['eta'] = 'asymmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(numpy.allclose(model.eta, [[0.630602], [0.369398]]))

        kwargs['eta'] = 0.3
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == numpy.array([[0.3], [0.3]])))

        kwargs['eta'] = 3
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == numpy.array([[3], [3]])))

        kwargs['eta'] = [[0.3], [0.3]]
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == numpy.array([[0.3], [0.3]])))

        kwargs['eta'] = [0.3, 0.3]
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == numpy.array([[0.3], [0.3]])))

        kwargs['eta'] = numpy.array([0.3, 0.3])
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == numpy.array([[0.3], [0.3]])))

        # should be ok with num_topics x num_terms
        testeta = numpy.array([[0.5] * len(dictionary)] * 2)
        kwargs['eta'] = testeta
        self.class_(**kwargs)

        # all should raise an exception for being wrong shape
        kwargs['eta'] = testeta.reshape(tuple(reversed(testeta.shape)))
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['eta'] = [0.3, 0.3, 0.3]
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['eta'] = [0.3]
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['eta'] = "gensim is cool"
        self.assertRaises(ValueError, self.class_, **kwargs)

    def testTopTopics(self):
        top_topics = self.model.top_topics(self.corpus)

        for topic, score in top_topics:
            self.assertTrue(isinstance(topic, list))
            self.assertTrue(isinstance(score, float))

            for v, k in topic:
                self.assertTrue(isinstance(k, six.string_types))
                self.assertTrue(isinstance(v, float))

    def testGetTopicTerms(self):
        topic_terms = self.model.get_topic_terms(1)

        for k, v in topic_terms:
            self.assertTrue(isinstance(k, numbers.Integral))
            self.assertTrue(isinstance(v, float))

    def testGetDocumentTopics(self):

        model = self.class_(self.corpus, id2word=dictionary, num_topics=2, passes= 100, random_state=numpy.random.seed(0))

        doc_topics = model.get_document_topics(self.corpus)

        for topic in doc_topics:
            self.assertTrue(isinstance(topic, list))
            for k, v in topic:
                self.assertTrue(isinstance(k, int))
                self.assertTrue(isinstance(v, float))

        doc_topics, word_topics, word_phis = model.get_document_topics(self.corpus[1], per_word_topics=True)

        for k, v in doc_topics:
            self.assertTrue(isinstance(k, int))
            self.assertTrue(isinstance(v, float))

        for w, topic_list in word_topics:
            self.assertTrue(isinstance(w, int))
            self.assertTrue(isinstance(topic_list, list))

        for w, phi_values in word_phis:
            self.assertTrue(isinstance(w, int))
            self.assertTrue(isinstance(phi_values, list))

        # word_topics looks like this: ({word_id => [topic_id_most_probable, topic_id_second_most_probable, ...]).
        # we check one case in word_topics, i.e of the first word in the doc, and it's likely topics.
        expected_word = 0
        # FIXME: Fails on osx and win
        # self.assertEqual(word_topics[0][0], expected_word)
        # self.assertTrue(0 in word_topics[0][1])

    def testTermTopics(self):

        model = self.class_(self.corpus, id2word=dictionary, num_topics=2, passes=100, random_state=numpy.random.seed(0))

        # check with word_type
        result = model.get_term_topics(2)
        for topic_no, probability in result:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(probability, float))

        # checks if topic '1' is in the result list
         # FIXME: Fails on osx and win
         # self.assertTrue(1 in result[0])


        # if user has entered word instead, check with word
        result = model.get_term_topics(str(model.id2word[2]))
        for topic_no, probability in result:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(probability, float))

        # checks if topic '1' is in the result list
         # FIXME: Fails on osx and win
         # self.assertTrue(1 in result[0])


    def testPasses(self):
        # long message includes the original error message with a custom one
        self.longMessage = True

        # construct what we expect when passes aren't involved
        test_rhots = list()
        model = self.class_(id2word=dictionary, chunksize=1, num_topics=2)
        final_rhot = lambda: pow(model.offset + (1 * model.num_updates) / model.chunksize, -model.decay)

        # generate 5 updates to test rhot on
        for x in range(5):
            model.update(self.corpus)
            test_rhots.append(final_rhot())

        for passes in [1, 5, 10, 50, 100]:
            model = self.class_(id2word=dictionary, chunksize=1, num_topics=2, passes=passes)
            self.assertEqual(final_rhot(), 1.0)
            # make sure the rhot matches the test after each update
            for test_rhot in test_rhots:
                model.update(self.corpus)

                msg = ", ".join(map(str, [passes, model.num_updates, model.state.numdocs]))
                self.assertAlmostEqual(final_rhot(), test_rhot, msg=msg)

            self.assertEqual(model.state.numdocs, len(corpus) * len(test_rhots))
            self.assertEqual(model.num_updates, len(corpus) * len(test_rhots))

    # def testTopicSeeding(self):
    #     for topic in range(2):
    #         passed = False
    #         for i in range(5):  # restart at most this many times, to mitigate LDA randomness
    #             # try seeding it both ways round, check you get the same
    #             # topics out but with which way round they are depending
    #             # on the way round they're seeded
    #             eta = numpy.ones((2, len(dictionary))) * 0.5
    #             system = dictionary.token2id[u'system']
    #             trees = dictionary.token2id[u'trees']

    #             # aggressively seed the word 'system', in one of the
    #             # two topics, 10 times higher than the other words
    #             eta[topic, system] *= 10.0

    #             model = self.class_(id2word=dictionary, num_topics=2, passes=200, eta=eta)
    #             model.update(self.corpus)

    #             topics = [dict((word, p) for p, word in model.show_topic(j, topn=None)) for j in range(2)]

    #             # check that the word 'system' in the topic we seeded got a high weight,
    #             # and the word 'trees' (the main word in the other topic) a low weight --
    #             # and vice versa for the other topic (which we didn't seed with 'system')
    #             passed = (
    #                 (topics[topic][u'system'] > topics[topic][u'trees'])
    #                 and
    #                 (topics[1 - topic][u'system'] < topics[1 - topic][u'trees'])
    #             )
    #             if passed:
    #                 break
    #             logging.warning("LDA failed to converge on attempt %i (got %s)", i, topics)
    #         self.assertTrue(passed)

    def testPersistence(self):
        fname = testfile()
        model = self.model
        model.save(fname)
        model2 = self.class_.load(fname)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(numpy.allclose(model.expElogbeta, model2.expElogbeta))
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector

    def testPersistenceIgnore(self):
        fname = testfile()
        model = ldamodel.LdaModel(self.corpus, num_topics=2)
        model.save(fname, ignore='id2word')
        model2 = ldamodel.LdaModel.load(fname)
        self.assertTrue(model2.id2word is None)

        model.save(fname, ignore=['id2word'])
        model2 = ldamodel.LdaModel.load(fname)
        self.assertTrue(model2.id2word is None)

    def testPersistenceCompressed(self):
        fname = testfile() + '.gz'
        model = self.model
        model.save(fname)
        model2 = self.class_.load(fname, mmap=None)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(numpy.allclose(model.expElogbeta, model2.expElogbeta))
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector

    def testLargeMmap(self):
        fname = testfile()
        model = self.model

        # simulate storing large arrays separately
        model.save(testfile(), sep_limit=0)

        # test loading the large model arrays with mmap
        model2 = self.class_.load(testfile(), mmap='r')
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(isinstance(model2.expElogbeta, numpy.memmap))
        self.assertTrue(numpy.allclose(model.expElogbeta, model2.expElogbeta))
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector

    def testLargeMmapCompressed(self):
        fname = testfile() + '.gz'
        model = self.model

        # simulate storing large arrays separately
        model.save(fname, sep_limit=0)

        # test loading the large model arrays with mmap
        self.assertRaises(IOError, self.class_.load, fname, mmap='r')

#endclass TestLdaModel


class TestLdaMulticore(TestLdaModel):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = ldamulticore.LdaMulticore
        self.model = self.class_(corpus, id2word=dictionary, num_topics=2, passes=100)

    # override LdaModel because multicore does not allow alpha=auto
    def testAlphaAuto(self):
        self.assertRaises(RuntimeError, self.class_, alpha='auto')


#endclass TestLdaMulticore


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
