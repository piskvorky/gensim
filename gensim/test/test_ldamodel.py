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
import numpy as np

from gensim.corpora import mmcorpus, Dictionary
from gensim.models import ldamodel, ldamulticore
from gensim import matutils, utils
from gensim.test import basetmtests

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
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


def testfile(test_fname=''):
    # temporary data will be stored to this file
    fname = 'gensim_models_' + test_fname + '.tst'
    return os.path.join(tempfile.gettempdir(), fname)


def testRandomState():
    testcases = [np.random.seed(0), None, np.random.RandomState(0), 0]
    for testcase in testcases:
        assert(isinstance(utils.get_random_state(testcase), np.random.RandomState))


class TestLdaModel(unittest.TestCase, basetmtests.TestBaseTopicModel):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = ldamodel.LdaModel
        self.model = self.class_(corpus, id2word=dictionary, num_topics=2, passes=100)

    def testTransform(self):
        passed = False
        # sometimes, LDA training gets stuck at a local minimum
        # in that case try re-training the model from scratch, hoping for a
        # better random initialization
        for i in range(25):  # restart at most 5 times
            # create the transformation model
            model = self.class_(id2word=dictionary, num_topics=2, passes=100)
            model.update(self.corpus)

            # transform one document
            doc = list(corpus)[0]
            transformed = model[doc]

            vec = matutils.sparse2full(transformed, 2)  # convert to dense vector, for easier equality tests
            expected = [0.13, 0.87]
            passed = np.allclose(sorted(vec), sorted(expected), atol=1e-1)  # must contain the same values, up to re-ordering
            if passed:
                break
            logging.warning("LDA failed to converge on attempt %i (got %s, expected %s)", i, sorted(vec), sorted(expected))
        self.assertTrue(passed)

    def testAlphaAuto(self):
        model1 = self.class_(corpus, id2word=dictionary, alpha='symmetric', passes=10)
        modelauto = self.class_(corpus, id2word=dictionary, alpha='auto', passes=10)

        # did we learn something?
        self.assertFalse(all(np.equal(model1.alpha, modelauto.alpha)))

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
        self.assertTrue(all(model.alpha == np.array([0.5, 0.5])))

        kwargs['alpha'] = 'asymmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(np.allclose(model.alpha, [0.630602, 0.369398]))

        kwargs['alpha'] = 0.3
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == np.array([0.3, 0.3])))

        kwargs['alpha'] = 3
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == np.array([3, 3])))

        kwargs['alpha'] = [0.3, 0.3]
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == np.array([0.3, 0.3])))

        kwargs['alpha'] = np.array([0.3, 0.3])
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == np.array([0.3, 0.3])))

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
        self.assertFalse(all(np.equal(model1.eta, modelauto.eta)))

    def testEta(self):
        kwargs = dict(
            id2word=dictionary,
            num_topics=2,
            eta=None
        )
        num_terms = len(dictionary)
        expected_shape = (num_terms,)

        # should not raise anything
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.5] * num_terms)))

        kwargs['eta'] = 'symmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.5] * num_terms)))

        kwargs['eta'] = 0.3
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.3] * num_terms)))

        kwargs['eta'] = 3
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([3] * num_terms)))

        kwargs['eta'] = [0.3] * num_terms
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.3] * num_terms)))

        kwargs['eta'] = np.array([0.3] * num_terms)
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.3] * num_terms)))

        # should be ok with num_topics x num_terms
        testeta = np.array([[0.5] * len(dictionary)] * 2)
        kwargs['eta'] = testeta
        self.class_(**kwargs)

        # all should raise an exception for being wrong shape
        kwargs['eta'] = testeta.reshape(tuple(reversed(testeta.shape)))
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['eta'] = [0.3]
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['eta'] = [0.3] * (num_terms + 1)
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['eta'] = "gensim is cool"
        self.assertRaises(ValueError, self.class_, **kwargs)

        kwargs['eta'] = "asymmetric"
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

        model = self.class_(self.corpus, id2word=dictionary, num_topics=2, passes=100, random_state=np.random.seed(0))

        doc_topics = model.get_document_topics(self.corpus)

        for topic in doc_topics:
            self.assertTrue(isinstance(topic, list))
            for k, v in topic:
                self.assertTrue(isinstance(k, int))
                self.assertTrue(isinstance(v, float))

        # Test case to use the get_document_topic function for the corpus
        all_topics = model.get_document_topics(self.corpus, per_word_topics=True)

        self.assertEqual(model.state.numdocs, len(corpus))

        for topic in all_topics:
            self.assertTrue(isinstance(topic, tuple))
            for k, v in topic[0]:  # list of doc_topics
                self.assertTrue(isinstance(k, int))
                self.assertTrue(isinstance(v, float))

            for w, topic_list in topic[1]:  # list of word_topics
                self.assertTrue(isinstance(w, int))
                self.assertTrue(isinstance(topic_list, list))

            for w, phi_values in topic[2]:  # list of word_phis
                self.assertTrue(isinstance(w, int))
                self.assertTrue(isinstance(phi_values, list))

        # Test case to check the filtering effect of minimum_probability and minimum_phi_value
        doc_topic_count_na = 0
        word_phi_count_na = 0

        all_topics = model.get_document_topics(self.corpus, minimum_probability=0.8, minimum_phi_value=1.0, per_word_topics=True)

        self.assertEqual(model.state.numdocs, len(corpus))

        for topic in all_topics:
            self.assertTrue(isinstance(topic, tuple))
            for k, v in topic[0]:  # list of doc_topics
                self.assertTrue(isinstance(k, int))
                self.assertTrue(isinstance(v, float))
                if len(topic[0]) != 0:
                    doc_topic_count_na += 1

            for w, topic_list in topic[1]:  # list of word_topics
                self.assertTrue(isinstance(w, int))
                self.assertTrue(isinstance(topic_list, list))

            for w, phi_values in topic[2]:  # list of word_phis
                self.assertTrue(isinstance(w, int))
                self.assertTrue(isinstance(phi_values, list))
                if len(phi_values) != 0:
                    word_phi_count_na += 1

        self.assertTrue(model.state.numdocs > doc_topic_count_na)
        self.assertTrue(sum([len(i) for i in corpus]) > word_phi_count_na)

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

        # FIXME: Fails on osx and win
        # expected_word = 0
        # self.assertEqual(word_topics[0][0], expected_word)
        # self.assertTrue(0 in word_topics[0][1])

    def testTermTopics(self):

        model = self.class_(self.corpus, id2word=dictionary, num_topics=2, passes=100, random_state=np.random.seed(0))

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

                msg = ", ".join(str(x) for x in [passes, model.num_updates, model.state.numdocs])
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
    #             eta = np.ones((2, len(dictionary))) * 0.5
    #             system = dictionary.token2id[u'system']
    #             trees = dictionary.token2id[u'trees']

    #             # aggressively seed the word 'system', in one of the
    #             # two topics, 10 times higher than the other words
    #             eta[topic, system] *= 10.0

    #             model = self.class_(id2word=dictionary, num_topics=2, passes=200, eta=eta)
    #             model.update(self.corpus)

    #             topics = [{word: p for p, word in model.show_topic(j, topn=None)} for j in range(2)]

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
        self.assertTrue(np.allclose(model.expElogbeta, model2.expElogbeta))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector

    def testModelCompatibilityWithPythonVersions(self):
        fname_model_2_7 = datapath('ldamodel_python_2_7')
        model_2_7 = self.class_.load(fname_model_2_7)
        fname_model_3_5 = datapath('ldamodel_python_3_5')
        model_3_5 = self.class_.load(fname_model_3_5)
        self.assertEqual(model_2_7.num_topics, model_3_5.num_topics)
        self.assertTrue(np.allclose(model_2_7.expElogbeta, model_3_5.expElogbeta))
        tstvec = []
        self.assertTrue(np.allclose(model_2_7[tstvec], model_3_5[tstvec]))  # try projecting an empty vector
        id2word_2_7 = dict(model_2_7.id2word.iteritems())
        id2word_3_5 = dict(model_3_5.id2word.iteritems())
        self.assertEqual(set(id2word_2_7.keys()), set(id2word_3_5.keys()))

    def testPersistenceIgnore(self):
        fname = testfile('testPersistenceIgnore')
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
        self.assertTrue(np.allclose(model.expElogbeta, model2.expElogbeta))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector

    def testLargeMmap(self):
        fname = testfile()
        model = self.model

        # simulate storing large arrays separately
        model.save(fname, sep_limit=0)

        # test loading the large model arrays with mmap
        model2 = self.class_.load(fname, mmap='r')
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(isinstance(model2.expElogbeta, np.memmap))
        self.assertTrue(np.allclose(model.expElogbeta, model2.expElogbeta))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector

    def testLargeMmapCompressed(self):
        fname = testfile() + '.gz'
        model = self.model

        # simulate storing large arrays separately
        model.save(fname, sep_limit=0)

        # test loading the large model arrays with mmap
        self.assertRaises(IOError, self.class_.load, fname, mmap='r')

    def testRandomStateBackwardCompatibility(self):
        # load a model saved using a pre-0.13.2 version of Gensim
        pre_0_13_2_fname = datapath('pre_0_13_2_model')
        model_pre_0_13_2 = self.class_.load(pre_0_13_2_fname)

        # set `num_topics` less than `model_pre_0_13_2.num_topics` so that `model_pre_0_13_2.random_state` is used
        model_topics = model_pre_0_13_2.print_topics(num_topics=2, num_words=3)

        for i in model_topics:
            self.assertTrue(isinstance(i[0], int))
            self.assertTrue(isinstance(i[1], six.string_types))

        # save back the loaded model using a post-0.13.2 version of Gensim
        post_0_13_2_fname = testfile('post_0_13_2_model')
        model_pre_0_13_2.save(post_0_13_2_fname)

        # load a model saved using a post-0.13.2 version of Gensim
        model_post_0_13_2 = self.class_.load(post_0_13_2_fname)
        model_topics_new = model_post_0_13_2.print_topics(num_topics=2, num_words=3)

        for i in model_topics_new:
            self.assertTrue(isinstance(i[0], int))
            self.assertTrue(isinstance(i[1], six.string_types))

# endclass TestLdaModel


class TestLdaMulticore(TestLdaModel):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = ldamulticore.LdaMulticore
        self.model = self.class_(corpus, id2word=dictionary, num_topics=2, passes=100)

    # override LdaModel because multicore does not allow alpha=auto
    def testAlphaAuto(self):
        self.assertRaises(RuntimeError, self.class_, alpha='auto')


# endclass TestLdaMulticore


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
