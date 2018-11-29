#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Timofey Yefimov <anotherbugmaster@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""

import logging
import unittest
import numbers

import six
import numpy as np

from gensim.corpora import mmcorpus
from gensim.models import nmf
from gensim import matutils
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary

dictionary = common_dictionary
corpus = common_corpus


class TestNmf(unittest.TestCase, basetmtests.TestBaseTopicModel):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = nmf.Nmf
        self.model = self.class_(corpus, id2word=dictionary, num_topics=2, passes=100)

    def testTransform(self):
        # create the transformation model
        model = self.class_(id2word=dictionary, num_topics=2, passes=100)
        model.update(self.corpus)

        # transform one document
        doc = list(corpus)[0]
        transformed = model[doc]

        vec = matutils.sparse2full(transformed, 2)  # convert to dense vector, for easier equality tests
        expected = [0., 1.]
        # must contain the same values, up to re-ordering
        self.assertTrue(np.allclose(sorted(vec), sorted(expected), atol=1e-1))

    @unittest.skip('top topics is not implemented')
    def testTopTopics(self):
        top_topics = self.model.top_topics(self.corpus)

        for topic, score in top_topics:
            self.assertTrue(isinstance(topic, list))
            self.assertTrue(isinstance(score, float))

            for v, k in topic:
                self.assertTrue(isinstance(k, six.string_types))
                self.assertTrue(np.issubdtype(v, float))

    def testGetTopicTerms(self):
        topic_terms = self.model.get_topic_terms(1)

        for k, v in topic_terms:
            self.assertTrue(isinstance(k, numbers.Integral))
            self.assertTrue(np.issubdtype(v, float))

    def testGetDocumentTopics(self):

        model = self.class_(
            self.corpus, id2word=dictionary, num_topics=2, passes=100, random_state=np.random.seed(0)
        )

        doc_topics = model.get_document_topics(self.corpus)

        for topic in doc_topics:
            self.assertTrue(isinstance(topic, list))
            for k, v in topic:
                self.assertTrue(isinstance(k, numbers.Integral))
                self.assertTrue(np.issubdtype(v, float))

        # Test case to use the get_document_topic function for the corpus
        all_topics = model.get_document_topics(self.corpus)

        print(list(all_topics))

        for topic in all_topics:
            self.assertTrue(isinstance(topic, list))
            for k, v in topic:  # list of doc_topics
                self.assertTrue(isinstance(k, numbers.Integral))
                self.assertTrue(np.issubdtype(v, float))

        # FIXME: Fails on osx and win
        # expected_word = 0
        # self.assertEqual(word_topics[0][0], expected_word)
        # self.assertTrue(0 in word_topics[0][1])

    def testTermTopics(self):

        model = self.class_(
            self.corpus, id2word=dictionary, num_topics=2, passes=100, random_state=np.random.seed(0)
        )

        # check with word_type
        result = model.get_term_topics(2)
        for topic_no, probability in result:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(np.issubdtype(probability, float))

        # checks if topic '1' is in the result list
        # FIXME: Fails on osx and win
        # self.assertTrue(1 in result[0])

        # if user has entered word instead, check with word
        result = model.get_term_topics(str(model.id2word[2]))
        for topic_no, probability in result:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(np.issubdtype(probability, float))

        # checks if topic '1' is in the result list
        # FIXME: Fails on osx and win
        # self.assertTrue(1 in result[0])

    @unittest.skip("There's no offset member")
    def testPasses(self):
        # long message includes the original error message with a custom one
        self.longMessage = True
        # construct what we expect when passes aren't involved
        test_rhots = list()
        model = self.class_(id2word=dictionary, chunksize=1, num_topics=2)

        def final_rhot(model):
            return pow(model.offset + (1 * model.num_updates) / model.chunksize, -model.decay)

        # generate 5 updates to test rhot on
        for x in range(5):
            model.update(self.corpus)
            test_rhots.append(final_rhot(model))

        for passes in [1, 5, 10, 50, 100]:
            model = self.class_(id2word=dictionary, chunksize=1, num_topics=2, passes=passes)
            self.assertEqual(final_rhot(model), 1.0)
            # make sure the rhot matches the test after each update
            for test_rhot in test_rhots:
                model.update(self.corpus)

                msg = ", ".join(str(x) for x in [passes, model.num_updates, model.state.numdocs])
                self.assertAlmostEqual(final_rhot(model), test_rhot, msg=msg)

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
        fname = get_tmpfile('gensim_models_nmf.tst')
        model = self.model
        model.save(fname)
        model2 = self.class_.load(fname)
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector

    @unittest.skip("There're no pickled models")
    def testModelCompatibilityWithPythonVersions(self):
        fname_model_2_7 = datapath('nmf_python_2_7')
        model_2_7 = self.class_.load(fname_model_2_7)
        fname_model_3_5 = datapath('nmf_python_3_5')
        model_3_5 = self.class_.load(fname_model_3_5)
        self.assertEqual(model_2_7.num_topics, model_3_5.num_topics)
        self.assertTrue(np.allclose(model_2_7.expElogbeta, model_3_5.expElogbeta))
        tstvec = []
        self.assertTrue(np.allclose(model_2_7[tstvec], model_3_5[tstvec]))  # try projecting an empty vector
        id2word_2_7 = dict(model_2_7.id2word.iteritems())
        id2word_3_5 = dict(model_3_5.id2word.iteritems())
        self.assertEqual(set(id2word_2_7.keys()), set(id2word_3_5.keys()))

    def testPersistenceCompressed(self):
        fname = get_tmpfile('gensim_models_nmf.tst.gz')
        model = self.model
        model.save(fname)
        model2 = self.class_.load(fname, mmap=None)
        self.assertEqual(model.num_topics, model2.num_topics)
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector

    def testLargeMmap(self):
        fname = get_tmpfile('gensim_models_nmf.tst')
        model = self.model

        # simulate storing large arrays separately
        model.save(fname, sep_limit=0)

        # test loading the large model arrays with mmap
        model2 = self.class_.load(fname, mmap='r')
        self.assertEqual(model.num_topics, model2.num_topics)
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector

    def testLargeMmapCompressed(self):
        fname = get_tmpfile('gensim_models_nmf.tst.gz')
        model = self.model

        # simulate storing large arrays separately
        model.save(fname, sep_limit=0)

        # test loading the large model arrays with mmap
        self.assertRaises(IOError, self.class_.load, fname, mmap='r')

    @unittest.skip("NMF has no state")
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
        post_0_13_2_fname = get_tmpfile('gensim_models_nmf_post_0_13_2_model.tst')
        model_pre_0_13_2.save(post_0_13_2_fname)

        # load a model saved using a post-0.13.2 version of Gensim
        model_post_0_13_2 = self.class_.load(post_0_13_2_fname)
        model_topics_new = model_post_0_13_2.print_topics(num_topics=2, num_words=3)

        for i in model_topics_new:
            self.assertTrue(isinstance(i[0], int))
            self.assertTrue(isinstance(i[1], six.string_types))

    @unittest.skip('different output format than lda')
    def testDtypeBackwardCompatibility(self):
        nmf_3_6_0_fname = datapath('nmf_3_6_0_model')
        test_doc = [(0, 1), (1, 1), (2, 1)]
        expected_topics = [(0, 0.87005886977475178), (1, 0.12994113022524822)]

        # save model to use in test
        self.model.save(nmf_3_6_0_fname)

        # load a model saved using a 3.0.1 version of Gensim
        model = self.class_.load(nmf_3_6_0_fname)

        # and test it on a predefined document
        topics = model[test_doc]
        self.assertTrue(np.allclose(expected_topics, topics))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    unittest.main()
