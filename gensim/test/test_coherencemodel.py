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
import multiprocessing as mp
from functools import partial

import numpy as np
from gensim.matutils import argsort
from gensim.models.coherencemodel import CoherenceModel, BOOLEAN_DOCUMENT_BASED
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import get_tmpfile, common_texts, common_dictionary, common_corpus


class TestCoherenceModel(unittest.TestCase):

    # set up vars used in testing ("Deerwester" from the web tutorial)
    texts = common_texts
    dictionary = common_dictionary
    corpus = common_corpus

    def setUp(self):
        # Suppose given below are the topics which two different LdaModels come up with.
        # `topics1` is clearly better as it has a clear distinction between system-human
        # interaction and graphs. Hence both the coherence measures for `topics1` should be
        # greater.
        self.topics1 = [
            ['human', 'computer', 'system', 'interface'],
            ['graph', 'minors', 'trees', 'eps']
        ]
        self.topics2 = [
            ['user', 'graph', 'minors', 'system'],
            ['time', 'graph', 'survey', 'minors']
        ]
        self.topics3 = [
            ['token', 'computer', 'system', 'interface'],
            ['graph', 'minors', 'trees', 'eps']
        ]
        # using this list the model should be unable to interpret topic
        # as either a list of tokens or a list of ids
        self.topics4 = [
            ['not a token', 'not an id', 'tests using', "this list"],
            ['should raise', 'an error', 'to pass', 'correctly']
        ]
        self.topicIds1 = []
        for topic in self.topics1:
            self.topicIds1.append([self.dictionary.token2id[token] for token in topic])

        self.ldamodel = LdaModel(
            corpus=self.corpus, id2word=self.dictionary, num_topics=2,
            passes=0, iterations=0
        )

    def check_coherence_measure(self, coherence):
        """Check provided topic coherence algorithm on given topics"""
        if coherence in BOOLEAN_DOCUMENT_BASED:
            kwargs = dict(corpus=self.corpus, dictionary=self.dictionary, coherence=coherence)
        else:
            kwargs = dict(texts=self.texts, dictionary=self.dictionary, coherence=coherence)

        cm1 = CoherenceModel(topics=self.topics1, **kwargs)
        cm2 = CoherenceModel(topics=self.topics2, **kwargs)
        cm3 = CoherenceModel(topics=self.topics3, **kwargs)
        cm4 = CoherenceModel(topics=self.topicIds1, **kwargs)
        self.assertRaises(ValueError, lambda: CoherenceModel(topics=self.topics4, **kwargs))
        self.assertEqual(cm1.get_coherence(), cm4.get_coherence())
        self.assertIsInstance(cm3.get_coherence(), np.double)
        self.assertGreater(cm1.get_coherence(), cm2.get_coherence())

    def testUMass(self):
        """Test U_Mass topic coherence algorithm on given topics"""
        self.check_coherence_measure('u_mass')

    def testCv(self):
        """Test C_v topic coherence algorithm on given topics"""
        self.check_coherence_measure('c_v')

    def testCuci(self):
        """Test C_uci topic coherence algorithm on given topics"""
        self.check_coherence_measure('c_uci')

    def testCnpmi(self):
        """Test C_npmi topic coherence algorithm on given topics"""
        self.check_coherence_measure('c_npmi')

    def testUMassLdaModel(self):
        """Perform sanity check to see if u_mass coherence works with LDA Model"""
        # Note that this is just a sanity check because LDA does not guarantee a better coherence
        # value on the topics if iterations are increased. This can be seen here:
        # https://gist.github.com/dsquareindia/60fd9ab65b673711c3fa00509287ddde
        CoherenceModel(model=self.ldamodel, corpus=self.corpus, coherence='u_mass')

    def testCvLdaModel(self):
        """Perform sanity check to see if c_v coherence works with LDA Model"""
        CoherenceModel(model=self.ldamodel, texts=self.texts, coherence='c_v')

    def testCw2vLdaModel(self):
        """Perform sanity check to see if c_w2v coherence works with LDAModel."""
        CoherenceModel(model=self.ldamodel, texts=self.texts, coherence='c_w2v')

    def testCuciLdaModel(self):
        """Perform sanity check to see if c_uci coherence works with LDA Model"""
        CoherenceModel(model=self.ldamodel, texts=self.texts, coherence='c_uci')

    def testCnpmiLdaModel(self):
        """Perform sanity check to see if c_npmi coherence works with LDA Model"""
        CoherenceModel(model=self.ldamodel, texts=self.texts, coherence='c_npmi')

    def testErrors(self):
        """Test if errors are raised on bad input"""
        # not providing dictionary
        self.assertRaises(
            ValueError, CoherenceModel, topics=self.topics1, corpus=self.corpus,
            coherence='u_mass'
        )
        # not providing texts for c_v and instead providing corpus
        self.assertRaises(
            ValueError, CoherenceModel, topics=self.topics1, corpus=self.corpus,
            dictionary=self.dictionary, coherence='c_v'
        )
        # not providing corpus or texts for u_mass
        self.assertRaises(
            ValueError, CoherenceModel, topics=self.topics1, dictionary=self.dictionary,
            coherence='u_mass'
        )

    def testProcesses(self):
        get_model = partial(CoherenceModel,
            topics=self.topics1, corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass'
        )

        model, used_cpus = get_model(), mp.cpu_count() - 1
        self.assertEqual(model.processes, used_cpus)
        for p in range(-2, 1):
            self.assertEqual(get_model(processes=p).processes, used_cpus)

        for p in range(1, 4):
            self.assertEqual(get_model(processes=p).processes, p)

    def testPersistence(self):
        fname = get_tmpfile('gensim_models_coherence.tst')
        model = CoherenceModel(
            topics=self.topics1, corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass'
        )
        model.save(fname)
        model2 = CoherenceModel.load(fname)
        self.assertTrue(model.get_coherence() == model2.get_coherence())

    def testPersistenceCompressed(self):
        fname = get_tmpfile('gensim_models_coherence.tst.gz')
        model = CoherenceModel(
            topics=self.topics1, corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass'
        )
        model.save(fname)
        model2 = CoherenceModel.load(fname)
        self.assertTrue(model.get_coherence() == model2.get_coherence())

    def testPersistenceAfterProbabilityEstimationUsingCorpus(self):
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        model = CoherenceModel(
            topics=self.topics1, corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass'
        )
        model.estimate_probabilities()
        model.save(fname)
        model2 = CoherenceModel.load(fname)
        self.assertIsNotNone(model2._accumulator)
        self.assertTrue(model.get_coherence() == model2.get_coherence())

    def testPersistenceAfterProbabilityEstimationUsingTexts(self):
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        model = CoherenceModel(
            topics=self.topics1, texts=self.texts, dictionary=self.dictionary, coherence='c_v'
        )
        model.estimate_probabilities()
        model.save(fname)
        model2 = CoherenceModel.load(fname)
        self.assertIsNotNone(model2._accumulator)
        self.assertTrue(model.get_coherence() == model2.get_coherence())

    def testAccumulatorCachingSameSizeTopics(self):
        kwargs = dict(corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass')
        cm1 = CoherenceModel(topics=self.topics1, **kwargs)
        cm1.estimate_probabilities()
        accumulator = cm1._accumulator
        self.assertIsNotNone(accumulator)
        cm1.topics = self.topics1
        self.assertEqual(accumulator, cm1._accumulator)
        cm1.topics = self.topics2
        self.assertEqual(None, cm1._accumulator)

    def testAccumulatorCachingTopicSubsets(self):
        kwargs = dict(corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass')
        cm1 = CoherenceModel(topics=self.topics1, **kwargs)
        cm1.estimate_probabilities()
        accumulator = cm1._accumulator
        self.assertIsNotNone(accumulator)
        cm1.topics = [t[:2] for t in self.topics1]
        self.assertEqual(accumulator, cm1._accumulator)
        cm1.topics = self.topics1
        self.assertEqual(accumulator, cm1._accumulator)

    def testAccumulatorCachingWithModelSetting(self):
        kwargs = dict(corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass')
        cm1 = CoherenceModel(topics=self.topics1, **kwargs)
        cm1.estimate_probabilities()
        self.assertIsNotNone(cm1._accumulator)
        cm1.model = self.ldamodel
        topics = []
        for topic in self.ldamodel.state.get_lambda():
            bestn = argsort(topic, topn=cm1.topn, reverse=True)
            topics.append(bestn)
        self.assertTrue(np.array_equal(topics, cm1.topics))
        self.assertIsNone(cm1._accumulator)

    def testAccumulatorCachingWithTopnSettingGivenTopics(self):
        kwargs = dict(corpus=self.corpus, dictionary=self.dictionary, topn=5, coherence='u_mass')
        cm1 = CoherenceModel(topics=self.topics1, **kwargs)
        cm1.estimate_probabilities()
        self.assertIsNotNone(cm1._accumulator)

        accumulator = cm1._accumulator
        topics_before = cm1._topics
        cm1.topn = 3
        self.assertEqual(accumulator, cm1._accumulator)
        self.assertEqual(3, len(cm1.topics[0]))
        self.assertEqual(topics_before, cm1._topics)

        # Topics should not have been truncated, so topn settings below 5 should work
        cm1.topn = 4
        self.assertEqual(accumulator, cm1._accumulator)
        self.assertEqual(4, len(cm1.topics[0]))
        self.assertEqual(topics_before, cm1._topics)

        with self.assertRaises(ValueError):
            cm1.topn = 6  # can't expand topics any further without model

    def testAccumulatorCachingWithTopnSettingGivenModel(self):
        kwargs = dict(corpus=self.corpus, dictionary=self.dictionary, topn=5, coherence='u_mass')
        cm1 = CoherenceModel(model=self.ldamodel, **kwargs)
        cm1.estimate_probabilities()
        self.assertIsNotNone(cm1._accumulator)

        accumulator = cm1._accumulator
        topics_before = cm1._topics
        cm1.topn = 3
        self.assertEqual(accumulator, cm1._accumulator)
        self.assertEqual(3, len(cm1.topics[0]))
        self.assertEqual(topics_before, cm1._topics)

        cm1.topn = 6  # should be able to expand given the model
        self.assertEqual(6, len(cm1.topics[0]))

    def testCompareCoherenceForTopics(self):
        topics = [self.topics1, self.topics2]
        cm = CoherenceModel.for_topics(
            topics, dictionary=self.dictionary, texts=self.texts, coherence='c_v')
        self.assertIsNotNone(cm._accumulator)

        # Accumulator should have all relevant IDs.
        for topic_list in topics:
            cm.topics = topic_list
            self.assertIsNotNone(cm._accumulator)

        (coherence_topics1, coherence1), (coherence_topics2, coherence2) = \
            cm.compare_model_topics(topics)

        self.assertAlmostEqual(np.mean(coherence_topics1), coherence1, 4)
        self.assertAlmostEqual(np.mean(coherence_topics2), coherence2, 4)
        self.assertGreater(coherence1, coherence2)

    def testCompareCoherenceForModels(self):
        models = [self.ldamodel, self.ldamodel]
        cm = CoherenceModel.for_models(
            models, dictionary=self.dictionary, texts=self.texts, coherence='c_v')
        self.assertIsNotNone(cm._accumulator)

        # Accumulator should have all relevant IDs.
        for model in models:
            cm.model = model
            self.assertIsNotNone(cm._accumulator)

        (coherence_topics1, coherence1), (coherence_topics2, coherence2) = \
            cm.compare_models(models)

        self.assertAlmostEqual(np.mean(coherence_topics1), coherence1, 4)
        self.assertAlmostEqual(np.mean(coherence_topics2), coherence2, 4)
        self.assertAlmostEqual(coherence1, coherence2, places=4)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
