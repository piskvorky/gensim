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

import numpy as np

from gensim.corpora import mmcorpus, Dictionary
from gensim.models.wrappers import ldamallet
from gensim import matutils
from gensim.models import ldamodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile, common_texts

dictionary = Dictionary(common_texts)
corpus = [dictionary.doc2bow(text) for text in common_texts]


class TestLdaMallet(unittest.TestCase, basetmtests.TestBaseTopicModel):
    def setUp(self):
        mallet_home = os.environ.get('MALLET_HOME', None)
        self.mallet_path = os.path.join(mallet_home, 'bin', 'mallet') if mallet_home else None
        if not self.mallet_path:
            raise unittest.SkipTest("MALLET_HOME not specified. Skipping Mallet tests.")
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))

        # self.model is used in TestBaseTopicModel
        self.model = ldamallet.LdaMallet(self.mallet_path, corpus, id2word=dictionary, num_topics=2, iterations=1)

    def testTransform(self):
        if not self.mallet_path:
            return
        passed = False
        for i in range(5):  # restart at most 5 times
            # create the transformation model
            model = ldamallet.LdaMallet(self.mallet_path, corpus, id2word=dictionary, num_topics=2, iterations=200)
            # transform one document
            doc = list(corpus)[0]
            transformed = model[doc]
            vec = matutils.sparse2full(transformed, 2)  # convert to dense vector, for easier equality tests
            expected = [0.49, 0.51]
            passed = np.allclose(sorted(vec), sorted(expected), atol=1e-1)  # must contain the same values, up to re-ordering
            if passed:
                break
            logging.warning(
                "LDA failed to converge on attempt %i (got %s, expected %s)",
                i, sorted(vec), sorted(expected)
            )
        self.assertTrue(passed)

    def testSparseTransform(self):
        if not self.mallet_path:
            return
        passed = False
        for i in range(5):  # restart at most 5 times
            # create the sparse transformation model with the appropriate topic_threshold
            model = ldamallet.LdaMallet(
                self.mallet_path, corpus, id2word=dictionary, num_topics=2, iterations=200, topic_threshold=0.5
            )
            # transform one document
            doc = list(corpus)[0]
            transformed = model[doc]
            vec = matutils.sparse2full(transformed, 2)  # convert to dense vector, for easier equality tests
            expected = [1.0, 0.0]
            passed = np.allclose(sorted(vec), sorted(expected), atol=1e-2)  # must contain the same values, up to re-ordering
            if passed:
                break
            logging.warning(
                "LDA failed to converge on attempt %i (got %s, expected %s)",
                i, sorted(vec), sorted(expected)
            )
        self.assertTrue(passed)

    def testMallet2Model(self):
        if not self.mallet_path:
            return

        tm1 = ldamallet.LdaMallet(self.mallet_path, corpus=corpus, num_topics=2, id2word=dictionary)
        tm2 = ldamallet.malletmodel2ldamodel(tm1)
        for document in corpus:
            element1_1, element1_2 = tm1[document][0]
            element2_1, element2_2 = tm2[document][0]
            self.assertAlmostEqual(element1_1, element2_1)
            self.assertAlmostEqual(element1_2, element2_2, 1)
            element1_1, element1_2 = tm1[document][1]
            element2_1, element2_2 = tm2[document][1]
            self.assertAlmostEqual(element1_1, element2_1)
            self.assertAlmostEqual(element1_2, element2_2, 1)
            logging.debug('%d %d', element1_1, element2_1)
            logging.debug('%d %d', element1_2, element2_2)
            logging.debug('%d %d', tm1[document][1], tm2[document][1])

    def testPersistence(self):
        if not self.mallet_path:
            return
        fname = get_tmpfile('gensim_models_lda_mallet.tst')
        model = ldamallet.LdaMallet(self.mallet_path, self.corpus, num_topics=2, iterations=100)
        model.save(fname)
        model2 = ldamallet.LdaMallet.load(fname)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.word_topics, model2.word_topics))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector

    def testPersistenceCompressed(self):
        if not self.mallet_path:
            return
        fname = get_tmpfile('gensim_models_lda_mallet.tst.gz')
        model = ldamallet.LdaMallet(self.mallet_path, self.corpus, num_topics=2, iterations=100)
        model.save(fname)
        model2 = ldamallet.LdaMallet.load(fname, mmap=None)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.word_topics, model2.word_topics))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector

    def testLargeMmap(self):
        if not self.mallet_path:
            return
        fname = get_tmpfile('gensim_models_lda_mallet.tst')
        model = ldamallet.LdaMallet(self.mallet_path, self.corpus, num_topics=2, iterations=100)

        # simulate storing large arrays separately
        model.save(fname, sep_limit=0)

        # test loading the large model arrays with mmap
        model2 = ldamodel.LdaModel.load(fname, mmap='r')
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(isinstance(model2.word_topics, np.memmap))
        self.assertTrue(np.allclose(model.word_topics, model2.word_topics))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector

    def testLargeMmapCompressed(self):
        if not self.mallet_path:
            return
        fname = get_tmpfile('gensim_models_lda_mallet.tst.gz')
        model = ldamallet.LdaMallet(self.mallet_path, self.corpus, num_topics=2, iterations=100)

        # simulate storing large arrays separately
        model.save(fname, sep_limit=0)

        # test loading the large model arrays with mmap
        self.assertRaises(IOError, ldamodel.LdaModel.load, fname, mmap='r')
# endclass TestLdaMallet


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
