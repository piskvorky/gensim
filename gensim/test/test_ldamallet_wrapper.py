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
            # must contain the same values, up to re-ordering
            passed = np.allclose(sorted(vec), sorted(expected), atol=1e-1)
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
            # must contain the same values, up to re-ordering
            passed = np.allclose(sorted(vec), sorted(expected), atol=1e-2)
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

    def test_load_model(self):
        if not self.mallet_path:
            return    
        # to conduct the test these directories and files should exist
        model_save_path = ".\\test_data\\Mallet_TMP\\"
        model_save_name = model_save_path + "Mallet_pre_rs.mdl"
        # the saved models temporary files need to be in a common directory, 
        # they are being named according to the model name to minimize conflicts
        prefix = model_save_path + "MTMP\\pre_rs_"
        
        if not (os.path.exists(model_save_name) & os.path.exists(prefix + "corpus.mallet")):
            logging.warning("Pre-existing model files not found. Skipping test loading of them.")  
            return
        model = ldamodel.LdaModel.load(model_save_name)
        
        # Test loaded model works on a new corpus, made of previously unseen documents.
        other_texts = [['computer', 'time', 'graph'],
                       ['survey', 'response', 'eps'],
                       ['human', 'system', 'computer']]
        other_corpus = [dictionary.doc2bow(text) for text in other_texts]

        unseen_doc = other_corpus[0]
        vector = model[unseen_doc] # get topic probability distribution for a document
        self.assertTrue(sum(n for _, n in vector) == 1)
        
    def test_random_seed(self):
        if not self.mallet_path:
            return 
        # test that 2 models created with the same random_seed are equal in their topics treatment
        SEED = 10 
        tm1 = ldamallet.LdaMallet(self.mallet_path, 
                                  corpus=corpus, 
                                  num_topics=2, 
                                  id2word=dictionary, 
                                  random_seed = SEED)
        tm2 = ldamallet.LdaMallet(self.mallet_path, 
                                  corpus=corpus, 
                                  num_topics=2, 
                                  id2word=dictionary, 
                                  random_seed = SEED)        
        self.assertTrue(np.allclose(tm1.word_topics, tm2.word_topics))
        for doc in corpus:
            self.assertTrue(np.allclose(
                        sorted(matutils.sparse2full(tm1[doc], 2)), 
                        sorted(matutils.sparse2full(tm2[doc], 2)), 
                        atol=1e-1))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
