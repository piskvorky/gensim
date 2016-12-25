#!/usr/bin/env python
# encoding: utf-8
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated test to check similarity functions and isbow function.

"""


import logging
import unittest

from gensim import matutils
from scipy.sparse import csr_matrix
import numpy as np
import math
import os
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import ldamodel

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



class TestIsBow(unittest.TestCase):
    def test_None(self):
        # test None
        result = matutils.isbow(None)
        expected = False
        self.assertEqual(expected, result)

    def test_bow(self):
        # test list words

        # one bag of words
        potentialbow = [(0, 0.4)]
        result = matutils.isbow(potentialbow)
        expected = True
        self.assertEqual(expected, result)

        # multiple bags
        potentialbow = [(0, 4.), (1, 2.), (2, 5.), (3, 8.)]
        result = matutils.isbow(potentialbow)
        expected = True
        self.assertEqual(expected, result)

        # checking empty input
        potentialbow = []
        result = matutils.isbow(potentialbow)
        expected = True
        self.assertEqual(expected, result)     

        # checking corpus; should return false
        potentialbow = [[(2, 1), (3, 1), (4, 1), (5, 1), (1, 1), (7, 1)]]
        result = matutils.isbow(potentialbow)
        expected = False
        self.assertEqual(expected, result)

        # not a bag of words, should return false
        potentialbow = [(1, 3, 6)]
        result = matutils.isbow(potentialbow)
        expected = False
        self.assertEqual(expected, result)

        # checking sparse matrix format bag of words
        potentialbow = csr_matrix([[1, 0.4], [0, 0.3], [2, 0.1]])
        result = matutils.isbow(potentialbow)
        expected = True
        self.assertEqual(expected, result)

        # checking np array format bag of words
        potentialbow = np.array([[1, 0.4], [0, 0.2],[2, 0.2]])
        result = matutils.isbow(potentialbow)
        expected = True
        self.assertEqual(expected, result)

class TestHellinger(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = ldamodel.LdaModel
        self.model = self.class_(corpus, id2word=dictionary, num_topics=2, passes=100)

    def test_inputs(self):

        # checking empty inputs
        vec_1 = []
        vec_2 = []
        result = matutils.hellinger(vec_1, vec_2)
        expected = 0.0
        self.assertEqual(expected, result)

        # checking np array and list input
        vec_1 = np.array([])
        vec_2 = []
        result = matutils.hellinger(vec_1, vec_2)
        expected = 0.0
        self.assertEqual(expected, result)

        # checking scipy csr matrix and list input
        vec_1 = csr_matrix([])
        vec_2 = []
        result = matutils.hellinger(vec_1, vec_2)
        expected = 0.0
        self.assertEqual(expected, result)

    def test_distributions(self):

        # checking bag of words as inputs
        vec_1 = [(2, 0.1), (3, 0.4), (4, 0.1), (5, 0.1), (1, 0.1), (7, 0.2)]
        vec_2 = [(1, 0.1), (3, 0.8), (4, 0.1)]
        result = matutils.hellinger(vec_1, vec_2)
        expected = 0.185241936534
        self.assertAlmostEqual(expected, result)


        # checking ndarray, csr_matrix as inputs
        vec_1 = np.array([[1, 0.3], [0, 0.4], [2, 0.3]])
        vec_2 = csr_matrix([[1, 0.4], [0, 0.2], [2, 0.2]])
        result = matutils.hellinger(vec_1, vec_2)
        expected = 0.160618030536
        self.assertAlmostEqual(expected, result)

        # checking ndarray, list as inputs
        vec_1 = np.array([0.6, 0.1, 0.1, 0.2])
        vec_2 = [0.2, 0.2, 0.1, 0.5]
        result = matutils.hellinger(vec_1, vec_2)
        expected = 0.309742984153
        self.assertAlmostEqual(expected, result)

         # testing LDA distribution vectors
        np.random.seed(0)
        model = self.class_(self.corpus, id2word=dictionary, num_topics=2, passes= 100)
        lda_vec1 = model[[(1, 2), (2, 3)]]
        lda_vec2 = model[[(2, 2), (1, 3)]]
        result = matutils.hellinger(lda_vec1, lda_vec2)
        expected = 1.0406845281146034e-06
        self.assertAlmostEqual(expected, result)

class TestKL(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = ldamodel.LdaModel
        self.model = self.class_(corpus, id2word=dictionary, num_topics=2, passes=100)

    def test_inputs(self):

        # checking empty inputs
        vec_1 = []
        vec_2 = []
        result = matutils.kullback_leibler(vec_1, vec_2)
        expected = 0.0
        self.assertEqual(expected, result)

        # checking np array and list input
        vec_1 = np.array([])
        vec_2 = []
        result = matutils.kullback_leibler(vec_1, vec_2)
        expected = 0.0
        self.assertEqual(expected, result)

        # checking scipy csr matrix and list input
        vec_1 = csr_matrix([])
        vec_2 = []
        result = matutils.kullback_leibler(vec_1, vec_2)
        expected = 0.0
        self.assertEqual(expected, result)

    def test_distributions(self):

        # checking bag of words as inputs
        vec_1 = [(2, 0.1), (3, 0.4), (4, 0.1), (5, 0.1), (1, 0.1), (7, 0.2)]
        vec_2 = [(1, 0.1), (3, 0.8), (4, 0.1)]
        result = matutils.kullback_leibler(vec_2, vec_1, 8)
        expected = 0.55451775
        self.assertAlmostEqual(expected, result)

        # KL is not symetric; vec1 compared with vec2 will contain log of zeros and return infinity
        vec_1 = [(2, 0.1), (3, 0.4), (4, 0.1), (5, 0.1), (1, 0.1), (7, 0.2)]
        vec_2 = [(1, 0.1), (3, 0.8), (4, 0.1)]
        result = matutils.kullback_leibler(vec_1, vec_2, 8)
        self.assertTrue(math.isinf(result))

        # checking ndarray, csr_matrix as inputs
        vec_1 = np.array([[1, 0.3], [0, 0.4], [2, 0.3]])
        vec_2 = csr_matrix([[1, 0.4], [0, 0.2], [2, 0.2]])
        result = matutils.kullback_leibler(vec_1, vec_2, 3)
        expected = 0.0894502
        self.assertAlmostEqual(expected, result)

        # checking ndarray, list as inputs
        vec_1 = np.array([0.6, 0.1, 0.1, 0.2])
        vec_2 = [0.2, 0.2, 0.1, 0.5]
        result = matutils.kullback_leibler(vec_1, vec_2)
        expected = 0.40659450877
        self.assertAlmostEqual(expected, result)

        # testing LDA distribution vectors
        np.random.seed(0)
        model = self.class_(self.corpus, id2word=dictionary, num_topics=2, passes= 100)
        lda_vec1 = model[[(1, 2), (2, 3)]]
        lda_vec2 = model[[(2, 2), (1, 3)]]
        result = matutils.kullback_leibler(lda_vec1, lda_vec2)
        expected = 4.283407e-12
        self.assertAlmostEqual(expected, result)

class TestJaccard(unittest.TestCase):
    def test_inputs(self):

        # all empty inputs will give a divide by zero exception
        vec_1 = []
        vec_2 = []
        self.assertRaises(ZeroDivisionError, matutils.jaccard , vec_1, vec_2)

    def test_distributions(self):

        # checking bag of words as inputs
        vec_1 = [(2, 1), (3, 4), (4, 1), (5, 1), (1, 1), (7, 2)]
        vec_2 = [(1, 1), (3, 8), (4, 1)]
        result = matutils.jaccard(vec_2, vec_1)
        expected = 1 - 0.3
        self.assertAlmostEqual(expected, result)

        # checking ndarray, csr_matrix as inputs
        vec_1 = np.array([[1, 3], [0, 4], [2, 3]])
        vec_2 = csr_matrix([[1, 4], [0, 2], [2, 2]])
        result = matutils.jaccard(vec_1, vec_2)
        expected = 1 - 0.388888888889
        self.assertAlmostEqual(expected, result)

        # checking ndarray, list as inputs
        vec_1 = np.array([6, 1, 2, 3])
        vec_2 = [4, 3, 2, 5]
        result = matutils.jaccard(vec_1, vec_2)
        expected = 1 - 0.333333333333
        self.assertAlmostEqual(expected, result)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
