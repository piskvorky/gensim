#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import logging
import unittest
import os
import os.path
import tempfile
import numbers

import six
import numpy as np
import scipy.linalg

from gensim import matutils, utils
from gensim.test import basetests

def gen_topics(rows):
    topics = []
    topic_base = np.concatenate((np.ones((1, rows)) * (1/rows),
                                 np.zeros((rows-1, rows))), axis=0).ravel()
    for i in range(rows):
        topics.append(np.roll(topic_base, i * rows))
    topic_base = np.concatenate((np.ones((rows, 1)) * (1/rows),
                                 np.zeros((rows, rows-1))), axis=1).ravel()
    for i in range(rows):
        topics.append(np.roll(topic_base, i))
    return np.array(topics)

def gen_doc(seed, K, N, thetas, V, topics, D):
    topic_assignments = np.array([np.random.choice(range(K), size=N, p=theta)
                                  for theta in thetas])
    word_assignments = \
        np.array([[np.random.choice(range(V), size=1,
                                    p=topics[topic_assignments[d, n]])[0]
                   for n in range(N)] for d in range(D)])
    return np.array([np.histogram(word_assignments[d], bins=V,
                                  range=(0, V - 1))[0] for d in range(D)])

ef language(document_size):
    # Generate topics
    rows = 3
    V = rows * rows
    K = rows * 2
    N = K * K
    D = document_size
    seed = 42
    topics = gen_topics(rows)

    # Generate documents from topics
    alpha = np.ones(K)
    np.random.seed(seed)
    thetas = gen_thetas(alpha, D)
    doc_term_matrix = gen_doc(seed, K, N, thetas, V, topics, D)
    return {'V': V, 'K': K, 'D': D, 'seed': seed, 'alpha': alpha,
            'topics': topics, 'thetas': thetas,
            'doc_term_matrix': doc_term_matrix, 'n_report_iters': 100}


def assert_probablity_distribution(results):
    assert (results >= 0).all()
    assert results.sum(axis=1).all()

def test_slda():
    l = language(10000)
    n_iter = 2000
    KL_thresh = 0.001

    nu2 = l['K']
    sigma2 = 1
    np.random.seed(l['seed'])
    eta = np.random.normal(scale=nu2, size=l['K'])
    y = [np.dot(eta, l['thetas'][i]) for i in range(l['D'])] + \
        np.random.normal(scale=sigma2, size=l['D'])
    _beta = np.repeat(0.01, l['V'])
    _mu = 0
    slda = SLDA(l['K'], l['alpha'], _beta, _mu, nu2, sigma2, n_iter,
                seed=l['seed'], n_report_iter=l['n_report_iters'])
    slda.fit(l['doc_term_matrix'], y)

    assert_probablity_distribution(slda.phi)
         
         
def testRandomState():
    testcases = [np.random.seed(0), None, np.random.RandomState(0), 0]
    for testcase in testcases:
        assert(
            isinstance(
                utils.get_random_state(testcase),
                np.random.RandomState))


def testAlpha(self):
    kwargs = dict(id2word=dictionary, num_topics=2, alpha=None)
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

    kwargs['alpha'] = "gensim is cool"
    self.assertRaises(ValueError, self.class_, **kwargs)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
