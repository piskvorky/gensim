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

# needed because sample data files are located in the same folder
module_path = os.path.dirname(__file__)


def datapath(fname): return os.path.join(module_path, 'test_data', fname)


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
