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

import six
import numpy
import scipy.linalg

from gensim.corpora import mmcorpus, Dictionary
from gensim.models.wrappers import ldamallet
from gensim import matutils
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


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_models.tst')


class TestLdaMallet(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        mallet_home = os.environ.get('MALLET_HOME', None)
        self.mallet_path = os.path.join(mallet_home, 'bin', 'mallet') if mallet_home else None

    def testTransform(self):
        if not self.mallet_path:
            return
        passed = False
        for i in range(5): # restart at most 5 times
            # create the transformation model
            model = ldamallet.LdaMallet(self.mallet_path, corpus, id2word=dictionary, num_topics=2, iterations=200)

            # transform one document
            doc = list(corpus)[0]
            transformed = model[doc]

            vec = matutils.sparse2full(transformed, 2) # convert to dense vector, for easier equality tests
            expected = [0.49, 0.51]
            passed = numpy.allclose(sorted(vec), sorted(expected), atol=1e-2) # must contain the same values, up to re-ordering
            if passed:
                break
            logging.warning("LDA failed to converge on attempt %i (got %s, expected %s)" %
                            (i, sorted(vec), sorted(expected)))
        self.assertTrue(passed)


    def testPersistence(self):
        if not self.mallet_path:
            return
        fname = testfile()
        model = ldamallet.LdaMallet(self.mallet_path, self.corpus, num_topics=2, iterations=100)
        model.save(fname)
        model2 = ldamallet.LdaMallet.load(fname)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(numpy.allclose(model.wordtopics, model2.wordtopics))
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector

    def testPersistenceCompressed(self):
        if not self.mallet_path:
            return
        fname = testfile() + '.gz'
        model = ldamallet.LdaMallet(self.mallet_path, self.corpus, num_topics=2, iterations=100)
        model.save(fname)
        model2 = ldamallet.LdaMallet.load(fname, mmap=None)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(numpy.allclose(model.wordtopics, model2.wordtopics))
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector

    def testLargeMmap(self):
        if not self.mallet_path:
            return
        fname = testfile()
        model = ldamallet.LdaMallet(self.mallet_path, self.corpus, num_topics=2, iterations=100)

        # simulate storing large arrays separately
        model.save(testfile(), sep_limit=0)

        # test loading the large model arrays with mmap
        model2 = ldamodel.LdaModel.load(testfile(), mmap='r')
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(isinstance(model2.wordtopics, numpy.memmap))
        self.assertTrue(numpy.allclose(model.wordtopics, model2.wordtopics))
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector

    def testLargeMmapCompressed(self):
        if not self.mallet_path:
            return
        fname = testfile() + '.gz'
        model = ldamallet.LdaMallet(self.mallet_path, self.corpus, num_topics=2, iterations=100)

        # simulate storing large arrays separately
        model.save(fname, sep_limit=0)

        # test loading the large model arrays with mmap
        self.assertRaises(IOError, ldamodel.LdaModel.load, fname, mmap='r')
#endclass TestLdaMallet

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
