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

import numpy as np
import scipy.linalg

from gensim import matutils
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import lsimodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile


class TestLsiModel(unittest.TestCase, basetmtests.TestBaseTopicModel):
    def setUp(self):
        self.corpus = MmCorpus(datapath('testcorpus.mm'))
        self.model = lsimodel.LsiModel(self.corpus, num_topics=2)

    def test_transform(self):
        """Test lsi[vector] transformation."""
        # create the transformation model
        model = self.model

        # make sure the decomposition is enough accurate
        u, s, vt = scipy.linalg.svd(matutils.corpus2dense(self.corpus, self.corpus.num_terms), full_matrices=False)
        self.assertTrue(np.allclose(s[:2], model.projection.s))  # singular values must match

        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, 2)  # convert to dense vector, for easier equality tests
        expected = np.array([-0.6594664, 0.142115444])  # scaled LSI version
        # expected = np.array([-0.1973928, 0.05591352])  # non-scaled LSI version
        self.assertTrue(np.allclose(abs(vec), abs(expected)))  # transformed entries must be equal up to sign

    def test_transform_float32(self):
        """Test lsi[vector] transformation."""
        # create the transformation model
        model = lsimodel.LsiModel(self.corpus, num_topics=2, dtype=np.float32)

        # make sure the decomposition is enough accurate
        u, s, vt = scipy.linalg.svd(matutils.corpus2dense(self.corpus, self.corpus.num_terms), full_matrices=False)
        self.assertTrue(np.allclose(s[:2], model.projection.s))  # singular values must match
        self.assertEqual(model.projection.u.dtype, np.float32)
        self.assertEqual(model.projection.s.dtype, np.float32)

        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, 2)  # convert to dense vector, for easier equality tests
        expected = np.array([-0.6594664, 0.142115444])  # scaled LSI version
        # transformed entries must be equal up to sign
        self.assertTrue(np.allclose(abs(vec), abs(expected), atol=1.e-5))

    def test_corpus_transform(self):
        """Test lsi[corpus] transformation."""
        model = self.model
        got = np.vstack([matutils.sparse2full(doc, 2) for doc in model[self.corpus]])
        expected = np.array([
            [0.65946639, 0.14211544],
            [2.02454305, -0.42088759],
            [1.54655361, 0.32358921],
            [1.81114125, 0.5890525],
            [0.9336738, -0.27138939],
            [0.01274618, -0.49016181],
            [0.04888203, -1.11294699],
            [0.08063836, -1.56345594],
            [0.27381003, -1.34694159]
        ])
        self.assertTrue(np.allclose(abs(got), abs(expected)))  # must equal up to sign

    def test_online_transform(self):
        corpus = list(self.corpus)
        doc = corpus[0]  # use the corpus' first document for testing

        # create the transformation model
        model2 = lsimodel.LsiModel(corpus=corpus, num_topics=5)  # compute everything at once
        # start with no documents, we will add them later
        model = lsimodel.LsiModel(corpus=None, id2word=model2.id2word, num_topics=5)

        # train model on a single document
        model.add_documents([corpus[0]])

        # transform the testing document with this partial transformation
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, model.num_topics)  # convert to dense vector, for easier equality tests
        expected = np.array([-1.73205078, 0.0, 0.0, 0.0, 0.0])  # scaled LSI version
        self.assertTrue(np.allclose(abs(vec), abs(expected), atol=1e-6))  # transformed entries must be equal up to sign

        # train on another 4 documents
        model.add_documents(corpus[1:5], chunksize=2)  # train on 4 extra docs, in chunks of 2 documents, for the lols

        # transform a document with this partial transformation
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, model.num_topics)  # convert to dense vector, for easier equality tests
        expected = np.array([-0.66493785, -0.28314203, -1.56376302, 0.05488682, 0.17123269])  # scaled LSI version
        self.assertTrue(np.allclose(abs(vec), abs(expected), atol=1e-6))  # transformed entries must be equal up to sign

        # train on the rest of documents
        model.add_documents(corpus[5:])

        # make sure the final transformation is the same as if we had decomposed the whole corpus at once
        vec1 = matutils.sparse2full(model[doc], model.num_topics)
        vec2 = matutils.sparse2full(model2[doc], model2.num_topics)
        # the two LSI representations must equal up to sign
        self.assertTrue(np.allclose(abs(vec1), abs(vec2), atol=1e-5))

    def test_persistence(self):
        fname = get_tmpfile('gensim_models_lsi.tst')
        model = self.model
        model.save(fname)
        model2 = lsimodel.LsiModel.load(fname)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.projection.u, model2.projection.u))
        self.assertTrue(np.allclose(model.projection.s, model2.projection.s))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector

    def test_persistence_compressed(self):
        fname = get_tmpfile('gensim_models_lsi.tst.gz')
        model = self.model
        model.save(fname)
        model2 = lsimodel.LsiModel.load(fname, mmap=None)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.projection.u, model2.projection.u))
        self.assertTrue(np.allclose(model.projection.s, model2.projection.s))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector

    def test_large_mmap(self):
        fname = get_tmpfile('gensim_models_lsi.tst')
        model = self.model

        # test storing the internal arrays into separate files
        model.save(fname, sep_limit=0)

        # now load the external arrays via mmap
        model2 = lsimodel.LsiModel.load(fname, mmap='r')
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(isinstance(model2.projection.u, np.memmap))
        self.assertTrue(isinstance(model2.projection.s, np.memmap))
        self.assertTrue(np.allclose(model.projection.u, model2.projection.u))
        self.assertTrue(np.allclose(model.projection.s, model2.projection.s))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))  # try projecting an empty vector

    def test_large_mmap_compressed(self):
        fname = get_tmpfile('gensim_models_lsi.tst.gz')
        model = self.model

        # test storing the internal arrays into separate files
        model.save(fname, sep_limit=0)

        # now load the external arrays via mmap
        return

        # turns out this test doesn't exercise this because there are no arrays
        # to be mmaped!
        self.assertRaises(IOError, lsimodel.LsiModel.load, fname, mmap='r')

    def test_docs_processed(self):
        self.assertEqual(self.model.docs_processed, 9)
        self.assertEqual(self.model.docs_processed, self.corpus.num_docs)

    def test_get_topics(self):
        topics = self.model.get_topics()
        vocab_size = len(self.model.id2word)
        for topic in topics:
            self.assertTrue(isinstance(topic, np.ndarray))
            self.assertEqual(topic.dtype, np.float64)
            self.assertEqual(vocab_size, topic.shape[0])
            # LSI topics are not probability distributions
            # self.assertAlmostEqual(np.sum(topic), 1.0, 5)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
