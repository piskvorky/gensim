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

import numpy

from gensim.corpora import mmcorpus, Dictionary
from gensim.models import lsimodel, ldamodel, tfidfmodel, rpmodel, logentropy_model
from gensim import matutils


module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)


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


class TestLsiModel(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))

    def testTransform(self):
        """Test lsi[vector] transformation."""
        # create the transformation model
        model = lsimodel.LsiModel(self.corpus, num_topics=2)

        # make sure the decomposition is enough accurate
        u, s, vt = numpy.linalg.svd(matutils.corpus2dense(self.corpus, self.corpus.num_terms), full_matrices=False)
        self.assertTrue(numpy.allclose(s[:2], model.projection.s)) # singular values must match

        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, 2) # convert to dense vector, for easier equality tests
        expected = numpy.array([-0.6594664, 0.142115444]) # scaled LSI version
        # expected = numpy.array([-0.1973928, 0.05591352]) # non-scaled LSI version
        self.assertTrue(numpy.allclose(abs(vec), abs(expected))) # transformed entries must be equal up to sign


    def testCorpusTransform(self):
        """Test lsi[corpus] transformation."""
        model = lsimodel.LsiModel(self.corpus, num_topics=2)
        got = numpy.vstack(matutils.sparse2full(doc, 2) for doc in model[corpus])
        expected = numpy.array([
            [ 0.65946639,  0.14211544],
            [ 2.02454305, -0.42088759],
            [ 1.54655361,  0.32358921],
            [ 1.81114125,  0.5890525 ],
            [ 0.9336738 , -0.27138939],
            [ 0.01274618, -0.49016181],
            [ 0.04888203, -1.11294699],
            [ 0.08063836, -1.56345594],
            [ 0.27381003, -1.34694159]])
        self.assertTrue(numpy.allclose(abs(got), abs(expected))) # must equal up to sign


    def testOnlineTransform(self):
        corpus = list(self.corpus)
        doc = corpus[0] # use the corpus' first document for testing

        # create the transformation model
        model2 = lsimodel.LsiModel(corpus=corpus, num_topics=5) # compute everything at once
        model = lsimodel.LsiModel(corpus=None, id2word=model2.id2word, num_topics=5) # start with no documents, we will add then later

        # train model on a single document
        model.add_documents([corpus[0]])
        #model.print_debug()

        # transform the testing document with this partial transformation
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, model.num_topics) # convert to dense vector, for easier equality tests
        expected = numpy.array([-1.73205078, 0.0, 0.0, 0.0, 0.0]) # scaled LSI version
        self.assertTrue(numpy.allclose(abs(vec), abs(expected), atol=1e-6)) # transformed entries must be equal up to sign

        # train on another 4 documents
        model.add_documents(corpus[1:5], chunksize=2) # train on 4 extra docs, in chunks of 2 documents, for the lols
        #model.print_debug()

        # transform a document with this partial transformation
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, model.num_topics) # convert to dense vector, for easier equality tests
        expected = numpy.array([-0.66493785, -0.28314203, -1.56376302, 0.05488682, 0.17123269]) # scaled LSI version
        self.assertTrue(numpy.allclose(abs(vec), abs(expected), atol=1e-6)) # transformed entries must be equal up to sign

        # train on the rest of documents
        model.add_documents(corpus[5:])
        #model.print_debug()

        # make sure the final transformation is the same as if we had decomposed the whole corpus at once
        vec1 = matutils.sparse2full(model[doc], model.num_topics)
        vec2 = matutils.sparse2full(model2[doc], model2.num_topics)
        self.assertTrue(numpy.allclose(abs(vec1), abs(vec2), atol=1e-5)) # the two LSI representations must equal up to sign


    def testPersistence(self):
        model = lsimodel.LsiModel(self.corpus, num_topics=2)
        model.save(testfile())
        model2 = lsimodel.LsiModel.load(testfile())
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(numpy.allclose(model.projection.u, model2.projection.u))
        self.assertTrue(numpy.allclose(model.projection.s, model2.projection.s))
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector
#endclass TestLsiModel


class TestRpModel(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))

    def testTransform(self):
        # create the transformation model
        numpy.random.seed(13) # HACK; set fixed seed so that we always get the same random matrix (and can compare against expected results)
        model = rpmodel.RpModel(self.corpus, num_topics=2)

        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, 2) # convert to dense vector, for easier equality tests

        expected = numpy.array([-0.70710677, 0.70710677])
        self.assertTrue(numpy.allclose(vec, expected)) # transformed entries must be equal up to sign


    def testPersistence(self):
        model = rpmodel.RpModel(self.corpus, num_topics=2)
        model.save(testfile())
        model2 = rpmodel.RpModel.load(testfile())
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(numpy.allclose(model.projection, model2.projection))
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector
#endclass TestRpModel


class TestLdaModel(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))

    def testTransform(self):
        passed = False
        # sometimes, LDA training gets stuck at a local minimum
        # in that case try re-training the model from scratch, hoping for a
        # better random initialization
        for i in xrange(5): # restart at most 5 times
            # create the transformation model
            model = ldamodel.LdaModel(id2word=dictionary, num_topics=2, passes=100)
            model.update(corpus)

            # transform one document
            doc = list(corpus)[0]
            transformed = model[doc]

            vec = matutils.sparse2full(transformed, 2) # convert to dense vector, for easier equality tests
            expected = [0.049, 0.951]
            passed = numpy.allclose(sorted(vec), sorted(expected), atol=1e-2) # must contain the same values, up to re-ordering
            if passed:
                break
            logging.warning("LDA failed to converge on attempt %i (got %s, expected %s)" %
                            (i, sorted(vec), sorted(expected)))
        self.assertTrue(passed)


    def testPersistence(self):
        model = ldamodel.LdaModel(self.corpus, num_topics=2)
        model.save(testfile())
        model2 = ldamodel.LdaModel.load(testfile())
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(numpy.allclose(model.expElogbeta, model2.expElogbeta))
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector
#endclass TestLdaModel


class TestTfidfModel(unittest.TestCase):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))

    def testTransform(self):
        # create the transformation model
        model = tfidfmodel.TfidfModel(self.corpus, normalize=True)

        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]

        expected =  [(0, 0.57735026918962573), (1, 0.57735026918962573), (2, 0.57735026918962573)]
        self.assertTrue(numpy.allclose(transformed, expected))


    def testInit(self):
        # create the transformation model by analyzing a corpus
        # uses the global `corpus`!
        model1 = tfidfmodel.TfidfModel(corpus)

        # make sure the dfs<->idfs transformation works
        dfs = tfidfmodel.idfs2dfs(model1.idfs, len(corpus))
        self.assertEqual(dfs, dictionary.dfs)
        self.assertEqual(model1.idfs, tfidfmodel.dfs2idfs(dfs, len(corpus)))

        # create the transformation model by directly supplying a term->docfreq
        # mapping from the global var `dictionary`.
        model2 = tfidfmodel.TfidfModel(dictionary=dictionary)
        self.assertEqual(model1.idfs, model2.idfs)


    def testPersistence(self):
        model = tfidfmodel.TfidfModel(self.corpus, normalize=True)
        model.save(testfile())
        model2 = tfidfmodel.TfidfModel.load(testfile())
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec])) # try projecting an empty vector
#endclass TestTfidfModel


class TestLogEntropyModel(unittest.TestCase):
    def setUp(self):
        self.corpus_small = mmcorpus.MmCorpus(datapath('test_corpus_small.mm'))
        self.corpus_ok = mmcorpus.MmCorpus(datapath('test_corpus_ok.mm'))


    def testTransform(self):
        # create the transformation model
        model = logentropy_model.LogEntropyModel(self.corpus_ok, normalize=False)

        # transform one document
        doc = list(self.corpus_ok)[0]
        transformed = model[doc]
        expected =  [(0, 0.29155145321295795),
                     (1, 0.024757785476437949),
                     (3, 1.0569257878828748)]
        self.assertTrue(numpy.allclose(transformed, expected))


    def testPersistence(self):
        model = logentropy_model.LogEntropyModel(self.corpus_ok, normalize=True)
        model.save(testfile())
        model2 = logentropy_model.LogEntropyModel.load(testfile())
        self.assertTrue(model.entr == model2.entr)
        tstvec = []
        self.assertTrue(numpy.allclose(model[tstvec], model2[tstvec]))
#endclass TestLogEntropyModel



if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
