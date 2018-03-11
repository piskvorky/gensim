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

from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import tfidfmodel
from gensim.test.utils import datapath, get_tmpfile, common_dictionary, common_corpus

from gensim.corpora import Dictionary

texts = [
    ['complier', 'system', 'computer'],
    ['eulerian', 'node', 'cycle', 'graph', 'tree', 'path'],
    ['graph', 'flow', 'network', 'graph'],
    ['loading', 'computer', 'system'],
    ['user', 'server', 'system'],
    ['tree', 'hamiltonian'],
    ['graph', 'trees'],
    ['computer', 'kernel', 'malfunction', 'computer'],
    ['server', 'system', 'computer'],
]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


class TestTfidfModel(unittest.TestCase):
    def setUp(self):
        self.corpus = MmCorpus(datapath('testcorpus.mm'))

    def test_transform(self):
        # create the transformation model
        model = tfidfmodel.TfidfModel(self.corpus, normalize=True)

        # transform one document
        doc = list(self.corpus)[0]
        transformed = model[doc]

        expected = [(0, 0.57735026918962573), (1, 0.57735026918962573), (2, 0.57735026918962573)]
        self.assertTrue(np.allclose(transformed, expected))

    def test_init(self):
        # create the transformation model by analyzing a corpus
        # uses the global `corpus`!
        model1 = tfidfmodel.TfidfModel(common_corpus)
        dfs = common_dictionary.dfs

        # make sure the dfs<->idfs transformation works
        self.assertEqual(model1.dfs, dfs)
        self.assertEqual(model1.idfs, tfidfmodel.precompute_idfs(model1.wglobal, dfs, len(common_corpus)))

        # create the transformation model by directly supplying a term->docfreq
        # mapping from the global var `dictionary`.
        model2 = tfidfmodel.TfidfModel(dictionary=common_dictionary)
        self.assertEqual(model1.idfs, model2.idfs)

    def test_persistence(self):
        # Test persistence without using `smartirs`
        fname = get_tmpfile('gensim_models.tst')
        model = tfidfmodel.TfidfModel(self.corpus, normalize=True)
        model.save(fname)
        model2 = tfidfmodel.TfidfModel.load(fname)
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model[tstvec[0]], model2[tstvec[0]]))
        self.assertTrue(np.allclose(model[tstvec[1]], model2[tstvec[1]]))
        self.assertTrue(np.allclose(model[[]], model2[[]]))  # try projecting an empty vector

        # Test persistence with using `smartirs`
        fname = get_tmpfile('gensim_models_smartirs.tst')
        model = tfidfmodel.TfidfModel(self.corpus, smartirs="ntc")
        model.save(fname)
        model2 = tfidfmodel.TfidfModel.load(fname)
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model[tstvec[0]], model2[tstvec[0]]))
        self.assertTrue(np.allclose(model[tstvec[1]], model2[tstvec[1]]))
        self.assertTrue(np.allclose(model[[]], model2[[]]))  # try projecting an empty vector

        # Test persistence between Gensim v3.2.0 and current model.
        model3 = tfidfmodel.TfidfModel(self.corpus, smartirs="ntc")
        model4 = tfidfmodel.TfidfModel.load(datapath('tfidf_model.tst'))
        self.assertTrue(model3.idfs == model4.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model3[tstvec[0]], model4[tstvec[0]]))
        self.assertTrue(np.allclose(model3[tstvec[1]], model4[tstvec[1]]))
        self.assertTrue(np.allclose(model3[[]], model4[[]]))  # try projecting an empty vector

        # Test persistence with using pivoted normalization
        fname = get_tmpfile('gensim_models_smartirs.tst')
        model = tfidfmodel.TfidfModel(self.corpus, pivot_norm=True, pivot=0, slope=1)
        model.save(fname)
        model2 = tfidfmodel.TfidfModel.load(fname, mmap=None)
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model[tstvec[0]], model2[tstvec[0]]))
        self.assertTrue(np.allclose(model[tstvec[1]], model2[tstvec[1]]))

        # Test persistence between Gensim v3.2.0 and pivoted normalization compressed model.
        model3 = tfidfmodel.TfidfModel(self.corpus, pivot_norm=True, pivot=0, slope=1)
        model4 = tfidfmodel.TfidfModel.load(datapath('tfidf_model.tst'))
        self.assertTrue(model3.idfs == model4.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model3[tstvec[0]], model4[tstvec[0]]))
        self.assertTrue(np.allclose(model3[tstvec[1]], model4[tstvec[1]]))

    def test_persistence_compressed(self):
        # Test persistence without using `smartirs`
        fname = get_tmpfile('gensim_models.tst.gz')
        model = tfidfmodel.TfidfModel(self.corpus, normalize=True)
        model.save(fname)
        model2 = tfidfmodel.TfidfModel.load(fname, mmap=None)
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model[tstvec[0]], model2[tstvec[0]]))
        self.assertTrue(np.allclose(model[tstvec[1]], model2[tstvec[1]]))
        self.assertTrue(np.allclose(model[[]], model2[[]]))  # try projecting an empty vector

        # Test persistence with using `smartirs`
        fname = get_tmpfile('gensim_models_smartirs.tst.gz')
        model = tfidfmodel.TfidfModel(self.corpus, smartirs="ntc")
        model.save(fname)
        model2 = tfidfmodel.TfidfModel.load(fname, mmap=None)
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model[tstvec[0]], model2[tstvec[0]]))
        self.assertTrue(np.allclose(model[tstvec[1]], model2[tstvec[1]]))
        self.assertTrue(np.allclose(model[[]], model2[[]]))  # try projecting an empty vector

        # Test persistence between Gensim v3.2.0 and current compressed model.
        model3 = tfidfmodel.TfidfModel(self.corpus, smartirs="ntc")
        model4 = tfidfmodel.TfidfModel.load(datapath('tfidf_model.tst.bz2'))
        self.assertTrue(model3.idfs == model4.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model3[tstvec[0]], model4[tstvec[0]]))
        self.assertTrue(np.allclose(model3[tstvec[1]], model4[tstvec[1]]))
        self.assertTrue(np.allclose(model3[[]], model4[[]]))  # try projecting an empty vector

        # Test persistence with using pivoted normalization
        fname = get_tmpfile('gensim_models_smartirs.tst.gz')
        model = tfidfmodel.TfidfModel(self.corpus, pivot_norm=True, pivot=0, slope=1)
        model.save(fname)
        model2 = tfidfmodel.TfidfModel.load(fname, mmap=None)
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model[tstvec[0]], model2[tstvec[0]]))
        self.assertTrue(np.allclose(model[tstvec[1]], model2[tstvec[1]]))

        # Test persistence between Gensim v3.2.0 and pivoted normalization compressed model.
        model3 = tfidfmodel.TfidfModel(self.corpus, pivot_norm=True, pivot=0, slope=1)
        model4 = tfidfmodel.TfidfModel.load(datapath('tfidf_model.tst.bz2'))
        self.assertTrue(model3.idfs == model4.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model3[tstvec[0]], model4[tstvec[0]]))
        self.assertTrue(np.allclose(model3[tstvec[1]], model4[tstvec[1]]))

    def test_consistency(self):
        docs = [corpus[1], corpus[2]]

        # Test if `ntc` yields the default docs.
        model = tfidfmodel.TfidfModel(self.corpus, smartirs='ntc')
        transformed_docs = [model[docs[0]], model[docs[1]]]

        model = tfidfmodel.TfidfModel(self.corpus)
        expected_docs = [model[docs[0]], model[docs[1]]]

        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))

        # Testing all the variations of `wlocal`
        # nnn
        model = tfidfmodel.TfidfModel(self.corpus, smartirs='nnn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 2),
                          (4, 2),
                          (5, 3),
                          (6, 2),
                          (7, 3),
                          (8, 2)],
                         [(5, 6),
                          (9, 3),
                          (10, 3)]]

        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))

        # lnn
        model = tfidfmodel.TfidfModel(self.corpus, smartirs='lnn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 2.0),
                          (4, 2.0),
                          (5, 3.0),
                          (6, 2.0),
                          (7, 3.0),
                          (8, 2.0)],
                         [(5, 6.0),
                          (9, 3.0),
                          (10, 3.0)]]

        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))

        # ann
        model = tfidfmodel.TfidfModel(self.corpus, smartirs='ann')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 2.0),
                          (4, 2.0),
                          (5, 3.0),
                          (6, 2.0),
                          (7, 3.0),
                          (8, 2.0)],
                         [(5, 3.0),
                          (9, 2.25),
                          (10, 2.25)]]

        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))

        # bnn
        model = tfidfmodel.TfidfModel(self.corpus, smartirs='bnn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 2),
                          (4, 2),
                          (5, 3),
                          (6, 2),
                          (7, 3),
                          (8, 2)],
                         [(5, 3),
                          (9, 3),
                          (10, 3)]]

        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))

        # Lnn
        model = tfidfmodel.TfidfModel(self.corpus, smartirs='Lnn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 1.4635792826230198),
                          (4, 1.4635792826230198),
                          (5, 2.19536892393453),
                          (6, 1.4635792826230198),
                          (7, 2.19536892393453),
                          (8, 1.4635792826230198)],
                         [(5, 3.627141918134611),
                          (9, 1.8135709590673055),
                          (10, 1.8135709590673055)]]

        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))

        # Testing all the variations of `glocal`
        # ntn
        model = tfidfmodel.TfidfModel(self.corpus, smartirs='ntn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 2.1699250014423126),
                          (4, 2.1699250014423126),
                          (5, 1.5849625007211563),
                          (6, 2.1699250014423126),
                          (7, 1.5849625007211563),
                          (8, 2.1699250014423126)],
                         [(5, 3.1699250014423126),
                          (9, 1.5849625007211563),
                          (10, 1.5849625007211563)]]

        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))

        # npn
        model = tfidfmodel.TfidfModel(self.corpus, smartirs='npn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 1.8073549220576042),
                          (4, 1.8073549220576042),
                          (5, 1.0),
                          (6, 1.8073549220576042),
                          (7, 1.0),
                          (8, 1.8073549220576042)],
                         [(5, 2.0),
                          (9, 1.0),
                          (10, 1.0)]]

        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))

        # Testing all the variations of `normalize`
        # nnc
        model = tfidfmodel.TfidfModel(self.corpus, smartirs='nnc')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 0.34299717028501764),
                          (4, 0.34299717028501764),
                          (5, 0.51449575542752646),
                          (6, 0.34299717028501764),
                          (7, 0.51449575542752646),
                          (8, 0.34299717028501764)],
                         [(5, 0.81649658092772603),
                          (9, 0.40824829046386302),
                          (10, 0.40824829046386302)]]

        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))

        model = tfidfmodel.TfidfModel(self.corpus, wlocal=lambda x: x, wglobal=lambda x, y: x * x, smartirs='nnc')

        transformed_docs = [model[docs[0]], model[docs[1]]]

        model = tfidfmodel.TfidfModel(self.corpus, wlocal=lambda x: x * x, wglobal=lambda x, y: x, smartirs='nnc')
        expected_docs = [model[docs[0]], model[docs[1]]]

        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))

    def test_pivoted_normalization(self):
        docs = [corpus[1], corpus[2]]

        # Test if slope=1 yields the default docs for pivoted normalization.
        model = tfidfmodel.TfidfModel(self.corpus)
        transformed_docs = [model[docs[0]], model[docs[1]]]

        model = tfidfmodel.TfidfModel(self.corpus, pivot_norm=True, pivot=0, slope=1)
        expected_docs = [model[docs[0]], model[docs[1]]]

        self.assertTrue(np.allclose(sorted(transformed_docs[0]), sorted(expected_docs[0])))
        self.assertTrue(np.allclose(sorted(transformed_docs[1]), sorted(expected_docs[1])))

        # Test if pivoted model is consistent
        model = tfidfmodel.TfidfModel(self.corpus, pivot_norm=True, pivot=0, slope=0.5)
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(8, 0.8884910505493495), (7, 0.648974041227711), (6, 0.8884910505493495),
            (5, 0.648974041227711), (4, 0.8884910505493495), (3, 0.8884910505493495)],
            [(10, 0.8164965809277263), (9, 0.8164965809277263), (5, 1.6329931618554525)]
            ]

        self.assertTrue(np.allclose(sorted(transformed_docs[0]), sorted(expected_docs[0])))
        self.assertTrue(np.allclose(sorted(transformed_docs[1]), sorted(expected_docs[1])))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
