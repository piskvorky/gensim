#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking processing/storing large inputs.
"""


import logging
import unittest
import os

import numpy as np

import gensim
from gensim.test.utils import get_tmpfile


class BigCorpus(object):
    """A corpus of a large number of docs & large vocab"""

    def __init__(self, words_only=False, num_terms=200000, num_docs=1000000, doc_len=100):
        self.dictionary = gensim.utils.FakeDict(num_terms)
        self.words_only = words_only
        self.num_docs = num_docs
        self.doc_len = doc_len

    def __iter__(self):
        for _ in range(self.num_docs):
            doc_len = np.random.poisson(self.doc_len)
            ids = np.random.randint(0, len(self.dictionary), doc_len)
            if self.words_only:
                yield [str(idx) for idx in ids]
            else:
                weights = np.random.poisson(3, doc_len)
                yield sorted(zip(ids, weights))


if os.environ.get('GENSIM_BIG', False):
    class TestLargeData(unittest.TestCase):
        """Try common operations, using large models. You'll need ~8GB RAM to run these tests"""

        def testWord2Vec(self):
            corpus = BigCorpus(words_only=True, num_docs=100000, num_terms=3000000, doc_len=200)
            tmpf = get_tmpfile('gensim_big.tst')
            model = gensim.models.Word2Vec(corpus, size=300, workers=4)
            model.save(tmpf, ignore=['syn1'])
            del model
            gensim.models.Word2Vec.load(tmpf)

        def testLsiModel(self):
            corpus = BigCorpus(num_docs=50000)
            tmpf = get_tmpfile('gensim_big.tst')
            model = gensim.models.LsiModel(corpus, num_topics=500, id2word=corpus.dictionary)
            model.save(tmpf)
            del model
            gensim.models.LsiModel.load(tmpf)

        def testLdaModel(self):
            corpus = BigCorpus(num_docs=5000)
            tmpf = get_tmpfile('gensim_big.tst')
            model = gensim.models.LdaModel(corpus, num_topics=500, id2word=corpus.dictionary)
            model.save(tmpf)
            del model
            gensim.models.LdaModel.load(tmpf)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
