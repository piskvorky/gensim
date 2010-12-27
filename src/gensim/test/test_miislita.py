#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this file replicates the miislita vector spaces f

from __future__ import division  # always use floats

import logging
import unittest

import os
from gensim import corpora, models, similarities

# sample data files are located in the same folder
module_path = os.path.dirname(__file__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.WARNING)


class TestMiislita(unittest.TestCase):
    def setUp(self):
        pass

    def test_miislita_high_level(self):
        # load plain text from file
        corpusName = os.path.join(module_path, 'miIslita.cor')
        with open(corpusName) as corpusfile:
            texts = corpusfile.readlines()

        # filter texts with a stopwordlist
        stoplist = set('for a of the and to in on'.split())
        texts = [[word for word in doc.lower().split() if word not in stoplist]
                 for doc in texts]

        # store the dictionary, for future reference; not really needed.
        dictionary = corpora.Dictionary.fromDocuments(texts)
        dictionary.save(corpusName + '.dict')

        corpusMiislita = [dictionary.doc2bow(text) for text in texts]

        # create a corpus object (not LoL, but a scipy matrix). For this we
        # need to create an index to get a sparse matrix
        #
        # 1: initialize a model
        # 2: counterintuitive -- create an index. The index object has the
        #    tfidf matrix.
        #    index_tfidf.corpus gives the doc vectors as rows. Note that
        #    naming is not very fortunate.
        # 3: See that the counts are the same as the mm file (write a test)

        tfidf = models.TfidfModel(corpusMiislita, dictionary.id2token,
                normalize=False)
        index_tfidf = similarities.SparseMatrixSimilarity(
                tfidf[corpusMiislita])
        index_tfidf.save(corpusName + '.index')

        # compare to query
        query = 'latent semantic indexing'
        vec_bow = dictionary.doc2bow(query.lower().split())

        # similarities, ordered
        # perform a similarity query against the corpus
        sims_tfidf = index_tfidf[vec_bow]

        # NOTE: it does not matter if we use the raw counts (vec_bow) or the
        # tfidf counts for the query here (vec_tfidf). The resulting cosines
        # are the same.
        # sims_tfidf = sorted(list(enumerate(sims_tfidf)), key=lambda item:
        #       -item[1])

        # for the expected results see the acticle
        expected = [0, 0.2560, 0.7022, 0.1524, 0.3334]
        for i, value in enumerate(expected):
            self.assertAlmostEqual(sims_tfidf[i], value, 2)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
