#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module replicates the miislita vector spaces from
"A Linear Algebra Approach to the Vector Space Model -- A Fast Track Tutorial" 
by Dr. E. Garcia, admin@miislita.com
See http://www.miislita.com for further details.
"""

from __future__ import division  # always use floats
from __future__ import with_statement

import logging
import unittest

import os
from gensim import corpora, models, similarities

# sample data files are located in the same folder
module_path = os.path.dirname(__file__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.WARNING)


class CorpusMiislita(corpora.TextCorpus):
    stoplist = set('for a of the and to in on'.split())

    def get_texts(self):
        """
        Parse documents from the .cor file provided in the constructor. Lowercase
        each document and ignore some stopwords.
        
        .cor format: one document per line, words separated by whitespace.
        """
        for doc in self.getstream():
            yield [word for word in doc.lower().split()
                    if word not in CorpusMiislita.stoplist]

    def __len__(self):
        """Define this so we can use `len(corpus)`"""
        if 'length' not in self.__dict__:
            # cache corpus size (number of documents)
             self.length = sum(1 for doc in self.get_texts())
        return self.length


class TestMiislita(unittest.TestCase):
    def test_miislita_high_level(self):
        # construct corpus from file
        corpusname = os.path.join(module_path, 'miIslita.cor')
        miislita = CorpusMiislita(corpusname)

        # initialize tfidf transformation and similarity index
        tfidf = models.TfidfModel(miislita, miislita.dictionary, normalize=False)
        index = similarities.SparseMatrixSimilarity(tfidf[miislita])
#        index.save(corpusname + '.simindex')

        # compare to query
        query = 'latent semantic indexing'
        vec_bow = miislita.dictionary.doc2bow(query.lower().split())
        vec_tfidf = tfidf[vec_bow]

        # perform a similarity query against the corpus
        sims_tfidf = index[vec_tfidf]

        # for the expected results see the article
        expected = [0.0, 0.2560, 0.7022, 0.1524, 0.3334]
        for i, value in enumerate(expected):
            self.assertAlmostEqual(sims_tfidf[i], value, 2)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
