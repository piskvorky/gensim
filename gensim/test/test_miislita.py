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
import os
import unittest

from gensim import utils, corpora, models, similarities
from gensim.test.utils import datapath, get_tmpfile

logger = logging.getLogger('test_miislita')


class CorpusMiislita(corpora.TextCorpus):
    stoplist = set('for a of the and to in on'.split())

    def get_texts(self):
        """
        Parse documents from the .cor file provided in the constructor. Lowercase
        each document and ignore some stopwords.

        .cor format: one document per line, words separated by whitespace.

        """
        for doc in self.getstream():
            yield [word for word in utils.to_unicode(doc).lower().split()
                    if word not in CorpusMiislita.stoplist]

    def __len__(self):
        """Define this so we can use `len(corpus)`"""
        if 'length' not in self.__dict__:
            logger.info("caching corpus size (calculating number of documents)")
            self.length = sum(1 for _ in self.get_texts())
        return self.length


class TestMiislita(unittest.TestCase):
    def test_textcorpus(self):
        """Make sure TextCorpus can be serialized to disk. """
        # construct corpus from file
        miislita = CorpusMiislita(datapath('head500.noblanks.cor.bz2'))

        # make sure serializing works
        ftmp = get_tmpfile('test_textcorpus.mm')
        corpora.MmCorpus.save_corpus(ftmp, miislita)
        self.assertTrue(os.path.exists(ftmp))

        # make sure deserializing gives the same result
        miislita2 = corpora.MmCorpus(ftmp)
        self.assertEqual(list(miislita), list(miislita2))

    def test_save_load_ability(self):
        """
        Make sure we can save and load (un/pickle) TextCorpus objects (as long
        as the underlying input isn't a file-like object; we cannot pickle those).
        """
        # construct corpus from file
        corpusname = datapath('miIslita.cor')
        miislita = CorpusMiislita(corpusname)

        # pickle to disk
        tmpf = get_tmpfile('tc_test.cpickle')
        miislita.save(tmpf)

        miislita2 = CorpusMiislita.load(tmpf)

        self.assertEqual(len(miislita), len(miislita2))
        self.assertEqual(miislita.dictionary.token2id, miislita2.dictionary.token2id)

    def test_miislita_high_level(self):
        # construct corpus from file
        miislita = CorpusMiislita(datapath('miIslita.cor'))

        # initialize tfidf transformation and similarity index
        tfidf = models.TfidfModel(miislita, miislita.dictionary, normalize=False)
        index = similarities.SparseMatrixSimilarity(tfidf[miislita], num_features=len(miislita.dictionary))

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
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
