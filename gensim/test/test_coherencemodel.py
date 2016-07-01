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

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaMallet
from gensim.models.wrappers import LdaVowpalWabbit
from gensim.corpora.dictionary import Dictionary

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

class TestCoherenceModel(unittest.TestCase):
    def setUp(self):
        # Suppose given below are the topics which two different LdaModels come up with.
        # `topics1` is clearly better as it has a clear distinction between system-human
        # interaction and graphs. Hence both the coherence measures for `topics1` should be
        # greater.
        self.topics1 = [['human', 'computer', 'system', 'interface'],
                        ['graph', 'minors', 'trees', 'eps']]
        self.topics2 = [['user', 'graph', 'minors', 'system'],
                        ['time', 'graph', 'survey', 'minors']]

    def testUMass(self):
        """Test U_Mass topic coherence algorithm on given topics"""
        cm1 = CoherenceModel(topics=self.topics1, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        cm2 = CoherenceModel(topics=self.topics2, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        self.assertTrue(cm1.get_coherence() > cm2.get_coherence())

    def testCv(self):
        """Test C_v topic coherence algorithm on given topics"""
        cm1 = CoherenceModel(topics=self.topics1, texts=texts, dictionary=dictionary, coherence='c_v')
        cm2 = CoherenceModel(topics=self.topics2, texts=texts, dictionary=dictionary, coherence='c_v')
        self.assertTrue(cm1.get_coherence() > cm2.get_coherence())

    def testErrors(self):
        """Test if errors are raised on bad input"""
        # not providing dictionary
        self.assertRaises(ValueError, CoherenceModel, topics=self.topics1, corpus=corpus, coherence='u_mass')
        # not providing texts for c_v and instead providing corpus
        self.assertRaises(ValueError, CoherenceModel, topics=self.topics1, corpus=corpus, dictionary=dictionary, coherence='c_v')
        # not providing corpus or texts for u_mass
        self.assertRaises(ValueError, CoherenceModel, topics=self.topics1, dictionary=dictionary, coherence='u_mass')

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
