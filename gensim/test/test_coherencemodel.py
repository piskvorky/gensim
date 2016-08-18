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

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
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
boolean_document_based = ['u_mass']
sliding_window_based = ['c_v', 'c_uci', 'c_npmi']


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_models.tst')

def checkCoherenceMeasure(topics1, topics2, coherence):
    """Check provided topic coherence algorithm on given topics"""
    if coherence in boolean_document_based:
        cm1 = CoherenceModel(topics=topics1, corpus=corpus, dictionary=dictionary, coherence=coherence)
        cm2 = CoherenceModel(topics=topics2, corpus=corpus, dictionary=dictionary, coherence=coherence)
    else:
        cm1 = CoherenceModel(topics=topics1, texts=texts, dictionary=dictionary, coherence=coherence)
        cm2 = CoherenceModel(topics=topics2, texts=texts, dictionary=dictionary, coherence=coherence)
    return cm1.get_coherence() > cm2.get_coherence()

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
        self.ldamodel = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, passes=0, iterations=0)
        mallet_home = os.environ.get('MALLET_HOME', None)
        self.mallet_path = os.path.join(mallet_home, 'bin', 'mallet') if mallet_home else None
        if self.mallet_path:
            self.malletmodel = LdaMallet(mallet_path=self.mallet_path, corpus=corpus, id2word=dictionary, num_topics=2, iterations=0)
        vw_path = os.environ.get('VOWPAL_WABBIT_PATH', None)
        if not vw_path:
            msg = "Environment variable 'VOWPAL_WABBIT_PATH' not specified, skipping sanity checks for LDA Model"
            logging.info(msg)
            self.vw_path = None
        else:
            self.vw_path = vw_path
            self.vwmodel = LdaVowpalWabbit(self.vw_path, corpus=corpus, id2word=dictionary, num_topics=2, passes=0)

    def testUMass(self):
        """Test U_Mass topic coherence algorithm on given topics"""
        self.assertTrue(checkCoherenceMeasure(self.topics1, self.topics2, 'u_mass'))

    def testCv(self):
        """Test C_v topic coherence algorithm on given topics"""
        self.assertTrue(checkCoherenceMeasure(self.topics1, self.topics2, 'c_v'))

    def testCuci(self):
        """Test C_uci topic coherence algorithm on given topics"""
        self.assertTrue(checkCoherenceMeasure(self.topics1, self.topics2, 'c_uci'))

    def testCnpmi(self):
        """Test C_npmi topic coherence algorithm on given topics"""
        self.assertTrue(checkCoherenceMeasure(self.topics1, self.topics2, 'c_npmi'))

    def testUMassLdaModel(self):
        """Perform sanity check to see if u_mass coherence works with LDA Model"""
        # Note that this is just a sanity check because LDA does not guarantee a better coherence
        # value on the topics if iterations are increased. This can be seen here:
        # https://gist.github.com/dsquareindia/60fd9ab65b673711c3fa00509287ddde
        try:
            cm = CoherenceModel(model=self.ldamodel, corpus=corpus, coherence='u_mass')
        except:
            raise

    def testCvLdaModel(self):
        """Perform sanity check to see if c_v coherence works with LDA Model"""
        try:
            cm = CoherenceModel(model=self.ldamodel, texts=texts, coherence='c_v')
        except:
            raise

    def testCuciLdaModel(self):
        """Perform sanity check to see if c_uci coherence works with LDA Model"""
        try:
            cm = CoherenceModel(model=self.ldamodel, texts=texts, coherence='c_uci')
        except:
            raise

    def testCnpmiLdaModel(self):
        """Perform sanity check to see if c_npmi coherence works with LDA Model"""
        try:
            cm = CoherenceModel(model=self.ldamodel, texts=texts, coherence='c_npmi')
        except:
            raise

    def testUMassMalletModel(self):
        """Perform sanity check to see if u_mass coherence works with LDA Mallet gensim wrapper"""
        if not self.mallet_path:
            return
        try:
            cm = CoherenceModel(model=self.malletmodel, corpus=corpus, coherence='u_mass')
        except:
            raise

    def testCvMalletModel(self):
        """Perform sanity check to see if c_v coherence works with LDA Mallet gensim wrapper"""
        if not self.mallet_path:
            return
        try:
            cm = CoherenceModel(model=self.malletmodel, texts=texts, coherence='c_v')
        except:
            raise

    def testCuciMalletModel(self):
        """Perform sanity check to see if c_uci coherence works with LDA Mallet gensim wrapper"""
        if not self.mallet_path:
            return
        try:
            cm = CoherenceModel(model=self.malletmodel, texts=texts, coherence='c_uci')
        except:
            raise

    def testCnpmiMalletModel(self):
        """Perform sanity check to see if c_npmi coherence works with LDA Mallet gensim wrapper"""
        if not self.mallet_path:
            return
        try:
            cm = CoherenceModel(model=self.malletmodel, texts=texts, coherence='c_npmi')
        except:
            raise

    def testUMassVWModel(self):
        """Perform sanity check to see if u_mass coherence works with LDA VW gensim wrapper"""
        if not self.vw_path:
            return
        try:
            cm = CoherenceModel(model=self.vwmodel, corpus=corpus, coherence='u_mass')
        except:
            raise

    def testCvVWModel(self):
        """Perform sanity check to see if c_v coherence works with LDA VW gensim wrapper"""
        if not self.vw_path:
            return
        try:
            cm = CoherenceModel(model=self.vwmodel, texts=texts, coherence='c_v')
        except:
            raise

    def testCuciVWModel(self):
        """Perform sanity check to see if c_uci coherence works with LDA VW gensim wrapper"""
        if not self.vw_path:
            return
        try:
            cm = CoherenceModel(model=self.vwmodel, texts=texts, coherence='c_uci')
        except:
            raise

    def testCnpmiVWModel(self):
        """Perform sanity check to see if c_npmi coherence works with LDA VW gensim wrapper"""
        if not self.vw_path:
            return
        try:
            cm = CoherenceModel(model=self.vwmodel, texts=texts, coherence='c_npmi')
        except:
            raise

    def testErrors(self):
        """Test if errors are raised on bad input"""
        # not providing dictionary
        self.assertRaises(ValueError, CoherenceModel, topics=self.topics1, corpus=corpus, coherence='u_mass')
        # not providing texts for c_v and instead providing corpus
        self.assertRaises(ValueError, CoherenceModel, topics=self.topics1, corpus=corpus, dictionary=dictionary, coherence='c_v')
        # not providing corpus or texts for u_mass
        self.assertRaises(ValueError, CoherenceModel, topics=self.topics1, dictionary=dictionary, coherence='u_mass')

    def testPersistence(self):
        fname = testfile()
        model = CoherenceModel(topics=self.topics1, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        model.save(fname)
        model2 = CoherenceModel.load(fname)
        self.assertTrue(model.get_coherence() == model2.get_coherence())

    def testPersistenceCompressed(self):
        fname = testfile() + '.gz'
        model = CoherenceModel(topics=self.topics1, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        model.save(fname)
        model2 = CoherenceModel.load(fname)
        self.assertTrue(model.get_coherence() == model2.get_coherence())

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
