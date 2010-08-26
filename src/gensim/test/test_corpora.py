#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking corpus I/O formats (the corpora package).
"""

import logging
import os
import os.path
import unittest
import tempfile

from gensim.corpora import dmlcorpus, bleicorpus, mmcorpus, lowcorpus, svmlightcorpus, dictionary

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.WARNING)


module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_corpus.tst')


class CorpusTesterABC(object):
    def __init__(self):
        raise NotImplementedError("cannot instantiate Abstract Base Class")
        self.corpusClass = None # to be overridden with a particular class
        self.fileExtension = None # file 'testcorpus.fileExtension' must exist and be in the format of corpusClass
    
    
    def testLoad(self):
        fname = os.path.join(module_path, 'testcorpus.' + self.fileExtension.lstrip('.'))
        corpus = self.corpusClass(fname)
        docs = list(corpus)
        self.assertEqual(len(docs), 9) # the deerwester corpus always has nine documents, no matter what format

    
    def testSave(self):
        corpus = [[(1, 1.0)], [], [(0, 0.5), (2, 1.0)], []]
        
        # make sure the corpus can be saved
        self.corpusClass.saveCorpus(testfile(), corpus)
        
        # and loaded back, resulting in exactly the same corpus
        corpus2 = list(self.corpusClass(testfile()))
        self.assertEqual(corpus, corpus2)
        
        # delete the temporary file
        os.remove(testfile())
#endclass CorpusTesterABC


class TestMmCorpus(unittest.TestCase, CorpusTesterABC):
    def setUp(self):
        self.corpusClass = mmcorpus.MmCorpus
        self.fileExtension = '.mm'
#endclass TestMmCorpus


class TestSvmLightCorpus(unittest.TestCase, CorpusTesterABC):
    def setUp(self):
        self.corpusClass = svmlightcorpus.SvmLightCorpus
        self.fileExtension = '.svmlight'
#endclass TestSvmLightCorpus


class TestBleiCorpus(unittest.TestCase, CorpusTesterABC):
    def setUp(self):
        self.corpusClass = bleicorpus.BleiCorpus
        self.fileExtension = '.blei'
#endclass TestBleiCorpus


class TestDictionary(unittest.TestCase):
    def setUp(self):
        self.texts = [
                      ['human', 'interface', 'computer'],
                      ['survey', 'user', 'computer', 'system', 'response', 'time'],
                      ['eps', 'user', 'interface', 'system'],
                      ['system', 'human', 'system', 'eps'],
                      ['user', 'response', 'time'],
                      ['trees'],
                      ['graph', 'trees'],
                      ['graph', 'minors', 'trees'],
                      ['graph', 'minors', 'survey']]

    def testBuild(self):
        d = dictionary.Dictionary(self.texts)
        expected = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 2, 7: 3, 8: 2, 9: 3, 10: 3, 11: 2}
        self.assertEqual(d.docFreq, expected)
        expected = {'computer': 0, 'eps': 8, 'graph': 10, 'human': 1, 'interface': 2,
                    'minors': 11, 'response': 3, 'survey': 4, 'system': 5,
                    'time': 6, 'trees': 9, 'user': 7}
        self.assertEqual(d.token2id, expected)
        expected = dict((v, k) for k, v in expected.iteritems())
        self.assertEqual(d.id2token, expected)
    
    def testFilter(self):
        d = dictionary.Dictionary(self.texts)
        d.filterExtremes(noBelow = 2, noAbove = 1.0, keepN = 4)
        expected = {0: 3, 1: 3, 2: 3, 3: 3}
        self.assertEqual(d.docFreq, expected)
#endclass TestDictionary


if __name__ == '__main__':
    logging.basicConfig(level = logging.WARNING)
    unittest.main()
