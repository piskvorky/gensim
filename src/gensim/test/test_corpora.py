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

from gensim.corpora import dmlcorpus, bleicorpus, mmcorpus, lowcorpus, svmlightcorpus


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


if __name__ == '__main__':
    logging.basicConfig(level = logging.WARNING)
    unittest.main()
