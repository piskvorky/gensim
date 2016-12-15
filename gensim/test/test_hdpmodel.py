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

import six
import scipy.linalg

from gensim.corpora import mmcorpus, Dictionary
from gensim.models import hdpmodel
from gensim import matutils
from gensim.test import basetests


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


class TestHdpModel(unittest.TestCase, basetests.TestBaseTopicModel):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = hdpmodel.HdpModel
        self.model = self.class_(corpus, id2word=dictionary)

    def testShowTopic(self):
        # TODO create show_topic in HdpModel and then test
        return

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
