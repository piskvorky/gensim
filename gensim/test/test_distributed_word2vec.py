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
import tempfile

import numpy

from gensim.models import Distributed_Word2Vec

module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)

def testfile():
    return os.path.join(tempfile.gettempdir(), 'gensim_distributed.test')

class TestTensorFlow(unittest.TestCase):
    def setUp(self):
        try:
            import tensorflow as tf
        except ImportError:
            raise unittest.SkipTest("TensorFlow not installed. Skipping tensorflow tests")
        self.corpus_file = datapath('lee.cor')
        self.out_path = 'testmodel'
        self.tf_file = datapath('test_glove.txt')
        vocab_path = '.'
        self.test_model = Distributed_Word2Vec(self.corpus_file, epochs_to_train=1, embedding_size=100, batch_size=100000, save_path=vocab_path)

    
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
