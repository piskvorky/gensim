#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Prakhar Pratyush (er.prakhar2b@gmail.com)

import logging
import unittest
import os
from gensim import utils

from gensim.models import FastText

module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)

class LeeCorpus(object):
    def __iter__(self):
        with open(datapath('lee_background.cor')) as f:
            for line in f:
                yield utils.simple_preprocess(line)

class TestNativeFastText(unittest.TestCase):

	def testTraining(self):
		sentences = LeeCorpus()
		self.assertTrue(FastText(sentences, min_count=100, size=100, workers=3))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
