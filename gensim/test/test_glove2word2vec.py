#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Test for gensim.scripts.glove2word2vec.py."""

import logging
import unittest
import os
import tempfile

import numpy
import gensim

from gensim.utils import check_output

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


def testfile():
    # temporary model will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'glove2word2vec.test')


class TestGlove2Word2Vec(unittest.TestCase):
    def setUp(self):
        self.datapath = datapath('test_glove.txt')
        self.output_file = testfile()

    def testConversion(self):
        output = check_output(args=['python', '-m', 'gensim.scripts.glove2word2vec', '--input', self.datapath, '--output', self.output_file])  # noqa:F841
        # test that the converted model loads successfully
        try:
            self.test_model = gensim.models.KeyedVectors.load_word2vec_format(self.output_file)
            self.assertTrue(numpy.allclose(self.test_model.n_similarity(['the', 'and'], ['and', 'the']), 1.0))
        except Exception:
            if os.path.isfile(os.path.join(self.output_file)):
                self.fail('model file %s was created but could not be loaded.' % self.output_file)
            else:
                self.fail('model file %s creation failed, check the parameters and input file format.' % self.output_file)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
