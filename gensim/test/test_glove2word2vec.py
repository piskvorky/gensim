#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Test for gensim.scripts.glove2word2vec.py."""


# import unittest
# import os
# import sys
# import gensim
# from gensim.utils import check_output

# class TestGlove2Word2Vec(unittest.TestCase):
#     def setUp(self):
#         self.module_path = os.path.dirname(gensim.__file__)
#         self.datapath = os.path.join(self.module_path, 'test', 'test_data', 'test_glove.txt')  # Sample data files are located in the same folder
#         self.output_file = 'sample_word2vec_out.txt'

#     def testConversion(self):
#         output = check_output(['python', '-m', 'gensim.scripts.glove2word2vec', '-i', self.datapath, '-o', self.output_file])
#         if sys.version_info < (3,):
#             self.assertEqual(output, '')
#         else:
#             self.assertEqual(output, b'')
