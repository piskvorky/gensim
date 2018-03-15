#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2018 Aakaash Rao   <aakaash@uchicago.edu>
# Copyright (C) 2018 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Test for gensim.scripts.word2vec2tensor.py."""

import logging
import unittest
import os
import sys, pdb

import numpy
import gensim

from gensim.utils import check_output, smart_open
from gensim.test.utils import datapath, get_tmpfile


class TestWord2Vec2Tensor(unittest.TestCase):
	def setUp(self):
		#self.datapath = datapath('test_word2vec.txt')
		self.datapath = 'test/test_data/test_word2vec.txt' #temporary
		self.output_folder = get_tmpfile('')
		self.metadata_file = self.output_folder + '_metadata.tsv'
		self.tensor_file = self.output_folder + '_tensor.tsv'
		self.vector_file = self.output_folder + '_vector.tsv'

	def testConversion(self):
		output = check_output(args=[
			sys.executable, '-m', 'scripts.word2vec2tensor', #'gensim.scripts.word2vec2tensor',
			'--input', self.datapath, '--output', self.output_folder
		])
		try:
			with smart_open(self.metadata_file, 'rb') as f:
				self.metadata = f.readlines()
		except FileNotFoundError:
			self.fail(
				'Metadata file %s creation failed. Check the parameters and input file format.' %self.metadata_file
				)
		try:
			with smart_open(self.tensor_file, 'rb') as f:
				self.vectors = f.readlines()
		except FileNotFoundError:
			self.fail(
				'Tensor file %s creation failed. Check the parameters and input file format.' %self.tensor_file
				)

		# check if number of words and vector size in tensor file line up with word2vec
		with smart_open(self.datapath, 'rb') as f:
			self.number_words, self.vector_size = map(int, f.readline().strip().split(b' '))
		if not len(self.metadata) == len(self.vectors) == self.number_words:
			self.fail(
				'Metadata file %s and tensor file %s imply different number of rows.' % (self.metadata_file, self.tensor_file)
				)

		# write word2vec to file
		self.metadata = [word.strip() for word in self.metadata]
		self.vectors  = [vector.replace(b'\t', b' ') for vector in self.vectors]
		self.word2veclines = [self.metadata[i] + b' ' + self.vectors[i] for i in range(len(self.metadata))]
		with smart_open(self.vector_file, 'wb') as f:
			# write header
			f.write(gensim.utils.any2utf8(str(self.number_words) + ' ' + str(self.vector_size) + '\n'))
			f.writelines(self.word2veclines)

		# test that the converted model loads successfully		
		self.test_model = gensim.models.KeyedVectors.load_word2vec_format(self.vector_file, binary=False)
		self.assertTrue(numpy.allclose(self.test_model.n_similarity(['the', 'and'], ['and', 'the']), 1.0))


if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
	unittest.main()
