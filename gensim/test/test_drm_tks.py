#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""

import unittest
import gensim.downloader as api
from gensim.test.utils import datapath, get_tmpfile
from gensim.models.experimental import DRMM_TKS

class TestDrmmTksModel(unittest.TestCase):
	def testLoadModel():
		model = DRMM_TKS.load(datapath('drmm_tks'))
		self.assertTrue(model.model != None)
		self.assertTrue(model._get_pair_list != None)
		self.assertTrue(model._get_batch_iter != None)

	def testSaveModel():
		model = DRMM_TKS.load(datapath('drmm_tks'))
		model.save(get_tmpfile('temp_drmm_tks_model'))

	def testTrainModel():
		queries = ["When was World War 1 fought ?".lower().split(),
		           "When was Gandhi born ?".lower().split()]
		docs = [["The world war was bad".lower().split(),
		    "It was fought in 1996".lower().split()],
		    ["Gandhi was born in the 18th century".lower().split(),
		     "He fought for the Indian freedom movement".lower().split(),
		     "Gandhi was assasinated".lower().split()]]
		labels = [[0, 1], [1, 0, 0]]
		word_embeddings_kv = api.load('glove-wiki-gigaword-50')
		model = DRMM_TKS(queries, docs, labels, word_embedding=word_embeddings_kv, verbose=0)
