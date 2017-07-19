#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Anmol Gulati <anmol01gulati@gmail.com>
# Copyright (C) 2017 Radim Rehurek <radimrehurek@seznam.cz>
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for VarEmbed wrapper.
"""

import logging
import os
import sys

import numpy as np

 import unittest

from gensim.models.wrappers import varembed

try:
    import morfessor  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Test requires Morfessor to be installed, which is not available")

# needed because sample data files are located in the same folder
module_path = os.path.dirname(__file__)
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)
varembed_model_vector_file = datapath('varembed_vectors.pkl')
varembed_model_morfessor_file = datapath('varembed_morfessor.bin')


class TestVarembed(unittest.TestCase):
    def testLoadVarembedFormat(self):
        """Test storing/loading the entire model."""
        model = varembed.VarEmbed.load_varembed_format(vectors=varembed_model_vector_file)
        self.model_sanity(model)

    def testSimilarity(self):
        """Test n_similarity for vocab words"""
        model = varembed.VarEmbed.load_varembed_format(vectors=varembed_model_vector_file)
        self.assertTrue(model.n_similarity(['result'], ['targets']) == model.similarity('result', 'targets'))

    def model_sanity(self, model):
        """Check vocabulary and vector size"""
        self.assertEqual(model.syn0.shape, (model.vocab_size, model.vector_size))
        self.assertTrue(model.syn0.shape[0] == len(model.vocab))

    @unittest.skipIf(sys.version_info < (2, 7), 'Supported only on Python 2.7 and above')
    def testAddMorphemesToEmbeddings(self):
        """Test add morphemes to Embeddings
           Test only in Python 2.7 and above. Add Morphemes is not supported in earlier versions.
        """
        model = varembed.VarEmbed.load_varembed_format(vectors=varembed_model_vector_file)
        model_with_morphemes = varembed.VarEmbed.load_varembed_format(
            vectors=varembed_model_vector_file, morfessor_model=varembed_model_morfessor_file)
        self.model_sanity(model_with_morphemes)
        # Check syn0 is different for both models.
        self.assertFalse(np.allclose(model.syn0, model_with_morphemes.syn0))
    def testLookup(self):
        """Test lookup of vector for a particular word and list"""
        model = varembed.VarEmbed.load_varembed_format(vectors=varembed_model_vector_file)
        self.assertTrue(np.allclose(model['language'], model[['language']]))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
