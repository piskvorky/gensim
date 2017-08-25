#!/usr/bin/env python
# encoding: utf-8
import os
import unittest
import tempfile
import numpy as np

from gensim import utils
from gensim.models import translation_matrix
from gensim.models import KeyedVectors

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


def temp_save_file():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'transmat-en-it.pkl')


class TestTranslationMatrix(unittest.TestCase):
    def setUp(self):
        self.train_file = datapath("OPUS_en_it_europarl_train_one2ten.txt")

        self.source_word_vec_file = datapath("EN.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        self.target_word_vec_file = datapath("IT.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")

        with utils.smart_open(self.train_file, "r") as f:
            self.word_pair = [tuple(utils.to_unicode(line).strip().split()) for line in f]

        self.source_word_vec = KeyedVectors.load_word2vec_format(self.source_word_vec_file, binary=False)
        self.target_word_vec = KeyedVectors.load_word2vec_format(self.target_word_vec_file, binary=False)

    def test_translation_matrix(self):
        model = translation_matrix.TranslationMatrix(self.word_pair, self.source_word_vec, self.target_word_vec)
        transmat = model.train(self.word_pair)
        self.assertEqual(transmat.shape, (300, 300))

    def testPersistence(self):
        """Test storing/loading the entire model."""
        model = translation_matrix.TranslationMatrix(self.word_pair, self.source_word_vec, self.target_word_vec)
        transmat = model.train(self.word_pair)
        model.save(temp_save_file())

        loaded_model = translation_matrix.TranslationMatrix.load(temp_save_file())
        self.assertTrue(np.allclose(model.translation_matrix, loaded_model.translation_matrix))

    def test_translate_nn(self):
        # test the nearest neighbor retrieval method
        model = translation_matrix.TranslationMatrix(self.word_pair, self.source_word_vec, self.target_word_vec)
        model.train(self.word_pair)

        test_word_pair = [("one", "uno"), ("two", "due"), ("apple", "mela"), ("orange", "aranicione"), ("dog", "cane"), ("pig", "maiale"), ("cat", "gatto")]
        test_source_word, test_target_word = zip(*test_word_pair)
        translated_words = model.translate(test_source_word, topn=3)

        self.assertTrue("uno" in translated_words["one"])
        self.assertTrue("due" in translated_words["two"])
        self.assertTrue("mela" in translated_words["apple"])
        self.assertTrue("arancione" in translated_words["orange"])
        self.assertTrue("cane" in translated_words["dog"])
        self.assertTrue("maiale" in translated_words["pig"])
        self.assertTrue("gatto" in translated_words["cat"])

    def test_translate_gc(self):
        # test globally corrected neighbour retrieval method
        model = translation_matrix.TranslationMatrix(self.word_pair, self.source_word_vec, self.target_word_vec)
        model.train(self.word_pair)

        test_word_pair = [("one", "uno"), ("two", "due"), ("apple", "mela"), ("orange", "aranicione"), ("dog", "cane"), ("pig", "maiale"), ("cat", "gatto")]
        test_source_word, test_target_word = zip(*test_word_pair)
        translated_words = model.translate(test_source_word, topn=3, gc=1, additional=10)

        self.assertTrue("uno" in translated_words["one"])
        self.assertTrue("due" in translated_words["two"])
        self.assertTrue("mela" in translated_words["apple"])
        self.assertTrue("arancione" in translated_words["orange"])
        self.assertTrue("cane" in translated_words["dog"])
        self.assertTrue("maiale" in translated_words["pig"])
        self.assertTrue("gatto" in translated_words["cat"])
