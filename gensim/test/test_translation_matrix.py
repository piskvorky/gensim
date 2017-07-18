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
    def test_translation_matrix(self):
        train_file = datapath("OPUS_en_it_europarl_train_one2ten.txt")
        with utils.smart_open(train_file, "r") as f:
            word_pair = [tuple(utils.to_unicode(line).strip().split()) for line in f]

        source_word_vec_file = datapath("EN.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        source_word_vec = KeyedVectors.load_word2vec_format(source_word_vec_file, binary=False)

        target_word_vec_file = datapath("IT.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        target_word_vec = KeyedVectors.load_word2vec_format(target_word_vec_file, binary=False)

        translation_matrix.TranslationMatrix(word_pair, source_word_vec, target_word_vec)

    def testPersistence(self):
        """Test storing/loading the entire model."""
        train_file = datapath("OPUS_en_it_europarl_train_one2ten.txt")
        with utils.smart_open(train_file, "r") as f:
            word_pair = [tuple(utils.to_unicode(line).strip().split()) for line in f]

        source_word_vec_file = datapath("EN.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        source_word_vec = KeyedVectors.load_word2vec_format(source_word_vec_file, binary=False)

        target_word_vec_file = datapath("IT.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        target_word_vec = KeyedVectors.load_word2vec_format(target_word_vec_file, binary=False)

        transmat = translation_matrix.TranslationMatrix(word_pair, source_word_vec, target_word_vec)
        transmat.save(temp_save_file())

        loaded_transmat = translation_matrix.TranslationMatrix.load(temp_save_file())

        self.assertTrue(np.allclose(transmat.translation_matrix, loaded_transmat.translation_matrix))

    def test_translate_NN(self):
        train_file = datapath("OPUS_en_it_europarl_train_one2ten.txt")
        with utils.smart_open(train_file, "r") as f:
            word_pair = [tuple(utils.to_unicode(line).strip().split()) for line in f]

        source_word_vec_file = datapath("EN.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        source_word_vec = KeyedVectors.load_word2vec_format(source_word_vec_file, binary=False)

        target_word_vec_file = datapath("IT.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        target_word_vec = KeyedVectors.load_word2vec_format(target_word_vec_file, binary=False)

        transmat = translation_matrix.TranslationMatrix(word_pair, source_word_vec, target_word_vec)

        test_word_pair = [("one", "uno"), ("two", "due")]
        test_source_word, test_target_word = zip(*test_word_pair)
        transmat.translate(test_source_word, topn=3)

    def test_translate_GC(self):
        train_file = datapath("OPUS_en_it_europarl_train_one2ten.txt")
        with utils.smart_open(train_file, "r") as f:
            word_pair = [tuple(utils.to_unicode(line).strip().split()) for line in f]

        source_word_vec_file = datapath("EN.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        source_word_vec = KeyedVectors.load_word2vec_format(source_word_vec_file, binary=False)

        target_word_vec_file = datapath("IT.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        target_word_vec = KeyedVectors.load_word2vec_format(target_word_vec_file, binary=False)

        transmat = translation_matrix.TranslationMatrix(word_pair, source_word_vec, target_word_vec)

        test_word_pair = [("one", "uno"), ("two", "due")]
        test_source_word, test_target_word = zip(*test_word_pair)
        transmat.translate(test_source_word, topn=3, additional=10)
