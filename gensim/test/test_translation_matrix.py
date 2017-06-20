import os
import numpy as np
import unittest
from gensim import utils

from gensim.models import translation_matrix
from gensim.models import KeyedVectors

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


class TesttranslationMatrix(unittest.TestCase):
    def test_translation_matrix(self):
        train_file = datapath("OPUS_en_it_europarl_train_5K.txt")
        with utils.smart_open(train_file, "r") as f:
            word_pair = [tuple(utils.to_unicode(line).strip().split()) for line in f]

        source_word_vec_file = datapath("EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        source_word_vec = KeyedVectors.load_word2vec_format(source_word_vec_file, binary=False)

        target_word_vec_file = datapath("IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        target_word_vec = KeyedVectors.load_word2vec_format(target_word_vec_file, binary=False)

        translation_matrix.TranslationMatrix(word_pair, source_word_vec, target_word_vec)
