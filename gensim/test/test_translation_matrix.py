#!/usr/bin/env python
# encoding: utf-8
import os
import unittest
import tempfile
import numpy as np
import gensim
import math

from scipy.spatial.distance import cosine
from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec
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
        self.source_word_vec_file = datapath("EN.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")
        self.target_word_vec_file = datapath("IT.1-10.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt")

        self.word_pairs = [("one", "uno"), ("two", "due"), ("three", "tre"),
            ("four", "quattro"), ("five", "cinque"), ("seven", "sette"), ("eight", "otto"),
            ("dog", "cane"), ("pig", "maiale"), ("fish", "cavallo"), ("birds", "uccelli"),
            ("apple", "mela"), ("orange", "arancione"), ("grape", "acino"), ("banana", "banana")
        ]

        self.test_word_pairs = [("ten", "dieci"), ("cat", "gatto")]

        self.source_word_vec = KeyedVectors.load_word2vec_format(self.source_word_vec_file, binary=False)
        self.target_word_vec = KeyedVectors.load_word2vec_format(self.target_word_vec_file, binary=False)

    def test_translation_matrix(self):
        model = translation_matrix.TranslationMatrix(self.source_word_vec, self.target_word_vec, self.word_pairs)
        model.train(self.word_pairs)
        self.assertEqual(model.translation_matrix.shape, (300, 300))

    def testPersistence(self):
        """Test storing/loading the entire model."""
        model = translation_matrix.TranslationMatrix(self.source_word_vec, self.target_word_vec, self.word_pairs)
        model.train(self.word_pairs)
        model.save(temp_save_file())

        loaded_model = translation_matrix.TranslationMatrix.load(temp_save_file())
        self.assertTrue(np.allclose(model.translation_matrix, loaded_model.translation_matrix))

    def test_translate_nn(self):
        # Test the nearest neighbor retrieval method
        model = translation_matrix.TranslationMatrix(self.source_word_vec, self.target_word_vec, self.word_pairs)
        model.train(self.word_pairs)

        test_source_word, test_target_word = zip(*self.test_word_pairs)
        translated_words = model.translate(test_source_word, topn=5, source_lang_vec=self.source_word_vec, target_lang_vec=self.target_word_vec)

        for idx, item in enumerate(self.test_word_pairs):
            self.assertTrue(item[1] in translated_words[item[0]])

    def test_translate_gc(self):
        # Test globally corrected neighbour retrieval method
        model = translation_matrix.TranslationMatrix(self.source_word_vec, self.target_word_vec, self.word_pairs)
        model.train(self.word_pairs)

        test_source_word, test_target_word = zip(*self.test_word_pairs)
        translated_words = model.translate(test_source_word, topn=5, gc=1, sample_num=3, source_lang_vec=self.source_word_vec, target_lang_vec=self.target_word_vec)

        for idx, item in enumerate(self.test_word_pairs):
            self.assertTrue(item[1] in translated_words[item[0]])


def read_sentiment_docs(filename):
    sentiment_document = namedtuple('SentimentDocument', 'words tags')
    alldocs = []  # will hold all docs in original order
    with gensim.utils.smart_open(filename, encoding='utf-8') as alldata:
        for line_no, line in enumerate(alldata):
            tokens = gensim.utils.to_unicode(line).split()
            words = tokens
            tags = str(line_no)
            alldocs.append(sentiment_document(words, tags))
    return alldocs


class TestBackMappingTranslationMatrix(unittest.TestCase):
    def setUp(self):
        filename = datapath("alldata-id-10.txt")
        train_docs = read_sentiment_docs(filename)
        self.train_docs = train_docs
        self.source_doc_vec_file = datapath("small_tag_doc_5_iter50")
        self.target_doc_vec_file = datapath("large_tag_doc_10_iter50")

        self.source_doc_vec = Doc2Vec.load(self.source_doc_vec_file)
        self.target_doc_vec = Doc2Vec.load(self.target_doc_vec_file)

    def test_translation_matrix(self):
        model = translation_matrix.BackMappingTranslationMatrix(self.train_docs[:5], self.source_doc_vec, self.target_doc_vec)
        transmat = model.train(self.train_docs[:5])
        self.assertEqual(transmat.shape, (100, 100))

    def test_infer_vector(self):
        model = translation_matrix.BackMappingTranslationMatrix(self.train_docs[:5], self.source_doc_vec, self.target_doc_vec)
        model.train(self.train_docs[:5])
        infered_vec = model.infer_vector(self.target_doc_vec.docvecs[self.train_docs[5].tags])
        self.assertEqual(infered_vec.shape, (100, ))

        expected = 0.6453547135
        eps = 1e-6
        caculated = cosine(self.target_doc_vec.docvecs[self.train_docs[5].tags], infered_vec)
        self.assertLessEqual(math.fabs(caculated - expected), eps)
