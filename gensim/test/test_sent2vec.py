#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import unittest
import os

import numpy as np

from gensim.models.sent2vec import Sent2Vec
from gensim.test.test_fasttext import new_sentences
from gensim.test.utils import get_tmpfile, common_texts as sentences

logger = logging.getLogger(__name__)

test_sentences = [
    ['How', 'are', 'you', 'today'],
    ['Coding', 'is', 'fun'],
    ['This', 'is', 'a', 'great', 'gensim', 'model'],
    ['Would', 'you', 'like', 'to', 'go', 'for', 'a', 'swim']
]


class TestSent2VecModel(unittest.TestCase):

    def setUp(self):
        self.s2v_path = os.path.join('gensim/models', 'sent2vec')

    def test_training(self):
        model = Sent2Vec(size=5, min_count=1, neg=5, seed=42, workers=1)
        model.build_vocab(sentences)

        model.train(sentences)

        self.assertEqual(model.wi.shape, (2000012, 5))
        self.assertEqual(model.dict.size, 12)
        self.assertEqual(model.wi.shape, (model.dict.size + model.bucket, model.vector_size))

        # build vocab and train in one step; must be the same as above
        model2 = Sent2Vec(sentences, size=5, min_count=1, neg=5, seed=42, workers=1)
        self.models_equal(model, model2)

    def models_equal(self, model, model2):
        self.assertEqual(model.dict.size, model2.dict.size)
        self.assertTrue(np.allclose(model.wi, model2.wi))
        most_common_word = max(model.dict.words, key=lambda item: item.count).word
        self.assertTrue(np.allclose(model.dict.word2int[model.dict.find(
                most_common_word)], model2.dict.word2int[model2.dict.find(most_common_word)]))

    def test_persistence(self):
        tmpf = get_tmpfile('gensim_sent2vec.tst')
        model = Sent2Vec(sentences, size=5, min_count=0, seed=42, neg=5, workers=1)
        model.save(tmpf)
        loaded_model = Sent2Vec.load(tmpf)
        self.models_equal(model, loaded_model)

    def test_online_learning(self):
        model = Sent2Vec(sentences, size=5, min_count=1, seed=42, neg=5, workers=1)
        self.assertTrue(model.dict.size, 12)
        self.assertTrue(model.dict.words[model.dict.word2int[model.dict.find('graph')]].count, 3)
        model.build_vocab(new_sentences, update=True)  # update vocab
        self.assertEqual(model.dict.size, 14)
        self.assertTrue(model.dict.words[model.dict.word2int[model.dict.find('graph')]].count, 4)
        self.assertTrue(model.dict.words[model.dict.word2int[model.dict.find('artificial')]].count, 3)

    def test_online_learning_after_save(self):
        tmpf = get_tmpfile('gensim_sent2vec.tst')
        model = Sent2Vec(sentences, size=5, min_count=0, seed=42, neg=5, workers=1)
        model.save(tmpf)
        model = Sent2Vec.load(tmpf)
        self.assertTrue(model.dict.size, 12)
        model.build_vocab(new_sentences, update=True)  # update vocab
        model.train(new_sentences)
        self.assertEqual(model.dict.size, 14)

    def test_sent2vec_for_document(self):
        model = Sent2Vec(sentences, size=5, min_count=0, seed=42, neg=5, workers=1)
        for sentence in test_sentences:
            sent_vec = model.sentence_vectors(sentence)
            self.assertEqual(len(sent_vec), 5)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
