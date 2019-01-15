#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import unittest
import os

import numpy as np

from gensim.models.sent2vec import Sent2Vec
from gensim.test.utils import get_tmpfile

logger = logging.getLogger(__name__)

test_sentences = [
    [u'How', u'are', u'you', u'today'],
    [u'Coding', u'is', u'fun'],
    [u'This', u'is', u'a', u'great', u'gensim', u'model'],
    [u'Would', u'you', u'like', u'to', u'go', u'for', u'a', u'swim']
]

new_sentences = [
    [u'computer', u'artificial', u'intelligence'],
    [u'artificial', u'trees'],
    [u'human', u'intelligence'],
    [u'artificial', u'graph'],
    [u'intelligence'],
    [u'artificial', u'intelligence', u'system']
]

sentences = [
    [u'human', u'interface', u'computer'],
    [u'survey', u'user', u'computer', u'system', u'response', u'time'],
    [u'eps', u'user', u'interface', u'system'],
    [u'system', u'human', u'system',u'eps'],
    [u'user', u'response', u'time'],
    [u'trees'],
    [u'graph', u'trees'],
    [u'graph', u'minors', u'trees'],
    [u'graph', u'minors', u'survey']
]


class TestSent2VecModel(unittest.TestCase):

    def setUp(self):
        self.s2v_path = os.path.join('gensim/models', 'sent2vec')

    def test_training(self):
        model = Sent2Vec(size=5, min_count=1, negative=5, seed=42, workers=1, max_vocab_size=100)
        model.build_vocab(sentences)

        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

        self.assertEqual(model.wi.shape, (2000012, 5))
        self.assertEqual(model.vocabulary.size, 12)
        self.assertEqual(model.wi.shape, (model.vocabulary.size + model.bucket, model.vector_size))

        # build vocab and train in one step; must be the same as above
        model2 = Sent2Vec(sentences, size=5, min_count=1, negative=5, seed=42, workers=1, max_vocab_size=100)
        self.models_equal(model, model2)

    def models_equal(self, model, model2):
        self.assertEqual(model.vocabulary.size, model2.vocabulary.size)
        self.assertTrue(np.allclose(model.wi, model2.wi))
        most_common_word = max(model.vocabulary.words, key=lambda item: item.count).word
        self.assertTrue(np.allclose(model.vocabulary.word2int[model.vocabulary.find(
                most_common_word)], model2.vocabulary.word2int[model2.vocabulary.find(most_common_word)]))

    def test_persistence(self):
        tmpf = get_tmpfile('gensim_sent2vec.tst')
        model = Sent2Vec(sentences, size=5, min_count=0, seed=42, negative=5, workers=1, max_vocab_size=100)
        model.save(tmpf)
        loaded_model = Sent2Vec.load(tmpf)
        self.models_equal(model, loaded_model)

    def test_online_learning(self):
        model = Sent2Vec(sentences, size=5, min_count=1, seed=42, negative=5, workers=1, max_vocab_size=100)
        self.assertTrue(model.vocabulary.size, 12)
        self.assertTrue(model.vocabulary.words[model.vocabulary.word2int[model.vocabulary.find(u'graph')]].count, 3)
        model.build_vocab(new_sentences, update=True)  # update vocab
        self.assertEqual(model.vocabulary.size, 14)
        self.assertTrue(model.vocabulary.words[model.vocabulary.word2int[model.vocabulary.find(u'graph')]].count, 4)
        self.assertTrue(
            model.vocabulary.words[model.vocabulary.word2int[model.vocabulary.find(u'artificial')]].count, 3)

    def test_online_learning2(self):
        model = Sent2Vec(sentences, size=5, min_count=1, seed=42, negative=5, workers=1, max_vocab_size=100)
        model.train(new_sentences, total_examples=len(new_sentences), epochs=1)

    def test_online_learning_after_save(self):
        tmpf = get_tmpfile('gensim_sent2vec.tst')
        model = Sent2Vec(sentences, size=5, min_count=0, seed=42, negative=5, workers=1, max_vocab_size=100)
        model.save(tmpf)
        model = Sent2Vec.load(tmpf)
        self.assertTrue(model.vocabulary.size, 12)
        model.train(new_sentences, total_examples=len(new_sentences), epochs=model.epochs)
        model.build_vocab(new_sentences, update=True)  # update vocab
        model.train(new_sentences, total_examples=model.corpus_count, epochs=model.epochs)
        self.assertEqual(model.vocabulary.size, 14)

    def test_sent2vec_for_document(self):
        model1 = Sent2Vec(sentences, size=5, min_count=0, seed=42, negative=5, workers=1, max_vocab_size=100)
        model2 = Sent2Vec(sentences, size=5, min_count=0, seed=42, negative=5, workers=1, max_vocab_size=100)
        for sentence in test_sentences:
            sent_vec1 = model1[sentence]
            sent_vec2 = model2[sentence]
            self.assertTrue(np.allclose(sent_vec1, sent_vec2))
            self.assertEqual(len(sent_vec1), 5)
            self.assertEqual(len(sent_vec2), 5)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
