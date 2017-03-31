import six
import unittest
import os
import codecs
import pickle

import numpy as np
from gensim.keras_integration.keras_wrapper_gensim_word2vec import KerasWrapperWord2VecModel
from gensim import utils
from keras.engine import Input
from keras.models import Model
from keras.layers import merge

sentences = [
    ['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey']
]

class TestKerasWord2VecWrapper(unittest.TestCase):
    def setUp(self):
        self.model = KerasWrapperWord2VecModel(sentences, size=100, min_count=1, hs=1)

    def testWord2VecTraining(self):
        """Test word2vec training."""
        model = self.model
        self.assertTrue(model.wv.syn0.shape == (len(model.wv.vocab), 100))
        self.assertTrue(model.syn1.shape == (len(model.wv.vocab), 100))
        sims = model.most_similar('graph', topn=10)
        # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # test querying for "most similar" by vector
        graph_vector = model.wv.syn0norm[model.wv.vocab['graph'].index]
        sims2 = model.most_similar(positive=[graph_vector], topn=11)
        sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
        self.assertEqual(sims, sims2)

    def testEmbeddingLayer(self):
        """Test Keras 'Embedding' layer returned by 'get_embedding_layer' function."""
        keras_w2v_model = self.model

        embedding_layer = keras_w2v_model.get_embedding_layer()

        input_a = Input(shape=(1,), dtype='int32', name='input_a')
        input_b = Input(shape=(1,), dtype='int32', name='input_b')
        embedding_a = embedding_layer(input_a)
        embedding_b = embedding_layer(input_b)
        similarity = merge([embedding_a, embedding_b], mode='cos', dot_axes=2)

        model = Model(input=[input_a, input_b], output=similarity)
        model.compile(optimizer='sgd', loss='mse')

        word_a = 'graph'
        word_b = 'trees'
        output = model.predict([np.asarray([keras_w2v_model.wv.vocab[word_a].index]), np.asarray([keras_w2v_model.wv.vocab[word_b].index])])    #prob of occuring together
        self.assertTrue(type(output[0][0][0][0]) == np.float32)     #verify that  a float is returned

if __name__ == '__main__':
    unittest.main()
