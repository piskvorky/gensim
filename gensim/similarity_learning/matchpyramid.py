# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Merge, Dot
from keras.optimizers import Adam
# from utils.utility import *

from dynamMP import *

class MatchPyramid:
    def __init__(self, embedding, vocab_size, embed_trainable=False, target_mode='ranking',
                 kernel_size=[3, 3], kernel_count=32, dpool_size=[3, 10], dropout_rate=0., text_maxlen=100):

        self.embedding = embedding
        self.embedding_dim = 50
        self.vocab_size = vocab_size
        self.embed_trainable = embed_trainable
        self.kernel_count = kernel_count
        self.kernel_size = kernel_size
        self.dpool_size = dpool_size
        self.dropout_rate = dropout_rate
        self.text_maxlen = text_maxlen
        self.target_mode = target_mode
        self.build()

    def build(self):
        query = Input(name='query', shape=(self.text_maxlen,))
        doc = Input(name='doc', shape=(self.text_maxlen,))
        
        dpool_index = Input(name='dpool_index', shape=[self.text_maxlen, self.text_maxlen, 3], dtype='int32')


        embedding = Embedding(self.vocab_size, self.embedding_dim, weights=[self.embedding], trainable=self.embed_trainable)
        q_embed = embedding(query)
        d_embed = embedding(doc)

        cross = Dot(axes=[2, 2], normalize=False)([q_embed, d_embed])

        cross_reshape = Reshape((self.text_maxlen, self.text_maxlen, 1))(cross)


        conv2d = Conv2D(self.kernel_count, self.kernel_size, padding='same', activation='relu')
        dpool = DynamicMaxPooling(self.dpool_size[0], self.dpool_size[1])

        conv1 = conv2d(cross_reshape)
        pool1 = dpool([conv1, dpool_index])
        pool1_flat = Flatten()(pool1)
        pool1_flat_drop = Dropout(rate=self.dropout_rate)(pool1_flat)

        if self.target_mode == 'classification':
            out_ = Dense(2, activation='softmax')(pool1_flat_drop)
        elif self.target_mode in ['regression', 'ranking']:
            out_ = Dense(1)(pool1_flat_drop)

        self.model = Model(inputs=[query, doc, dpool_index], outputs=out_)
    
    def get_model(self):
        return self.model
