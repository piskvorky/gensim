import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.activations import softmax

class DRMM:
    def __init__(self, text_maxlen, vocab_size, embedding_matrix, hist_size=60, dropout_rate=0.,
                hidden_sizes=[20, 1], target_mode='ranking'):
        self.initializer_fc = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=11)
        self.initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)
        self.text_maxlen = text_maxlen
        self.hist_size = hist_size
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.target_mode = target_mode
        self.build()

    def build(self):
        query = Input(name='query', shape=(self.text_maxlen,))
        doc = Input(name='doc', shape=(self.text_maxlen, self.hist_size))

        embedding = Embedding(self.embedding_matrix.shape[0], self.embedding_matrix.shape[1],
                                weights=[self.embedding_matrix], trainable = False)

        q_embed = embedding(query)
        
        q_w = Dense(1, kernel_initializer=self.initializer_gate, use_bias=False)(q_embed)
        
        q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.text_maxlen, ))(q_w)
        
        z = doc
        z = Dropout(rate=self.dropout_rate)(z)
        
        for i in range(self.num_layers - 1):
            z = Dense(self.hidden_sizes[i], kernel_initializer=self.initializer_fc)(z)
            z = Activation('tanh')(z)
            
        z = Dense(self.hidden_sizes[self.num_layers - 1], kernel_initializer=self.initializer_fc)(z)
        z = Permute((2, 1))(z)
        z = Reshape((self.text_maxlen,))(z)
        q_w = Reshape((self.text_maxlen,))(q_w)
        

        out_ = Dot( axes= [1, 1])([z, q_w])
        if self.target_mode == 'classification':
            out_ = Dense(2, activation='softmax')(out_)
        

        self.model = Model(inputs=[query, doc], outputs=[out_])
    
    def get_model(self):
        return self.model
