from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dot, Dropout
from keras import regularizers

class DSSM(object):
    def __init__(self, config, vocab_size):
        self.__name = 'DSSM'
        self.config = config

        self.vocab_size = vocab_size # config['vocab_size']
        self.hidden_sizes = config['hidden_sizes']
        self.regularizer_rate = config['reg_rate']
        self.dropout_rate = config['dropout_rate']
        self.target_mode = config['target_mode']

    def get_model(self):
        # TODO check this show_layer business

        query = Input(name='query', shape=(self.vocab_size,))#, sparse=True)
        doc = Input(name='doc', shape=(self.vocab_size,))#, sparse=True)

        def mlp_work(input_dim):
            seq = Sequential()
            num_hidden_layers = len(self.hidden_sizes)

            if num_hidden_layers == 1:
                seq.add(Dense(self.hidden_sizes[0], input_shape=(input_dim,), activity_regularizer=regularizers.l2(self.regularizer_rate)))
            else:
                seq.add(Dense(self.hidden_sizes[0], activation='tanh', input_shape=(input_dim,), activity_regularizer=regularizers.l2(self.regularizer_rate)))
                for i in range(num_hidden_layers-2):
                    seq.add(Dense(self.config['hidden_sizes'][i+1], activation='tanh', activity_regularizer=regularizers.l2(self.regularizer_rate)))
                    seq.add(Dropout(rate=self.dropout_rate))
                seq.add(Dense(self.hidden_sizes[num_hidden_layers-1], activity_regularizer=regularizers.l2(self.regularizer_rate)))
                seq.add(Dropout(rate=self.dropout_rate))
            return seq

        mlp = mlp_work(self.vocab_size)
        rq = mlp(query)
        rd = mlp(doc)

        out_ = Dot( axes= [1, 1], normalize=True)([rq, rd])
        if self.target_mode == 'classification':
            out_ = Dense(2, activation='softmax')(out_)
            show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=[out_])
        return model