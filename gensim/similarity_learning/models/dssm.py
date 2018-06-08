from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dot, Dropout
from keras import regularizers


class DSSM(object):
    """Class for the Deep Structured Semantic Model for Similarity Learning
    described here
    https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/ # noqa

    Usage:
    dssm = DSSM(...)
    model = dssm.get_model()

    Currently a WIP, so it doesn't work perfectly and needs a lot of tuning.
    """

    def __init__(self, vocab_size, hidden_sizes=[300, 128], regularizer_rate=0.0, dropout_rate=0.5,
                 target_mode='ranking'):
        """
        parameters:
        ==========
        vocab_size : int
            the vocabulary size counted on all the character level trigrams

        hidden_sizes : list of ints
            the network architecture in terms of fully connected feed forward neural network layers

        regularizer_rate : float (TODO check if it should be bounded between {0, 1})
            the rate used by the regularizer while training

        dropout_rate : float between {0, 1}
            the rate of dropout used by the network while training

        target_mode : string {'ranking', 'classification'}
            train it either to rank or classify
            TODO check working

        """
        self.vocab_size = vocab_size
        self.hidden_sizes = hidden_sizes
        self.regularizer_rate = regularizer_rate
        self.dropout_rate = dropout_rate
        self.target_mode = target_mode

    def build_model(self):
        query = Input(name='query', shape=(self.vocab_size,))  # , sparse=True)
        doc = Input(name='doc', shape=(self.vocab_size,))  # , sparse=True)

        def mlp_work(input_dim):
            seq = Sequential()
            num_hidden_layers = len(self.hidden_sizes)

            if num_hidden_layers == 1:
                seq.add(Dense(self.hidden_sizes[0], input_shape=(input_dim,),
                              activity_regularizer=regularizers.l2(self.regularizer_rate)))
            else:
                seq.add(Dense(self.hidden_sizes[0], activation='tanh', input_shape=(input_dim,),
                              activity_regularizer=regularizers.l2(self.regularizer_rate)))
                for i in range(num_hidden_layers - 2):
                    seq.add(Dense(self.config['hidden_sizes'][i + 1], activation='tanh',
                                  activity_regularizer=regularizers.l2(self.regularizer_rate)))
                    seq.add(Dropout(rate=self.dropout_rate))
                seq.add(Dense(self.hidden_sizes[num_hidden_layers - 1],
                              activity_regularizer=regularizers.l2(self.regularizer_rate)))
                seq.add(Dropout(rate=self.dropout_rate))
            return seq

        mlp = mlp_work(self.vocab_size)
        rq = mlp(query)
        rd = mlp(doc)

        out_ = Dot(axes=[1, 1], normalize=True)([rq, rd])
        if self.target_mode == 'classification':
            out_ = Dense(2, activation='softmax')(out_)

        self.model = Model(inputs=[query, doc], outputs=[out_])

    def train(self, queries, docs, labels, epochs=10, optimizer='rmsprop', loss='mse', metrics=['accuracy']):
        # check these params
        # TODO add batching in WikiQAExtractor and here
        self.build_model()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model.fit(x={'query': queries, 'doc': docs}, y=labels)

    def get_model(self):
        if self.model:
            return self.model
        else:
            print('No model built!')
            # TODO might have to raise an Exception here
