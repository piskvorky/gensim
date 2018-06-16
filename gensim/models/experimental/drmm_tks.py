import keras.backend as K
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Dense, Lambda, Reshape, Dropout
from keras.activations import softmax


class DRMM_TKS:
    """This is a variant version of DRMM, which applied topk pooling in the matching matrix.
    It has the following steps:
    1. embed queries into embedding vector named 'q_embed' and 'd_embed' respectively
    2. computing 'q_embed' and 'd_embed' with element-wise multiplication
    3. computing output of upper layer with dense layer operation
    4. take softmax operation on the output of this layer named 'g' and find the k largest entries named 'mm_k'.
    5. input 'mm_k' into hidden layers, with specified length of layers and activation function
    6. compute 'g' and 'mm_k' with element-wise multiplication.

    # Returns
        Score list between queries and documents.
    """

    def __init__(self, embedding, vocab_size, embed_trainable=False, target_mode='ranking',
                 topk=50, dropout_rate=0.5, text_maxlen=100, hidden_sizes=[100, 1]):
        """Initializes the model
        Parameters:
        ----------
        embedding: numpy array matrix
            A numpy array matrix which has the embeddings extracted from a pretrained
            word embedding like Stanford Glove
            This is fed to the Embedding Layer which then outputs the word embedding

        vocab_size: int
            The number of unique words in the corpus

        embed_trainable: boolean
            Whether the embeddings should be trained
            if True, the embeddings are trianed

        target_mode: 'training', 'ranking' or 'classification'
            Indicates the mode in which the model will be used and thus changes the topology

        topk: int
            Used for topk pooling in the matching matrix

        dropout_rate: float between 0 and 1
            The probability of making a neuron dead
            Used for regularization

        text_maxlen: int
            The maximum possible length of a sentence
            used for deiciding matrix dimensions

        hidden_sizes: list of ints
            The list of hidden sizes for the fully connected layers connected to the matching matrix
            For example
                hidden_sizes = [10, 20, 30]
            will add 3 fully connected layers of 10, 20 and 30 hidden neurons
        """
        self.embedding = embedding
        self.embed_dim = embedding.shape[1]
        self.embed_trainable = embed_trainable
        self.topk = topk
        self.dropout_rate = dropout_rate
        self.text_maxlen = text_maxlen
        self.vocab_size = vocab_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(self.hidden_sizes)
        self.target_mode = target_mode
        self.build()

    def build(self):
        """Builds the model based on parameters set during initialization"""
        query = Input(name='query', shape=(self.text_maxlen,))
        doc = Input(name='doc', shape=(self.text_maxlen,))
        embedding = Embedding(self.embedding.shape[0], self.embedding.shape[1], weights=[self.embedding],
                                trainable=self.embed_trainable)

        q_embed = embedding(query)
        d_embed = embedding(doc)

        mm = Dot(axes=[2, 2], normalize=True, name="mm_q_embed_DOT_d_embed")([q_embed, d_embed])

        # compute term gating
        w_g = Dense(1, name="w_g_Dense_1_q_embed")(q_embed)

        g = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.text_maxlen, ), name="g_Softmax_w_g")(w_g)
        g = Reshape((self.text_maxlen,), name="g_Reshape_maxlen_w_g")(g)

        mm_k = Lambda(lambda x: K.tf.nn.top_k(x, k=self.topk, sorted=True)[0], name="mm_k_topk_mm")(mm)

        for i in range(self.num_layers):
            mm_k = Dense(self.hidden_sizes[i], activation='softplus', kernel_initializer='he_uniform',
                         bias_initializer='zeros', name="mm_k_Dense_%d_mm_k" % self.hidden_sizes[i])(mm_k)

        mm_k_dropout = Dropout(rate=self.dropout_rate, name="mm_k_dropout_Dropout_mm_k")(mm_k)

        mm_reshape = Reshape((self.text_maxlen,), name="mm_reshape_Reshape_maxlen_mm_k_dropout")(mm_k_dropout)

        mean = Dot(axes=[1, 1], normalize=True, name="mean_mm_reshape_DOT_g")([mm_reshape, g])

        if self.target_mode == 'classification':
            out_ = Dense(2, activation='softmax')(mean)
        elif self.target_mode in ['regression', 'ranking']:
            out_ = Reshape((1,), name="out_Reshape_mean")(mean)

        self.model = Model(inputs=[query, doc], outputs=out_)

    def get_model(self):
        return self.model
