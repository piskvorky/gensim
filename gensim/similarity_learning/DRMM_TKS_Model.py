from gensim.models.experimental import DRMM_TKS
from collections import Counter
from pprint import pprint
import numpy as np
import six
import logging
from keras import optimizers
from keras.losses import hinge
import keras.backend as K
from gensim.similarity_learning import rank_hinge_loss
from sklearn.preprocessing import normalize
from gensim.similarity_learning import ValidationCallback
import random
random.seed(101010)
import numpy
numpy.random.seed(101010)
import tensorflow
tensorflow.set_random_seed(101010)

logger = logging.getLogger(__name__)


class DRMM_TKS_Model: 
    """User friendly model for training on similarity learning data.
    You only have to provide sentences in the data as a list of words.

    Example Usage:
    -------------
    drmm_tks_model = DRMM_TKS_Model(queries, docs, labels, word_embedding_path)
    drmm_tks_model.predict(test_queries, test_docs)

    The data should have the format:
    queries = ["When was World Wat 1 fought ?".split(),
             "When was Gandhi born ?".split()]

    docs = [
            ["The world war was bad".split(),
            "It was fought in 1996".split()],
            ["Gandhi was born in the 18th century".split(),
             "He fought for the Indian freedom movement".split(),
             "Gandhi was assasinated".split()]
           ]

    labels = [[0, 1],
              [1, 0, 0]]
    """

    def __init__(self, queries, docs, labels, word_embedding_path=None,
        text_maxlen=200, keep_full_embedding=True, normalize_embeddings=True,
        epochs=10, unk_handle_method='zero', validation_data=None):
        """Initializes the model and trains it

        Parameters:
        -----------
        queries: list of list of string words
            The questions for the similarity learning model
            
            Example:
            queries=["When was World Wat 1 fought ?".split(),
                     "When was Gandhi born ?".split()],

        docs: list of list of list of string words
            The candidate answers for the similarity learning model

            Example:
            docs = [
                    ["The world war was bad".split(),
                    "It was fought in 1996".split()],
                    ["Gandhi was born in the 18th century".split(),
                     "He fought for the Indian freedom movement".split(),
                     "Gandhi was assasinated".split()]
                   ]

        labels: list of list of ints
            Indicates when a candidate document is relevant to a query
            1 : relevant
            0 : irrelevant

            Example:
            labels = [[0, 1],
                      [1, 0, 0]]

        word_embedding_path: str
            path to the Glove vectors which have the embeddings in a .txt format
            If unset, random word embeddings will be used

        text_maxlen: int
            The maximum possible length of a query or a document
            This is used for padding.

        keep_full_embedding: boolean
            Whether the full embedding should be built or only the words in the dataset's vocab
            This becomes important for checking validation and test sets

        normalize_embeddings: boolean
            Whether the word embeddings provided should be normalized

        epochs: int
            The number of epochs for which the model should train on the data

        unk_handle_method: string {'zero', 'random'}
            The method for handling unkown words
            'zero': unknown words are given a zero vector
            'random': unknown words are given a uniformly random vector

        validation_data: list of the form [test_queries, test_docs, test_labels]
            where test_queries, test_docs  and test_labels are of the same form as
            their counter parts stated above

        """
        self.queries = queries
        self.docs = docs
        self.labels = labels
        self.word_counter = Counter()
        self.text_maxlen = text_maxlen
        self.hist_size = hist_size
        self.word_embedding_path = word_embedding_path
        self.word2index, self.index2word = {}, {}
        self.keep_full_embedding = keep_full_embedding
        self.additional_word2index = {}
        self.normalize_embeddings = normalize_embeddings
        self.model = None
        self.epochs = epochs
        self.validation_data = validation_data

        if unk_handle_method not in ['random', 'zero']:
            raise ValueError("Unkown token handling method %s" % str(unk_handle_method))
        self.unk_handle_method = unk_handle_method

        self.build_vocab()
        self.pair_list = self.get_pair_list()
        self.indexed_pair_list = self.make_indexed_pair_list()
        self.train_model()

    def build_vocab(self):
        """Indexes all the words and makes an embedding_matrix which
        can be fed directly into an Embedding layer"""

        logger.info("Starting Vocab Build")

        for q in self.queries:
            self.word_counter.update(q)
        for doc in self.docs:
            for d in doc: 
                self.word_counter.update(d)
        for i, word in enumerate(self.word_counter.keys()):
            self.word2index[word] = i
            self.index2word[i] = word

        self.vocab_size = len(self.word2index)
        logger.info("Vocab Build Complete")
        logger.info("Vocab Size is %d" % self.vocab_size)

        logger.info("Building embedding index using pretrained word embeddings")
        embeddings_index = {}
        with open(self.word_embedding_path) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        self.embedding_dim = six.next(
            six.itervalues(embeddings_index)).shape[0]
        embedding_vocab_size = len(embeddings_index)

        logger.info("The embeddings_index built from the given file has %d words of %d dimensions" %
                    (embedding_vocab_size, self.embedding_dim))

        if self.keep_full_embedding:
            if self.unk_handle_method == 'random':
                self.embedding_matrix = np.random.uniform(-0.2, 0.2,
                    (self.vocab_size + 1, self.embedding_dim))  # one for ignore vec
            elif  self.unk_handle_method == 'zero':
                self.embedding_matrix = np.zeros((self.vocab_size + 1, self.embedding_dim))  # one for ignore vec
        else:
            # one for pad, one for ignore vec
            if self.unk_handle_method == 'random':
                self.embedding_matrix = np.random.uniform(-0.2, 0.2,
                    (self.vocab_size + 2, self.embedding_dim))
            elif  self.unk_handle_method == 'zero':
                self.embedding_matrix = np.zeros((self.vocab_size + 2, self.embedding_dim))


        # We add 1 for the padding word
        logger.info("Embedding Matrix for Embedding Layer has shape %s " %
                    str(self.embedding_matrix.shape))

        n_non_embedding_words = 0
        for word, i in self.word2index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
            else:
                n_non_embedding_words += 1

        logger.info("There are %d words not in the embeddings. Setting them to zero" %
                    n_non_embedding_words)

        if self.keep_full_embedding:
            logger.info(
                       "Adding additional dimensions from the embedding file to embedding matrix")
            i = self.vocab_size
            extra_embeddings = []
            for word in embeddings_index.keys():
                if word not in self.word2index:
                    # Stack the new word's vector and index it
                    extra_embeddings.append(embeddings_index.get(word))
                    # We also need to keep an additional indexing of these
                    # words
                    self.additional_word2index[word] = i
                    i += 1

            if self.unk_handle_method == 'random':
                unk_embedding_row = np.random.uniform(-0.2, 0.2, (1, self.embedding_dim))
            elif self.unk_handle_method == 'zero':
                unk_embedding_row = np.zeros((1, self.embedding_dim))

            pad_embedding_row = np.random.uniform(-0.2, 0.2, (1, self.embedding_dim))

            self.embedding_matrix = np.vstack(
                [self.embedding_matrix, np.array(extra_embeddings),
                 pad_embedding_row, unk_embedding_row])

            # Last word is kept as the pad word
            # Here that is the last word in the embedding matrix
            self.pad_word_index = i
            self.unk_word_index = i + 1

        else:
            self.pad_word_index = self.vocab_size
            self.unk_word_index = self.vocab_size + 1

        if self.normalize_embeddings:
            logger.info("Normalizing the word embeddings")
            self.embedding_matrix = normalize(self.embedding_matrix)

        logger.info("Embedding Matrix now has shape %s" %
                    str(self.embedding_matrix.shape))
        logger.info("Pad word has been set to index %d" % self.pad_word_index)
        logger.info("Embedding index build complete")

    def preprocess(self, sentence):
        """Preprocess an input string to allow only alphabets and numbers

        Parameters:
        ----------
        sentence: str
            The sentence to preprocess
        """
        return re.sub("[^a-zA-Z0-9]", " ", sentence.lower())

    def make_indexed(self, sentence):
        """Returns the indexed version of the sentence based on self.word2index
        in the form of a list

        Parameters:
        -----------
        sentence str
            The sentence to be indexed

        Raises:
        -------
        ValueError: If the sentence has a lenght more than text_maxlen
        """
        indexed_sent = [self.word2index[word] for word in sentence]
        if len(indexed_sent)  > self.text_maxlen:
            raise ValueError("text_maxlen: %d isn't big enough. Error at sentence of length %d. Sentence is %s" % 
            (self.text_maxlen, len(sentence), sentence))

        indexed_sent = indexed_sent + [self.pad_word_index]*(self.text_maxlen - len(indexed_sent))
        return indexed_sent

    def get_full_batch(self):
        """Provides all the data points int the format: X1, X2, y with
        alternate positive and negative examples

        X1: the queries
            shape : (num_samples, text_maxlen)
        X2: the docs
            shape : (num_samples, text_maxlen)
        y: int {0, 1}
            The relation between X1[i] and X2[j]
            1 : X2[i] is relevant to X1[i]
            0 : X2[i] is not relevant to X1[i]
        """

        num_samples = len(self.indexed_pair_list)
        X1 = np.zeros((num_samples * 2, self.text_maxlen))
        X2 = np.zeros((num_samples * 2, self.text_maxlen))

        # To be uncommented when histogram support is included
        # if self.hist_size is not None:
        #     X2 = np.zeros((num_samples * 2, self.text_maxlen, self.hist_size))

        y = np.zeros((num_samples * 2, 1))

        X1[:] = self.pad_word_index
        X2[:] = self.pad_word_index
        y[::2] = 1

        for i, (query, pos_doc, neg_doc) in enumerate(self.indexed_pair_list):
            query_len = min(self.text_maxlen, len(query))
            pos_doc_len = min(self.text_maxlen, len(pos_doc))
            neg_doc_len = min(self.text_maxlen, len(neg_doc))


            X1[i * 2, :query_len] = query[:query_len]
            X2[i * 2, :pos_doc_len] = pos_doc[:pos_doc_len]
            X1[i * 2 + 1, :query_len] = query[:query_len]
            X2[i * 2 + 1, :neg_doc_len] = neg_doc[:neg_doc_len]

        return X1, X2, y

    def get_pair_list(self):
        """Returns a list with query document pairs in the format
        (query, positive_doc, negative_doc)

        Example output:
        -------------
        [(q1, d+, d-), (q2, d+, d-), (q3, d+, d-), ..., (qn, d+, d-)]

             where each query or document is a list of ints
    
        Example:
        -------
        [(['When', 'was', 'Abraham', 'Lincoln', 'born', '?'],
          ['He', 'was', 'born', 'in', '1809'],
          ['Abraham', 'Lincoln', 'was', 'the', 'president',
           'of', 'the', 'United', 'States', 'of', 'America']),

         (['When', 'was', 'the', 'first', 'World', 'War', '?'],
          ['It', 'was', 'fought', 'in', '1914'],
          ['There', 'were', 'over', 'a', 'million', 'deaths']),

         (['When', 'was', 'the', 'first', 'World', 'War', '?'],
          ['It', 'was', 'fought', 'in', '1914'],
          ['The', 'first', 'world', 'war', 'was', 'bad'])
        ]

        """
        pair_list = []
        for q, doc, label in zip(self.queries, self.docs, self.labels):
            doc, label = (list(t) for t in zip(*sorted(zip(doc, label), reverse=True)))
            for item in zip(doc, label):
                if item[1] == 1:
                    for new_item in zip(doc, label):
                        if new_item[1] == 0:
                            pair_list.append((q, item[0], new_item[0]))
        return pair_list

    def make_indexed_pair_list(self): 
        """Converts the existing word based pair list into an indexed format

        Note: pair_list needs to be first created using get_pair_list"""
        indexed_pair_list = []
        for q, d_pos, d_neg in self.pair_list:
            indexed_pair_list.append([self.make_indexed(q),
                self.make_indexed(d_pos), self.make_indexed(d_neg)])
        return indexed_pair_list

    def train_model(self):
        """Trains a DRMM_TKS model using specified parameters"""
        X1_train, X2_train, y_train = self.get_full_batch()
        drmm_tks = DRMM_TKS(
                    embedding=self.embedding_matrix, vocab_size=self.embedding_matrix.shape[0],
                    text_maxlen=self.text_maxlen)

        self.model = drmm_tks.get_model()
        self.model.summary()

        optimizer = 'adam'
        optimizer = optimizers.get(optimizer)
        K.set_value(optimizer.lr, 0.0001)

        # either one can be selected. Currently, the choice is manual.
        loss = rank_hinge_loss
        loss = hinge
        loss = 'mse'

        val_callback = None
        if self.validation_data is not None:
            test_queries, test_docs, test_labels = self.validation_data
            doc_lens = []
            long_doc_list = []
            long_test_labels = []

            for label, doc in zip(test_labels, test_docs):
                for l, d in zip(label, doc):
                    long_doc_list.append(d)
                    long_test_labels.append(l)
                doc_lens.append(len(doc))

            long_queries = []
            for doc_len, q in zip(doc_lens, test_queries):
                for i in range(doc_len):
                    long_queries.append(q)
            

            indexed_long_queries = self.translate_user_data(long_queries)
            indexed_long_doc_list = self.translate_user_data(long_doc_list)   

            val_callback = ValidationCallback({"X1": indexed_long_queries,
                "X2": indexed_long_doc_list, "doc_lengths": doc_lens, "y": long_test_labels})


        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        self.model.fit(x={"query": X1_train, "doc": X2_train}, y=y_train, batch_size=5,
                  verbose=1, epochs=self.epochs, shuffle=True, callbacks=[val_callback])

    def translate_user_data(self, data):
        """Translates given user data (as a list of words) into an indexed
        format which the model understands

        Parameters:
        ----------
        data: list of list of string words
            The data to be tranlsated
            Example:
                data = [["Hello World".split(),
                        "Translate this sentence".split()]
                       ]
                should return something like:
                [[12, 54],
                 [65, 23, 21]"""
        translated_data = []
        for sentence in data:
            translated_sentence = []
            for word in sentence:
                try:
                    translated_sentence.append(self.word2index[word])
                except KeyError:
                    # If the key isn't there give it the zero word index
                    translated_sentence.append(self.unk_word_index)
            if len(sentence) > self.text_maxlen:
                logger.info("text_maxlen: %d isn't big enough. Error at sentence of length %d. Sentence is %s" % (self.text_maxlen, len(sentence), str(sentence)))
            translated_sentence = translated_sentence + (self.text_maxlen - len(sentence))*[self.pad_word_index]
            translated_data.append(np.array(translated_sentence))
        return np.array(translated_data)

    def predict(self, queries, docs):
        """Predcits on the input paramters using the trained model

        Parameters:
        -----------
        queries: list of list of string words
            The questions for the similarity learning model
            
            Example:
            queries=["When was World Wat 1 fought ?".split(),
                     "When was Gandhi born ?".split()],

        docs: list of list of list of string words
            The candidate answers for the similarity learning model

            Example:
            docs = [
                    ["The world war was bad".split(),
                    "It was fought in 1996".split()],
                    ["Gandhi was born in the 18th century".split(),
                     "He fought for the Indian freedom movement".split(),
                     "Gandhi was assasinated".split()]
                   ]
        """
        doc_lens = []
        long_doc_list = []
        for doc in docs:
            long_doc_list.append(doc)
            doc_lens.append(len(doc))

        long_queries = []
        for doc_len, q in zip(doc_lens, queries):
            for i in range(len(docs)):
                long_queries.append(q)

        indexed_long_queries = self.translate_user_data(long_queries)
        indexed_long_doc_list = self.translate_user_data(long_doc_list)
        print(self.model.predict(x={'query': indexed_long_queries, 'doc': indexed_long_doc_list}))
