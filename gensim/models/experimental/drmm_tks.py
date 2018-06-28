#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Aneesh Joshi <aneeshyjoshi@gmail.com>
# Copyright (C) 2018 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Makes a trainable and usable model for getting similarity between documents using the DRMM_TKS model which
is a variant of DRMM model

Once the model is trained with the query-candidate-relevance data, the model can provide a vector for each new
document which is entered into it. The similarity between any 2 documents can then be measured using the
cosine similarty between the vectors.

Abbreviations:
DRMM : Deep Relevance Matching Model
TKS : Top K Solutions

About DRMM_TKS
--------------
This is a variant version of DRMM, which applied topk pooling in the matching matrix.
It has the following steps:
1. embed queries and docs into embedding vector named 'q_embed' and 'd_embed' respectively
2. computing 'q_embed' and 'd_embed' with element-wise multiplication
3. computing output of upper layer with dense layer operation
4. take softmax operation on the output of this layer named 'g' and find the k largest entries named 'mm_k'.
5. input 'mm_k' into hidden layers, with specified length of layers and activation function
6. compute 'g' and 'mm_k' with element-wise multiplication.

On predicting, the model returns the score list between queries and documents.


Initialize a model with e.g.::

    >>> model = DRMM_TKS(queries, docs, labels, word_embedding=word_embedding_path)

Train the model with e.g.::

    >>> model.train(epochs=12)

Persist a model to disk with::

    >>> model.save(fname)
    >>> model = DRMM_TKS.load(fname)

The trained model can predict on new data like e.g.::
  >>> queries = ["how are glacier caves formed ?".lower().split()]
  >>> docs = ["A partly submerged glacier cave on Perito Moreno Glacier".lower().split(),
              "A glacier cave is a cave formed within the ice of a glacier".lower().split()]
  >>> drmm_tks_model.predict(queries, docs)
  [[0.5416601]
   [0.6190841]]

.. [1] Jiafeng Guo, Yixing Fan, Qingyao Ai, W. Bruce Croft
       A Deep Relevance Matching Model for Ad-hoc Retrieval
       http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf
.. [2] MatchZoo Repository
       https://github.com/faneshion/MatchZoo
.. [3] Similarity Learning
       https://en.wikipedia.org/wiki/Similarity_learning

"""

import logging
import numpy as np
import hashlib
from numpy import random as np_random
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from collections import Counter
from custom_losses import rank_hinge_loss
from custom_layers import TopKLayer
from sklearn.preprocessing import normalize
from custom_callbacks import ValidationCallback
from gensim import utils
from collections import Iterable
import random as rn
try:
    import keras.backend as K
    from keras import optimizers
    from keras.losses import hinge
    from keras.models import Model
    from keras.layers import Input, Embedding, Dot, Dense, Reshape, Dropout
    import tensorflow as tf
    import os
    # For understanding why random seeding has been done as below, refer to
    # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(1234)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DRMM_TKS(utils.SaveLoad):
    """Model for training a Similarity Learning Model using the DRMM TKS model.
    You only have to provide sentences in the data as a list of words.

    Examples
    --------
    >>> queries = ["When was World Wat 1 fought ?".lower().split(),
             "When was Gandhi born ?".lower().split()]

    >>> docs = [
            ["The world war was bad".lower().split(),
            "It was fought in 1996".lower().split()],
            ["Gandhi was born in the 18th century".lower().split(),
             "He fought for the Indian freedom movement".lower().split(),
             "Gandhi was assasinated".lower().split()]
           ]

    >>> labels = [[0, 1],
                 [1, 0, 0]]

    >>> drmm_tks_model = DRMM_TKS_Model(queries, docs, labels, word_embedding)
    >>> drmm_tks_model.predict(["What is AWS ?".lower().split()],
                               ["AWS is costly .".lower().split(), "It stands for Amazon Web Services".lower().split()])
    """

    def __init__(self, queries=None, docs=None, labels=None, word_embedding=None,
                 text_maxlen=200, normalize_embeddings=True, epochs=10, unk_handle_method='zero',
                 validation_data=None, topk=50, target_mode='ranking'):
        """Initializes the model and trains it

        Parameters
        ----------
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
        word_embedding : str or :class:`~gensim.models.keyedvectors.KeyedVectors`
            path to the Glove vectors which have the embeddings in a .txt format OR
            a KeyedVector object which has the embeddings pre-loaded
            If unset, random word embeddings will be used
        text_maxlen : int
            The maximum possible length of a query or a document
            This is used for padding sentences.
        normalize_embeddings : bool
            Whether the word embeddings provided should be normalized
        epochs : int
            The number of epochs for which the model should train on the data
        unk_handle_method : {'zero', 'random'}
            The method for handling unkown words
            'zero' : unknown words are given a zero vector
            'random' : unknown words are given a uniformly random vector bassed on the word string hash.
        validation_data: list of the form [test_queries, test_docs, test_labels]
            where test_queries, test_docs  and test_labels are of the same form as
            their counter parts stated above.
        topk : int
            the k topmost values in the interaction matrix between the queries and the docs
        target_mode : {'ranking', 'classification'}
            the way the model should be trained, either to rank or classify
        """
        self.queries = queries
        self.docs = docs
        self.labels = labels
        self.word_counter = Counter()
        self.text_maxlen = text_maxlen
        self.topk = topk
        self.word_embedding = word_embedding
        self.word2index, self.index2word = {}, {}
        self.normalize_embeddings = normalize_embeddings
        self.model = None
        self.epochs = epochs
        self.validation_data = validation_data
        self.target_mode = target_mode

        if self.target_mode not in ['ranking', 'classification']:
            raise ValueError("Unkown target_mode %s. It must be either 'ranking' or 'classification'" %
                             self.target_mode)

        if unk_handle_method not in ['random', 'zero']:
            raise ValueError("Unkown token handling method %s" %
                             str(unk_handle_method))
        self.unk_handle_method = unk_handle_method

        if self.queries is not None and self.docs is not None and self.labels is not None:
            self.build_vocab()
            self.train()
        else:
            logger.info("Vocab won't be built and Model won't be trained"
                        " as data is either not provided or is incomplete.")

    def build_vocab(self):
        """Indexes all the words and makes an embedding_matrix which
        can be fed directly into an Embedding layer"""

        logger.info("Starting Vocab Build")

        # get all the vocab words
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
        if type(self.word_embedding) == str:
            # Use KeyedVectors for easy and quick access of word embeddings
            glove_file = self.word_embedding
            tmp_file = get_tmpfile("tmp_word2vec.txt")
            embedding_vocab_size, self.embedding_dim = glove2word2vec(
                glove_file, tmp_file)
            kv_model = KeyedVectors.load_word2vec_format(tmp_file)
        elif type(self.word_embedding) == KeyedVectors:
            kv_model = self.word_embedding
            embedding_vocab_size, self.embedding_dim = len(
                kv_model.vocab), kv_model.vector_size
        else:
            raise ValueError("Unknown value of word_embedding : %s."
                             "Must be either a string path to Glove Embedding file or a KeyedVector"
                             )

        logger.info("The embeddings_index built from the given file has %d words of %d dimensions" %
                    (embedding_vocab_size, self.embedding_dim))

        logger.info(
            "Building the Embedding Matrix for the model's Embedding Layer")

        # Initialize the embedding matrix
        # UNK word gets the vector based on the method
        if self.unk_handle_method == 'random':
            self.embedding_matrix = np.random.uniform(-0.2, 0.2,
                                                      (self.vocab_size, self.embedding_dim))
        elif self.unk_handle_method == 'zero':
            self.embedding_matrix = np.zeros(
                (self.vocab_size, self.embedding_dim))

        n_non_embedding_words = 0
        for word, i in self.word2index.items():
            if word in kv_model:
                # words not found in keyed vectors will get the vector based on unk_handle_method
                self.embedding_matrix[i] = kv_model[word]
            else:
                if self.unk_handle_method == 'random':
                    # Creates the same random vector for the given string each time
                    self.embedding_matrix[i] = self._seeded_vector(
                        word, self.embedding_dim)
                n_non_embedding_words += 1
        logger.info("There are %d words out of %d (%.2f%%) not in the embeddings. Setting them to %s" %
                    (n_non_embedding_words, self.vocab_size, n_non_embedding_words * 100 / self.vocab_size,
                     self.unk_handle_method))

        # Include embeddings for words in embedding file but not in the train vocab
        # It will be useful for embedding words encountered in validation and test set
        logger.info(
            "Adding additional words from the embedding file to embedding matrix"
        )

        # The point where vocab words end
        vocab_offset = self.vocab_size
        extra_embeddings = []
        # Take the words in the embedding file which aren't there int the train vocab
        for word in list(kv_model.vocab):
            if word not in self.word2index:
                # Add the new word's vector and index it
                extra_embeddings.append(kv_model[word])
                # We also need to keep an additional indexing of these
                # words
                self.word2index[word] = vocab_offset
                vocab_offset += 1

        # Set the pad and unk word to second last and last index
        self.pad_word_index = vocab_offset
        self.unk_word_index = vocab_offset + 1

        if self.unk_handle_method == 'random':
            unk_embedding_row = np.random.uniform(
                -0.2, 0.2, (1, self.embedding_dim))
        elif self.unk_handle_method == 'zero':
            unk_embedding_row = np.zeros((1, self.embedding_dim))

        pad_embedding_row = np.random.uniform(-0.2,
                                              0.2, (1, self.embedding_dim))

        self.embedding_matrix = np.vstack(
            [self.embedding_matrix, np.array(extra_embeddings),
             pad_embedding_row, unk_embedding_row]
        )

        if self.normalize_embeddings:
            logger.info("Normalizing the word embeddings")
            self.embedding_matrix = normalize(self.embedding_matrix)

        logger.info("Embedding Matrix build complete. It now has shape %s" %
                    str(self.embedding_matrix.shape))
        logger.info("Pad word has been set to index %d" % self.pad_word_index)
        logger.info("Unknown word has been set to index %d" %
                    self.unk_word_index)
        logger.info("Embedding index build complete")

    def _string2numeric_hash(text):
        "Gets a numeric hash for a given string"
        return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)

    def _seeded_vector(self, seed_string, vector_size):
        """Create one 'random' vector (but deterministic by seed_string)"""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = np_random.RandomState(
            self._string2numeric_hash(seed_string) & 0xffffffff)
        return (once.rand(vector_size) - 0.5) / vector_size

    def _make_indexed(self, sentence):
        """Gets the indexed version of the sentence based on the self.word2index dict
        in the form of a list

        Parameters
        ----------
        sentence : iterable list of list of str
            The sentence to be indexed

        Raises
        ------
        ValueError : If the sentence has a lenght more than text_maxlen
        """

        indexed_sent = [self.word2index[word] for word in sentence]
        if len(indexed_sent) > self.text_maxlen:
            raise ValueError(
                "text_maxlen: %d isn't big enough. Error at sentence of length %d."
                "Sentence is %s" % (self.text_maxlen, len(sentence), sentence)
            )
        indexed_sent = indexed_sent + \
            [self.pad_word_index] * (self.text_maxlen - len(indexed_sent))
        return indexed_sent

    def _get_full_batch(self):
        """Provides all the data points int the format: X1, X2, y with
        alternate positive and negative examples

        Returns
        -------
        X1 : numpy array of shape (num_samples, text_maxlen)
            the queries
        X2 : numpy array of shape (num_samples, text_maxlen)
            the docs
        y : numpy array with {0, 1} of shape (num_samples,)
            The relation between X1[i] and X2[j]
            1 : X2[i] is relevant to X1[i]
            0 : X2[i] is not relevant to X1[i]
        """
        X1, X2, y = [], [], []
        for i, (query, pos_doc, neg_doc) in enumerate(self.pair_list):
            X1.append(query)
            X2.append(pos_doc)
            y.append(1)
            X1.append(query)
            X2.append(neg_doc)
            y.append(0)
        return np.array(X1), np.array(X2), np.array(y)

    def _get_full_batch_iter(self, batch_size):
        """Provides all the data points int the format: X1, X2, y with
        alternate positive and negative examples of `batch_size` in a streamable format.

        Yields
        -------
        X1 : numpy array of shape (batch_size * 2, text_maxlen)
            the queries
        X2 : numpy array of shape (batch_size * 2, text_maxlen)
            the docs
        y : numpy array with {0, 1} of shape (batch_size * 2, 1)
            The relation between X1[i] and X2[j]
            1 : X2[i] is relevant to X1[i]
            0 : X2[i] is not relevant to X1[i]
        """

        X1, X2, y = [], [], []
        while True:
            for i, (query, pos_doc, neg_doc) in enumerate(self.pair_list):
                X1.append(query)
                X2.append(pos_doc)
                y.append(1)
                X1.append(query)
                X2.append(neg_doc)
                y.append(0)
                if i % batch_size == 0 and i != 0:
                    yield ({'query': np.array(X1), 'doc': np.array(X2)}, np.array(y))
                    X1, X2, y = [], [], []

    def _get_pair_list(self):
        """Yields a tuple with query document pairs in the format
        (query, positive_doc, negative_doc)

        [(q1, d+, d-), (q2, d+, d-), (q3, d+, d-), ..., (qn, d+, d-)]
            where each query or document is a list of ints

        Example
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

        for q, doc, label in zip(self.queries, self.docs, self.labels):
            doc, label = (list(t)
                          for t in zip(*sorted(zip(doc, label), reverse=True)))
            for item in zip(doc, label):
                if item[1] == 1:
                    for new_item in zip(doc, label):
                        if new_item[1] == 0:
                            yield(self._make_indexed(q), self._make_indexed(item[0]), self._make_indexed(new_item[0]))

    def train(self, queries=None, docs=None, labels=None, word_embedding=None,
              text_maxlen=None, normalize_embeddings=None, epochs=None, unk_handle_method=None,
              validation_data=None, topk=None, target_mode=None):
        """Trains a DRMM_TKS model using specified parameters"""

        # In case the user wants to initialize and train the model in different phases
        self.queries = queries or self.queries
        self.docs = docs or self.docs
        self.labels = labels or self.labels
        # TODO this won't update anything!
        self.word_embedding = word_embedding or self.word_embedding
        self.text_maxlen = text_maxlen or self.text_maxlen
        self.normalize_embeddings = normalize_embeddings or self.normalize_embeddings
        self.epochs = epochs or self.epochs
        self.unk_handle_method = unk_handle_method or self.unk_handle_method
        self.validation_data = validation_data or self.validation_data
        self.topk = topk or self.topk
        self.target_mode = target_mode or self.target_mode

        if self.queries is None and self.docs is None and self.labels is None:
            raise ValueError("queries, docs and labels have to be specified")
        # We need to build these each time since any of the parameters can change from each train to trian

        # TODO A check for queries, docs and labels
        is_iterable = False
        if isinstance(self.queries, Iterable) and not isinstance(self.queries, list):
            is_iterable = True

        self.pair_list = self._get_pair_list()

        if is_iterable:
            train_generator = self._get_full_batch_iter(32)
        else:
            X1_train, X2_train, y_train = self._get_full_batch()

        self.model = self._get_keras_model()
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

            indexed_long_queries = self._translate_user_data(long_queries)
            indexed_long_doc_list = self._translate_user_data(long_doc_list)

            val_callback = ValidationCallback({"X1": indexed_long_queries, "X2": indexed_long_doc_list,
                                               "doc_lengths": doc_lens, "y": long_test_labels})

        self.model.compile(optimizer=optimizer, loss=loss,
                           metrics=['accuracy'])
        if is_iterable:
            self.model.fit_generator(train_generator, steps_per_epoch=128, callbacks=[val_callback], epochs=self.epochs)
        else:
            self.model.fit(x={"query": X1_train, "doc": X2_train}, y=y_train, batch_size=5,
                           verbose=1, epochs=self.epochs, shuffle=True, callbacks=[val_callback])

    def _translate_user_data(self, data):
        """Translates given user data into an indexed format which the model understands.

        Parameters
        ----------
        data : list of list of string words
            The data to be tranlsated

        Example
        -------
        >>> data = [["Hello World".split(),
                     "Translate this sentence".split()]
                    ]
        >>> _translate_user_data(data)
        [[12, 54],
         [65, 23, 21]

        """

        translated_data = []
        n_skipped_words = 0
        for sentence in data:
            translated_sentence = []
            for word in sentence:
                if word in self.word2index:
                    translated_sentence.append(self.word2index[word])
                else:
                    # If the key isn't there give it the zero word index
                    translated_sentence.append(self.unk_word_index)
                    n_skipped_words += 1
            if len(sentence) > self.text_maxlen:
                logger.info("text_maxlen: %d isn't big enough. Error at sentence of length %d."
                            "Sentence is %s" % (
                                self.text_maxlen, len(sentence), str(sentence))
                            )
            translated_sentence = translated_sentence + \
                (self.text_maxlen - len(sentence)) * [self.pad_word_index]
            translated_data.append(np.array(translated_sentence))

        logger.info("Found %d unknown words. Set them to unknown word index : %d" %
                    (n_skipped_words, self.unk_word_index))
        return np.array(translated_data)

    def predict(self, queries, docs):
        """Predcits the similarity between a query-document pair
        based on the trained DRMM TKS model

        Parameters
        ----------
        queries : list of list of str
            The questions for the similarity learning model
            Example :
            queries=["When was World Wat 1 fought ?".split(),
                     "When was Gandhi born ?".split()]
        docs : list of list of list of str
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

        indexed_long_queries = self._translate_user_data(long_queries)
        indexed_long_doc_list = self._translate_user_data(long_doc_list)
        return self.model.predict(
            x={'query': indexed_long_queries, 'doc': indexed_long_doc_list})

    def save(self, fname, *args, **kwargs):
        """Save the model.
        This saved model can be loaded again using :func:`~gensim.models.experimental.drmm_tks.DRMM_TKS.load`
        The keras model shouldn't be serialized using pickle or cPickle. So, the non-keras
        variables will be saved using gensim's SaveLoad and the keras model will be saved using
        the keras save method with ".keras" prefix.

        Also see :func:`~gensim.models.experimental.drmm_tks.DRMM_TKS.load`

        Parameters
        ----------
        fname : str
            Path to the file.

        """
        # don't save the keras model as it needs to be saved with a keras function
        kwargs['ignore'] = kwargs.get('ignore', ['model'])
        kwargs['fname_or_handle'] = fname
        super(DRMM_TKS, self).save(*args, **kwargs)
        self.model.save(fname + ".keras")

    @classmethod
    def load(cls, *args, **kwargs):
        """Loads a previously saved `DRMM TKS` model. Also see `save()`.
        Collects the gensim and the keras models and returns it as on gensim model.

        Parameters
        ----------
        fname : str
            Path to the saved file.

        Returns
        -------
        :obj: `~gensim.models.experimental.DRMM_TKS`
            Returns the loaded model as an instance of :class: `~gensim.models.experimental.DRMM_TKS`.
        """
        from keras.models import load_model
        fname = args[0]
        gensim_model = super(DRMM_TKS, cls).load(*args, **kwargs)
        keras_model = load_model(
            fname + '.keras', custom_objects={'TopKLayer': TopKLayer})
        gensim_model.model = keras_model
        return gensim_model

    def _get_keras_model(self, embed_trainable=False, dropout_rate=0.5, hidden_sizes=[100, 1]):
        """Builds and returns the keras class for drmm tks model

        About DRMM_TKS
        --------------
        This is a variant version of DRMM, which applied topk pooling in the matching matrix.
        It has the following steps:
        1. embed queries into embedding vector named 'q_embed' and 'd_embed' respectively
        2. computing 'q_embed' and 'd_embed' with element-wise multiplication
        3. computing output of upper layer with dense layer operation
        4. take softmax operation on the output of this layer named 'g' and find the k largest entries named 'mm_k'.
        5. input 'mm_k' into hidden layers, with specified length of layers and activation function
        6. compute 'g' and 'mm_k' with element-wise multiplication.

        On predicting, the model returns the score list between queries and documents.

        Parameters
        ----------
        embed_trainable : bool
            Whether the embeddings should be trained
            if True, the embeddings are trianed
        dropout_rate : float between 0 and 1
            The probability of making a neuron dead
            Used for regularization.
        hidden_sizes : list of ints
            The list of hidden sizes for the fully connected layers connected to the matching matrix
            Example :
                hidden_sizes = [10, 20, 30]
            will add 3 fully connected layers of 10, 20 and 30 hidden neurons

        """

        if not KERAS_AVAILABLE:
            raise ImportError("Please install Keras to use this model")

        n_layers = len(hidden_sizes)

        query = Input(name='query', shape=(self.text_maxlen,))
        doc = Input(name='doc', shape=(self.text_maxlen,))
        embedding = Embedding(self.embedding_matrix.shape[0], self.embedding_dim,
                              weights=[self.embedding_matrix], trainable=embed_trainable)

        q_embed = embedding(query)
        d_embed = embedding(doc)

        mm = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])

        # compute term gating
        w_g = Dense(1, activation='softmax')(q_embed)
        g = Reshape((self.text_maxlen,))(w_g)

        mm_k = TopKLayer(topk=self.topk, output_dim=(
            self.text_maxlen, self.embedding_dim))(mm)

        for i in range(n_layers):
            mm_k = Dense(hidden_sizes[i], activation='softplus', kernel_initializer='he_uniform',
                         bias_initializer='zeros')(mm_k)

        mm_k_dropout = Dropout(rate=dropout_rate)(mm_k)

        mm_reshape = Reshape(
            (self.text_maxlen,))(mm_k_dropout)

        mean = Dot(axes=[1, 1], normalize=True)([mm_reshape, g])

        if self.target_mode == 'classification':
            out_ = Dense(2, activation='softmax')(mean)
        elif self.target_mode in ['regression', 'ranking']:
            out_ = Reshape((1,))(mean)

        model = Model(inputs=[query, doc], outputs=out_)
        return model
