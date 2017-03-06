# Copyright (C) 2017 Parul Sethi <parul1sethi@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import tensorflow as tf
from tensorflow.models.embedding.word2vec_optimized import Word2Vec as tfw2v
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.word2vec import Vocab
from gensim import utils
from six import string_types
import os
import tempfile
import logging
logger = logging.getLogger(__name__)


class Options(object):
    """Options class that doesn't use FLAGS"""

    def __init__(self, train_data=None, save_path=None, eval_data=None,
        embedding_size=200, epochs_to_train=15, learning_rate=0.025,
        num_neg_samples=25, batch_size=500, concurrent_steps=12,
        window_size=5, min_count=5, subsample=1e-3):
        """
        train_data: Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.

        save_path: Directory to write the model.

        eval_data: Analogy questions.

        embedding_size: The embedding dimension size.

        epochs_to_train: Number of epochs to train. Each epoch processes the training data once

        learning_rate: Initial learning rate

        batch_size: Numbers of training examples each step processes

        concurrent_steps: The number of concurrent training steps.

        window_size: The number of worlds to predict to the left and right of the target word.

        min_count: The minimum number of word occurrences for it to be included in the vocabulary.

        subsample: Subsample threshold for word occurrence. Words that appear with higher frequency
                   will be randomly down-sampled. Set to 0 to disable.
        """

        if train_data is None:
            raise ValueError("train_data must be specified.")

        # Model options.

        # Embedding dimension.
        self.emb_dim = embedding_size

        # Training options.

        # The training text file.
        self.train_data = train_data

        # Number of negative samples per example.
        self.num_samples = num_neg_samples

        # The initial learning rate.
        self.learning_rate = learning_rate

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.epochs_to_train = epochs_to_train

        # Concurrent training steps.
        self.concurrent_steps = concurrent_steps

        # Number of examples for one training step.
        self.batch_size = batch_size

        # The number of words to predict to the left and right of the target word.
        self.window_size = window_size

        # The minimum number of word occurrences for it to be included in the
        # vocabulary.
        self.min_count = min_count

        # Subsampling threshold for word occurrence.
        self.subsample = subsample

        # Where to write out summaries.
        self.save_path = save_path

        # Eval options.

        # The text file for eval.
        self.eval_data = eval_data


def modified_tfw2v_init(self, options, session):
    self._options = options
    self._session = session
    self._word2id = {}
    self._id2word = []
    self.build_graph()
    self.build_eval_graph()
    if options.save_path is not None:
        self.save_vocab()
    if options.eval_data is not None:
        self.read_analogies()


tfw2v.__init__ = modified_tfw2v_init


class TfWord2Vec(KeyedVectors):

    def __init__(self, train_data=None, save_path=None, eval_data=None,
        embedding_size=200, epochs_to_train=15, learning_rate=0.025,
        num_neg_samples=25, batch_size=500, concurrent_steps=12,
        window_size=5, min_count=5, subsample=1e-3, gpus=1):

        self.options = Options(train_data, save_path=save_path, eval_data=eval_data,
            embedding_size=embedding_size, epochs_to_train=epochs_to_train,
            learning_rate=learning_rate, num_neg_samples=num_neg_samples,
            batch_size=batch_size, concurrent_steps=concurrent_steps,
            window_size=window_size, min_count=min_count, subsample=subsample)

        self.convert_input(train_data)
        self.gpus = gpus
        self.train()
        self.vocab = {}
        self.create_vocab()

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'syn0_all_norm'])
        super(TfWord2Vec, self).save(*args, **kwargs)

    def train(self):

        with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as session:
            gpu_devices = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
            for i in range(self.gpus):
                with tf.device(gpu_devices[i]):
                    self.model = tfw2v(self.options, session)
            for _ in range(self.options.epochs_to_train):
                self.model.train()  # Process one epoch
                if self.options.eval_data is not None:
                    self.model.eval()  # Eval analogies.

            self.syn0 = self.model._w_in
            self.syn0norm = session.run(tf.nn.l2_normalize(self.model._w_in, 1))
            self.index2word = self.model._id2word

    @classmethod
    def load_tf_model(cls, model_file):
        glove2word2vec(model_file, model_file+'.w2vformat')
        model = KeyedVectors.load_word2vec_format('%s.w2vformat' % model_file)
        return model

    def create_vocab(self):
        for word in self.options.vocab_words:
            self.vocab[word] = Vocab(index=self.model._word2id[word])

    def convert_input(self, corpus):
        """
        Converts gensim corpus to a file that can be used by tf word2vec

        """
        # assumes that the string represents a file extension
        if not isinstance(corpus, string_types):
            fout = tempfile.TemporaryFile()
            for sentences in corpus:
                fout.write(u' '.join(sentences))
                fout.write("\n")
            fout.seek(0)
            self.options.train_data = fout

    def __getitem__(self, words):
        if isinstance(words, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.syn0norm[self.model._word2id[words]]

        ids = [self.model._word2id[word] for word in words]
        return [self.syn0norm[id] for id in ids]
