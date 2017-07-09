# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.word2vec import Vocab
from gensim import utils
from six import string_types
import os
import tempfile
import logging
logger = logging.getLogger(__name__)


FLAGS = None

class TfWord2Vec(KeyedVectors):

    def __init__(self, train_data=None, save_path=None, size=100, alpha=0.025,
                 window=5, min_count=5, sample=1e-3, negative=25,
                 batch_size=500, train_epochs=15, concurrent_steps=12):

        """
        'train_data' is training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.
        
        'save_path' is directory to write the model.
        
        `size` is the dimensionality of the feature vectors.
        
        `alpha` is the initial learning rate (will linearly drop to `min_alpha` as training progresses).
        
        `window` is the maximum distance between the current and predicted word within a sentence.
        
        `min_count` = ignore all words with total frequency lower than this.
        
        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
            default is 1e-3, useful range is (0, 1e-5).
        
        `negative` = if > 0, negative sampling will be used, the int for negative
        specifies how many "noise words" should be drawn (usually between 5-20).
        Default is 5. If set to 0, no negative samping is used.
        
        'batch_size' is numbers of training examples each step processes.
        
        'train_epochs' is number of epochs to train, each epoch processes the training data once.
        
        'concurrent_steps' is the number of concurrent training steps.
        
        """
        if train_data is None:
            raise ValueError("Train_data must be specified")

        self.train_data = train_data
        self.vector_size = int(size)
        self.sample = sample
        self.negative = negative
        self.alpha = float(alpha)
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.concurrent_steps = concurrent_steps
        self.window = window
        self.min_count = min_count
        self.save_path = save_path

        self.vocab_size = 5000

        self.build_dataset(train_data, self.vocab_size)
        self.data_index = 0                                                     #TODO
        self.train()
        self.vocab = {}

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'syn0_all_norm'])
        super(TfWord2Vec, self).save(*args, **kwargs)

    def build_dataset(self, words, n_words):
        """
        Build the dictionary and replace rare words with UNK token.
        
        """
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(n_words - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(
            zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reversed_dictionary

    def generate_batch(self, batch_size, num_skips, skip_window):
        """
        Function to generate a training batch for the skip-gram model.
        
        """
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = (data_index + len(data) - span) % len(data)
        return batch, labels

    def train(self):
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

        server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                                     task_index=FLAGS.task_index)
        if FLAGS.job_name == "ps":
            server.join()
        elif FLAGS.job_name == "worker":
            #Build graph
            with tf.device(tf.train.replica_device_setter(
                    worker_device='/job:worker/task:%d' % FLAGS.task_index,
                    cluster=cluster, ps_ops = ['Variable', 'Placeholder'])):

                global_step = tf.contrib.framework.get_or_create_global_step()

                # Input data.
                with tf.name_scope('input'):
                    train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                    train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

                # Look up embeddings for inputs.
                with tf.name_scope('embeddings'):
                    embeddings = tf.Variable(
                        tf.random_uniform([self.vocab_size, self.vector_size], -1.0,
                                          1.0))
                    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                with tf.name_scope('nce_loss'):
                    nce_weights = tf.Variable(
                        tf.truncated_normal([self.vocab_size, self.vector_size],
                                            stddev=1.0 / math.sqrt(self.vector_size)))
                    nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

                    # Compute the average NCE loss for the batch.
                    # tf.nce_loss automatically draws a new sample of the
                    # negative labels each time we evaluate the loss.
                    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                    biases=nce_biases,
                                                    labels=train_labels,
                                                    inputs=embed,
                                                    num_sampled=self.sample,
                                                    num_classes=self.vocab_size))

                # Construct the SGD optimizer using a learning rate of 1.0.
                with tf.name_scope('train'):
                    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

                # Add variable initializer.
                init = tf.global_variables_initializer()

            # The supervisor takes care of session initialization, restoring from
            # a checkpoint, and closing when done or an error occurs.
            sv = tf.train.Supervisor(is_chief = (FLAGS.task_index == 0),
                                     global_step = global_step,
                                     init_op = init)
            average_loss = 0
            with sv.prepare_or_wait_for_session(server.target) as sess:
                for step in xrange(self.concurrent_steps):
                    batch_inputs, batch_labels = self.generate_batch(
                                            batch_size = self.batch_size,
                                            num_skips = self.min_count,
                                            skip_window = self.window)
                    feed_dict = {train_inputs: batch_inputs,
                                 train_labels: batch_labels}
                    _, loss_val = sess.run(optimizer, feed_dict = feed_dict)
                    average_loss += loss_val

                    if step % 2000 == 0 and step > 0:
                        average_loss /= 2000
                        # The average loss is an estimate of the loss over the last
                        # 2000 batches.
                        print('Average loss at step ', step, ': ', average_loss)
                        average_loss = 0

        self.syn0norm = loss_val

    @classmethod
    def load_tf_model(cls, model_file):
        glove2word2vec(model_file, model_file+'.w2vformat')
        model = KeyedVectors.load_word2vec_format('%s.w2vformat' % model_file)
        return model

    def __getitem__(self, words):
        if isinstance(words, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.syn0norm[self.model._word2id[words]]

        ids = [self.model._word2id[word] for word in words]
        return [self.syn0norm[id] for id in ids]
