# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import logging
logger = logging.getLogger(__name__)


class TfWord2Vec(KeyedVectors):

    def __init__(self, train_data=None, eval_data=None, save_path=None, size=100, alpha=0.025,
                 window=5, num_skips=2, min_count=5, sample=1e-3, negative=25,
                 batch_size=500, train_epochs=15, concurrent_steps=12, FLAGS=None):

        """Word2vec as TensorFlow graph.
            Args:
            train_data: Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.
            eval_data: Data for evaluation.
            save_path: Directory to write the model.
            size: The dimensionality of the feature vectors.
            alpha: The initial learning rate (will linearly drop to `min_alpha` as training progresses).
            window: The maximum distance between the current and predicted word within a sentence.
            num_skips: How many times to reuse an input to generate a label
            min_count: Ignore all words with total frequency lower than this.
            sample: Threshold for configuring which higher-frequency words are randomly downsampled;
                default is 1e-3, useful range is (0, 1e-5).
            negative: If > 0, negative sampling will be used, the int for negative
                specifies how many "noise words" should be drawn (usually between 5-20).
                Default is 5. If set to 0, no negative samping is used.
            batch_size: Numbers of training examples each step processes.
            train_epochs: Number of epochs to train, each epoch processes the training data once.
            concurrent_steps: The number of concurrent training steps.
        """
        if train_data is None:
            raise ValueError("Train_data must be specified")

        self.train_data = train_data
        self.eval_data = eval_data
        self.vector_size = int(size)
        self.sample = sample
        self.negative = negative
        self.alpha = float(alpha)
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.concurrent_steps = concurrent_steps
        self.window = window
        self.num_skips = num_skips
        self.min_count = min_count
        self.save_path = save_path
        self.FLAGS = FLAGS

        self.vocab_size = 5000

        self.build_dataset(train_data, self.vocab_size)
        self.data_index = 0
        self.train()

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'syn0_all_norm'])
        super(TfWord2Vec, self).save(*args, **kwargs)

    def build_dataset(self, words, n_words):
        """Build the dictionary and replace rare words with UNK token."""
        self.count = [['UNK', -1]]
        self.count.extend(collections.Counter(words).most_common(n_words - 1))
        self.dict = dict()
        for word, _ in self.count:
            self.dict[word] = len(self.dict)
        self.data = list()
        unk_count = 0
        for word in words:
            if word in self.dict:
                index = self.dict[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            self.data.append(index)
        self.count[0][1] = unk_count
        self.reversed_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def read_analogies(self):
        """Reads through the analogy question file.

        Returns:
          questions: a [n, 4] numpy array containing the analogy question's
                     word ids.
          questions_skipped: questions skipped due to unknown words.
        """
        questions = []
        questions_skipped = 0
        with open(self.eval_data, "r") as analogy_f:
            for line in analogy_f:
                if line.startswith(":"):  # Skip comments.
                    continue
                words = line.strip().lower().split(" ")
                ids = [self.dict.get(w.strip()) for w in words]
                if None in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))
        print("Eval analogy file: ", self.eval_data)
        print("Questions: ", len(questions))
        print("Skipped: ", questions_skipped)
        self.analogy_questions = np.array(questions, dtype=np.int32)

    def generate_batch(self, batch_size, num_skips, skip_window):
        """Function to generate a training batch for the skip-gram model."""
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels

    def train(self):
        ps_hosts = self.FLAGS.ps_hosts.split(',')
        worker_hosts = self.FLAGS.worker_hosts.split(',')
        cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

        server = tf.train.Server(cluster, job_name=self.FLAGS.job_name,
                                     task_index=self.FLAGS.task_index)
        if self.FLAGS.job_name == "ps":
            server.join()
        elif self.FLAGS.job_name == "worker":
            # Build graph
            with tf.device(tf.train.replica_device_setter(
                    worker_device='/job:worker/task:%d' % self.FLAGS.task_index,
                    cluster=cluster, ps_ops=['Variable', 'Placeholder'])):

                global_step = tf.contrib.framework.get_or_create_global_step()

                # Input data.
                with tf.name_scope('input'):
                    train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                    train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

                # Look up embeddings for inputs.
                with tf.name_scope('embeddings'):
                    embeddings = tf.Variable(
                        tf.random_uniform([self.vocab_size, self.vector_size],
                                                                    -1.0, 1.0))
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
                                                    num_sampled=self.negative,
                                                    num_classes=self.vocab_size))

                # Construct the SGD optimizer using a learning rate of 1.0.
                with tf.name_scope('train'):
                    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

                norm = tf.sqrt(
                    tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

                # Add variable initializer.
                init = tf.global_variables_initializer()

            # The supervisor takes care of session initialization, restoring from
            # a checkpoint, and closing when done or an error occurs.
            sv = tf.train.Supervisor(is_chief=(self.FLAGS.task_index == 0),
                                     global_step=global_step,
                                     init_op=init)

            average_loss = 0
            norm_vec = []
            embed_vec = []
            with sv.prepare_or_wait_for_session(server.target) as sess:
                for step in xrange(self.concurrent_steps):
                    batch_inputs, batch_labels = self.generate_batch(
                                                    batch_size=self.batch_size,
                                                    num_skips=self.num_skips,
                                                    skip_window=self.window)
                    feed_dict = {train_inputs: batch_inputs,
                                 train_labels: batch_labels}
                    _, loss_val, norm_vec, embed_vec = sess.run(
                        [optimizer, loss, norm, embeddings], feed_dict=feed_dict)
                    average_loss += loss_val

                    if step % 2000 == 0 and step > 0:
                        average_loss /= 2000
                        # The average loss is an estimate of the loss over the last
                        # 2000 batches.
                        logger.info('Average loss at step %d: %.5f', step, average_loss)
                        print('Average loss at step %d: %.5f', step, average_loss)
                        average_loss = 0

            self.syn0norm = norm_vec
            self.syn0 = embed_vec

    @classmethod
    def load_tf_model(cls, model_file):
        glove2word2vec(model_file, model_file + '.w2vformat')
        model = KeyedVectors.load_word2vec_format('%s.w2vformat' % model_file)
        return model

    def eval_graph(self, analogy):
        """Build the eval graph and predict the top 4 answers for analogy
        questions."""

        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.

        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to
        # evaluate in one batch.
        eval_graph = tf.Graph()
        with eval_graph.as_default():
            analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
            analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
            analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

            # Normalized word embeddings of shape [vocab_size, emb_dim].
            nemb = tf.nn.l2_normalize(self.syn0, 1)

            # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
            # They all have the shape [N, emb_dim]
            a_emb = tf.gather(nemb, analogy_a)  # a's embs
            b_emb = tf.gather(nemb, analogy_b)  # b's embs
            c_emb = tf.gather(nemb, analogy_c)  # c's embs

            # We expect that d's embedding vectors on the unit hyper-sphere is
            # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
            target = c_emb + (b_emb - a_emb)

            # Compute cosine distance between each pair of target and vocab.
            # dist has shape [N, vocab_size].
            dist = tf.matmul(target, nemb, transpose_b=True)

            # For each question (row in dist), find the top 4 words.
            _, pred_idx = tf.nn.top_k(dist, 4)

        with tf.Session(graph=eval_graph) as session:
            idx, = session.run([pred_idx], {analogy_a: analogy[:, 0],
                                            analogy_b: analogy[:, 1],
                                            analogy_c: analogy[:, 2]
                                            })
        return idx

    def eval(self):
        """Evaluate analogy questions and reports accuracy."""
        if self.eval_data is None:
            raise ValueError("Need to read analogy questions.")

        self.read_analogies()
        correct = 0
        total = self.analogy_questions.shape[0]
        start = 0
        while start < total:
            limit = start + 2500
            sub = self.analogy_questions[start:limit, :]
            idx = self.eval_graph(sub)
            start = limit
            for question in xrange(sub.shape[0]):
                for j in xrange(4):
                    if idx[question, j] == sub[question, 3]:
                        correct += 1
                    else:
                        continue

        print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                                  correct * 100.0 / total))
