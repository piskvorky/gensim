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
from gensim.utils import keep_vocab_item
import time


class TfWord2Vec(KeyedVectors):

    def __init__(self, train_data=None, eval_data=None, save_path=None, size=100,
                 window=5, num_skips=2, min_count=5, negative=25, alpha=0.025,
                 batch_size=500, train_epochs=1, FLAGS=None):

        """Word2vec as TensorFlow graph.
            Args:
            train_data: Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.
            eval_data: Data for evaluation.
            save_path: Directory to write the model.
            size: The dimensionality of the feature vectors.
            window: The maximum distance between the current and predicted word within a sentence.
            num_skips: How many times to reuse an input to generate a label
            min_count: Ignore all words with total frequency lower than this.
            negative: If > 0, negative sampling will be used, the int for negative
                specifies how many "noise words" should be drawn (usually between 5-20).
                Default is 5. If set to 0, no negative samping is used.
            batch_size: Numbers of training examples each step processes.
            alpha: learning rate
            train_epochs: Number of epochs to train, each epoch processes the training data once.
        """
        if train_data is None:
            raise ValueError("Train_data must be specified")

        self.train_data = train_data
        self.eval_data = eval_data
        self.vector_size = int(size)
        self.negative = negative
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.window = window
        self.num_skips = num_skips
        self.min_count = min_count
        self.save_path = save_path
        self.alpha = alpha
        self.FLAGS = FLAGS

        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        self.valid_size = 16     # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)

        self.build_dataset(train_data)
        ps_hosts = self.FLAGS.ps_hosts.split(',')
        worker_hosts = self.FLAGS.worker_hosts.split(',')

        # Create a cluster from the parameter server and worker hosts.
        self.cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
        self.num_workers = len(worker_hosts)
        self.data_size = len(self.data) // self.num_workers
        self.data_index = self.FLAGS.task_index * self.data_size
        self.max_index = self.data_index + self.data_size
        self.start_index = self.data_index

        self.train()

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'syn0_all_norm'])
        super(TfWord2Vec, self).save(*args, **kwargs)

    def build_dataset(self, words):
        """Build the dictionary"""
        vocab = collections.Counter()
        for word in words:
            vocab[word] += 1
        self.vocab = vocab.most_common()
        self.dict = dict()
        rare_words = 0
        for word, count in self.vocab:
            if keep_vocab_item(word, count, self.min_count):
                self.dict[word] = len(self.dict)
            else:
                rare_words += 1
        self.vocab_size = len(self.dict)
        self.data = list()
        for word in words:
            if word in self.dict:
                index = self.dict[word]
            self.data.append(index)
        self.reversed_dict = dict(zip(self.dict.values(), self.dict.keys()))
        print("Count of vocabulary: {}. Count of rare words: {}. Data contains {} words.".format(self.vocab_size, rare_words, len(words)))

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
        if self.data_index + (batch_size // num_skips) > self.max_index:
            self.max_index = -1
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
        self.data_index = self.data_index - span
        return batch, labels

    def train(self):
        # Create and start a server for the local task.
        server = tf.train.Server(self.cluster, job_name=self.FLAGS.job_name,
                                     task_index=self.FLAGS.task_index)

        if self.FLAGS.job_name == "ps":
            server.join()
        elif self.FLAGS.job_name == "worker":
            # Build graph
            with tf.device(tf.train.replica_device_setter(
                    worker_device='/job:worker/task:%d' % self.FLAGS.task_index,
                    cluster=self.cluster)):

                global_step = tf.contrib.framework.get_or_create_global_step()

                # Input data.
                train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

                valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

                # Look up embeddings for inputs.
                embeddings = tf.Variable(
                        tf.random_uniform([self.vocab_size, self.vector_size],
                                                                    -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
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

                # Construct the SGD optimizer using a learning rate.
                optimizer = tf.train.GradientDescentOptimizer(self.alpha).minimize(loss)

                norm = tf.sqrt(
                    tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

                normalized_embeddings = embeddings / norm
                valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
                similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

                # Add variable initializer.
                init = tf.global_variables_initializer()

            # The supervisor takes care of session initialization, restoring from
            # a checkpoint, and closing when done or an error occurs.
            sv = tf.train.Supervisor(is_chief=(self.FLAGS.task_index == 0),
                                     global_step=global_step,
                                     init_op=init)

            average_loss = 0
            with sv.prepare_or_wait_for_session(server.target, config=None) as sess:
                start_time = time.time()
                for epoch in xrange(self.train_epochs):
                    step = 0
                    while self.data_index < self.max_index:
                        batch_inputs, batch_labels = self.generate_batch(
                                                        batch_size=self.batch_size,
                                                        num_skips=self.num_skips,
                                                        skip_window=self.window)
                        feed_dict = {train_inputs: batch_inputs,
                                     train_labels: batch_labels}
                        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
                        average_loss += loss_val

                        step += 1
                        if step % 500 == 0 and step > 0:
                            average_loss /= 500
                            # The average loss is an estimate of the loss over the last batches.
                            print('Task: {}. Average loss at step {}: {}. Data index: {}. Time: {}'.format(
                                   self.FLAGS.task_index, step, average_loss, self.data_index, time.time() - start_time))
                            average_loss = 0

                    sim = similarity.eval()
                    for i in xrange(self.valid_size):
                        valid_word = self.reversed_dict[self.valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in xrange(top_k):
                            close_word = self.reversed_dict[nearest[k]]
                            log_str = '%s %s,' % (log_str, close_word)
                        print(log_str)

                    self.syn0norm = normalized_embeddings.eval()
                    self.syn0 = embeddings.eval()
                    self.data_index = self.start_index
                    self.max_index = self.data_index + self.data_size

    @classmethod
    def load_tf_model(cls, model_file):
        glove2word2vec(model_file, model_file + '.w2vformat')
        model = KeyedVectors.load_word2vec_format('%s.w2vformat' % model_file)
        return model

    def build_eval_graph(self):
        """Build the eval graph and predict the top 4 answers for analogy
        questions."""

        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.

        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to
        # evaluate in one batch.

        tf.reset_default_graph()
        self.eval_graph = tf.Graph()
        with self.eval_graph.as_default():
            analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
            analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
            analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

            # Normalized word embeddings of shape [vocab_size, emb_dim].
            nemb = tf.Variable(self.syn0norm)

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

            # Nodes in the construct graph which are used by training and
            # evaluation to run/feed/fetch.
            self.analogy_a = analogy_a
            self.analogy_b = analogy_b
            self.analogy_c = analogy_c
            self.analogy_pred_idx = pred_idx

    def predict(self, analogy, session):
        """Predict the top 4 answers for analogy questions."""
        idx, = session.run([self.analogy_pred_idx], {
            self.analogy_a: analogy[:, 0],
            self.analogy_b: analogy[:, 1],
            self.analogy_c: analogy[:, 2]})
        return idx

    def eval(self):
        """Evaluate analogy questions and reports accuracy."""
        if self.eval_data is None:
            raise ValueError("Need to read analogy questions.")

        self.read_analogies()
        correct = 0
        total = self.analogy_questions.shape[0]
        start = 0
        self.build_eval_graph()
        with tf.Session(graph=self.eval_graph) as session:
            session.run(tf.global_variables_initializer())
            while start < total:
                limit = start + 2500
                sub = self.analogy_questions[start:limit, :]
                idx = self.predict(sub, session)
                start = limit
                for question in xrange(sub.shape[0]):
                    for j in xrange(4):
                        if idx[question, j] == sub[question, 3]:
                            # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
                            correct += 1
                            break
                        elif idx[question, j] in sub[question, :3]:
                            # We need to skip words already in the question.
                            continue
                        else:
                            # The correct label is not the precision@1
                            break
        print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                                  correct * 100.0 / total))
