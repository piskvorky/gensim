#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayantjain1992@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Python implementation of Poincare Embeddings, an embedding to capture hierarchical information,
described in [1]

This module allows training a Poincare Embedding from a training file containing relations from
a transitive closure.


.. [1] https://arxiv.org/pdf/1705.08039.pdf

"""


import csv
import itertools
import logging
import os
import random
import time

from autograd import numpy as np, grad
from numpy import random as np_random
from smart_open import smart_open

from gensim import utils
from gensim.models.keyedvectors import KeyedVectors, Vocab
from gensim.models.word2vec import Word2Vec

logger = logging.getLogger(__name__)


class PoincareExample(object):
    """
    Class for computing Poincare distances and gradients for a training example,
    and storing intermediate state to avoid recomputing multiple times
    """
    def __init__(self, vector_u, vectors_v):
        """Initialize instance with sets of vectors for which distances are to be computed

        Args:
            vectors_u (numpy array): expected shape (dim,)
            vectors_v (numpy array): expected shape (1 + neg_size, dim)
        """
        self.vector_u = vector_u[np.newaxis, :]  # (1, dim)
        self.vectors_v = vectors_v  # (1 + neg_size, dim)

        self.poincare_dists = None
        self.euclidean_dists = None

        self.norm_u = None
        self.norms_v = None
        self.alpha = None
        self.beta = None
        self.gamma = None

        self.gradients_u = None
        self.distance_gradients_u = None
        self.gradients_v = None
        self.distance_gradients_v = None

        self.loss = None

        self.distances_computed = False
        self.gradients_computed = False
        self.distance_gradients_computed = False
        self.loss_computed = False

    def compute_all(self):
        """Convenience method to perform all computations"""
        self.compute_distances()
        self.compute_distance_gradients()
        self.compute_gradients()
        self.compute_loss()

    def compute_distances(self):
        """Compute and store norms, euclidean distances and poincare distances between input vectors"""
        if self.distances_computed:
            return

        euclidean_dists = np.linalg.norm(self.vector_u - self.vectors_v, axis=1)  # (1 + neg_size,)
        norm_u = np.linalg.norm(self.vector_u, axis=1)  # (1,)
        norms_v = np.linalg.norm(self.vectors_v, axis=1)  # (1 + neg_size,)
        alpha = 1 - norm_u ** 2
        beta = 1 - norms_v ** 2
        gamma = 1 + 2 * (
                (euclidean_dists ** 2) / (alpha * beta)
            )  # (1 + neg_size,)
        poincare_dists = np.arccosh(gamma)  # (1 + neg_size,)

        self.euclidean_dists = euclidean_dists
        self.poincare_dists = poincare_dists
        self.gamma = gamma
        self.norm_u = norm_u
        self.alpha = alpha
        self.norms_v = norms_v
        self.beta = beta

        self.distances_computed = True

    def compute_gradients(self):
        """Compute and store gradients of poincare distance for all input vectors"""
        if self.gradients_computed:
            return
        self.compute_distances()
        self.compute_distance_gradients()

        exp_negative_distances = np.exp(-self.poincare_dists)
        Z = exp_negative_distances.sum()

        gradients_v = -exp_negative_distances[:, np.newaxis] * self.distance_gradients_v
        gradients_v /= Z
        gradients_v[0] += self.distance_gradients_v[0]

        gradients_u = -exp_negative_distances[:, np.newaxis] * self.distance_gradients_u
        gradients_u /= Z
        gradients_u = gradients_u.sum(axis=0)
        gradients_u += self.distance_gradients_u[0]

        assert(not np.isnan(gradients_u).any())
        assert(not np.isnan(gradients_v).any())

        self.exp_negative_distances = exp_negative_distances
        self.Z = Z
        self.gradients_u = gradients_u
        self.gradients_v = gradients_v

        self.gradients_computed = True

    def compute_distance_gradients(self):
        """Compute and store partial derivatives of d(u, v) w.r.t u and all v"""
        if self.distance_gradients_computed:
            return
        self.compute_distances()

        euclidean_dists_squared = self.euclidean_dists ** 2
        c = (4 / (self.alpha * self.beta * np.sqrt(self.gamma ** 2 - 1)))[:, np.newaxis]
        u_coeffs = ((euclidean_dists_squared + self.alpha) / self.alpha)[:, np.newaxis]
        distance_gradients_u = u_coeffs * self.vector_u - self.vectors_v
        distance_gradients_u *= c
        nan_gradients = self.gamma == 1
        if nan_gradients.any():
            distance_gradients_u[nan_gradients] = 0
        self.distance_gradients_u = distance_gradients_u

        v_coeffs = ((euclidean_dists_squared + self.beta) / self.beta)[:, np.newaxis]
        distance_gradients_v = v_coeffs * self.vectors_v - self.vector_u
        distance_gradients_v *= c
        if nan_gradients.any():
            distance_gradients_v[nan_gradients] = 0
        self.distance_gradients_v = distance_gradients_v

        self.distance_gradients_computed = True

    def compute_loss(self):
        if self.loss_computed:
            return
        self.loss = -np.log(self.exp_negative_distances[0] / self.Z)
        self.loss_computed = True


class PoincareBatch(object):
    # TODO: cleanup to reduce repeated code in this class,
    # `train_batchwise` and other batch-related methods in PoincareModel
    """
    Class for computing Poincare distances and gradients for a training batch,
    and storing intermediate state to avoid recomputing multiple times
    """
    def __init__(self, vectors_u, vectors_v):
        """Initialize instance with sets of vectors for which distances are to be computed

        Args:
            vectors_u (numpy array): expected shape (dim, batch_size)
            vectors_v (numpy array): expected shape (1 + neg_size, dim, batch_size)
        """
        self.vectors_u = vectors_u[np.newaxis, :, :]  # (1, dim, batch_size)
        self.vectors_v = vectors_v  # (1 + neg_size, dim, batch_size)

        self.poincare_dists = None
        self.euclidean_dists = None

        self.norms_u = None
        self.norms_v = None
        self.alpha = None
        self.beta = None
        self.gamma = None

        self.gradients_u = None
        self.distance_gradients_u = None
        self.gradients_v = None
        self.distance_gradients_v = None

        self.loss = None

        self.distances_computed = False
        self.gradients_computed = False
        self.distance_gradients_computed = False
        self.loss_computed = False

    def compute_all(self):
        """Convenience method to perform all computations"""
        self.compute_distances()
        self.compute_distance_gradients()
        self.compute_gradients()
        self.compute_loss()

    def compute_distances(self):
        """Compute and store norms, euclidean distances and poincare distances between input vectors"""
        if self.distances_computed:
            return
        euclidean_dists = np.linalg.norm(self.vectors_u - self.vectors_v, axis=1)  # (1 + neg_size,)
        norms_u = np.linalg.norm(self.vectors_u, axis=1)  # (1,)
        norms_v = np.linalg.norm(self.vectors_v, axis=1)  # (1 + neg_size,)
        alpha = 1 - norms_u ** 2
        beta = 1 - norms_v ** 2
        gamma = 1 + 2 * (
                (euclidean_dists ** 2) / (alpha * beta)
            )  # (1 + neg_size,)
        poincare_dists = np.arccosh(gamma)  # (1 + neg_size,)

        example = PoincareExample(self.vectors_u[0, :, 0], self.vectors_v[:, :, 0])
        example.compute_all()

        self.example = example
        self.euclidean_dists = euclidean_dists
        self.poincare_dists = poincare_dists
        self.gamma = gamma
        self.norms_u = norms_u
        self.norms_v = norms_v
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.distances_computed = True

    def compute_gradients(self):
        """Compute and store gradients of poincare distance for all input vectors"""
        if self.gradients_computed:
            return
        self.compute_distances()
        self.compute_distance_gradients()

        exp_negative_distances = np.exp(-self.poincare_dists)
        Z = exp_negative_distances.sum(axis=0)
        gradients_v = -exp_negative_distances[:, np.newaxis, :] * self.distance_gradients_v
        gradients_v /= Z
        gradients_v[0] += self.distance_gradients_v[0]

        gradients_u = -exp_negative_distances[:, np.newaxis, :] * self.distance_gradients_u
        gradients_u /= Z
        gradients_u = gradients_u.sum(axis=0)
        gradients_u += self.distance_gradients_u[0]

        assert(not np.isnan(gradients_u).any())
        assert(not np.isnan(gradients_v).any())
        self.exp_negative_distances = exp_negative_distances
        self.Z = Z
        self.gradients_u = gradients_u
        self.gradients_v = gradients_v

        self.gradients_computed = True

    def compute_distance_gradients(self):
        """Compute and store partial derivatives of d(u, v) w.r.t u and all v"""
        if self.distance_gradients_computed:
            return
        self.compute_distances()

        euclidean_dists_squared = self.euclidean_dists ** 2
        c_ = (4 / (self.alpha * self.beta * np.sqrt(self.gamma ** 2 - 1)))[:, np.newaxis, :]
        u_coeffs = ((euclidean_dists_squared + self.alpha) / self.alpha)[:, np.newaxis, :]
        distance_gradients_u = u_coeffs * self.vectors_u - self.vectors_v
        distance_gradients_u *= c_
        nan_gradients = self.gamma == 1
        if nan_gradients.any():
            distance_gradients_u[nan_gradients] = 0
        self.distance_gradients_u = distance_gradients_u

        v_coeffs = ((euclidean_dists_squared + self.beta) / self.beta)[:, np.newaxis, :]
        distance_gradients_v = v_coeffs * self.vectors_v - self.vectors_u
        distance_gradients_v *= c_
        if nan_gradients.any():
            distance_gradients_v[nan_gradients] = 0
        self.distance_gradients_v = distance_gradients_v

        self.distance_gradients_computed = True

    def compute_loss(self):
        if self.loss_computed:
            return
        self.loss = -np.log(self.exp_negative_distances[0] / self.Z).sum()
        self.loss_computed = True


class PoincareKeyedVectors(KeyedVectors):
    """
    Class to contain vectors and vocab for the PoincareModel training class,
    can be used to perform operations on the vectors such as vector lookup, distance etc.

    """
    @staticmethod
    def poincare_dist(vector_1, vector_2):
        """Return poincare distance between two vectors"""
        norm_1 = np.linalg.norm(vector_1)
        norm_2 = np.linalg.norm(vector_2)
        euclidean_dist = np.linalg.norm(vector_1 - vector_2)
        if euclidean_dist == 0.0:
            return 0.0
        else:
            return np.arccosh(
                1 + 2 * (
                    (euclidean_dist ** 2) / ((1 - norm_1 ** 2) * (1 - norm_2 ** 2))
                )
            )


class PoincareModel(utils.SaveLoad):
    """
    Class for training, using and evaluating Poincare Embeddings described in https://arxiv.org/pdf/1705.08039.pdf

    The model can be stored/loaded via its `save()` and `load()` methods, or stored/loaded in the word2vec
    format via `wv.save_word2vec_format()` and `KeyedVectors.load_word2vec_format()`.

    """
    def __init__(
        self, train_file, size, alpha, min_alpha, negative,
        iter, workers, epsilon, burn_in, encoding='utf8', seed=0):
        """
        Initialize and train a Poincare embedding model from a file of transitive closure relations.

        Args:
            train_file (str): Path to tsv file containing relation pairs
            size (int): Number of dimensions of the trained model
            alpha (float): initial learning rate, decreases linearly to `min_alpha`
            negative (int): Number of negative samples to use
            iter (int): Number of iterations (epochs) over the corpus
            workers (int): Number of threads to use for training the model
            epsilon (float): Constant used for clipping embeddings below a norm of one
            burn_in (int): Number of epochs to use for burn-in initialization (0 means no burn-in)
            encoding (str): encoding of training file
            seed (int): seed for random to ensure reproducibility
        """
        self.train_file = train_file
        self.encoding = encoding
        self.wv = KeyedVectors()
        self.size = size
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.negative = negative
        self.iter = iter
        self.workers = workers
        self.epsilon = epsilon
        self.burn_in = burn_in
        self.seed = seed
        self.random = random.Random(seed)
        self.np_random = np_random.RandomState(seed)
        self.init_range = (-0.001, 0.001)
        self.loss_grad = grad(PoincareModel.loss_fn)
        self.load_relations()
        self.init_embeddings()

    def load_relations(self):
        """Load relations from the train file and build vocab"""
        vocab = {}
        index2word = []
        relations = []

        with smart_open(self.train_file, 'r', encoding=self.encoding) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                assert len(row) == 2, 'Relation pair has more than two items'
                for item in row:
                    if item in vocab:
                        vocab[item].count += 1
                    else:
                        vocab[item] = Vocab(count=1, index=len(index2word))
                        index2word.append(item)
                relations.append(tuple(row))
        self.wv.vocab = vocab
        self.wv.index2word = index2word
        self.relations = relations

    def init_embeddings(self):
        """Randomly initialize vectors for the items in the vocab"""
        shape = (len(self.wv.index2word), self.size)
        self.wv.syn0 = self.np_random.uniform(self.init_range[0], self.init_range[1], shape)

    def sample_negatives(self, _node_1):
        """Return a sample of negative examples for the given positive example"""
        # TODO: make sure returned nodes aren't positive relations for `_node_1`
        indices = self.random.sample(range(len(self.wv.index2word)), self.negative)
        return [self.wv.index2word[index] for index in indices]

    @staticmethod
    def loss_fn(matrix):
        """Given vectors for u, v and negative samples, computes loss value"""
        vector_u = matrix[0]
        vector_v = matrix[1]
        vectors_negative = matrix[2:]
        positive_distance = PoincareKeyedVectors.poincare_dist(vector_u, vector_v)
        negative_distances = np.array([
            PoincareKeyedVectors.poincare_dist(vector_u, vector_negative)
            for vector_negative in vectors_negative
        ])
        exp_negative_distances = np.exp(-negative_distances)
        exp_positive_distance = np.exp(-positive_distance)
        return -np.log(exp_positive_distance / (exp_positive_distance + exp_negative_distances.sum()))

    def compute_gradients(self, relation, negatives, check_gradients=False):
        """Computes gradients for vectors of positively related nodes and negatively sampled nodes"""
        u, v = relation
        vector_u = self.wv.word_vec(u)
        v_indices = [self.wv.vocab[v].index] + [self.wv.vocab[neg].index for neg in negatives]
        vectors_v = self.wv.syn0[v_indices]
        example = PoincareExample(vector_u, vectors_v)
        example.compute_all()
        if check_gradients:
            auto_gradients = self.loss_grad(np.vstack((vector_u, vectors_v)))
            computed_gradients = np.vstack((example.gradients_u, example.gradients_v))
            max_diff = np.abs(auto_gradients - computed_gradients).max()
            print('Max difference between gradients: %.10f' % max_diff)
            assert max_diff < 1e-10, 'Max difference greater than tolerance'
        return example

    def train_on_example(self, relation, check_gradients=False):
        """Performs training for a single training example"""
        u, v = relation
        negatives = self.sample_negatives(u)
        example = self.compute_gradients(relation, negatives, check_gradients)
        u_index = self.wv.vocab[u].index
        v_indices = [self.wv.vocab[v].index]
        for negative in negatives:
            v_indices.append(self.wv.vocab[negative].index)
        self.update_vectors(example, u_index, v_indices)
        return example

    def update_vectors(self, example, u_index, v_indices):
        grad_u, grad_v = example.gradients_u, example.gradients_v

        self.wv.syn0[u_index] -= self.alpha * (example.alpha ** 2) / 4 * grad_u
        self.wv.syn0[u_index] = self.clip_vectors(self.wv.syn0[u_index], self.epsilon)

        self.wv.syn0[v_indices] -= self.alpha * (example.beta ** 2)[:, np.newaxis] / 4 * grad_v
        self.wv.syn0[v_indices] = self.clip_vectors(self.wv.syn0[v_indices], self.epsilon)


    @staticmethod
    def clip_vectors(vectors, epsilon):
        """Clip vectors to have a norm of less than one"""
        one_d = len(vectors.shape) == 1
        if one_d:
            norm = np.linalg.norm(vectors)
            if norm < 1:
                return vectors
            else:
                return vectors / norm - (np.sign(vectors) * epsilon)
        else:
            norms = np.linalg.norm(vectors, axis=1)
            if (norms < 1).all():
                return vectors
            else:
                vectors[norms >= 1] /= norms[norms >= 1][:, np.newaxis]
                vectors[norms >= 1] -= np.sign(vectors[norms >= 1]) * epsilon
                return vectors

    @staticmethod
    def loss_fn_batch(matrix):
        """Given vectors for a batch, computes loss value"""
        vectors_u = matrix[0, :, :].T
        all_distances = PoincareKeyedVectors.poincare_dist_batch(vectors_u, matrix)
        exp_negative_distances = np.exp(-all_distances)
        return (-np.log(exp_negative_distances[:, 0] / exp_negative_distances.sum(axis=1))).sum()

    def compute_gradients_batch(self, relations, all_negatives):
        """Computes gradients for vectors of positively related nodes and negatively sampled nodes"""
        all_vectors = []
        u_indices, v_indices = [], []
        for relation, negatives in zip(relations, all_negatives):
            u, v = relation
            u_indices.append(self.wv.vocab[u].index)
            v_indices.append(
                [self.wv.vocab[v].index] +
                [self.wv.vocab[negative].index for negative in negatives]
            )
        v_indices = list(itertools.chain.from_iterable(v_indices))
        vectors_u = self.wv.syn0[u_indices].T
        vectors_v = self.wv.syn0[v_indices].reshape(1 + self.negative, self.size, self.batch_size)
        batch = PoincareBatch(vectors_u, vectors_v)
        batch.compute_all()
        return u_indices, v_indices, batch

    def sample_negatives_batch(self, _nodes):
        """Return a sample of negative examples for the given positive example"""
        # TODO: make sure returned nodes aren't positive relations for `_node_1`
        all_indices = [
            self.random.sample(range(len(self.wv.index2word)), self.negative)
            for _node in _nodes
        ]
        return [
            [self.wv.index2word[index] for index in indices]
            for indices in all_indices
        ]

    def train_on_batch(self, relations):
        """Performs training for a single training batch"""
        all_negatives = self.sample_negatives_batch([relation[0] for relation in relations])
        u_indices, v_indices, batch = self.compute_gradients_batch(relations, all_negatives)
        self.update_vectors_batch(batch, u_indices, v_indices)
        return batch

    def update_vectors_batch(self, batch, u_indices, v_indices):
        grad_u, grad_v = batch.gradients_u, batch.gradients_v

        self.wv.syn0[u_indices] -= (self.alpha * (batch.alpha ** 2) / 4 * grad_u).T
        self.wv.syn0[u_indices] = self.clip_vectors(self.wv.syn0[u_indices], self.epsilon)

        v_updates = self.alpha * (batch.beta ** 2)[:, np.newaxis] / 4 * grad_v
        v_updates = v_updates.reshape(((1 + self.negative) * self.batch_size, self.size))
        self.wv.syn0[v_indices] -= v_updates
        self.wv.syn0[v_indices] = self.clip_vectors(self.wv.syn0[v_indices], self.epsilon)
        if np.isnan(self.wv.syn0[v_indices]).any() or np.isnan(self.wv.syn0[u_indices]).any():
            import ipdb
            ipdb.set_trace()

    def train_examplewise(self, num_examples=None, print_every=10000):
        """Trains Poincare embeddings using loaded relations"""
        if self.workers > 1:
            raise NotImplementedError("Multi-threaded version not implemented yet")
        last_time = time.time()
        for epoch in range(1, self.iter + 1):
            indices = list(range(len(self.relations)))
            self.np_random.shuffle(indices)
            for i, idx in enumerate(indices, start=1):
                relation = self.relations[idx]
                print_check = not (i % print_every)
                result = self.train_on_example(relation, check_gradients=print_check)
                if print_check:
                    time_taken = time.time() - last_time
                    speed = print_every / time_taken
                    print(
                        'Training on epoch %d, example #%d %s, loss: %.2f'
                        % (epoch, i, relation, result.loss))
                    print(
                        'Time taken for %d examples: %.2f s, %.2f examples / s'
                        % (print_every, time_taken, speed))
                    last_time = time.time()
                if num_examples and i >= num_examples:
                    return

    def train_batchwise(self, num_examples=None, batch_size=2, print_every=1000):
        """Trains Poincare embeddings using loaded relations"""
        self.batch_size = batch_size
        if self.workers > 1:
            raise NotImplementedError("Multi-threaded version not implemented yet")
        last_time = time.time()
        for epoch in range(1, self.iter + 1):
            indices = list(range(len(self.relations)))
            self.np_random.shuffle(indices)
            for batch_num, i in enumerate(range(0, len(indices), batch_size), start=1):
                print_check = not (batch_num % print_every)
                batch_indices = indices[i:i+batch_size]
                relations = [self.relations[idx] for idx in batch_indices]
                result = self.train_on_batch(relations)
                if print_check:
                    time_taken = time.time() - last_time
                    speed = print_every * batch_size / time_taken
                    print(
                        'Training on epoch %d, examples #%d-#%d, loss: %.2f'
                        % (epoch, i, i + batch_size, result.loss))
                    print(
                        'Time taken for %d examples: %.2f s, %.2f examples / s'
                        % (print_every * batch_size, time_taken, speed))
                    last_time = time.time()
                if num_examples and i >= num_examples:
                    return

