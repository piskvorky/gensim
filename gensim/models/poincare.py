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
import logging
import os
import random

import numpy as np
from numpy import random as np_random
from smart_open import smart_open

from gensim import utils
from gensim.models.keyedvectors import KeyedVectors, Vocab
from gensim.models.word2vec import Word2Vec

logger = logging.getLogger(__name__)


class PoincareDistance(object):
    """
    Class for computing Poincare distances between sets of vectors, and storing intermediate
    state to avoid recomputing multiple times
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
        u_coeffs = ((self.euclidean_dists ** 2 + self.alpha) / self.alpha)[:, np.newaxis]
        distance_gradients_u = u_coeffs * self.vector_u - self.vectors_v
        distance_gradients_u *= (4 / (self.alpha * self.beta * np.sqrt(self.gamma ** 2 - 1)))[:, np.newaxis]
        np.nan_to_num(distance_gradients_u, copy=False)
        self.distance_gradients_u = distance_gradients_u

        v_coeffs = ((self.euclidean_dists ** 2 + self.beta) / self.beta)[:, np.newaxis]
        distance_gradients_v = v_coeffs * self.vectors_v - self.vector_u
        distance_gradients_v *= (4 / (self.alpha * self.beta * np.sqrt(self.gamma ** 2 - 1)))[:, np.newaxis]
        np.nan_to_num(distance_gradients_v, copy=False)
        self.distance_gradients_v = distance_gradients_v

        self.distance_gradients_computed = True

    def compute_loss(self):
        if self.loss_computed:
            return
        self.loss = -np.log(self.exp_negative_distances[0] / self.Z)
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

    def compute_gradients(self, relation, negatives):
        """Computes gradients for vectors of positively related nodes and negatively sampled nodes"""
        u, v = relation
        vector_u = self.wv.word_vec(u)
        vector_v = self.wv.word_vec(v)
        vectors_negative = self.wv[negatives]
        vectors_v = np.vstack((vector_v, vectors_negative))
        # TODO: better naming, some refactoring
        distances = PoincareDistance(vector_u, vectors_v)
        distances.compute_all()
        return distances

    def train_on_example(self, relation):
        """Performs training for a single training example"""
        u, v = relation
        negatives = self.sample_negatives(u)
        distances = self.compute_gradients(relation, negatives)
        grad_u, grad_v = distances.gradients_u, distances.gradients_v
        u_index = self.wv.vocab[u].index
        v_indices = [self.wv.vocab[v].index]
        for negative in negatives:
            v_indices.append(self.wv.vocab[negative].index)

        self.wv.syn0[u_index] -= self.alpha * (distances.alpha ** 2) / 4 * grad_u
        self.wv.syn0[u_index] = self.clip_vectors(self.wv.syn0[u_index], self.epsilon)

        self.wv.syn0[v_indices] -= self.alpha * (distances.beta ** 2)[:, np.newaxis] / 4 * grad_v
        self.wv.syn0[v_indices] = self.clip_vectors(self.wv.syn0[v_indices], self.epsilon)
        print('Loss: %.2f' % distances.loss)


    @staticmethod
    def clip_vectors(vectors, epsilon):
        """Clip vectors to have a norm of less than one"""
        # TODO: correct implementation
        one_d = len(vectors.shape) == 1
        if one_d:
            norm = np.linalg.norm(vectors)
            if norm < 1:
                return vectors
            else:
                return vector / norm - epsilon
        else:
            norms = np.linalg.norm(vectors, axis=1)
            if (norms < 1).all():
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
        for relation, negatives in zip(relations, all_negatives):
            u, v = relation
            vectors = self.wv[[u, v] + negatives]
            all_vectors.append(vectors)
        matrix = np.dstack(tuple(all_vectors))
        loss = self.loss_fn_batch(matrix)
        print('Loss: %.2f' % loss)
        gradients = self.batch_loss_grad(matrix)
        return gradients

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
        gradients = self.compute_gradients_batch(relations, all_negatives)
        # TODO: use gradients to perform updates

    def train_examplewise(self, num_examples=None):
        """Trains Poincare embeddings using loaded relations"""
        if self.workers > 1:
            raise NotImplementedError("Multi-threaded version not implemented yet")
        for epoch in range(1, self.iter + 1):
            indices = list(range(len(self.relations)))
            self.np_random.shuffle(indices)
            for i, idx in enumerate(indices, start=1):
                relation = self.relations[idx]
                print('Training on example #%d %s' % (i, relation))
                self.train_on_example(relation)
                if num_examples and i >= num_examples:
                    return

    def train_batchwise(self, num_examples=100, batch_size=2):
        """Trains Poincare embeddings using loaded relations"""
        self.batch_size = batch_size
        if self.workers > 1:
            raise NotImplementedError("Multi-threaded version not implemented yet")
        for epoch in range(1, self.iter + 1):
            indices = list(range(len(self.relations)))
            self.np_random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                relations = [self.relations[idx] for idx in batch_indices]
                print('Training on example #%d-%d' % (i, i+batch_size))
                self.train_on_batch(relations)
                if i >= num_examples:
                    return

