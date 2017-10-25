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

from autograd import numpy as np
from autograd import grad
from numpy import random as np_random
from smart_open import smart_open

from gensim import utils
from gensim.models.keyedvectors import KeyedVectors, Vocab
from gensim.models.word2vec import Word2Vec

logger = logging.getLogger(__name__)


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

    def compute_gradients(self, relation, negatives):
        """Computes gradients for vectors of positively related nodes and negatively sampled nodes"""
        u, v = relation
        vector_u = self.wv.word_vec(u)
        vector_v = self.wv.word_vec(v)
        vectors_negative = self.wv[negatives]
        matrix = np.vstack((vector_u, vector_v, vectors_negative))
        loss = self.loss_fn(matrix)
        print('Loss: %.2f' % loss)
        gradients = self.loss_grad(matrix)
        return gradients

    def train_on_example(self, relation):
        """Performs training for a single training example"""
        u, v = relation
        negatives = self.sample_negatives(u)
        gradients = self.compute_gradients(relation, negatives)
        vector_indices = [self.wv.vocab[u].index, self.wv.vocab[v].index]
        for negative in negatives:
            vector_indices.append(self.wv.vocab[negative].index)
        for vector_index, gradient in zip(vector_indices, gradients):
            vector = self.wv.syn0[vector_index]
            norm_squared = np.linalg.norm(vector) ** 2
            self.wv.syn0[vector_index] -= self.alpha * ((1.0 - norm_squared) ** 2)/4.0 * gradient

    def train(self):
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

