#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayantjain1992@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""Python implementation of Poincare Embeddings [1]_, an embedding that is better at capturing latent hierarchical
information better than traditional Euclidean embeddings. The method is described in more detail in [1]_.

The main use-case is to automatically learn hierarchical representations of nodes from a tree-like structure,
such as a Directed Acyclic Graph, using the transitive closure of the relations. Representations of nodes in a
symmetric graph can also be learned, using an iterable of the relations in the graph.

This module allows training a Poincare Embedding from a training file containing relations of graph in a
csv-like format.

.. [1] Maximilian Nickel, Douwe Kiela - "Poincaré Embeddings for Learning Hierarchical Representations"
    https://arxiv.org/pdf/1705.08039.pdf

Examples:
---------
Initialize and train a model from a list:

>>> from gensim.models.poincare import PoincareModel
>>> relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
>>> model = PoincareModel(relations, negative=2)
>>> model.train(epochs=50)

Initialize and train a model from a file containing one relation per line:

>>> from gensim.models.poincare import PoincareModel, PoincareRelations
>>> from gensim.test.utils import datapath
>>> file_path = datapath('poincare_hypernyms.tsv')
>>> model = PoincareModel(PoincareRelations(file_path), negative=2)
>>> model.train(epochs=50)

"""


import csv
import logging
import sys
import time

import numpy as np
from collections import defaultdict, Counter
from numpy import random as np_random
from smart_open import smart_open

from gensim import utils
from gensim.models.keyedvectors import KeyedVectors, Vocab


logger = logging.getLogger(__name__)


class PoincareModel(utils.SaveLoad):
    """Class for training, using and evaluating Poincare Embeddings.

    The model can be stored/loaded via its :meth:`~gensim.models.poincare.PoincareModel.save`
    and :meth:`~gensim.models.poincare.PoincareModel.load` methods, or stored/loaded in the word2vec format
    via `model.kv.save_word2vec_format` and :meth:`~gensim.models.keyedvectors.KeyedVectors.load_word2vec_format`.

    Note that training cannot be resumed from a model loaded via `load_word2vec_format`, if you wish to train further,
    use :meth:`~gensim.models.poincare.PoincareModel.save` and :meth:`~gensim.models.poincare.PoincareModel.load`
    methods instead.

    """
    def __init__(self, train_data, size=50, alpha=0.1, negative=10, workers=1, epsilon=1e-5,
                 burn_in=10, burn_in_alpha=0.01, init_range=(-0.001, 0.001), seed=0):
        """Initialize and train a Poincare embedding model from an iterable of relations.

        Parameters
        ----------
        train_data : iterable of (str, str)
            Iterable of relations, e.g. a list of tuples, or a PoincareRelations instance streaming from a file.
            Note that the relations are treated as ordered pairs, i.e. a relation (a, b) does not imply the
            opposite relation (b, a). In case the relations are symmetric, the data should contain both relations
            (a, b) and (b, a).
        size : int, optional
            Number of dimensions of the trained model.
        alpha : float, optional
            Learning rate for training.
        negative : int, optional
            Number of negative samples to use.
        workers : int, optional
            Number of threads to use for training the model.
        epsilon : float, optional
            Constant used for clipping embeddings below a norm of one.
        burn_in : int, optional
            Number of epochs to use for burn-in initialization (0 means no burn-in).
        burn_in_alpha : float, optional
            Learning rate for burn-in initialization, ignored if `burn_in` is 0.
        init_range : 2-tuple (float, float)
            Range within which the vectors are randomly initialized.
        seed : int, optional
            Seed for random to ensure reproducibility.

        Examples
        --------
        Initialize a model from a list:

        >>> from gensim.models.poincare import PoincareModel
        >>> relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
        >>> model = PoincareModel(relations, negative=2)

        Initialize a model from a file containing one relation per line:

        >>> from gensim.models.poincare import PoincareModel, PoincareRelations
        >>> from gensim.test.utils import datapath
        >>> file_path = datapath('poincare_hypernyms.tsv')
        >>> model = PoincareModel(PoincareRelations(file_path), negative=2)

        See :class:`~gensim.models.poincare.PoincareRelations` for more options.

        """
        self.train_data = train_data
        self.kv = KeyedVectors()
        self.size = size
        self.train_alpha = alpha  # Learning rate for training
        self.burn_in_alpha = burn_in_alpha  # Learning rate for burn-in
        self.alpha = alpha  # Current learning rate
        self.negative = negative
        self.workers = workers
        self.epsilon = epsilon
        self.burn_in = burn_in
        self._burn_in_done = False
        self.seed = seed
        self._np_random = np_random.RandomState(seed)
        self.init_range = init_range
        self._loss_grad = None
        self._load_relations()
        self._init_embeddings()

    def _load_relations(self):
        """Load relations from the train data and build vocab."""
        vocab = {}
        index2word = []
        all_relations = []  # List of all relation pairs
        node_relations = defaultdict(set)  # Mapping from node index to its related node indices

        logger.info("Loading relations from train data..")
        for relation in self.train_data:
            if len(relation) != 2:
                raise ValueError('Relation pair "%s" should have exactly two items' % repr(relation))
            for item in relation:
                if item in vocab:
                    vocab[item].count += 1
                else:
                    vocab[item] = Vocab(count=1, index=len(index2word))
                    index2word.append(item)
            node_1, node_2 = relation
            node_1_index, node_2_index = vocab[node_1].index, vocab[node_2].index
            node_relations[node_1_index].add(node_2_index)
            relation = (node_1_index, node_2_index)
            all_relations.append(relation)
        logger.info("Loaded %d relations from train data, %d unique terms", len(all_relations), len(vocab))
        self.kv.vocab = vocab
        self.kv.index2word = index2word
        self.indices_set = set((range(len(index2word))))  # Set of all node indices
        self.indices_array = np.array(range(len(index2word)))  # Numpy array of all node indices
        counts = np.array([self.kv.vocab[index2word[i]].count for i in range(len(index2word))], dtype=np.float64)
        self._node_probabilities = counts / counts.sum()
        self._node_probabilities_cumsum = np.cumsum(self._node_probabilities)
        self.all_relations = all_relations
        self.node_relations = node_relations
        self._negatives_buffer = NegativesBuffer([])  # Buffer for negative samples, to reduce calls to sampling method
        self._negatives_buffer_size = 2000

    def _init_embeddings(self):
        """Randomly initialize vectors for the items in the vocab."""
        shape = (len(self.kv.index2word), self.size)
        self.kv.syn0 = self._np_random.uniform(self.init_range[0], self.init_range[1], shape)

    def _get_candidate_negatives(self):
        """Returns candidate negatives of size `self.negative` from the negative examples buffer.

        Returns
        --------
        numpy.array
            Array of shape (`self.negative`,) containing indices of negative nodes.

        """

        if self._negatives_buffer.num_items() < self.negative:
            # Note: np.random.choice much slower than random.sample for large populations, possible bottleneck
            uniform_numbers = self._np_random.random_sample(self._negatives_buffer_size)
            cumsum_table_indices = np.searchsorted(self._node_probabilities_cumsum, uniform_numbers)
            self._negatives_buffer = NegativesBuffer(cumsum_table_indices)
        return self._negatives_buffer.get_items(self.negative)

    def _sample_negatives(self, node_index):
        """Return a sample of negatives for the given node.

        Parameters
        ----------
        node_index : int
            Index of the positive node for which negative samples are to be returned.

        Returns
        --------
        numpy.array
            Array of shape (self.negative,) containing indices of negative nodes for the given node index.

        """
        node_relations = self.node_relations[node_index]
        num_remaining_nodes = len(self.kv.vocab) - len(node_relations)
        if num_remaining_nodes < self.negative:
            raise ValueError(
                'Cannot sample %d negative items from a set of %d items' %
                (self.negative, num_remaining_nodes)
            )

        positive_fraction = len(node_relations) / len(self.kv.vocab)
        if positive_fraction < 0.01:
            # If number of positive relations is a small fraction of total nodes
            # re-sample till no positively connected nodes are chosen
            indices = self._get_candidate_negatives()
            unique_indices = set(indices)
            times_sampled = 1
            while (len(indices) != len(unique_indices)) or (unique_indices & node_relations):
                times_sampled += 1
                indices = self._get_candidate_negatives()
                unique_indices = set(indices)
            if times_sampled > 1:
                logger.debug('Sampled %d times, positive fraction %.5f', times_sampled, positive_fraction)
        else:
            # If number of positive relations is a significant fraction of total nodes
            # subtract positively connected nodes from set of choices and sample from the remaining
            valid_negatives = np.array(list(self.indices_set - node_relations))
            probs = self._node_probabilities[valid_negatives]
            probs /= probs.sum()
            indices = self._np_random.choice(valid_negatives, size=self.negative, p=probs, replace=False)

        return list(indices)

    @staticmethod
    def _loss_fn(matrix):
        """Given a numpy array with vectors for u, v and negative samples, computes loss value.

        Parameters
        ----------
        matrix : numpy.array
            Array containing vectors for u, v and negative samples, of shape (2 + negative_size, dim).

        Returns
        -------
        float
            Computed loss value.

        Warnings
        --------
        Only used for autograd gradients, since autograd requires a specific function signature.

        """
        # Loaded only if gradients are to be checked to avoid dependency
        from autograd import numpy as grad_np

        vector_u = matrix[0]
        vectors_v = matrix[1:]
        euclidean_dists = grad_np.linalg.norm(vector_u - vectors_v, axis=1)
        norm = grad_np.linalg.norm(vector_u)
        all_norms = grad_np.linalg.norm(vectors_v, axis=1)
        poincare_dists = grad_np.arccosh(
            1 + 2 * (
                (euclidean_dists ** 2) / ((1 - norm ** 2) * (1 - all_norms ** 2))
            )
        )
        exp_negative_distances = grad_np.exp(-poincare_dists)
        return -grad_np.log(exp_negative_distances[0] / (exp_negative_distances.sum()))

    @staticmethod
    def _clip_vectors(vectors, epsilon):
        """Clip vectors to have a norm of less than one.

        Parameters
        ----------
        vectors : numpy.array
            Can be 1-D,or 2-D (in which case the norm for each row is checked).
        epsilon : float
            Parameter for numerical stability, each dimension of the vector is reduced by `epsilon`
            if the norm of the vector is greater than or equal to 1.

        Returns
        -------
        numpy.array
            Array with norms clipped below 1.

        """
        one_d = len(vectors.shape) == 1
        threshold = 1 - epsilon
        if one_d:
            norm = np.linalg.norm(vectors)
            if norm < threshold:
                return vectors
            else:
                return vectors / norm - (np.sign(vectors) * epsilon)
        else:
            norms = np.linalg.norm(vectors, axis=1)
            if (norms < threshold).all():
                return vectors
            else:
                vectors[norms >= threshold] *= (threshold / norms[norms >= threshold])[:, np.newaxis]
                vectors[norms >= threshold] -= np.sign(vectors[norms >= threshold]) * epsilon
                return vectors

    def save(self, *args, **kwargs):
        """Save complete model to disk, inherited from :class:`gensim.utils.SaveLoad`."""
        self._loss_grad = None  # Can't pickle autograd fn to disk
        super(PoincareModel, self).save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        """Load model from disk, inherited from :class:`~gensim.utils.SaveLoad`."""
        model = super(PoincareModel, cls).load(*args, **kwargs)
        return model

    def _prepare_training_batch(self, relations, all_negatives, check_gradients=False):
        """Creates training batch and computes gradients and loss for the batch.

        Parameters
        ----------

        relations : list of tuples
            List of tuples of positive examples of the form (node_1_index, node_2_index).
        all_negatives : list of lists
            List of lists of negative samples for each node_1 in the positive examples.
        check_gradients : bool, optional
            Whether to compare the computed gradients to autograd gradients for this batch.

        Returns
        -------
        :class:`~gensim.models.poincare.PoincareBatch`
            Contains node indices, computed gradients and loss for the batch.

        """
        batch_size = len(relations)
        indices_u, indices_v = [], []
        for relation, negatives in zip(relations, all_negatives):
            u, v = relation
            indices_u.append(u)
            indices_v.append(v)
            indices_v.extend(negatives)

        vectors_u = self.kv.syn0[indices_u]
        vectors_v = self.kv.syn0[indices_v].reshape((batch_size, 1 + self.negative, self.size))
        vectors_v = vectors_v.swapaxes(0, 1).swapaxes(1, 2)
        batch = PoincareBatch(vectors_u, vectors_v, indices_u, indices_v)
        batch.compute_all()

        if check_gradients:
            self._check_gradients(relations, all_negatives, batch)

        return batch

    def _check_gradients(self, relations, all_negatives, batch, tol=1e-8):
        """Compare computed gradients for batch to autograd gradients.

        Parameters
        ----------
        batch : PoincareBatch instance
            Batch for which computed gradients are to checked.
        relations : list of tuples
            List of tuples of positive examples of the form (node_1_index, node_2_index).
        all_negatives : list of lists
            List of lists of negative samples for each node_1 in the positive examples.

        """
        try:  # Loaded only if gradients are to be checked to avoid dependency
            from autograd import grad
        except ImportError:
            logger.warning('autograd could not be imported, skipping checking of gradients')
            logger.warning('please install autograd to enable gradient checking')
            return

        if self._loss_grad is None:
            self._loss_grad = grad(PoincareModel._loss_fn)

        max_diff = 0.0
        for i, (relation, negatives) in enumerate(zip(relations, all_negatives)):
            u, v = relation
            auto_gradients = self._loss_grad(np.vstack((self.kv.syn0[u], self.kv.syn0[[v] + negatives])))
            computed_gradients = np.vstack((batch.gradients_u[:, i], batch.gradients_v[:, :, i]))
            diff = np.abs(auto_gradients - computed_gradients).max()
            if diff > max_diff:
                max_diff = diff
        logger.info('Max difference between computed gradients and autograd gradients: %.10f', max_diff)
        assert max_diff < tol, (
                'Max difference between computed gradients and autograd gradients %.10f, '
                'greater than tolerance %.10f' % (max_diff, tol))

    def _sample_negatives_batch(self, nodes):
        """Return negative examples for each node in the given nodes.

        Parameters
        ----------
        nodes : list
            List of node indices for which negative samples are to be returned.

        Returns
        -------
        list of lists
            Each inner list is a list of negative sample for a single node in the input list.

        """
        all_indices = [self._sample_negatives(node) for node in nodes]
        return all_indices

    def _train_on_batch(self, relations, check_gradients=False):
        """Performs training for a single training batch.

        Parameters
        ----------
        relations : list of tuples
            List of tuples of positive examples of the form (node_1_index, node_2_index).
        check_gradients : bool, optional
            Whether to compare the computed gradients to autograd gradients for this batch.

        Returns
        -------
        :class:`~gensim.models.poincare.PoincareBatch`
            The batch that was just trained on, contains computed loss for the batch.

        """
        all_negatives = self._sample_negatives_batch([relation[0] for relation in relations])
        batch = self._prepare_training_batch(relations, all_negatives, check_gradients)
        self._update_vectors_batch(batch)
        return batch

    @staticmethod
    def _handle_duplicates(vector_updates, node_indices):
        """Handles occurrences of multiple updates to the same node in a batch of vector updates.

        Parameters
        ----------
        vector_updates : numpy.array
            Array with each row containing updates to be performed on a certain node.
        node_indices : list
            Node indices on which the above updates are to be performed on.

        Notes
        -----
        Mutates the `vector_updates` array.

        Required because vectors[[2, 1, 2]] += np.array([-0.5, 1.0, 0.5]) performs only the last update
        on the row at index 2.

        """
        counts = Counter(node_indices)
        for node_index, count in counts.items():
            if count == 1:
                continue
            positions = [i for i, index in enumerate(node_indices) if index == node_index]
            # Move all updates to the same node to the last such update, zeroing all the others
            vector_updates[positions[-1]] = vector_updates[positions].sum(axis=0)
            vector_updates[positions[:-1]] = 0

    def _update_vectors_batch(self, batch):
        """Updates vectors for nodes in the given batch.

        Parameters
        ----------
        batch : :class:`~gensim.models.poincare.PoincareBatch`
            Batch containing computed gradients and node indices of the batch for which updates are to be done.

        """
        grad_u, grad_v = batch.gradients_u, batch.gradients_v
        indices_u, indices_v = batch.indices_u, batch.indices_v
        batch_size = len(indices_u)

        u_updates = (self.alpha * (batch.alpha ** 2) / 4 * grad_u).T
        self._handle_duplicates(u_updates, indices_u)

        self.kv.syn0[indices_u] -= u_updates
        self.kv.syn0[indices_u] = self._clip_vectors(self.kv.syn0[indices_u], self.epsilon)

        v_updates = self.alpha * (batch.beta ** 2)[:, np.newaxis] / 4 * grad_v
        v_updates = v_updates.swapaxes(1, 2).swapaxes(0, 1)
        v_updates = v_updates.reshape(((1 + self.negative) * batch_size, self.size))
        self._handle_duplicates(v_updates, indices_v)

        self.kv.syn0[indices_v] -= v_updates
        self.kv.syn0[indices_v] = self._clip_vectors(self.kv.syn0[indices_v], self.epsilon)

    def train(self, epochs, batch_size=10, print_every=1000, check_gradients_every=None):
        """Trains Poincare embeddings using loaded data and model parameters.

        Parameters
        ----------

        batch_size : int, optional
            Number of examples to train on in a single batch.
        epochs : int
            Number of iterations (epochs) over the corpus.
        print_every : int, optional
            Prints progress and average loss after every `print_every` batches.
        check_gradients_every : int or None, optional
            Compares computed gradients and autograd gradients after every `check_gradients_every` batches.
            Useful for debugging, doesn't compare by default.

        Examples
        --------
        >>> from gensim.models.poincare import PoincareModel
        >>> relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
        >>> model = PoincareModel(relations, negative=2)
        >>> model.train(epochs=50)

        """
        if self.workers > 1:
            raise NotImplementedError("Multi-threaded version not implemented yet")

        logger.info(
            "training model of size %d with %d workers on %d relations for %d epochs and %d burn-in epochs, "
            "using lr=%.5f burn-in lr=%.5f negative=%d",
            self.size, self.workers, len(self.all_relations), epochs, self.burn_in,
            self.alpha, self.burn_in_alpha, self.negative
        )

        if self.burn_in > 0 and not self._burn_in_done:
            logger.info("Starting burn-in (%d epochs)----------------------------------------", self.burn_in)
            self.alpha = self.burn_in_alpha
            self._train_batchwise(
                epochs=self.burn_in, batch_size=batch_size, print_every=print_every,
                check_gradients_every=check_gradients_every)
            self._burn_in_done = True
            logger.info("Burn-in finished")

        self.alpha = self.train_alpha
        logger.info("Starting training (%d epochs)----------------------------------------", epochs)
        self._train_batchwise(
            epochs=epochs, batch_size=batch_size, print_every=print_every,
            check_gradients_every=check_gradients_every)
        logger.info("Training finished")

    def _train_batchwise(self, epochs, batch_size=10, print_every=1000, check_gradients_every=None):
        """Trains Poincare embeddings using specified parameters.

        Parameters
        ----------
        epochs : int
            Number of iterations (epochs) over the corpus.
        batch_size : int, optional
            Number of examples to train on in a single batch.
        print_every : int, optional
            Prints progress and average loss after every `print_every` batches.
        check_gradients_every : int or None, optional
            Compares computed gradients and autograd gradients after every `check_gradients_every` batches.
            Useful for debugging, doesn't compare by default.

        """
        if self.workers > 1:
            raise NotImplementedError("Multi-threaded version not implemented yet")
        for epoch in range(1, epochs + 1):
            indices = list(range(len(self.all_relations)))
            self._np_random.shuffle(indices)
            avg_loss = 0.0
            last_time = time.time()
            for batch_num, i in enumerate(range(0, len(indices), batch_size), start=1):
                should_print = not (batch_num % print_every)
                check_gradients = bool(check_gradients_every) and (batch_num % check_gradients_every) == 0
                batch_indices = indices[i:i + batch_size]
                relations = [self.all_relations[idx] for idx in batch_indices]
                result = self._train_on_batch(relations, check_gradients=check_gradients)
                avg_loss += result.loss
                if should_print:
                    avg_loss /= print_every
                    time_taken = time.time() - last_time
                    speed = print_every * batch_size / time_taken
                    logger.info(
                        'Training on epoch %d, examples #%d-#%d, loss: %.2f'
                        % (epoch, i, i + batch_size, avg_loss))
                    logger.info(
                        'Time taken for %d examples: %.2f s, %.2f examples / s'
                        % (print_every * batch_size, time_taken, speed))
                    last_time = time.time()
                    avg_loss = 0.0


class PoincareBatch(object):
    """Compute Poincare distances, gradients and loss for a training batch.

    Class for computing Poincare distances, gradients and loss for a training batch,
    and storing intermediate state to avoid recomputing multiple times.

    """
    def __init__(self, vectors_u, vectors_v, indices_u, indices_v):
        """
        Initialize instance with sets of vectors for which distances are to be computed.

        Parameters
        ----------
        vectors_u : numpy.array
            Vectors of all nodes `u` in the batch.
            Expected shape (batch_size, dim).
        vectors_v : numpy.array
            Vectors of all positively related nodes `v` and negatively sampled nodes `v'`,
            for each node `u` in the batch.
            Expected shape (1 + neg_size, dim, batch_size).
        indices_u : list
            List of node indices for each of the vectors in `vectors_u`.
        indices_v : list
            Nested list of lists, each of which is a  list of node indices
            for each of the vectors in `vectors_v` for a specific node `u`.

        """
        self.vectors_u = vectors_u.T[np.newaxis, :, :]  # (1, dim, batch_size)
        self.vectors_v = vectors_v  # (1 + neg_size, dim, batch_size)
        self.indices_u = indices_u
        self.indices_v = indices_v

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

        self._distances_computed = False
        self._gradients_computed = False
        self._distance_gradients_computed = False
        self._loss_computed = False

    def compute_all(self):
        """Convenience method to perform all computations."""
        self.compute_distances()
        self.compute_distance_gradients()
        self.compute_gradients()
        self.compute_loss()

    def compute_distances(self):
        """Compute and store norms, euclidean distances and poincare distances between input vectors."""
        if self._distances_computed:
            return
        euclidean_dists = np.linalg.norm(self.vectors_u - self.vectors_v, axis=1)  # (1 + neg_size, batch_size)
        norms_u = np.linalg.norm(self.vectors_u, axis=1)  # (1, batch_size)
        norms_v = np.linalg.norm(self.vectors_v, axis=1)  # (1 + neg_size, batch_size)
        alpha = 1 - norms_u ** 2  # (1, batch_size)
        beta = 1 - norms_v ** 2  # (1 + neg_size, batch_size)
        gamma = 1 + 2 * (
                (euclidean_dists ** 2) / (alpha * beta)
            )  # (1 + neg_size, batch_size)
        poincare_dists = np.arccosh(gamma)  # (1 + neg_size, batch_size)
        exp_negative_distances = np.exp(-poincare_dists)  # (1 + neg_size, batch_size)
        Z = exp_negative_distances.sum(axis=0)  # (batch_size)

        self.euclidean_dists = euclidean_dists
        self.poincare_dists = poincare_dists
        self.exp_negative_distances = exp_negative_distances
        self.Z = Z
        self.gamma = gamma
        self.norms_u = norms_u
        self.norms_v = norms_v
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self._distances_computed = True

    def compute_gradients(self):
        """Compute and store gradients of loss function for all input vectors."""
        if self._gradients_computed:
            return
        self.compute_distances()
        self.compute_distance_gradients()

        gradients_v = -self.exp_negative_distances[:, np.newaxis, :] * self.distance_gradients_v  # (1 + neg_size, dim, batch_size)
        gradients_v /= self.Z  # (1 + neg_size, dim, batch_size)
        gradients_v[0] += self.distance_gradients_v[0]

        gradients_u = -self.exp_negative_distances[:, np.newaxis, :] * self.distance_gradients_u  # (1 + neg_size, dim, batch_size)
        gradients_u /= self.Z  # (1 + neg_size, dim, batch_size)
        gradients_u = gradients_u.sum(axis=0)  # (dim, batch_size)
        gradients_u += self.distance_gradients_u[0]

        assert(not np.isnan(gradients_u).any())
        assert(not np.isnan(gradients_v).any())
        self.gradients_u = gradients_u
        self.gradients_v = gradients_v

        self._gradients_computed = True

    def compute_distance_gradients(self):
        """Compute and store partial derivatives of poincare distance d(u, v) w.r.t all u and all v."""
        if self._distance_gradients_computed:
            return
        self.compute_distances()

        euclidean_dists_squared = self.euclidean_dists ** 2  # (1 + neg_size, batch_size)
        c_ = (4 / (self.alpha * self.beta * np.sqrt(self.gamma ** 2 - 1)))[:, np.newaxis, :]  # (1 + neg_size, 1, batch_size)
        u_coeffs = ((euclidean_dists_squared + self.alpha) / self.alpha)[:, np.newaxis, :]  # (1 + neg_size, 1, batch_size)
        distance_gradients_u = u_coeffs * self.vectors_u - self.vectors_v  # (1 + neg_size, dim, batch_size)
        distance_gradients_u *= c_  # (1 + neg_size, dim, batch_size)

        nan_gradients = self.gamma == 1  # (1 + neg_size, batch_size)
        if nan_gradients.any():
            distance_gradients_u.swapaxes(1, 2)[nan_gradients] = 0
        self.distance_gradients_u = distance_gradients_u

        v_coeffs = ((euclidean_dists_squared + self.beta) / self.beta)[:, np.newaxis, :]  # (1 + neg_size, 1, batch_size)
        distance_gradients_v = v_coeffs * self.vectors_v - self.vectors_u  # (1 + neg_size, dim, batch_size)
        distance_gradients_v *= c_  # (1 + neg_size, dim, batch_size)

        if nan_gradients.any():
            distance_gradients_v.swapaxes(1, 2)[nan_gradients] = 0
        self.distance_gradients_v = distance_gradients_v

        self._distance_gradients_computed = True

    def compute_loss(self):
        """Compute and store loss value for the given batch of examples."""
        if self._loss_computed:
            return
        self.compute_distances()

        self.loss = -np.log(self.exp_negative_distances[0] / self.Z).sum()  # scalar
        self._loss_computed = True


class PoincareKeyedVectors(KeyedVectors):
    """Class to contain vectors and vocab for the :class:`~gensim.models.poincare.PoincareModel` training class.

    Used to perform operations on the vectors such as vector lookup, distance etc.

    """
    @staticmethod
    def poincare_dist(vector_1, vector_2):
        """Return poincare distance between two vectors."""
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
    # TODO: Add other KeyedVector supported methods - most_similar, etc.


class PoincareRelations(object):
    """Class to stream relations for `PoincareModel` from a tsv-like file."""

    def __init__(self, file_path, encoding='utf8', delimiter='\t'):
        """Initialize instance from file containing a pair of nodes (a relation) per line.

        Parameters
        ----------
        file_path : str
            Path to file containing a pair of nodes (a relation) per line, separated by `delimiter`.
        encoding : str, optional
            Character encoding of the input file.
        delimiter : str, optional
            Delimiter character for each relation.

        """

        self.file_path = file_path
        self.encoding = encoding
        self.delimiter = delimiter

    def __iter__(self):
        """Streams relations from self.file_path decoded into unicode strings.

        Yields
        -------
        2-tuple (unicode, unicode)
            Relation from input file.

        """
        if sys.version_info[0] < 3:
            lines = smart_open(self.file_path, 'rb')
        else:
            lines = (l.decode(self.encoding) for l in smart_open(self.file_path, 'rb'))
        # csv.reader requires bytestring input in python2, unicode input in python3
        reader = csv.reader(lines, delimiter=self.delimiter)
        for row in reader:
            if sys.version_info[0] < 3:
                row = [value.decode(self.encoding) for value in row]
            yield tuple(row)


class NegativesBuffer(object):
    """Class to buffer and return negative samples."""

    def __init__(self, items):
        """Initialize instance from list or numpy array of samples.

        Parameters
        ----------
        items : list/numpy.array
            List or array containing negative samples.

        """

        self._items = items
        self._current_index = 0

    def num_items(self):
        """Returns number of items remaining in the buffer.

        Returns
        -------
        int
            Number of items in the buffer that haven't been consumed yet.

        """
        return len(self._items) - self._current_index

    def get_items(self, num_items):
        """Returns next `num_items` from buffer.

        Parameters
        ----------
        num_items : int
            number of items to fetch.

        Returns
        -------
        numpy.array or list
            Slice containing `num_items` items from the original data.

        Notes
        -----
        No error is raised if less than `num_items` items are remaining,
        simply all the remaining items are returned.

        """
        start_index = self._current_index
        end_index = start_index + num_items
        self._current_index += num_items
        return self._items[start_index:end_index]
