#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayantjain1992@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Python implementation of Poincaré Embeddings.

These embeddings are better at capturing latent hierarchical information than traditional Euclidean embeddings.
The method is described in detail in `Maximilian Nickel, Douwe Kiela -
"Poincaré Embeddings for Learning Hierarchical Representations" <https://arxiv.org/abs/1705.08039>`_.

The main use-case is to automatically learn hierarchical representations of nodes from a tree-like structure,
such as a Directed Acyclic Graph (DAG), using a transitive closure of the relations. Representations of nodes in a
symmetric graph can also be learned.

This module allows training Poincaré Embeddings from a training file containing relations of graph in a
csv-like format, or from a Python iterable of relations.


Examples
--------
Initialize and train a model from a list

.. sourcecode:: pycon

    >>> from gensim.models.poincare import PoincareModel
    >>> relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
    >>> model = PoincareModel(relations, negative=2)
    >>> model.train(epochs=50)

Initialize and train a model from a file containing one relation per line

.. sourcecode:: pycon

    >>> from gensim.models.poincare import PoincareModel, PoincareRelations
    >>> from gensim.test.utils import datapath
    >>> file_path = datapath('poincare_hypernyms.tsv')
    >>> model = PoincareModel(PoincareRelations(file_path), negative=2)
    >>> model.train(epochs=50)

"""

import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter

import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
try:
    from autograd import grad  # Only required for optionally verifying gradients while training
    from autograd import numpy as grad_np
    AUTOGRAD_PRESENT = True
except ImportError:
    AUTOGRAD_PRESENT = False

from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors


logger = logging.getLogger(__name__)


class PoincareModel(utils.SaveLoad):
    """Train, use and evaluate Poincare Embeddings.

    The model can be stored/loaded via its :meth:`~gensim.models.poincare.PoincareModel.save`
    and :meth:`~gensim.models.poincare.PoincareModel.load` methods, or stored/loaded in the word2vec format
    via `model.kv.save_word2vec_format` and :meth:`~gensim.models.poincare.PoincareKeyedVectors.load_word2vec_format`.

    Notes
    -----
    Training cannot be resumed from a model loaded via `load_word2vec_format`, if you wish to train further,
    use :meth:`~gensim.models.poincare.PoincareModel.save` and :meth:`~gensim.models.poincare.PoincareModel.load`
    methods instead.

    An important attribute (that provides a lot of additional functionality when directly accessed) are the
    keyed vectors:

    self.kv : :class:`~gensim.models.poincare.PoincareKeyedVectors`
        This object essentially contains the mapping between nodes and embeddings, as well the vocabulary of the model
        (set of unique nodes seen by the model). After training, it can be used to perform operations on the vectors
        such as vector lookup, distance and similarity calculations etc.
        See the documentation of its class for usage examples.

    """
    def __init__(self, train_data, size=50, alpha=0.1, negative=10, workers=1, epsilon=1e-5, regularization_coeff=1.0,
                 burn_in=10, burn_in_alpha=0.01, init_range=(-0.001, 0.001), dtype=np.float64, seed=0):
        """Initialize and train a Poincare embedding model from an iterable of relations.

        Parameters
        ----------
        train_data : {iterable of (str, str), :class:`gensim.models.poincare.PoincareRelations`}
            Iterable of relations, e.g. a list of tuples, or a :class:`gensim.models.poincare.PoincareRelations`
            instance streaming from a file. Note that the relations are treated as ordered pairs,
            i.e. a relation (a, b) does not imply the opposite relation (b, a). In case the relations are symmetric,
            the data should contain both relations (a, b) and (b, a).
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
        regularization_coeff : float, optional
            Coefficient used for l2-regularization while training (0 effectively disables regularization).
        burn_in : int, optional
            Number of epochs to use for burn-in initialization (0 means no burn-in).
        burn_in_alpha : float, optional
            Learning rate for burn-in initialization, ignored if `burn_in` is 0.
        init_range : 2-tuple (float, float)
            Range within which the vectors are randomly initialized.
        dtype : numpy.dtype
            The numpy dtype to use for the vectors in the model (numpy.float64, numpy.float32 etc).
            Using lower precision floats may be useful in increasing training speed and reducing memory usage.
        seed : int, optional
            Seed for random to ensure reproducibility.

        Examples
        --------
        Initialize a model from a list:

        .. sourcecode:: pycon

            >>> from gensim.models.poincare import PoincareModel
            >>> relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
            >>> model = PoincareModel(relations, negative=2)

        Initialize a model from a file containing one relation per line:

        .. sourcecode:: pycon

            >>> from gensim.models.poincare import PoincareModel, PoincareRelations
            >>> from gensim.test.utils import datapath
            >>> file_path = datapath('poincare_hypernyms.tsv')
            >>> model = PoincareModel(PoincareRelations(file_path), negative=2)

        See :class:`~gensim.models.poincare.PoincareRelations` for more options.

        """
        self.train_data = train_data
        self.kv = PoincareKeyedVectors(size, 0)
        self.all_relations = []
        self.node_relations = defaultdict(set)
        self._negatives_buffer = NegativesBuffer([])
        self._negatives_buffer_size = 2000
        self.size = size
        self.train_alpha = alpha  # Learning rate for training
        self.burn_in_alpha = burn_in_alpha  # Learning rate for burn-in
        self.alpha = alpha  # Current learning rate
        self.negative = negative
        self.workers = workers
        self.epsilon = epsilon
        self.regularization_coeff = regularization_coeff
        self.burn_in = burn_in
        self._burn_in_done = False
        self.dtype = dtype
        self.seed = seed
        self._np_random = np_random.RandomState(seed)
        self.init_range = init_range
        self._loss_grad = None
        self.build_vocab(train_data)

    def build_vocab(self, relations, update=False):
        """Build the model's vocabulary from known relations.

        Parameters
        ----------
        relations : {iterable of (str, str), :class:`gensim.models.poincare.PoincareRelations`}
            Iterable of relations, e.g. a list of tuples, or a :class:`gensim.models.poincare.PoincareRelations`
            instance streaming from a file. Note that the relations are treated as ordered pairs,
            i.e. a relation (a, b) does not imply the opposite relation (b, a). In case the relations are symmetric,
            the data should contain both relations (a, b) and (b, a).
        update : bool, optional
            If true, only new nodes's embeddings are initialized.
            Use this when the model already has an existing vocabulary and you want to update it.
            If false, all node's embeddings are initialized.
            Use this when you're creating a new vocabulary from scratch.

        Examples
        --------
        Train a model and update vocab for online training:

        .. sourcecode:: pycon

            >>> from gensim.models.poincare import PoincareModel
            >>>
            >>> # train a new model from initial data
            >>> initial_relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal')]
            >>> model = PoincareModel(initial_relations, negative=1)
            >>> model.train(epochs=50)
            >>>
            >>> # online training: update the vocabulary and continue training
            >>> online_relations = [('striped_skunk', 'mammal')]
            >>> model.build_vocab(online_relations, update=True)
            >>> model.train(epochs=50)

        """
        old_index_to_key_len = len(self.kv.index_to_key)

        logger.info("loading relations from train data..")
        for relation in relations:
            if len(relation) != 2:
                raise ValueError('Relation pair "%s" should have exactly two items' % repr(relation))
            for item in relation:
                if item in self.kv.key_to_index:
                    self.kv.set_vecattr(item, 'count', self.kv.get_vecattr(item, 'count') + 1)
                else:
                    self.kv.key_to_index[item] = len(self.kv.index_to_key)
                    self.kv.index_to_key.append(item)
                    self.kv.set_vecattr(item, 'count', 1)

            node_1, node_2 = relation
            node_1_index, node_2_index = self.kv.key_to_index[node_1], self.kv.key_to_index[node_2]
            self.node_relations[node_1_index].add(node_2_index)
            relation = (node_1_index, node_2_index)
            self.all_relations.append(relation)
        logger.info("loaded %d relations from train data, %d nodes", len(self.all_relations), len(self.kv))
        self.indices_set = set(range(len(self.kv.index_to_key)))  # Set of all node indices
        self.indices_array = np.fromiter(range(len(self.kv.index_to_key)), dtype=int)  # Numpy array of all node indices
        self._init_node_probabilities()

        if not update:
            self._init_embeddings()
        else:
            self._update_embeddings(old_index_to_key_len)

    def _init_embeddings(self):
        """Randomly initialize vectors for the items in the vocab."""
        shape = (len(self.kv.index_to_key), self.size)
        self.kv.vectors = self._np_random.uniform(self.init_range[0], self.init_range[1], shape).astype(self.dtype)

    def _update_embeddings(self, old_index_to_key_len):
        """Randomly initialize vectors for the items in the additional vocab."""
        shape = (len(self.kv.index_to_key) - old_index_to_key_len, self.size)
        v = self._np_random.uniform(self.init_range[0], self.init_range[1], shape).astype(self.dtype)
        self.kv.vectors = np.concatenate([self.kv.vectors, v])

    def _init_node_probabilities(self):
        """Initialize a-priori probabilities."""
        counts = self.kv.expandos['count'].astype(np.float64)
        self._node_counts_cumsum = np.cumsum(counts)
        self._node_probabilities = counts / counts.sum()

    def _get_candidate_negatives(self):
        """Get candidate negatives of size `self.negative` from the negative examples buffer.

        Returns
        -------
        numpy.array
            Array of shape (`self.negative`,) containing indices of negative nodes.

        """
        if self._negatives_buffer.num_items() < self.negative:
            # cumsum table of counts used instead of the standard approach of a probability cumsum table
            # this is to avoid floating point errors that result when the number of nodes is very high
            # for reference: https://github.com/RaRe-Technologies/gensim/issues/1917
            max_cumsum_value = self._node_counts_cumsum[-1]
            uniform_numbers = self._np_random.randint(1, max_cumsum_value + 1, self._negatives_buffer_size)
            cumsum_table_indices = np.searchsorted(self._node_counts_cumsum, uniform_numbers)
            self._negatives_buffer = NegativesBuffer(cumsum_table_indices)
        return self._negatives_buffer.get_items(self.negative)

    def _sample_negatives(self, node_index):
        """Get a sample of negatives for the given node.

        Parameters
        ----------
        node_index : int
            Index of the positive node for which negative samples are to be returned.

        Returns
        -------
        numpy.array
            Array of shape (self.negative,) containing indices of negative nodes for the given node index.

        """
        node_relations = self.node_relations[node_index]
        num_remaining_nodes = len(self.kv) - len(node_relations)
        if num_remaining_nodes < self.negative:
            raise ValueError(
                'Cannot sample %d negative nodes from a set of %d negative nodes for %s' %
                (self.negative, num_remaining_nodes, self.kv.index_to_key[node_index])
            )

        positive_fraction = float(len(node_relations)) / len(self.kv)
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
                logger.debug('sampled %d times, positive fraction %.5f', times_sampled, positive_fraction)
        else:
            # If number of positive relations is a significant fraction of total nodes
            # subtract positively connected nodes from set of choices and sample from the remaining
            valid_negatives = np.array(list(self.indices_set - node_relations))
            probs = self._node_probabilities[valid_negatives]
            probs /= probs.sum()
            indices = self._np_random.choice(valid_negatives, size=self.negative, p=probs, replace=False)

        return list(indices)

    @staticmethod
    def _loss_fn(matrix, regularization_coeff=1.0):
        """Computes loss value.

        Parameters
        ----------
        matrix : numpy.array
            Array containing vectors for u, v and negative samples, of shape (2 + negative_size, dim).
        regularization_coeff : float, optional
            Coefficient to use for l2-regularization

        Returns
        -------
        float
            Computed loss value.

        Warnings
        --------
        Only used for autograd gradients, since autograd requires a specific function signature.

        """
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
        regularization_term = regularization_coeff * grad_np.linalg.norm(vectors_v[0]) ** 2
        return -grad_np.log(exp_negative_distances[0] / (exp_negative_distances.sum())) + regularization_term

    @staticmethod
    def _clip_vectors(vectors, epsilon):
        """Clip vectors to have a norm of less than one.

        Parameters
        ----------
        vectors : numpy.array
            Can be 1-D, or 2-D (in which case the norm for each row is checked).
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
        """Save complete model to disk, inherited from :class:`~gensim.utils.SaveLoad`.

        See also
        --------
        :meth:`~gensim.models.poincare.PoincareModel.load`

        Parameters
        ----------
        *args
            Positional arguments passed to :meth:`~gensim.utils.SaveLoad.save`.
        **kwargs
            Keyword arguments passed to :meth:`~gensim.utils.SaveLoad.save`.

        """
        self._loss_grad = None  # Can't pickle autograd fn to disk
        attrs_to_ignore = ['_node_probabilities', '_node_counts_cumsum']
        kwargs['ignore'] = set(list(kwargs.get('ignore', [])) + attrs_to_ignore)
        super(PoincareModel, self).save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        """Load model from disk, inherited from :class:`~gensim.utils.SaveLoad`.

        See also
        --------
        :meth:`~gensim.models.poincare.PoincareModel.save`

        Parameters
        ----------
        *args
            Positional arguments passed to :meth:`~gensim.utils.SaveLoad.load`.
        **kwargs
            Keyword arguments passed to :meth:`~gensim.utils.SaveLoad.load`.

        Returns
        -------
        :class:`~gensim.models.poincare.PoincareModel`
            The loaded model.

        """
        model = super(PoincareModel, cls).load(*args, **kwargs)
        model._init_node_probabilities()
        return model

    def _prepare_training_batch(self, relations, all_negatives, check_gradients=False):
        """Create a training batch and compute gradients and loss for the batch.

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
            Node indices, computed gradients and loss for the batch.

        """
        batch_size = len(relations)
        indices_u, indices_v = [], []
        for relation, negatives in zip(relations, all_negatives):
            u, v = relation
            indices_u.append(u)
            indices_v.append(v)
            indices_v.extend(negatives)

        vectors_u = self.kv.vectors[indices_u]
        vectors_v = self.kv.vectors[indices_v].reshape((batch_size, 1 + self.negative, self.size))
        vectors_v = vectors_v.swapaxes(0, 1).swapaxes(1, 2)
        batch = PoincareBatch(vectors_u, vectors_v, indices_u, indices_v, self.regularization_coeff)
        batch.compute_all()

        if check_gradients:
            self._check_gradients(relations, all_negatives, batch)

        return batch

    def _check_gradients(self, relations, all_negatives, batch, tol=1e-8):
        """Compare computed gradients for batch to autograd gradients.

        Parameters
        ----------
        relations : list of tuples
            List of tuples of positive examples of the form (node_1_index, node_2_index).
        all_negatives : list of lists
            List of lists of negative samples for each node_1 in the positive examples.
        batch : :class:`~gensim.models.poincare.PoincareBatch`
            Batch for which computed gradients are to be checked.
        tol : float, optional
            The maximum error between our computed gradients and the reference ones from autograd.

        """
        if not AUTOGRAD_PRESENT:
            logger.warning('autograd could not be imported, cannot do gradient checking')
            logger.warning('please install autograd to enable gradient checking')
            return

        if self._loss_grad is None:
            self._loss_grad = grad(PoincareModel._loss_fn)

        max_diff = 0.0
        for i, (relation, negatives) in enumerate(zip(relations, all_negatives)):
            u, v = relation
            auto_gradients = self._loss_grad(
                np.vstack((self.kv.vectors[u], self.kv.vectors[[v] + negatives])), self.regularization_coeff)
            computed_gradients = np.vstack((batch.gradients_u[:, i], batch.gradients_v[:, :, i]))
            diff = np.abs(auto_gradients - computed_gradients).max()
            if diff > max_diff:
                max_diff = diff
        logger.info('max difference between computed gradients and autograd gradients: %.10f', max_diff)
        assert max_diff < tol, (
            'Max difference between computed gradients and autograd gradients %.10f, '
            'greater than tolerance %.10f' % (max_diff, tol))

    def _sample_negatives_batch(self, nodes):
        """Get negative examples for each node.

        Parameters
        ----------
        nodes : iterable of int
            Iterable of node indices for which negative samples are to be returned.

        Returns
        -------
        list of lists
            Each inner list is a list of negative samples for a single node in the input list.

        """
        all_indices = [self._sample_negatives(node) for node in nodes]
        return all_indices

    def _train_on_batch(self, relations, check_gradients=False):
        """Perform training for a single training batch.

        Parameters
        ----------
        relations : list of tuples of (int, int)
            List of tuples of positive examples of the form (node_1_index, node_2_index).
        check_gradients : bool, optional
            Whether to compare the computed gradients to autograd gradients for this batch.

        Returns
        -------
        :class:`~gensim.models.poincare.PoincareBatch`
            The batch that was just trained on, contains computed loss for the batch.

        """
        all_negatives = self._sample_negatives_batch(relation[0] for relation in relations)
        batch = self._prepare_training_batch(relations, all_negatives, check_gradients)
        self._update_vectors_batch(batch)
        return batch

    @staticmethod
    def _handle_duplicates(vector_updates, node_indices):
        """Handle occurrences of multiple updates to the same node in a batch of vector updates.

        Parameters
        ----------
        vector_updates : numpy.array
            Array with each row containing updates to be performed on a certain node.
        node_indices : list of int
            Node indices on which the above updates are to be performed on.

        Notes
        -----
        Mutates the `vector_updates` array.

        Required because vectors[[2, 1, 2]] += np.array([-0.5, 1.0, 0.5]) performs only the last update
        on the row at index 2.

        """
        counts = Counter(node_indices)
        node_dict = defaultdict(list)
        for i, node_index in enumerate(node_indices):
            node_dict[node_index].append(i)
        for node_index, count in counts.items():
            if count == 1:
                continue
            positions = node_dict[node_index]
            # Move all updates to the same node to the last such update, zeroing all the others
            vector_updates[positions[-1]] = vector_updates[positions].sum(axis=0)
            vector_updates[positions[:-1]] = 0

    def _update_vectors_batch(self, batch):
        """Update vectors for nodes in the given batch.

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

        self.kv.vectors[indices_u] -= u_updates
        self.kv.vectors[indices_u] = self._clip_vectors(self.kv.vectors[indices_u], self.epsilon)

        v_updates = self.alpha * (batch.beta ** 2)[:, np.newaxis] / 4 * grad_v
        v_updates = v_updates.swapaxes(1, 2).swapaxes(0, 1)
        v_updates = v_updates.reshape(((1 + self.negative) * batch_size, self.size))
        self._handle_duplicates(v_updates, indices_v)

        self.kv.vectors[indices_v] -= v_updates
        self.kv.vectors[indices_v] = self._clip_vectors(self.kv.vectors[indices_v], self.epsilon)

    def train(self, epochs, batch_size=10, print_every=1000, check_gradients_every=None):
        """Train Poincare embeddings using loaded data and model parameters.

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

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.models.poincare import PoincareModel
            >>> relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
            >>> model = PoincareModel(relations, negative=2)
            >>> model.train(epochs=50)

        """
        if self.workers > 1:
            raise NotImplementedError("Multi-threaded version not implemented yet")
        # Some divide-by-zero results are handled explicitly
        old_settings = np.seterr(divide='ignore', invalid='ignore')

        logger.info(
            "training model of size %d with %d workers on %d relations for %d epochs and %d burn-in epochs, "
            "using lr=%.5f burn-in lr=%.5f negative=%d",
            self.size, self.workers, len(self.all_relations), epochs, self.burn_in,
            self.alpha, self.burn_in_alpha, self.negative
        )

        if self.burn_in > 0 and not self._burn_in_done:
            logger.info("starting burn-in (%d epochs)----------------------------------------", self.burn_in)
            self.alpha = self.burn_in_alpha
            self._train_batchwise(
                epochs=self.burn_in, batch_size=batch_size, print_every=print_every,
                check_gradients_every=check_gradients_every)
            self._burn_in_done = True
            logger.info("burn-in finished")

        self.alpha = self.train_alpha
        logger.info("starting training (%d epochs)----------------------------------------", epochs)
        self._train_batchwise(
            epochs=epochs, batch_size=batch_size, print_every=print_every,
            check_gradients_every=check_gradients_every)
        logger.info("training finished")

        np.seterr(**old_settings)

    def _train_batchwise(self, epochs, batch_size=10, print_every=1000, check_gradients_every=None):
        """Train Poincare embeddings using specified parameters.

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
                        'training on epoch %d, examples #%d-#%d, loss: %.2f'
                        % (epoch, i, i + batch_size, avg_loss))
                    logger.info(
                        'time taken for %d examples: %.2f s, %.2f examples / s'
                        % (print_every * batch_size, time_taken, speed))
                    last_time = time.time()
                    avg_loss = 0.0


class PoincareBatch:
    """Compute Poincare distances, gradients and loss for a training batch.

    Store intermediate state to avoid recomputing multiple times.

    """
    def __init__(self, vectors_u, vectors_v, indices_u, indices_v, regularization_coeff=1.0):
        """
        Initialize instance with sets of vectors for which distances are to be computed.

        Parameters
        ----------
        vectors_u : numpy.array
            Vectors of all nodes `u` in the batch. Expected shape (batch_size, dim).
        vectors_v : numpy.array
            Vectors of all positively related nodes `v` and negatively sampled nodes `v'`,
            for each node `u` in the batch. Expected shape (1 + neg_size, dim, batch_size).
        indices_u : list of int
            List of node indices for each of the vectors in `vectors_u`.
        indices_v : list of lists of int
            Nested list of lists, each of which is a  list of node indices
            for each of the vectors in `vectors_v` for a specific node `u`.
        regularization_coeff : float, optional
            Coefficient to use for l2-regularization

        """
        self.vectors_u = vectors_u.T[np.newaxis, :, :]  # (1, dim, batch_size)
        self.vectors_v = vectors_v  # (1 + neg_size, dim, batch_size)
        self.indices_u = indices_u
        self.indices_v = indices_v
        self.regularization_coeff = regularization_coeff

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

        # (1 + neg_size, dim, batch_size)
        gradients_v = -self.exp_negative_distances[:, np.newaxis, :] * self.distance_gradients_v
        gradients_v /= self.Z  # (1 + neg_size, dim, batch_size)
        gradients_v[0] += self.distance_gradients_v[0]
        gradients_v[0] += self.regularization_coeff * 2 * self.vectors_v[0]

        # (1 + neg_size, dim, batch_size)
        gradients_u = -self.exp_negative_distances[:, np.newaxis, :] * self.distance_gradients_u
        gradients_u /= self.Z  # (1 + neg_size, dim, batch_size)
        gradients_u = gradients_u.sum(axis=0)  # (dim, batch_size)
        gradients_u += self.distance_gradients_u[0]

        assert not np.isnan(gradients_u).any()
        assert not np.isnan(gradients_v).any()
        self.gradients_u = gradients_u
        self.gradients_v = gradients_v

        self._gradients_computed = True

    def compute_distance_gradients(self):
        """Compute and store partial derivatives of poincare distance d(u, v) w.r.t all u and all v."""
        if self._distance_gradients_computed:
            return
        self.compute_distances()

        euclidean_dists_squared = self.euclidean_dists ** 2  # (1 + neg_size, batch_size)
        # (1 + neg_size, 1, batch_size)
        c_ = (4 / (self.alpha * self.beta * np.sqrt(self.gamma ** 2 - 1)))[:, np.newaxis, :]
        # (1 + neg_size, 1, batch_size)
        u_coeffs = ((euclidean_dists_squared + self.alpha) / self.alpha)[:, np.newaxis, :]
        distance_gradients_u = u_coeffs * self.vectors_u - self.vectors_v  # (1 + neg_size, dim, batch_size)
        distance_gradients_u *= c_  # (1 + neg_size, dim, batch_size)

        nan_gradients = self.gamma == 1  # (1 + neg_size, batch_size)
        if nan_gradients.any():
            distance_gradients_u.swapaxes(1, 2)[nan_gradients] = 0
        self.distance_gradients_u = distance_gradients_u

        # (1 + neg_size, 1, batch_size)
        v_coeffs = ((euclidean_dists_squared + self.beta) / self.beta)[:, np.newaxis, :]
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
    """Vectors and vocab for the :class:`~gensim.models.poincare.PoincareModel` training class.

    Used to perform operations on the vectors such as vector lookup, distance calculations etc.

    (May be used to save/load final vectors in the plain word2vec format, via the inherited
    methods save_word2vec_format() and load_word2vec_format().)

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.test.utils import datapath
        >>>
        >>> # Read the sample relations file and train the model
        >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
        >>> model = PoincareModel(train_data=relations)
        >>> model.train(epochs=50)
        >>>
        >>> # Query the trained model.
        >>> wv = model.kv.get_vector('kangaroo.n.01')

    """
    def __init__(self, vector_size, vector_count, dtype=REAL):
        super(PoincareKeyedVectors, self).__init__(vector_size, vector_count, dtype=dtype)
        self.max_distance = 0

    def _load_specials(self, *args, **kwargs):
        super(PoincareKeyedVectors, self)._load_specials(*args, **kwargs)
        # fixup rename of syn0
        if not hasattr(self, 'vectors'):
            self.vectors = self.__dict__.pop('syn0')

    @staticmethod
    def vector_distance(vector_1, vector_2):
        """Compute poincare distance between two input vectors. Convenience method over `vector_distance_batch`.

        Parameters
        ----------
        vector_1 : numpy.array
            Input vector.
        vector_2 : numpy.array
            Input vector.

        Returns
        -------
        numpy.float
            Poincare distance between `vector_1` and `vector_2`.

        """
        return PoincareKeyedVectors.vector_distance_batch(vector_1, vector_2[np.newaxis, :])[0]

    @staticmethod
    def vector_distance_batch(vector_1, vectors_all):
        """Compute poincare distances between one vector and a set of other vectors.

        Parameters
        ----------
        vector_1 : numpy.array
            vector from which Poincare distances are to be computed, expected shape (dim,).
        vectors_all : numpy.array
            for each row in vectors_all, distance from vector_1 is computed, expected shape (num_vectors, dim).

        Returns
        -------
        numpy.array
            Poincare distance between `vector_1` and each row in `vectors_all`, shape (num_vectors,).

        """
        euclidean_dists = np.linalg.norm(vector_1 - vectors_all, axis=1)
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)
        return np.arccosh(
            1 + 2 * (
                (euclidean_dists ** 2) / ((1 - norm ** 2) * (1 - all_norms ** 2))
            )
        )

    def closest_child(self, node):
        """Get the node closest to `node` that is lower in the hierarchy than `node`.

        Parameters
        ----------
        node : {str, int}
            Key for node for which closest child is to be found.

        Returns
        -------
        {str, None}
            Node closest to `node` that is lower in the hierarchy than `node`.
            If there are no nodes lower in the hierarchy, None is returned.

        """
        all_distances = self.distances(node)
        all_norms = np.linalg.norm(self.vectors, axis=1)
        node_norm = all_norms[self.get_index(node)]
        mask = node_norm >= all_norms
        if mask.all():  # No nodes lower in the hierarchy
            return None
        all_distances = np.ma.array(all_distances, mask=mask)
        closest_child_index = np.ma.argmin(all_distances)
        return self.index_to_key[closest_child_index]

    def closest_parent(self, node):
        """Get the node closest to `node` that is higher in the hierarchy than `node`.

        Parameters
        ----------
        node : {str, int}
            Key for node for which closest parent is to be found.

        Returns
        -------
        {str, None}
            Node closest to `node` that is higher in the hierarchy than `node`.
            If there are no nodes higher in the hierarchy, None is returned.

        """
        all_distances = self.distances(node)
        all_norms = np.linalg.norm(self.vectors, axis=1)
        node_norm = all_norms[self.get_index(node)]
        mask = node_norm <= all_norms
        if mask.all():  # No nodes higher in the hierarchy
            return None
        all_distances = np.ma.array(all_distances, mask=mask)
        closest_child_index = np.ma.argmin(all_distances)
        return self.index_to_key[closest_child_index]

    def descendants(self, node, max_depth=5):
        """Get the list of recursively closest children from the given node, up to a max depth of `max_depth`.

        Parameters
        ----------
        node : {str, int}
            Key for node for which descendants are to be found.
        max_depth : int
            Maximum number of descendants to return.

        Returns
        -------
        list of str
            Descendant nodes from the node `node`.

        """
        depth = 0
        descendants = []
        current_node = node
        while depth < max_depth:
            descendants.append(self.closest_child(current_node))
            current_node = descendants[-1]
            depth += 1
        return descendants

    def ancestors(self, node):
        """Get the list of recursively closest parents from the given node.

        Parameters
        ----------
        node : {str, int}
            Key for node for which ancestors are to be found.

        Returns
        -------
        list of str
            Ancestor nodes of the node `node`.

        """
        ancestors = []
        current_node = node
        ancestor = self.closest_parent(current_node)
        while ancestor is not None:
            ancestors.append(ancestor)
            ancestor = self.closest_parent(ancestors[-1])
        return ancestors

    def distance(self, w1, w2):
        """Calculate Poincare distance between vectors for nodes `w1` and `w2`.

        Parameters
        ----------
        w1 : {str, int}
            Key for first node.
        w2 : {str, int}
            Key for second node.

        Returns
        -------
        float
            Poincare distance between the vectors for nodes `w1` and `w2`.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # What is the distance between the words 'mammal' and 'carnivore'?
            >>> model.kv.distance('mammal.n.01', 'carnivore.n.01')
            2.9742298803339304

        Raises
        ------
        KeyError
            If either of `w1` and `w2` is absent from vocab.

        """
        vector_1 = self.get_vector(w1)
        vector_2 = self.get_vector(w2)
        return self.vector_distance(vector_1, vector_2)

    def similarity(self, w1, w2):
        """Compute similarity based on Poincare distance between vectors for nodes `w1` and `w2`.

        Parameters
        ----------
        w1 : {str, int}
            Key for first node.
        w2 : {str, int}
            Key for second node.

        Returns
        -------
        float
            Similarity between the between the vectors for nodes `w1` and `w2` (between 0 and 1).

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # What is the similarity between the words 'mammal' and 'carnivore'?
            >>> model.kv.similarity('mammal.n.01', 'carnivore.n.01')
            0.25162107631176484

        Raises
        ------
        KeyError
            If either of `w1` and `w2` is absent from vocab.

        """
        return 1 / (1 + self.distance(w1, w2))

    def most_similar(self, node_or_vector, topn=10, restrict_vocab=None):
        """Find the top-N most similar nodes to the given node or vector, sorted in increasing order of distance.

        Parameters
        ----------
        node_or_vector : {str, int, numpy.array}
            node key or vector for which similar nodes are to be found.
        topn : int or None, optional
            Number of top-N similar nodes to return, when `topn` is int. When `topn` is None,
            then distance for all nodes are returned.
        restrict_vocab : int or None, optional
            Optional integer which limits the range of vectors which are searched for most-similar values.
            For example, restrict_vocab=10000 would only check the first 10000 node vectors in the vocabulary order.
            This may be meaningful if vocabulary is sorted by descending frequency.

        Returns
        --------
        list of (str, float) or numpy.array
            When `topn` is int, a sequence of (node, distance) is returned in increasing order of distance.
            When `topn` is None, then similarities for all words are returned as a one-dimensional numpy array with the
            size of the vocabulary.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # Which words are most similar to 'kangaroo'?
            >>> model.kv.most_similar('kangaroo.n.01', topn=2)
            [(u'kangaroo.n.01', 0.0), (u'marsupial.n.01', 0.26524229460827725)]

        """
        if isinstance(topn, Integral) and topn < 1:
            return []

        if not restrict_vocab:
            all_distances = self.distances(node_or_vector)
        else:
            nodes_to_use = self.index_to_key[:restrict_vocab]
            all_distances = self.distances(node_or_vector, nodes_to_use)

        if isinstance(node_or_vector, (str, int,)):
            node_index = self.get_index(node_or_vector)
        else:
            node_index = None
        if not topn:
            closest_indices = matutils.argsort(all_distances)
        else:
            closest_indices = matutils.argsort(all_distances, topn=1 + topn)
        result = [
            (self.index_to_key[index], float(all_distances[index]))
            for index in closest_indices if (not node_index or index != node_index)  # ignore the input node
        ]
        if topn:
            result = result[:topn]
        return result

    def distances(self, node_or_vector, other_nodes=()):
        """Compute Poincare distances from given `node_or_vector` to all nodes in `other_nodes`.
        If `other_nodes` is empty, return distance between `node_or_vector` and all nodes in vocab.

        Parameters
        ----------
        node_or_vector : {str, int, numpy.array}
            Node key or vector from which distances are to be computed.
        other_nodes : {iterable of str, iterable of int, None}, optional
            For each node in `other_nodes` distance from `node_or_vector` is computed.
            If None or empty, distance of `node_or_vector` from all nodes in vocab is computed (including itself).

        Returns
        -------
        numpy.array
            Array containing distances to all nodes in `other_nodes` from input `node_or_vector`,
            in the same order as `other_nodes`.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # Check the distances between a word and a list of other words.
            >>> model.kv.distances('mammal.n.01', ['carnivore.n.01', 'dog.n.01'])
            array([2.97422988, 2.83007402])

            >>> # Check the distances between a word and every other word in the vocab.
            >>> all_distances = model.kv.distances('mammal.n.01')

        Raises
        ------
        KeyError
            If either `node_or_vector` or any node in `other_nodes` is absent from vocab.

        """
        if isinstance(node_or_vector, str):
            input_vector = self.get_vector(node_or_vector)
        else:
            input_vector = node_or_vector
        if not other_nodes:
            other_vectors = self.vectors
        else:
            other_indices = [self.get_index(node) for node in other_nodes]
            other_vectors = self.vectors[other_indices]
        return self.vector_distance_batch(input_vector, other_vectors)

    def norm(self, node_or_vector):
        """Compute absolute position in hierarchy of input node or vector.
        Values range between 0 and 1. A lower value indicates the input node or vector is higher in the hierarchy.

        Parameters
        ----------
        node_or_vector : {str, int, numpy.array}
            Input node key or vector for which position in hierarchy is to be returned.

        Returns
        -------
        float
            Absolute position in the hierarchy of the input vector or node.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> # Get the norm of the embedding of the word `mammal`.
            >>> model.kv.norm('mammal.n.01')
            0.6423008703542398

        Notes
        -----
        The position in hierarchy is based on the norm of the vector for the node.

        """
        if isinstance(node_or_vector, str):
            input_vector = self.get_vector(node_or_vector)
        else:
            input_vector = node_or_vector
        return np.linalg.norm(input_vector)

    def difference_in_hierarchy(self, node_or_vector_1, node_or_vector_2):
        """Compute relative position in hierarchy of `node_or_vector_1` relative to `node_or_vector_2`.
        A positive value indicates `node_or_vector_1` is higher in the hierarchy than `node_or_vector_2`.

        Parameters
        ----------
        node_or_vector_1 : {str, int, numpy.array}
            Input node key or vector.
        node_or_vector_2 : {str, int, numpy.array}
            Input node key or vector.

        Returns
        -------
        float
            Relative position in hierarchy of `node_or_vector_1` relative to `node_or_vector_2`.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>>
            >>> # Read the sample relations file and train the model
            >>> relations = PoincareRelations(file_path=datapath('poincare_hypernyms_large.tsv'))
            >>> model = PoincareModel(train_data=relations)
            >>> model.train(epochs=50)
            >>>
            >>> model.kv.difference_in_hierarchy('mammal.n.01', 'dog.n.01')
            0.05382517902410999

            >>> model.kv.difference_in_hierarchy('dog.n.01', 'mammal.n.01')
            -0.05382517902410999

        Notes
        -----
        The returned value can be positive or negative, depending on whether `node_or_vector_1` is higher
        or lower in the hierarchy than `node_or_vector_2`.

        """
        return self.norm(node_or_vector_2) - self.norm(node_or_vector_1)


class PoincareRelations:
    """Stream relations for `PoincareModel` from a tsv-like file."""

    def __init__(self, file_path, encoding='utf8', delimiter='\t'):
        """Initialize instance from file containing a pair of nodes (a relation) per line.

        Parameters
        ----------
        file_path : str
            Path to file containing a pair of nodes (a relation) per line, separated by `delimiter`.
            Since the relations are asymmetric, the order of `u` and `v` nodes in each pair matters.
            To express a "u is v" relation, the lines should take the form `u delimeter v`.
            e.g: `kangaroo	mammal` is a tab-delimited line expressing a "`kangaroo is a mammal`" relation.

            For a full input file example, see `gensim/test/test_data/poincare_hypernyms.tsv
            <https://github.com/RaRe-Technologies/gensim/blob/master/gensim/test/test_data/poincare_hypernyms.tsv>`_.
        encoding : str, optional
            Character encoding of the input file.
        delimiter : str, optional
            Delimiter character for each relation.

        """

        self.file_path = file_path
        self.encoding = encoding
        self.delimiter = delimiter

    def __iter__(self):
        """Stream relations from self.file_path decoded into unicode strings.

        Yields
        -------
        (unicode, unicode)
            Relation from input file.

        """
        with utils.open(self.file_path, 'rb') as file_obj:
            if sys.version_info[0] < 3:
                lines = file_obj
            else:
                lines = (line.decode(self.encoding) for line in file_obj)
            # csv.reader requires bytestring input in python2, unicode input in python3
            reader = csv.reader(lines, delimiter=self.delimiter)
            for row in reader:
                if sys.version_info[0] < 3:
                    row = [value.decode(self.encoding) for value in row]
                yield tuple(row)


class NegativesBuffer:
    """Buffer and return negative samples."""

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
        """Get the number of items remaining in the buffer.

        Returns
        -------
        int
            Number of items in the buffer that haven't been consumed yet.

        """
        return len(self._items) - self._current_index

    def get_items(self, num_items):
        """Get the next `num_items` from buffer.

        Parameters
        ----------
        num_items : int
            Number of items to fetch.

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


class ReconstructionEvaluation:
    """Evaluate reconstruction on given network for given embedding."""

    def __init__(self, file_path, embedding):
        """Initialize evaluation instance with tsv file containing relation pairs and embedding to be evaluated.

        Parameters
        ----------
        file_path : str
            Path to tsv file containing relation pairs.
        embedding : :class:`~gensim.models.poincare.PoincareKeyedVectors`
            Embedding to be evaluated.

        """
        items = set()
        relations = defaultdict(set)
        with utils.open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                assert len(row) == 2, 'Hypernym pair has more than two items'
                item_1_index = embedding.get_index(row[0])
                item_2_index = embedding.get_index(row[1])
                relations[item_1_index].add(item_2_index)
                items.update([item_1_index, item_2_index])
        self.items = items
        self.relations = relations
        self.embedding = embedding

    @staticmethod
    def get_positive_relation_ranks_and_avg_prec(all_distances, positive_relations):
        """Compute ranks and Average Precision of positive relations.

        Parameters
        ----------
        all_distances : numpy.array of float
            Array of all distances (floats) for a specific item.
        positive_relations : list
            List of indices of positive relations for the item.

        Returns
        -------
        (list of int, float)
            The list contains ranks of positive relations in the same order as `positive_relations`.
            The float is the Average Precision of the ranking, e.g. ([1, 2, 3, 20], 0.610).

        """
        positive_relation_distances = all_distances[positive_relations]
        negative_relation_distances = np.ma.array(all_distances, mask=False)
        negative_relation_distances.mask[positive_relations] = True
        # Compute how many negative relation distances are less than each positive relation distance, plus 1 for rank
        ranks = (negative_relation_distances < positive_relation_distances[:, np.newaxis]).sum(axis=1) + 1
        map_ranks = np.sort(ranks) + np.arange(len(ranks))
        avg_precision = ((np.arange(1, len(map_ranks) + 1) / np.sort(map_ranks)).mean())
        return list(ranks), avg_precision

    def evaluate(self, max_n=None):
        """Evaluate all defined metrics for the reconstruction task.

        Parameters
        ----------
        max_n : int, optional
            Maximum number of positive relations to evaluate, all if `max_n` is None.

        Returns
        -------
        dict of (str, float)
            (metric_name, metric_value) pairs, e.g. {'mean_rank': 50.3, 'MAP': 0.31}.

        """
        mean_rank, map_ = self.evaluate_mean_rank_and_map(max_n)
        return {'mean_rank': mean_rank, 'MAP': map_}

    def evaluate_mean_rank_and_map(self, max_n=None):
        """Evaluate mean rank and MAP for reconstruction.

        Parameters
        ----------
        max_n : int, optional
            Maximum number of positive relations to evaluate, all if `max_n` is None.

        Returns
        -------
        (float, float)
            (mean_rank, MAP), e.g (50.3, 0.31).

        """
        ranks = []
        avg_precision_scores = []
        for i, item in enumerate(self.items, start=1):
            if item not in self.relations:
                continue
            item_relations = list(self.relations[item])
            item_term = self.embedding.index_to_key[item]
            item_distances = self.embedding.distances(item_term)
            positive_relation_ranks, avg_precision = \
                self.get_positive_relation_ranks_and_avg_prec(item_distances, item_relations)
            ranks += positive_relation_ranks
            avg_precision_scores.append(avg_precision)
            if max_n is not None and i > max_n:
                break
        return np.mean(ranks), np.mean(avg_precision_scores)


class LinkPredictionEvaluation:
    """Evaluate reconstruction on given network for given embedding."""

    def __init__(self, train_path, test_path, embedding):
        """Initialize evaluation instance with tsv file containing relation pairs and embedding to be evaluated.

        Parameters
        ----------
        train_path : str
            Path to tsv file containing relation pairs used for training.
        test_path : str
            Path to tsv file containing relation pairs to evaluate.
        embedding : :class:`~gensim.models.poincare.PoincareKeyedVectors`
            Embedding to be evaluated.

        """
        items = set()
        relations = {'known': defaultdict(set), 'unknown': defaultdict(set)}
        data_files = {'known': train_path, 'unknown': test_path}
        for relation_type, data_file in data_files.items():
            with utils.open(data_file, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    assert len(row) == 2, 'Hypernym pair has more than two items'
                    item_1_index = embedding.get_index(row[0])
                    item_2_index = embedding.get_index(row[1])
                    relations[relation_type][item_1_index].add(item_2_index)
                    items.update([item_1_index, item_2_index])
        self.items = items
        self.relations = relations
        self.embedding = embedding

    @staticmethod
    def get_unknown_relation_ranks_and_avg_prec(all_distances, unknown_relations, known_relations):
        """Compute ranks and Average Precision of unknown positive relations.

        Parameters
        ----------
        all_distances : numpy.array of float
            Array of all distances for a specific item.
        unknown_relations : list of int
            List of indices of unknown positive relations.
        known_relations : list of int
            List of indices of known positive relations.

        Returns
        -------
        tuple (list of int, float)
            The list contains ranks of positive relations in the same order as `positive_relations`.
            The float is the Average Precision of the ranking, e.g. ([1, 2, 3, 20], 0.610).

        """
        unknown_relation_distances = all_distances[unknown_relations]
        negative_relation_distances = np.ma.array(all_distances, mask=False)
        negative_relation_distances.mask[unknown_relations] = True
        negative_relation_distances.mask[known_relations] = True
        # Compute how many negative relation distances are less than each unknown relation distance, plus 1 for rank
        ranks = (negative_relation_distances < unknown_relation_distances[:, np.newaxis]).sum(axis=1) + 1
        map_ranks = np.sort(ranks) + np.arange(len(ranks))
        avg_precision = ((np.arange(1, len(map_ranks) + 1) / np.sort(map_ranks)).mean())
        return list(ranks), avg_precision

    def evaluate(self, max_n=None):
        """Evaluate all defined metrics for the link prediction task.

        Parameters
        ----------
        max_n : int, optional
            Maximum number of positive relations to evaluate, all if `max_n` is None.

        Returns
        -------
        dict of (str, float)
            (metric_name, metric_value) pairs, e.g. {'mean_rank': 50.3, 'MAP': 0.31}.

        """
        mean_rank, map_ = self.evaluate_mean_rank_and_map(max_n)
        return {'mean_rank': mean_rank, 'MAP': map_}

    def evaluate_mean_rank_and_map(self, max_n=None):
        """Evaluate mean rank and MAP for link prediction.

        Parameters
        ----------
        max_n : int, optional
            Maximum number of positive relations to evaluate, all if `max_n` is None.

        Returns
        -------
        tuple (float, float)
            (mean_rank, MAP), e.g (50.3, 0.31).

        """
        ranks = []
        avg_precision_scores = []
        for i, item in enumerate(self.items, start=1):
            if item not in self.relations['unknown']:  # No positive relations to predict for this node
                continue
            unknown_relations = list(self.relations['unknown'][item])
            known_relations = list(self.relations['known'][item])
            item_term = self.embedding.index_to_key[item]
            item_distances = self.embedding.distances(item_term)
            unknown_relation_ranks, avg_precision = \
                self.get_unknown_relation_ranks_and_avg_prec(item_distances, unknown_relations, known_relations)
            ranks += unknown_relation_ranks
            avg_precision_scores.append(avg_precision)
            if max_n is not None and i > max_n:
                break
        return np.mean(ranks), np.mean(avg_precision_scores)


class LexicalEntailmentEvaluation:
    """Evaluate reconstruction on given network for any embedding."""

    def __init__(self, filepath):
        """Initialize evaluation instance with HyperLex text file containing relation pairs.

        Parameters
        ----------
        filepath : str
            Path to HyperLex text file.

        """
        expected_scores = {}
        with utils.open(filepath, 'r') as f:
            reader = csv.DictReader(f, delimiter=' ')
            for row in reader:
                word_1, word_2 = row['WORD1'], row['WORD2']
                expected_scores[(word_1, word_2)] = float(row['AVG_SCORE'])
        self.scores = expected_scores
        self.alpha = 1000

    def score_function(self, embedding, trie, term_1, term_2):
        """Compute predicted score - extent to which `term_1` is a type of `term_2`.

        Parameters
        ----------
        embedding : :class:`~gensim.models.poincare.PoincareKeyedVectors`
            Embedding to use for computing predicted score.
        trie : :class:`pygtrie.Trie`
            Trie to use for finding matching vocab terms for input terms.
        term_1 : str
            Input term.
        term_2 : str
            Input term.

        Returns
        -------
        float
            Predicted score (the extent to which `term_1` is a type of `term_2`).

        """
        try:
            word_1_terms = self.find_matching_terms(trie, term_1)
            word_2_terms = self.find_matching_terms(trie, term_2)
        except KeyError:
            raise ValueError("No matching terms found for either %s or %s" % (term_1, term_2))
        min_distance = np.inf
        min_term_1, min_term_2 = None, None
        for term_1 in word_1_terms:
            for term_2 in word_2_terms:
                distance = embedding.distance(term_1, term_2)
                if distance < min_distance:
                    min_term_1, min_term_2 = term_1, term_2
                    min_distance = distance
        assert min_term_1 is not None and min_term_2 is not None
        vector_1, vector_2 = embedding.get_vector(min_term_1), embedding.get_vector(min_term_2)
        norm_1, norm_2 = np.linalg.norm(vector_1), np.linalg.norm(vector_2)
        return -1 * (1 + self.alpha * (norm_2 - norm_1)) * min_distance

    @staticmethod
    def find_matching_terms(trie, word):
        """Find terms in the `trie` beginning with the `word`.

        Parameters
        ----------
        trie : :class:`pygtrie.Trie`
            Trie to use for finding matching terms.
        word : str
            Input word to use for prefix search.

        Returns
        -------
        list of str
            List of matching terms.

        """
        matches = trie.items('%s.' % word)
        matching_terms = [''.join(key_chars) for key_chars, value in matches]
        return matching_terms

    @staticmethod
    def create_vocab_trie(embedding):
        """Create trie with vocab terms of the given embedding to enable quick prefix searches.

        Parameters
        ----------
        embedding : :class:`~gensim.models.poincare.PoincareKeyedVectors`
            Embedding for which trie is to be created.

        Returns
        -------
        :class:`pygtrie.Trie`
            Trie containing vocab terms of the input embedding.

        """
        try:
            from pygtrie import Trie
        except ImportError:
            raise ImportError(
                'pygtrie could not be imported, please install pygtrie in order to use LexicalEntailmentEvaluation')

        vocab_trie = Trie()
        for key in embedding.key_to_index:
            vocab_trie[key] = True
        return vocab_trie

    def evaluate_spearman(self, embedding):
        """Evaluate spearman scores for lexical entailment for given embedding.

        Parameters
        ----------
        embedding : :class:`~gensim.models.poincare.PoincareKeyedVectors`
            Embedding for which evaluation is to be done.

        Returns
        -------
        float
            Spearman correlation score for the task for input embedding.

        """
        predicted_scores = []
        expected_scores = []
        skipped = 0
        count = 0
        vocab_trie = self.create_vocab_trie(embedding)
        for (word_1, word_2), expected_score in self.scores.items():
            try:
                predicted_score = self.score_function(embedding, vocab_trie, word_1, word_2)
            except ValueError:
                skipped += 1
                continue
            count += 1
            predicted_scores.append(predicted_score)
            expected_scores.append(expected_score)
        logger.info('skipped pairs: %d out of %d' % (skipped, len(self.scores)))
        spearman = spearmanr(expected_scores, predicted_scores)
        return spearman.correlation
