#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod


class BaseVocabBuilder(object):
    """Base class to handle building and updating vocabulary (of any entity)."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def scan_vocab(self, entites, **kwargs):
        """Do an initial scan of all entities appearing in iterator."""
        pass

    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        return

    def update_weights(self):
        """Copy all the existing weights, and reset the weights for the newly added vocabulary."""
        return

    def sort_vocab(self):
        """Sort the vocabulary so the most frequent words have the lowest indexes."""
        return

    def scale_vocab(self, **kwargs):
        """Apply vocabulary settings for `min_count` (discarding less-frequent entities)
        and `sample` (controlling the downsampling of more-frequent entities).
        """
        return

    def finalize_vocab(self, **kwargs):
        """Initialize model weights based on final vocabulary settings."""
        return

    def build_vocab(self, entites, **kwargs):
        """Build vocabulary. Internally calls `scan_vocab`, `scale_vocab` and `finalize_vocab`."""
        return


class BaseVectorTrainer(object):
    """Base class for training any2vec model. This handles feeding data into queues and
    multi workers learning of vectors.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def _do_train_job(self, *args):
        """Train single batch of entities."""
        return

    def train(self, *ars, **kwargs):
        """Provide implementation for learning vectors using job producer and `worker_loop`."""
        return


class BaseWord2VecTypeModel(BaseVocabBuilder, BaseVectorTrainer):
    """Base Class for all "Word2Vec like" algorithms -- which use sg, cbow architecture with neg/hs loss.
    Should contain all repeated/reused code for presently implemented algorithms -- Doc2Vec, FastText, Word2Vec.
    """
    def __init__(self, *args, **kwargs):
        """Initialize common parameters."""
        return

    def train_sg_pair(*args, **kwagrs):
        return

    def train_cbow_pair(*args, **kwagrs):
        return
