#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from abc import ABCMeta, abstractmethod
from gensim import utils


# Public Interfaces
class BaseAny2VecModel(utils.SaveLoad):

    def __init__(self, data_iterator, **kwargs):
        """Initialize model parameters."""
        raise NotImplementedError

    def train(self, data_iterator, **kwargs):
        """Handles multiworker training."""
        raise NotImplementedError

    def build_vocab(self, data_iterator, **kwargs):
        """Scan through the vocab and create/update vocabulary.
        Should also initialize/reset vectors for new vocab entities."""
        raise NotImplementedError

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        model = super(BaseAny2VecModel, cls).load(fname_or_handle, **kwargs)
        return model

    def save(self, fname_or_handle, **kwargs):
        super(BaseAny2VecModel, self).save(fname_or_handle, **kwargs)


class BaseKeyedVectors(utils.SaveLoad):

    def __init__(self):
        self.syn0 = []
        self.vocab = {}
        self.index2entity = []
        self.vector_size = None

    def save(self, fname_or_handle, **kwargs):
        super(BaseKeyedVectors, self).save(fname_or_handle, **kwargs)

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        return super(BaseKeyedVectors, cls).load(fname_or_handle, **kwargs)

    def similarity(self, e1, e2):
        """Compute similarity between vectors of two input entities (words, documents, sentences etc.).
        To be implemented by child class.
        """
        raise NotImplementedError

    def most_similar(self, e1, **kwargs):
        """Find the top-N most similar entities.
        Possibly have `postive` and `negative` list of entities in `**kwargs`."""
        return NotImplementedError

    def distance(self, e1, e2):
        """Compute distance between vectors of two input words.
        To be implemented by child class.
        """
        raise NotImplementedError

    def distances(self, entity_or_vector, other_entities=()):
        """Compute distances from given entity or vector to all words in `other_entity`.
        If `other_entities` is empty, return distance between `entity_or_vectors` and all entities in vocab.
        To be implemented by child class.
        """
        raise NotImplementedError

    def get_vector(self, entity):
        """Accept a single entity as input.
        Returns the word's representations in vector space, as a 1D numpy array.
        """
        raise NotImplementedError

    def most_similar_to_given(self, e1, entities_list):
        """Return the entity from entities_list most similar to e1."""
        raise NotImplementedError

    def entities_closer_than(self, e1, e2):
        """Returns all words that are closer to `e1` than `e2` is to `e1`."""
        raise NotImplementedError

    def rank(self, e1, e2):
        """Rank of the distance of `e2` from `e1`, in relation to distances of all entities from `e1`."""
        raise NotImplementedError


class VocabItem(object):
    """A single vocabulary item, used internally for collecting per-entity frequency, and it's mapped index."""

    def __init__(self, count, index):
        self.count = count
        self.index = index

# class BaseVocabBuilder(object):
#     """Base class to handle building and updating vocabulary (of any entity)."""

#     __metaclass__ = ABCMeta

#     @abstractmethod
#     def scan_vocab(self, entites, **kwargs):
#         """Do an initial scan of all entities appearing in iterator."""
#         pass

#     def reset_weights(self):
#         """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
#         return

#     def update_weights(self):
#         """Copy all the existing weights, and reset the weights for the newly added vocabulary."""
#         return

#     def sort_vocab(self):
#         """Sort the vocabulary so the most frequent words have the lowest indexes."""
#         return

#     def scale_vocab(self, **kwargs):
#         """Apply vocabulary settings for `min_count` (discarding less-frequent entities)
#         and `sample` (controlling the downsampling of more-frequent entities).
#         """
#         return

#     def finalize_vocab(self, **kwargs):
#         """Initialize model weights based on final vocabulary settings."""
#         return

#     def build_vocab(self, entites, **kwargs):
#         """Build vocabulary. Internally calls `scan_vocab`, `scale_vocab` and `finalize_vocab`."""
#         return


# class BaseVectorTrainer(object):
#     """Base class for training any2vec model. This handles feeding data into queues and
#     multi workers learning of vectors.
#     """
#     __metaclass__ = ABCMeta

#     @abstractmethod
#     def _do_train_job(self, *args):
#         """Train single batch of entities."""
#         return

#     def train(self, *ars, **kwargs):
#         """Provide implementation for learning vectors using job producer and `worker_loop`."""
#         return


# class BaseWord2VecTypeModel(BaseVocabBuilder, BaseVectorTrainer):
#     """Base Class for all "Word2Vec like" algorithms -- which use sg, cbow architecture with neg/hs loss.
#     Should contain all repeated/reused code for presently implemented algorithms -- Doc2Vec, FastText, Word2Vec.
#     """
#     def __init__(self, *args, **kwargs):
#         """Initialize common parameters."""
#         return

#     def train_sg_pair(*args, **kwagrs):
#         return

#     def train_cbow_pair(*args, **kwagrs):
#         return
