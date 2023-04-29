#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Gensim Contributors
# Copyright (C) 2018 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""This module implements word vectors, and more generally sets of vectors keyed by lookup tokens/ints,
 and various similarity look-ups.

Since trained word vectors are independent from the way they were trained (:class:`~gensim.models.word2vec.Word2Vec`,
:class:`~gensim.models.fasttext.FastText` etc), they can be represented by a standalone structure,
as implemented in this module.

The structure is called "KeyedVectors" and is essentially a mapping between *keys*
and *vectors*. Each vector is identified by its lookup key, most often a short string token, so this is usually
a mapping between {str => 1D numpy array}.

The key is, in the original motivating case, a word (so the mapping maps words to 1D vectors),
but for some models, the key can also correspond to a document, a graph node etc.

(Because some applications may maintain their own integral identifiers, compact and contiguous
starting at zero, this class also supports use of plain ints as keys – in that case using them as literal
pointers to the position of the desired vector in the underlying array, and saving the overhead of
a lookup map entry.)

Why use KeyedVectors instead of a full model?
=============================================

+---------------------------+--------------+------------+-------------------------------------------------------------+
|        capability         | KeyedVectors | full model |                               note                          |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| continue training vectors | ❌           | ✅         | You need the full model to train or update vectors.         |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| smaller objects           | ✅           | ❌         | KeyedVectors are smaller and need less RAM, because they    |
|                           |              |            | don't need to store the model state that enables training.  |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| save/load from native     |              |            | Vectors exported by the Facebook and Google tools           |
| fasttext/word2vec format  | ✅           | ❌         | do not support further training, but you can still load     |
|                           |              |            | them into KeyedVectors.                                     |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| append new vectors        | ✅           | ✅         | Add new-vector entries to the mapping dynamically.          |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| concurrency               | ✅           | ✅         | Thread-safe, allows concurrent vector queries.              |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| shared RAM                | ✅           | ✅         | Multiple processes can re-use the same data, keeping only   |
|                           |              |            | a single copy in RAM using                                  |
|                           |              |            | `mmap <https://en.wikipedia.org/wiki/Mmap>`_.               |
+---------------------------+--------------+------------+-------------------------------------------------------------+
| fast load                 | ✅           | ✅         | Supports `mmap <https://en.wikipedia.org/wiki/Mmap>`_       |
|                           |              |            | to load data from disk instantaneously.                     |
+---------------------------+--------------+------------+-------------------------------------------------------------+

TL;DR: the main difference is that KeyedVectors do not support further training.
On the other hand, by shedding the internal data structures necessary for training, KeyedVectors offer a smaller RAM
footprint and a simpler interface.

How to obtain word vectors?
===========================

Train a full model, then access its `model.wv` property, which holds the standalone keyed vectors.
For example, using the Word2Vec algorithm to train the vectors

.. sourcecode:: pycon

    >>> from gensim.test.utils import lee_corpus_list
    >>> from gensim.models import Word2Vec
    >>>
    >>> model = Word2Vec(lee_corpus_list, vector_size=24, epochs=100)
    >>> word_vectors = model.wv

Persist the word vectors to disk with

.. sourcecode:: pycon

    >>> from gensim.models import KeyedVectors
    >>>
    >>> word_vectors.save('vectors.kv')
    >>> reloaded_word_vectors = KeyedVectors.load('vectors.kv')

The vectors can also be instantiated from an existing file on disk
in the original Google's word2vec C format as a KeyedVectors instance

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> wv_from_text = KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)  # C text format
    >>> wv_from_bin = KeyedVectors.load_word2vec_format(datapath("euclidean_vectors.bin"), binary=True)  # C bin format

What can I do with word vectors?
================================

You can perform various syntactic/semantic NLP word tasks with the trained vectors.
Some of them are already built-in

.. sourcecode:: pycon

    >>> import gensim.downloader as api
    >>>
    >>> word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data
    >>>
    >>> # Check the "most similar words", using the default "cosine similarity" measure.
    >>> result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
    >>> most_similar_key, similarity = result[0]  # look at the first match
    >>> print(f"{most_similar_key}: {similarity:.4f}")
    queen: 0.7699
    >>>
    >>> # Use a different similarity measure: "cosmul".
    >>> result = word_vectors.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
    >>> most_similar_key, similarity = result[0]  # look at the first match
    >>> print(f"{most_similar_key}: {similarity:.4f}")
    queen: 0.8965
    >>>
    >>> print(word_vectors.doesnt_match("breakfast cereal dinner lunch".split()))
    cereal
    >>>
    >>> similarity = word_vectors.similarity('woman', 'man')
    >>> similarity > 0.8
    True
    >>>
    >>> result = word_vectors.similar_by_word("cat")
    >>> most_similar_key, similarity = result[0]  # look at the first match
    >>> print(f"{most_similar_key}: {similarity:.4f}")
    dog: 0.8798
    >>>
    >>> sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
    >>> sentence_president = 'The president greets the press in Chicago'.lower().split()
    >>>
    >>> similarity = word_vectors.wmdistance(sentence_obama, sentence_president)
    >>> print(f"{similarity:.4f}")
    3.4893
    >>>
    >>> distance = word_vectors.distance("media", "media")
    >>> print(f"{distance:.1f}")
    0.0
    >>>
    >>> similarity = word_vectors.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
    >>> print(f"{similarity:.4f}")
    0.7067
    >>>
    >>> vector = word_vectors['computer']  # numpy vector of a word
    >>> vector.shape
    (100,)
    >>>
    >>> vector = word_vectors.wv.get_vector('office', norm=True)
    >>> vector.shape
    (100,)

Correlation with human opinion on word similarity

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))

And on word analogies

.. sourcecode:: pycon

    >>> analogy_scores = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))

and so on.

"""

import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable

from numpy import (
    dot, float32 as REAL, double, zeros, vstack, ndarray,
    sum as np_sum, prod, argmax, dtype, ascontiguousarray, frombuffer,
)
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated


logger = logging.getLogger(__name__)


_KEY_TYPES = (str, int, np.integer)

_EXTENDED_KEY_TYPES = (str, int, np.integer, np.ndarray)


def _ensure_list(value):
    """Ensure that the specified value is wrapped in a list, for those supported cases
    where we also accept a single key or vector."""
    if value is None:
        return []

    if isinstance(value, _KEY_TYPES) or (isinstance(value, ndarray) and len(value.shape) == 1):
        return [value]

    if isinstance(value, ndarray) and len(value.shape) == 2:
        return list(value)

    return value


class KeyedVectors(utils.SaveLoad):

    def __init__(self, vector_size, count=0, dtype=np.float32, mapfile_path=None):
        """Mapping between keys (such as words) and vectors for :class:`~gensim.models.Word2Vec`
        and related models.

        Used to perform operations on the vectors such as vector lookup, distance, similarity etc.

        To support the needs of specific models and other downstream uses, you can also set
        additional attributes via the :meth:`~gensim.models.keyedvectors.KeyedVectors.set_vecattr`
        and :meth:`~gensim.models.keyedvectors.KeyedVectors.get_vecattr` methods.
        Note that all such attributes under the same `attr` name must have compatible `numpy`
        types, as the type and storage array for such attributes is established by the 1st time such
        `attr` is set.

        Parameters
        ----------
        vector_size : int
            Intended number of dimensions for all contained vectors.
        count : int, optional
            If provided, vectors wil be pre-allocated for at least this many vectors. (Otherwise
            they can be added later.)
        dtype : type, optional
            Vector dimensions will default to `np.float32` (AKA `REAL` in some Gensim code) unless
            another type is provided here.
        mapfile_path : string, optional
            Currently unused.
        """
        self.vector_size = vector_size
        # pre-allocating `index_to_key` to full size helps avoid redundant re-allocations, esp for `expandos`
        self.index_to_key = [None] * count  # fka index2entity or index2word
        self.next_index = 0  # pointer to where next new entry will land
        self.key_to_index = {}

        self.vectors = zeros((count, vector_size), dtype=dtype)  # formerly known as syn0
        self.norms = None

        # "expandos" are extra attributes stored for each key: {attribute_name} => numpy array of values of
        # this attribute, with one array value for each vector key.
        # The same information used to be stored in a structure called Vocab in Gensim <4.0.0, but
        # with different indexing: {vector key} => Vocab object containing all attributes for the given vector key.
        #
        # Don't modify expandos directly; call set_vecattr()/get_vecattr() instead.
        self.expandos = {}

        self.mapfile_path = mapfile_path

    def __str__(self):
        return f"{self.__class__.__name__}<vector_size={self.vector_size}, {len(self)} keys>"

    def _load_specials(self, *args, **kwargs):
        """Handle special requirements of `.load()` protocol, usually up-converting older versions."""
        super(KeyedVectors, self)._load_specials(*args, **kwargs)
        if hasattr(self, 'doctags'):
            self._upconvert_old_d2vkv()
        # fixup rename/consolidation into index_to_key of older index2word, index2entity
        if not hasattr(self, 'index_to_key'):
            self.index_to_key = self.__dict__.pop('index2word', self.__dict__.pop('index2entity', None))
        # fixup rename into vectors of older syn0
        if not hasattr(self, 'vectors'):
            self.vectors = self.__dict__.pop('syn0', None)
            self.vector_size = self.vectors.shape[1]
        # ensure at least a 'None' in 'norms' to force recalc
        if not hasattr(self, 'norms'):
            self.norms = None
        # ensure at least an empty 'expandos'
        if not hasattr(self, 'expandos'):
            self.expandos = {}
        # fixup rename of vocab into map
        if 'key_to_index' not in self.__dict__:
            self._upconvert_old_vocab()
        # ensure older instances have next_index
        if not hasattr(self, 'next_index'):
            self.next_index = len(self)

    def _upconvert_old_vocab(self):
        """Convert a loaded, pre-gensim-4.0.0 version instance that had a 'vocab' dict of data objects."""
        old_vocab = self.__dict__.pop('vocab', None)
        self.key_to_index = {}
        for k in old_vocab.keys():
            old_v = old_vocab[k]
            self.key_to_index[k] = old_v.index
            for attr in old_v.__dict__.keys():
                self.set_vecattr(old_v.index, attr, old_v.__dict__[attr])
        # special case to enforce required type on `sample_int`
        if 'sample_int' in self.expandos:
            self.expandos['sample_int'] = self.expandos['sample_int'].astype(np.uint32)

    def allocate_vecattrs(self, attrs=None, types=None):
        """Ensure arrays for given per-vector extra-attribute names & types exist, at right size.

        The length of the index_to_key list is canonical 'intended size' of KeyedVectors,
        even if other properties (vectors array) hasn't yet been allocated or expanded.
        So this allocation targets that size.

        """
        # with no arguments, adjust lengths of existing vecattr arrays to match length of index_to_key
        if attrs is None:
            attrs = list(self.expandos.keys())
            types = [self.expandos[attr].dtype for attr in attrs]
        target_size = len(self.index_to_key)
        for attr, t in zip(attrs, types):
            if t is int:
                t = np.int64  # ensure 'int' type 64-bit (numpy-on-Windows https://github.com/numpy/numpy/issues/9464)
            if t is str:
                # Avoid typing numpy arrays as strings, because numpy would use its fixed-width `dtype=np.str_`
                # dtype, which uses too much memory!
                t = object
            if attr not in self.expandos:
                self.expandos[attr] = np.zeros(target_size, dtype=t)
                continue
            prev_expando = self.expandos[attr]
            if not np.issubdtype(t, prev_expando.dtype):
                raise TypeError(
                    f"Can't allocate type {t} for attribute {attr}, "
                    f"conflicts with its existing type {prev_expando.dtype}"
                )
            if len(prev_expando) == target_size:
                continue  # no resizing necessary
            prev_count = len(prev_expando)
            self.expandos[attr] = np.zeros(target_size, dtype=prev_expando.dtype)
            self.expandos[attr][: min(prev_count, target_size), ] = prev_expando[: min(prev_count, target_size), ]

    def set_vecattr(self, key, attr, val):
        """Set attribute associated with the given key to value.

        Parameters
        ----------

        key : str
            Store the attribute for this vector key.
        attr : str
            Name of the additional attribute to store for the given key.
        val : object
            Value of the additional attribute to store for the given key.

        Returns
        -------

        None

        """
        self.allocate_vecattrs(attrs=[attr], types=[type(val)])
        index = self.get_index(key)
        self.expandos[attr][index] = val

    def get_vecattr(self, key, attr):
        """Get attribute value associated with given key.

        Parameters
        ----------

        key : str
            Vector key for which to fetch the attribute value.
        attr : str
            Name of the additional attribute to fetch for the given key.

        Returns
        -------

        object
            Value of the additional attribute fetched for the given key.

        """
        index = self.get_index(key)
        return self.expandos[attr][index]

    def resize_vectors(self, seed=0):
        """Make underlying vectors match index_to_key size; random-initialize any new rows."""
        target_shape = (len(self.index_to_key), self.vector_size)
        self.vectors = prep_vectors(target_shape, prior_vectors=self.vectors, seed=seed)
        self.allocate_vecattrs()
        self.norms = None

    def __len__(self):
        return len(self.index_to_key)

    def __getitem__(self, key_or_keys):
        """Get vector representation of `key_or_keys`.

        Parameters
        ----------
        key_or_keys : {str, list of str, int, list of int}
            Requested key or list-of-keys.

        Returns
        -------
        numpy.ndarray
            Vector representation for `key_or_keys` (1D if `key_or_keys` is single key, otherwise - 2D).

        """
        if isinstance(key_or_keys, _KEY_TYPES):
            return self.get_vector(key_or_keys)

        return vstack([self.get_vector(key) for key in key_or_keys])

    def get_index(self, key, default=None):
        """Return the integer index (slot/position) where the given key's vector is stored in the
        backing vectors array.

        """
        val = self.key_to_index.get(key, -1)
        if val >= 0:
            return val
        elif isinstance(key, (int, np.integer)) and 0 <= key < len(self.index_to_key):
            return key
        elif default is not None:
            return default
        else:
            raise KeyError(f"Key '{key}' not present")

    def get_vector(self, key, norm=False):
        """Get the key's vector, as a 1D numpy array.

        Parameters
        ----------

        key : str
            Key for vector to return.
        norm : bool, optional
            If True, the resulting vector will be L2-normalized (unit Euclidean length).

        Returns
        -------

        numpy.ndarray
            Vector for the specified key.

        Raises
        ------

        KeyError
            If the given key doesn't exist.

        """
        index = self.get_index(key)
        if norm:
            self.fill_norms()
            result = self.vectors[index] / self.norms[index]
        else:
            result = self.vectors[index]

        result.setflags(write=False)  # disallow direct tampering that would invalidate `norms` etc
        return result

    @deprecated("Use get_vector instead")
    def word_vec(self, *args, **kwargs):
        """Compatibility alias for get_vector(); must exist so subclass calls reach subclass get_vector()."""
        return self.get_vector(*args, **kwargs)

    def get_mean_vector(self, keys, weights=None, pre_normalize=True, post_normalize=False, ignore_missing=True):
        """Get the mean vector for a given list of keys.

        Parameters
        ----------

        keys : list of (str or int or ndarray)
            Keys specified by string or int ids or numpy array.
        weights : list of float or numpy.ndarray, optional
            1D array of same size of `keys` specifying the weight for each key.
        pre_normalize : bool, optional
            Flag indicating whether to normalize each keyvector before taking mean.
            If False, individual keyvector will not be normalized.
        post_normalize: bool, optional
            Flag indicating whether to normalize the final mean vector.
            If True, normalized mean vector will be return.
        ignore_missing : bool, optional
            If False, will raise error if a key doesn't exist in vocabulary.

        Returns
        -------

        numpy.ndarray
            Mean vector for the list of keys.

        Raises
        ------

        ValueError
            If the size of the list of `keys` and `weights` doesn't match.
        KeyError
            If any of the key doesn't exist in vocabulary and `ignore_missing` is false.

        """
        if len(keys) == 0:
            raise ValueError("cannot compute mean with no input")
        if isinstance(weights, list):
            weights = np.array(weights)
        if weights is None:
            weights = np.ones(len(keys))
        if len(keys) != weights.shape[0]:  # weights is a 1-D numpy array
            raise ValueError(
                "keys and weights array must have same number of elements"
            )

        mean = np.zeros(self.vector_size, self.vectors.dtype)

        total_weight = 0
        for idx, key in enumerate(keys):
            if isinstance(key, ndarray):
                mean += weights[idx] * key
                total_weight += abs(weights[idx])
            elif self.__contains__(key):
                vec = self.get_vector(key, norm=pre_normalize)
                mean += weights[idx] * vec
                total_weight += abs(weights[idx])
            elif not ignore_missing:
                raise KeyError(f"Key '{key}' not present in vocabulary")

        if total_weight > 0:
            mean = mean / total_weight
        if post_normalize:
            mean = matutils.unitvec(mean).astype(REAL)
        return mean

    def add_vector(self, key, vector):
        """Add one new vector at the given key, into existing slot if available.

        Warning: using this repeatedly is inefficient, requiring a full reallocation & copy,
        if this instance hasn't been preallocated to be ready for such incremental additions.

        Parameters
        ----------

        key: str
            Key identifier of the added vector.
        vector: numpy.ndarray
            1D numpy array with the vector values.

        Returns
        -------
        int
            Index of the newly added vector, so that ``self.vectors[result] == vector`` and
            ``self.index_to_key[result] == key``.

        """
        target_index = self.next_index
        if target_index >= len(self) or self.index_to_key[target_index] is not None:
            # must append at end by expanding existing structures
            target_index = len(self)
            warnings.warn(
                "Adding single vectors to a KeyedVectors which grows by one each time can be costly. "
                "Consider adding in batches or preallocating to the required size.",
                UserWarning)
            self.add_vectors([key], [vector])
            self.allocate_vecattrs()  # grow any adjunct arrays
            self.next_index = target_index + 1
        else:
            # can add to existing slot
            self.index_to_key[target_index] = key
            self.key_to_index[key] = target_index
            self.vectors[target_index] = vector
            self.next_index += 1
        return target_index

    def add_vectors(self, keys, weights, extras=None, replace=False):
        """Append keys and their vectors in a manual way.
        If some key is already in the vocabulary, the old vector is kept unless `replace` flag is True.

        Parameters
        ----------
        keys : list of (str or int)
            Keys specified by string or int ids.
        weights: list of numpy.ndarray or numpy.ndarray
            List of 1D np.array vectors or a 2D np.array of vectors.
        replace: bool, optional
            Flag indicating whether to replace vectors for keys which already exist in the map;
            if True - replace vectors, otherwise - keep old vectors.

        """
        if isinstance(keys, _KEY_TYPES):
            keys = [keys]
            weights = np.array(weights).reshape(1, -1)
        elif isinstance(weights, list):
            weights = np.array(weights)
        if extras is None:
            extras = {}

        # TODO? warn if not matching extras already present?
        # initially allocate extras, check type compatibility
        self.allocate_vecattrs(extras.keys(), [extras[k].dtype for k in extras.keys()])

        in_vocab_mask = np.zeros(len(keys), dtype=bool)
        for idx, key in enumerate(keys):
            if key in self.key_to_index:
                in_vocab_mask[idx] = True

        # add new entities to the vocab
        for idx in np.nonzero(~in_vocab_mask)[0]:
            key = keys[idx]
            self.key_to_index[key] = len(self.index_to_key)
            self.index_to_key.append(key)

        # add vectors, extras for new entities
        self.vectors = vstack((self.vectors, weights[~in_vocab_mask].astype(self.vectors.dtype)))
        for attr, extra in extras:
            self.expandos[attr] = np.vstack((self.expandos[attr], extra[~in_vocab_mask]))

        # change vectors, extras for in_vocab entities if `replace` flag is specified
        if replace:
            in_vocab_idxs = [self.get_index(keys[idx]) for idx in np.nonzero(in_vocab_mask)[0]]
            self.vectors[in_vocab_idxs] = weights[in_vocab_mask]
            for attr, extra in extras:
                self.expandos[attr][in_vocab_idxs] = extra[in_vocab_mask]

    def __setitem__(self, keys, weights):
        """Add keys and theirs vectors in a manual way.
        If some key is already in the vocabulary, old vector is replaced with the new one.

        This method is an alias for :meth:`~gensim.models.keyedvectors.KeyedVectors.add_vectors`
        with `replace=True`.

        Parameters
        ----------
        keys : {str, int, list of (str or int)}
            keys specified by their string or int ids.
        weights: list of numpy.ndarray or numpy.ndarray
            List of 1D np.array vectors or 2D np.array of vectors.

        """
        if not isinstance(keys, list):
            keys = [keys]
            weights = weights.reshape(1, -1)

        self.add_vectors(keys, weights, replace=True)

    def has_index_for(self, key):
        """Can this model return a single index for this key?

        Subclasses that synthesize vectors for out-of-vocabulary words (like
        :class:`~gensim.models.fasttext.FastText`) may respond True for a
        simple `word in wv` (`__contains__()`) check but False for this
        more-specific check.

        """
        return self.get_index(key, -1) >= 0

    def __contains__(self, key):
        return self.has_index_for(key)

    def most_similar_to_given(self, key1, keys_list):
        """Get the `key` from `keys_list` most similar to `key1`."""
        return keys_list[argmax([self.similarity(key1, key) for key in keys_list])]

    def closer_than(self, key1, key2):
        """Get all keys that are closer to `key1` than `key2` is to `key1`."""
        all_distances = self.distances(key1)
        e1_index = self.get_index(key1)
        e2_index = self.get_index(key2)
        closer_node_indices = np.where(all_distances < all_distances[e2_index])[0]
        return [self.index_to_key[index] for index in closer_node_indices if index != e1_index]

    @deprecated("Use closer_than instead")
    def words_closer_than(self, word1, word2):
        return self.closer_than(word1, word2)

    def rank(self, key1, key2):
        """Rank of the distance of `key2` from `key1`, in relation to distances of all keys from `key1`."""
        return len(self.closer_than(key1, key2)) + 1

    @property
    def vectors_norm(self):
        raise AttributeError(
            "The `.vectors_norm` attribute is computed dynamically since Gensim 4.0.0. "
            "Use `.get_normed_vectors()` instead.\n"
            "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4"
        )

    @vectors_norm.setter
    def vectors_norm(self, _):
        pass   # ignored but must remain for backward serialization compatibility

    def get_normed_vectors(self):
        """Get all embedding vectors normalized to unit L2 length (euclidean), as a 2D numpy array.

        To see which key corresponds to which vector = which array row, refer
        to the :attr:`~gensim.models.keyedvectors.KeyedVectors.index_to_key` attribute.

        Returns
        -------
        numpy.ndarray:
            2D numpy array of shape ``(number_of_keys, embedding dimensionality)``, L2-normalized
            along the rows (key vectors).

        """
        self.fill_norms()
        return self.vectors / self.norms[..., np.newaxis]

    def fill_norms(self, force=False):
        """
        Ensure per-vector norms are available.

        Any code which modifies vectors should ensure the accompanying norms are
        either recalculated or 'None', to trigger a full recalculation later on-request.

        """
        if self.norms is None or force:
            self.norms = np.linalg.norm(self.vectors, axis=1)

    @property
    def index2entity(self):
        raise AttributeError(
            "The index2entity attribute has been replaced by index_to_key since Gensim 4.0.0.\n"
            "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4"
        )

    @index2entity.setter
    def index2entity(self, value):
        self.index_to_key = value  # must remain for backward serialization compatibility

    @property
    def index2word(self):
        raise AttributeError(
            "The index2word attribute has been replaced by index_to_key since Gensim 4.0.0.\n"
            "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4"
        )

    @index2word.setter
    def index2word(self, value):
        self.index_to_key = value  # must remain for backward serialization compatibility

    @property
    def vocab(self):
        raise AttributeError(
            "The vocab attribute was removed from KeyedVector in Gensim 4.0.0.\n"
            "Use KeyedVector's .key_to_index dict, .index_to_key list, and methods "
            ".get_vecattr(key, attr) and .set_vecattr(key, attr, new_val) instead.\n"
            "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4"
        )

    @vocab.setter
    def vocab(self, value):
        self.vocab()  # trigger above NotImplementedError

    def sort_by_descending_frequency(self):
        """Sort the vocabulary so the most frequent words have the lowest indexes."""
        if not len(self):
            return  # noop if empty
        count_sorted_indexes = np.argsort(self.expandos['count'])[::-1]
        self.index_to_key = [self.index_to_key[idx] for idx in count_sorted_indexes]
        self.allocate_vecattrs()
        for k in self.expandos:
            # Use numpy's "fancy indexing" to permutate the entire array in one step.
            self.expandos[k] = self.expandos[k][count_sorted_indexes]
        if len(self.vectors):
            logger.warning("sorting after vectors have been allocated is expensive & error-prone")
            self.vectors = self.vectors[count_sorted_indexes]
        self.key_to_index = {word: i for i, word in enumerate(self.index_to_key)}

    def save(self, *args, **kwargs):
        """Save KeyedVectors to a file.

        Parameters
        ----------
        fname : str
            Path to the output file.

        See Also
        --------
        :meth:`~gensim.models.keyedvectors.KeyedVectors.load`
            Load a previously saved model.

        """
        super(KeyedVectors, self).save(*args, **kwargs)

    def most_similar(
            self, positive=None, negative=None, topn=10, clip_start=0, clip_end=None,
            restrict_vocab=None, indexer=None,
        ):
        """Find the top-N most similar keys.
        Positive keys contribute positively towards the similarity, negative keys negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given keys and the vectors for each key in the model.
        The method corresponds to the `word-analogy` and `distance` scripts in the original
        word2vec implementation.

        Parameters
        ----------
        positive : list of (str or int or ndarray) or list of ((str,float) or (int,float) or (ndarray,float)), optional
            List of keys that contribute positively. If tuple, second element specifies the weight (default `1.0`)
        negative : list of (str or int or ndarray) or list of ((str,float) or (int,float) or (ndarray,float)), optional
            List of keys that contribute negatively. If tuple, second element specifies the weight (default `-1.0`)
        topn : int or None, optional
            Number of top-N similar keys to return, when `topn` is int. When `topn` is None,
            then similarities for all keys are returned.
        clip_start : int
            Start clipping index.
        clip_end : int
            End clipping index.
        restrict_vocab : int, optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 key vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.) If
            specified, overrides any values of ``clip_start`` or ``clip_end``.

        Returns
        -------
        list of (str, float) or numpy.array
            When `topn` is int, a sequence of (key, similarity) is returned.
            When `topn` is None, then similarities for all keys are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        if isinstance(topn, Integral) and topn < 1:
            return []

        # allow passing a single string-key or vector for the positive/negative arguments
        positive = _ensure_list(positive)
        negative = _ensure_list(negative)

        self.fill_norms()
        clip_end = clip_end or len(self.vectors)

        if restrict_vocab:
            clip_start = 0
            clip_end = restrict_vocab

        # add weights for each key, if not already present; default to 1.0 for positive and -1.0 for negative keys
        keys = []
        weight = np.concatenate((np.ones(len(positive)), -1.0 * np.ones(len(negative))))
        for idx, item in enumerate(positive + negative):
            if isinstance(item, _EXTENDED_KEY_TYPES):
                keys.append(item)
            else:
                keys.append(item[0])
                weight[idx] = item[1]

        # compute the weighted average of all keys
        mean = self.get_mean_vector(keys, weight, pre_normalize=True, post_normalize=True, ignore_missing=False)
        all_keys = [
            self.get_index(key) for key in keys if isinstance(key, _KEY_TYPES) and self.has_index_for(key)
        ]

        if indexer is not None and isinstance(topn, int):
            return indexer.most_similar(mean, topn)

        dists = dot(self.vectors[clip_start:clip_end], mean) / self.norms[clip_start:clip_end]
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_keys), reverse=True)
        # ignore (don't return) keys from the input
        result = [
            (self.index_to_key[sim + clip_start], float(dists[sim]))
            for sim in best if (sim + clip_start) not in all_keys
        ]
        return result[:topn]

    def similar_by_word(self, word, topn=10, restrict_vocab=None):
        """Compatibility alias for similar_by_key()."""
        return self.similar_by_key(word, topn, restrict_vocab)

    def similar_by_key(self, key, topn=10, restrict_vocab=None):
        """Find the top-N most similar keys.

        Parameters
        ----------
        key : str
            Key
        topn : int or None, optional
            Number of top-N similar keys to return. If topn is None, similar_by_key returns
            the vector of similarity scores.
        restrict_vocab : int, optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 key vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        list of (str, float) or numpy.array
            When `topn` is int, a sequence of (key, similarity) is returned.
            When `topn` is None, then similarities for all keys are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        return self.most_similar(positive=[key], topn=topn, restrict_vocab=restrict_vocab)

    def similar_by_vector(self, vector, topn=10, restrict_vocab=None):
        """Find the top-N most similar keys by vector.

        Parameters
        ----------
        vector : numpy.array
            Vector from which similarities are to be computed.
        topn : int or None, optional
            Number of top-N similar keys to return, when `topn` is int. When `topn` is None,
            then similarities for all keys are returned.
        restrict_vocab : int, optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 key vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        list of (str, float) or numpy.array
            When `topn` is int, a sequence of (key, similarity) is returned.
            When `topn` is None, then similarities for all keys are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        return self.most_similar(positive=[vector], topn=topn, restrict_vocab=restrict_vocab)

    def wmdistance(self, document1, document2, norm=True):
        """Compute the Word Mover's Distance between two documents.

        When using this code, please consider citing the following papers:

        * `Rémi Flamary et al. "POT: Python Optimal Transport"
          <https://jmlr.org/papers/v22/20-451.html>`_
        * `Matt Kusner et al. "From Word Embeddings To Document Distances"
          <http://proceedings.mlr.press/v37/kusnerb15.pdf>`_.

        Parameters
        ----------
        document1 : list of str
            Input document.
        document2 : list of str
            Input document.
        norm : boolean
            Normalize all word vectors to unit length before computing the distance?
            Defaults to True.

        Returns
        -------
        float
            Word Mover's distance between `document1` and `document2`.

        Warnings
        --------
        This method only works if `POT <https://pypi.org/project/POT/>`_ is installed.

        If one of the documents have no words that exist in the vocab, `float('inf')` (i.e. infinity)
        will be returned.

        Raises
        ------
        ImportError
            If `POT <https://pypi.org/project/POT/>`_  isn't installed.

        """
        # If POT is attempted to be used, but isn't installed, ImportError will be raised in wmdistance
        from ot import emd2

        # Remove out-of-vocabulary words.
        len_pre_oov1 = len(document1)
        len_pre_oov2 = len(document2)
        document1 = [token for token in document1 if token in self]
        document2 = [token for token in document2 if token in self]
        diff1 = len_pre_oov1 - len(document1)
        diff2 = len_pre_oov2 - len(document2)
        if diff1 > 0 or diff2 > 0:
            logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)

        if not document1 or not document2:
            logger.warning("At least one of the documents had no words that were in the vocabulary.")
            return float('inf')

        dictionary = Dictionary(documents=[document1, document2])
        vocab_len = len(dictionary)

        if vocab_len == 1:
            # Both documents are composed of a single unique token => zero distance.
            return 0.0

        doclist1 = list(set(document1))
        doclist2 = list(set(document2))
        v1 = np.array([self.get_vector(token, norm=norm) for token in doclist1])
        v2 = np.array([self.get_vector(token, norm=norm) for token in doclist2])
        doc1_indices = dictionary.doc2idx(doclist1)
        doc2_indices = dictionary.doc2idx(doclist2)

        # Compute distance matrix.
        distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
        distance_matrix[np.ix_(doc1_indices, doc2_indices)] = cdist(v1, v2)

        if abs(np_sum(distance_matrix)) < 1e-8:
            # `emd` gets stuck if the distance matrix contains only zeros.
            logger.info('The distance matrix is all zeros. Aborting (returning inf).')
            return float('inf')

        def nbow(document):
            d = zeros(vocab_len, dtype=double)
            nbow = dictionary.doc2bow(document)  # Word frequencies.
            doc_len = len(document)
            for idx, freq in nbow:
                d[idx] = freq / float(doc_len)  # Normalized word frequencies.
            return d

        # Compute nBOW representation of documents. This is what POT expects on input.
        d1 = nbow(document1)
        d2 = nbow(document2)

        # Compute WMD.
        return emd2(d1, d2, distance_matrix)

    def most_similar_cosmul(
            self, positive=None, negative=None, topn=10, restrict_vocab=None
        ):
        """Find the top-N most similar words, using the multiplicative combination objective,
        proposed by `Omer Levy and Yoav Goldberg "Linguistic Regularities in Sparse and Explicit Word Representations"
        <http://www.aclweb.org/anthology/W14-1618>`_. Positive words still contribute positively towards the similarity,
        negative words negatively, but with less susceptibility to one large distance dominating the calculation.
        In the common analogy-solving case, of two positive and one negative examples,
        this method is equivalent to the "3CosMul" objective (equation (4)) of Levy and Goldberg.

        Additional positive or negative examples contribute to the numerator or denominator,
        respectively - a potentially sensible but untested extension of the method.
        With a single positive example, rankings will be the same as in the default
        :meth:`~gensim.models.keyedvectors.KeyedVectors.most_similar`.

        Allows calls like most_similar_cosmul('dog', 'cat'), as a shorthand for
        most_similar_cosmul(['dog'], ['cat']) where 'dog' is positive and 'cat' negative

        Parameters
        ----------
        positive : list of str, optional
            List of words that contribute positively.
        negative : list of str, optional
            List of words that contribute negatively.
        topn : int or None, optional
            Number of top-N similar words to return, when `topn` is int. When `topn` is None,
            then similarities for all words are returned.
        restrict_vocab : int or None, optional
            Optional integer which limits the range of vectors which are searched for most-similar values.
            For example, restrict_vocab=10000 would only check the first 10000 node vectors in the vocabulary order.
            This may be meaningful if vocabulary is sorted by descending frequency.


        Returns
        -------
        list of (str, float) or numpy.array
            When `topn` is int, a sequence of (word, similarity) is returned.
            When `topn` is None, then similarities for all words are returned as a
            one-dimensional numpy array with the size of the vocabulary.

        """
        # TODO: Update to better match & share code with most_similar()
        if isinstance(topn, Integral) and topn < 1:
            return []

        # allow passing a single string-key or vector for the positive/negative arguments
        positive = _ensure_list(positive)
        negative = _ensure_list(negative)

        self.init_sims()

        if isinstance(positive, str):
            # allow calls like most_similar_cosmul('dog'), as a shorthand for most_similar_cosmul(['dog'])
            positive = [positive]

        if isinstance(negative, str):
            negative = [negative]

        all_words = {
            self.get_index(word) for word in positive + negative
            if not isinstance(word, ndarray) and word in self.key_to_index
        }

        positive = [
            self.get_vector(word, norm=True) if isinstance(word, str) else word
            for word in positive
        ]
        negative = [
            self.get_vector(word, norm=True) if isinstance(word, str) else word
            for word in negative
        ]

        if not positive:
            raise ValueError("cannot compute similarity with no input")

        # equation (4) of Levy & Goldberg "Linguistic Regularities...",
        # with distances shifted to [0,1] per footnote (7)
        pos_dists = [((1 + dot(self.vectors, term) / self.norms) / 2) for term in positive]
        neg_dists = [((1 + dot(self.vectors, term) / self.norms) / 2) for term in negative]
        dists = prod(pos_dists, axis=0) / (prod(neg_dists, axis=0) + 0.000001)

        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index_to_key[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def rank_by_centrality(self, words, use_norm=True):
        """Rank the given words by similarity to the centroid of all the words.

        Parameters
        ----------
        words : list of str
            List of keys.
        use_norm : bool, optional
            Whether to calculate centroid using unit-normed vectors; default True.

        Returns
        -------
        list of (float, str)
            Ranked list of (similarity, key), most-similar to the centroid first.

        """
        self.fill_norms()

        used_words = [word for word in words if word in self]
        if len(used_words) != len(words):
            ignored_words = set(words) - set(used_words)
            logger.warning("vectors for words %s are not present in the model, ignoring these words", ignored_words)
        if not used_words:
            raise ValueError("cannot select a word from an empty list")
        vectors = vstack([self.get_vector(word, norm=use_norm) for word in used_words]).astype(REAL)
        mean = self.get_mean_vector(vectors, post_normalize=True)
        dists = dot(vectors, mean)
        return sorted(zip(dists, used_words), reverse=True)

    def doesnt_match(self, words):
        """Which key from the given list doesn't go with the others?

        Parameters
        ----------
        words : list of str
            List of keys.

        Returns
        -------
        str
            The key further away from the mean of all keys.

        """
        return self.rank_by_centrality(words)[-1][1]

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        """Compute cosine similarities between one vector and a set of other vectors.

        Parameters
        ----------
        vector_1 : numpy.ndarray
            Vector from which similarities are to be computed, expected shape (dim,).
        vectors_all : numpy.ndarray
            For each row in vectors_all, distance from vector_1 is computed, expected shape (num_vectors, dim).

        Returns
        -------
        numpy.ndarray
            Contains cosine distance between `vector_1` and each row in `vectors_all`, shape (num_vectors,).

        """
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)
        dot_products = dot(vectors_all, vector_1)
        similarities = dot_products / (norm * all_norms)
        return similarities

    def distances(self, word_or_vector, other_words=()):
        """Compute cosine distances from given word or vector to all words in `other_words`.
        If `other_words` is empty, return distance between `word_or_vector` and all words in vocab.

        Parameters
        ----------
        word_or_vector : {str, numpy.ndarray}
            Word or vector from which distances are to be computed.
        other_words : iterable of str
            For each word in `other_words` distance from `word_or_vector` is computed.
            If None or empty, distance of `word_or_vector` from all words in vocab is computed (including itself).

        Returns
        -------
        numpy.array
            Array containing distances to all words in `other_words` from input `word_or_vector`.

        Raises
        -----
        KeyError
            If either `word_or_vector` or any word in `other_words` is absent from vocab.

        """
        if isinstance(word_or_vector, _KEY_TYPES):
            input_vector = self.get_vector(word_or_vector)
        else:
            input_vector = word_or_vector
        if not other_words:
            other_vectors = self.vectors
        else:
            other_indices = [self.get_index(word) for word in other_words]
            other_vectors = self.vectors[other_indices]
        return 1 - self.cosine_similarities(input_vector, other_vectors)

    def distance(self, w1, w2):
        """Compute cosine distance between two keys.
        Calculate 1 - :meth:`~gensim.models.keyedvectors.KeyedVectors.similarity`.

        Parameters
        ----------
        w1 : str
            Input key.
        w2 : str
            Input key.

        Returns
        -------
        float
            Distance between `w1` and `w2`.

        """
        return 1 - self.similarity(w1, w2)

    def similarity(self, w1, w2):
        """Compute cosine similarity between two keys.

        Parameters
        ----------
        w1 : str
            Input key.
        w2 : str
            Input key.

        Returns
        -------
        float
            Cosine similarity between `w1` and `w2`.

        """
        return dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))

    def n_similarity(self, ws1, ws2):
        """Compute cosine similarity between two sets of keys.

        Parameters
        ----------
        ws1 : list of str
            Sequence of keys.
        ws2: list of str
            Sequence of keys.

        Returns
        -------
        numpy.ndarray
            Similarities between `ws1` and `ws2`.

        """
        if not (len(ws1) and len(ws2)):
            raise ZeroDivisionError('At least one of the passed list is empty.')
        mean1 = self.get_mean_vector(ws1, pre_normalize=False)
        mean2 = self.get_mean_vector(ws2, pre_normalize=False)
        return dot(matutils.unitvec(mean1), matutils.unitvec(mean2))

    @staticmethod
    def _log_evaluate_word_analogies(section):
        """Calculate score by section, helper for
        :meth:`~gensim.models.keyedvectors.KeyedVectors.evaluate_word_analogies`.

        Parameters
        ----------
        section : dict of (str, (str, str, str, str))
            Section given from evaluation.

        Returns
        -------
        float
            Accuracy score if at least one prediction was made (correct or incorrect).

            Or return 0.0 if there were no predictions at all in this section.

        """
        correct, incorrect = len(section['correct']), len(section['incorrect'])

        if correct + incorrect == 0:
            return 0.0

        score = correct / (correct + incorrect)
        logger.info("%s: %.1f%% (%i/%i)", section['section'], 100.0 * score, correct, correct + incorrect)
        return score

    def evaluate_word_analogies(
            self, analogies, restrict_vocab=300000, case_insensitive=True,
            dummy4unknown=False, similarity_function='most_similar'):
        """Compute performance of the model on an analogy test set.

        The accuracy is reported (printed to log and returned as a score) for each section separately,
        plus there's one aggregate summary at the end.

        This method corresponds to the `compute-accuracy` script of the original C word2vec.
        See also `Analogy (State of the art) <https://aclweb.org/aclwiki/Analogy_(State_of_the_art)>`_.

        Parameters
        ----------
        analogies : str
            Path to file, where lines are 4-tuples of words, split into sections by ": SECTION NAME" lines.
            See `gensim/test/test_data/questions-words.txt` as example.
        restrict_vocab : int, optional
            Ignore all 4-tuples containing a word not in the first `restrict_vocab` words.
            This may be meaningful if you've sorted the model vocabulary by descending frequency (which is standard
            in modern word embedding models).
        case_insensitive : bool, optional
            If True - convert all words to their uppercase form before evaluating the performance.
            Useful to handle case-mismatch between training tokens and words in the test set.
            In case of multiple case variants of a single word, the vector for the first occurrence
            (also the most frequent if vocabulary is sorted) is taken.
        dummy4unknown : bool, optional
            If True - produce zero accuracies for 4-tuples with out-of-vocabulary words.
            Otherwise, these tuples are skipped entirely and not used in the evaluation.
        similarity_function : str, optional
            Function name used for similarity calculation.

        Returns
        -------
        score : float
            The overall evaluation score on the entire evaluation set
        sections : list of dict of {str : str or list of tuple of (str, str, str, str)}
            Results broken down by each section of the evaluation set. Each dict contains the name of the section
            under the key 'section', and lists of correctly and incorrectly predicted 4-tuples of words under the
            keys 'correct' and 'incorrect'.

        """
        ok_keys = self.index_to_key[:restrict_vocab]
        if case_insensitive:
            ok_vocab = {k.upper(): self.get_index(k) for k in reversed(ok_keys)}
        else:
            ok_vocab = {k: self.get_index(k) for k in reversed(ok_keys)}
        oov = 0
        logger.info("Evaluating word analogies for top %i words in the model on %s", restrict_vocab, analogies)
        sections, section = [], None
        quadruplets_no = 0
        with utils.open(analogies, 'rb') as fin:
            for line_no, line in enumerate(fin):
                line = utils.to_unicode(line)
                if line.startswith(': '):
                    # a new section starts => store the old section
                    if section:
                        sections.append(section)
                        self._log_evaluate_word_analogies(section)
                    section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
                else:
                    if not section:
                        raise ValueError("Missing section header before line #%i in %s" % (line_no, analogies))
                    try:
                        if case_insensitive:
                            a, b, c, expected = [word.upper() for word in line.split()]
                        else:
                            a, b, c, expected = [word for word in line.split()]
                    except ValueError:
                        logger.info("Skipping invalid line #%i in %s", line_no, analogies)
                        continue
                    quadruplets_no += 1
                    if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                        oov += 1
                        if dummy4unknown:
                            logger.debug('Zero accuracy for line #%d with OOV words: %s', line_no, line.strip())
                            section['incorrect'].append((a, b, c, expected))
                        else:
                            logger.debug("Skipping line #%i with OOV words: %s", line_no, line.strip())
                        continue
                    original_key_to_index = self.key_to_index
                    self.key_to_index = ok_vocab
                    ignore = {a, b, c}  # input words to be ignored
                    predicted = None
                    # find the most likely prediction using 3CosAdd (vector offset) method
                    # TODO: implement 3CosMul and set-based methods for solving analogies

                    sims = self.most_similar(positive=[b, c], negative=[a], topn=5, restrict_vocab=restrict_vocab)
                    self.key_to_index = original_key_to_index
                    for element in sims:
                        predicted = element[0].upper() if case_insensitive else element[0]
                        if predicted in ok_vocab and predicted not in ignore:
                            if predicted != expected:
                                logger.debug("%s: expected %s, predicted %s", line.strip(), expected, predicted)
                            break
                    if predicted == expected:
                        section['correct'].append((a, b, c, expected))
                    else:
                        section['incorrect'].append((a, b, c, expected))
        if section:
            # store the last section, too
            sections.append(section)
            self._log_evaluate_word_analogies(section)

        total = {
            'section': 'Total accuracy',
            'correct': list(itertools.chain.from_iterable(s['correct'] for s in sections)),
            'incorrect': list(itertools.chain.from_iterable(s['incorrect'] for s in sections)),
        }

        oov_ratio = float(oov) / quadruplets_no * 100
        logger.info('Quadruplets with out-of-vocabulary words: %.1f%%', oov_ratio)
        if not dummy4unknown:
            logger.info(
                'NB: analogies containing OOV words were skipped from evaluation! '
                'To change this behavior, use "dummy4unknown=True"'
            )
        analogies_score = self._log_evaluate_word_analogies(total)
        sections.append(total)
        # Return the overall score and the full lists of correct and incorrect analogies
        return analogies_score, sections

    @staticmethod
    def log_accuracy(section):
        correct, incorrect = len(section['correct']), len(section['incorrect'])
        if correct + incorrect > 0:
            logger.info(
                "%s: %.1f%% (%i/%i)",
                section['section'], 100.0 * correct / (correct + incorrect), correct, correct + incorrect,
            )

    @staticmethod
    def log_evaluate_word_pairs(pearson, spearman, oov, pairs):
        logger.info('Pearson correlation coefficient against %s: %.4f', pairs, pearson[0])
        logger.info('Spearman rank-order correlation coefficient against %s: %.4f', pairs, spearman[0])
        logger.info('Pairs with unknown words ratio: %.1f%%', oov)

    def evaluate_word_pairs(
            self, pairs, delimiter='\t', encoding='utf8',
            restrict_vocab=300000, case_insensitive=True, dummy4unknown=False,
        ):
        """Compute correlation of the model with human similarity judgments.

        Notes
        -----
        More datasets can be found at
        * http://technion.ac.il/~ira.leviant/MultilingualVSMdata.html
        * https://www.cl.cam.ac.uk/~fh295/simlex.html.

        Parameters
        ----------
        pairs : str
            Path to file, where lines are 3-tuples, each consisting of a word pair and a similarity value.
            See `test/test_data/wordsim353.tsv` as example.
        delimiter : str, optional
            Separator in `pairs` file.
        restrict_vocab : int, optional
            Ignore all 4-tuples containing a word not in the first `restrict_vocab` words.
            This may be meaningful if you've sorted the model vocabulary by descending frequency (which is standard
            in modern word embedding models).
        case_insensitive : bool, optional
            If True - convert all words to their uppercase form before evaluating the performance.
            Useful to handle case-mismatch between training tokens and words in the test set.
            In case of multiple case variants of a single word, the vector for the first occurrence
            (also the most frequent if vocabulary is sorted) is taken.
        dummy4unknown : bool, optional
            If True - produce zero accuracies for 4-tuples with out-of-vocabulary words.
            Otherwise, these tuples are skipped entirely and not used in the evaluation.

        Returns
        -------
        pearson : tuple of (float, float)
            Pearson correlation coefficient with 2-tailed p-value.
        spearman : tuple of (float, float)
            Spearman rank-order correlation coefficient between the similarities from the dataset and the
            similarities produced by the model itself, with 2-tailed p-value.
        oov_ratio : float
            The ratio of pairs with unknown words.

        """
        ok_keys = self.index_to_key[:restrict_vocab]
        if case_insensitive:
            ok_vocab = {k.upper(): self.get_index(k) for k in reversed(ok_keys)}
        else:
            ok_vocab = {k: self.get_index(k) for k in reversed(ok_keys)}

        similarity_gold = []
        similarity_model = []
        oov = 0

        original_key_to_index, self.key_to_index = self.key_to_index, ok_vocab
        try:
            with utils.open(pairs, encoding=encoding) as fin:
                for line_no, line in enumerate(fin):
                    if not line or line.startswith('#'):  # Ignore lines with comments.
                        continue
                    try:
                        if case_insensitive:
                            a, b, sim = [word.upper() for word in line.split(delimiter)]
                        else:
                            a, b, sim = [word for word in line.split(delimiter)]
                        sim = float(sim)
                    except (ValueError, TypeError):
                        logger.info('Skipping invalid line #%d in %s', line_no, pairs)
                        continue

                    if a not in ok_vocab or b not in ok_vocab:
                        oov += 1
                        if dummy4unknown:
                            logger.debug('Zero similarity for line #%d with OOV words: %s', line_no, line.strip())
                            similarity_model.append(0.0)
                            similarity_gold.append(sim)
                        else:
                            logger.info('Skipping line #%d with OOV words: %s', line_no, line.strip())
                        continue
                    similarity_gold.append(sim)  # Similarity from the dataset
                    similarity_model.append(self.similarity(a, b))  # Similarity from the model
        finally:
            self.key_to_index = original_key_to_index

        assert len(similarity_gold) == len(similarity_model)
        if not similarity_gold:
            raise ValueError(
                f"No valid similarity judgements found in {pairs}: either invalid format or "
                f"all are out-of-vocabulary in {self}"
            )
        spearman = stats.spearmanr(similarity_gold, similarity_model)
        pearson = stats.pearsonr(similarity_gold, similarity_model)
        if dummy4unknown:
            oov_ratio = float(oov) / len(similarity_gold) * 100
        else:
            oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100

        logger.debug('Pearson correlation coefficient against %s: %f with p-value %f', pairs, pearson[0], pearson[1])
        logger.debug(
            'Spearman rank-order correlation coefficient against %s: %f with p-value %f',
            pairs, spearman[0], spearman[1]
        )
        logger.debug('Pairs with unknown words: %d', oov)
        self.log_evaluate_word_pairs(pearson, spearman, oov_ratio, pairs)
        return pearson, spearman, oov_ratio

    @deprecated(
        "Use fill_norms() instead. "
        "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4"
    )
    def init_sims(self, replace=False):
        """Precompute data helpful for bulk similarity calculations.

        :meth:`~gensim.models.keyedvectors.KeyedVectors.fill_norms` now preferred for this purpose.

        Parameters
        ----------

        replace : bool, optional
            If True - forget the original vectors and only keep the normalized ones.

        Warnings
        --------

        You **cannot sensibly continue training** after doing a replace on a model's
        internal KeyedVectors, and a replace is no longer necessary to save RAM. Do not use this method.

        """
        self.fill_norms()
        if replace:
            logger.warning("destructive init_sims(replace=True) deprecated & no longer required for space-efficiency")
            self.unit_normalize_all()

    def unit_normalize_all(self):
        """Destructively scale all vectors to unit-length.

        You cannot sensibly continue training after such a step.

        """
        self.fill_norms()
        self.vectors /= self.norms[..., np.newaxis]
        self.norms = np.ones((len(self.vectors),))

    def relative_cosine_similarity(self, wa, wb, topn=10):
        """Compute the relative cosine similarity between two words given top-n similar words,
        by `Artuur Leeuwenberga, Mihaela Velab , Jon Dehdaribc, Josef van Genabithbc "A Minimally Supervised Approach
        for Synonym Extraction with Word Embeddings" <https://ufal.mff.cuni.cz/pbml/105/art-leeuwenberg-et-al.pdf>`_.

        To calculate relative cosine similarity between two words, equation (1) of the paper is used.
        For WordNet synonyms, if rcs(topn=10) is greater than 0.10 then wa and wb are more similar than
        any arbitrary word pairs.

        Parameters
        ----------
        wa: str
            Word for which we have to look top-n similar word.
        wb: str
            Word for which we evaluating relative cosine similarity with wa.
        topn: int, optional
            Number of top-n similar words to look with respect to wa.

        Returns
        -------
        numpy.float64
            Relative cosine similarity between wa and wb.

        """
        sims = self.similar_by_word(wa, topn)
        if not sims:
            raise ValueError("Cannot calculate relative cosine similarity without any similar words.")
        rcs = float(self.similarity(wa, wb)) / (sum(sim for _, sim in sims))

        return rcs

    def save_word2vec_format(
            self, fname, fvocab=None, binary=False, total_vec=None, write_header=True,
            prefix='', append=False, sort_attr='count',
        ):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            File path to save the vectors to.
        fvocab : str, optional
            File path to save additional vocabulary information to. `None` to not store the vocabulary.
        binary : bool, optional
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec : int, optional
            Explicitly specify total number of vectors
            (in case word vectors are appended with document vectors afterwards).
        write_header : bool, optional
            If False, don't write the 1st line declaring the count of vectors and dimensions.
            This is the format used by e.g. gloVe vectors.
        prefix : str, optional
            String to prepend in front of each stored word. Default = no prefix.
        append : bool, optional
            If set, open `fname` in `ab` mode instead of the default `wb` mode.
        sort_attr : str, optional
            Sort the output vectors in descending order of this attribute. Default: most frequent keys first.

        """
        if total_vec is None:
            total_vec = len(self.index_to_key)
        mode = 'wb' if not append else 'ab'

        if sort_attr in self.expandos:
            store_order_vocab_keys = sorted(self.key_to_index.keys(), key=lambda k: -self.get_vecattr(k, sort_attr))
        else:
            # This can happen even for the default `count`: the "native C word2vec" format does not store counts,
            # so models loaded via load_word2vec_format() do not have the "count" attribute set. They have
            # no attributes at all, and fall under this code path.
            if fvocab is not None:
                raise ValueError(f"Cannot store vocabulary with '{sort_attr}' because that attribute does not exist")
            logger.warning(
                "attribute %s not present in %s; will store in internal index_to_key order",
                sort_attr, self,
            )
            store_order_vocab_keys = self.index_to_key

        if fvocab is not None:
            logger.info("storing vocabulary in %s", fvocab)
            with utils.open(fvocab, mode) as vout:
                for word in store_order_vocab_keys:
                    vout.write(f"{prefix}{word} {self.get_vecattr(word, sort_attr)}\n".encode('utf8'))

        logger.info("storing %sx%s projection weights into %s", total_vec, self.vector_size, fname)
        assert (len(self.index_to_key), self.vector_size) == self.vectors.shape

        # After (possibly-empty) initial range of int-only keys in Doc2Vec,
        # store in sorted order: most frequent keys at the top.
        # XXX: get rid of this: not used much, too complex and brittle.
        # See https://github.com/RaRe-Technologies/gensim/pull/2981#discussion_r512969788
        index_id_count = 0
        for i, val in enumerate(self.index_to_key):
            if i != val:
                break
            index_id_count += 1
        keys_to_write = itertools.chain(range(0, index_id_count), store_order_vocab_keys)

        # Store the actual vectors to the output file, in the order defined by sort_attr.
        with utils.open(fname, mode) as fout:
            if write_header:
                fout.write(f"{total_vec} {self.vector_size}\n".encode('utf8'))
            for key in keys_to_write:
                key_vector = self[key]
                if binary:
                    fout.write(f"{prefix}{key} ".encode('utf8') + key_vector.astype(REAL).tobytes())
                else:
                    fout.write(f"{prefix}{key} {' '.join(repr(val) for val in key_vector)}\n".encode('utf8'))

    @classmethod
    def load_word2vec_format(
            cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
            limit=None, datatype=REAL, no_header=False,
        ):
        """Load KeyedVectors from a file produced by the original C word2vec-tool format.

        Warnings
        --------
        The information stored in the file is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.

        Parameters
        ----------
        fname : str
            The file path to the saved word2vec-format file.
        fvocab : str, optional
            File path to the vocabulary.Word counts are read from `fvocab` filename, if set
            (this is the file generated by `-save-vocab` flag of the original C tool).
        binary : bool, optional
            If True, indicates whether the data is in binary word2vec format.
        encoding : str, optional
            If you trained the C model using non-utf8 encoding for words, specify that encoding in `encoding`.
        unicode_errors : str, optional
            default 'strict', is a string suitable to be passed as the `errors`
            argument to the unicode() (Python 2.x) or str() (Python 3.x) function. If your source
            file may include word tokens truncated in the middle of a multibyte unicode character
            (as is common from the original word2vec.c tool), 'ignore' or 'replace' may help.
        limit : int, optional
            Sets a maximum number of word-vectors to read from the file. The default,
            None, means read all.
        datatype : type, optional
            (Experimental) Can coerce dimensions to a non-default float type (such as `np.float16`) to save memory.
            Such types may result in much slower bulk operations or incompatibility with optimized routines.)
        no_header : bool, optional
            Default False means a usual word2vec-format file, with a 1st line declaring the count of
            following vectors & number of dimensions. If True, the file is assumed to lack a declaratory
            (vocab_size, vector_size) header and instead start with the 1st vector, and an extra
            reading-pass will be used to discover the number of vectors. Works only with `binary=False`.

        Returns
        -------
        :class:`~gensim.models.keyedvectors.KeyedVectors`
            Loaded model.

        """
        return _load_word2vec_format(
            cls, fname, fvocab=fvocab, binary=binary, encoding=encoding, unicode_errors=unicode_errors,
            limit=limit, datatype=datatype, no_header=no_header,
        )

    def intersect_word2vec_format(self, fname, lockf=0.0, binary=False, encoding='utf8', unicode_errors='strict'):
        """Merge in an input-hidden weight matrix loaded from the original C word2vec-tool format,
        where it intersects with the current vocabulary.

        No words are added to the existing vocabulary, but intersecting words adopt the file's weights, and
        non-intersecting words are left alone.

        Parameters
        ----------
        fname : str
            The file path to load the vectors from.
        lockf : float, optional
            Lock-factor value to be set for any imported word-vectors; the
            default value of 0.0 prevents further updating of the vector during subsequent
            training. Use 1.0 to allow further training updates of merged vectors.
        binary : bool, optional
            If True, `fname` is in the binary word2vec C format.
        encoding : str, optional
            Encoding of `text` for `unicode` function (python2 only).
        unicode_errors : str, optional
            Error handling behaviour, used as parameter for `unicode` function (python2 only).

        """
        overlap_count = 0
        logger.info("loading projection weights from %s", fname)
        with utils.open(fname, 'rb') as fin:
            header = utils.to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = (int(x) for x in header.split())  # throws for invalid file format
            if not vector_size == self.vector_size:
                raise ValueError("incompatible vector size %d in file %s" % (vector_size, fname))
                # TODO: maybe mismatched vectors still useful enough to merge (truncating/padding)?
            if binary:
                binary_len = dtype(REAL).itemsize * vector_size
                for _ in range(vocab_size):
                    # mixed text and binary: read text first, then binary
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                            word.append(ch)
                    word = utils.to_unicode(b''.join(word), encoding=encoding, errors=unicode_errors)
                    weights = np.fromstring(fin.read(binary_len), dtype=REAL)
                    if word in self.key_to_index:
                        overlap_count += 1
                        self.vectors[self.get_index(word)] = weights
                        self.vectors_lockf[self.get_index(word)] = lockf  # lock-factor: 0.0=no changes
            else:
                for line_no, line in enumerate(fin):
                    parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % line_no)
                    word, weights = parts[0], [REAL(x) for x in parts[1:]]
                    if word in self.key_to_index:
                        overlap_count += 1
                        self.vectors[self.get_index(word)] = weights
                        self.vectors_lockf[self.get_index(word)] = lockf  # lock-factor: 0.0=no changes
        self.add_lifecycle_event(
            "intersect_word2vec_format",
            msg=f"merged {overlap_count} vectors into {self.vectors.shape} matrix from {fname}",
        )

    def vectors_for_all(self, keys: Iterable, allow_inference: bool = True,
                        copy_vecattrs: bool = False) -> 'KeyedVectors':
        """Produce vectors for all given keys as a new :class:`KeyedVectors` object.

        Notes
        -----
        The keys will always be deduplicated. For optimal performance, you should not pass entire
        corpora to the method. Instead, you should construct a dictionary of unique words in your
        corpus:

        >>> from collections import Counter
        >>> import itertools
        >>>
        >>> from gensim.models import FastText
        >>> from gensim.test.utils import datapath, common_texts
        >>>
        >>> model_corpus_file = datapath('lee_background.cor')  # train word vectors on some corpus
        >>> model = FastText(corpus_file=model_corpus_file, vector_size=20, min_count=1)
        >>> corpus = common_texts  # infer word vectors for words from another corpus
        >>> word_counts = Counter(itertools.chain.from_iterable(corpus))  # count words in your corpus
        >>> words_by_freq = (k for k, v in word_counts.most_common())
        >>> word_vectors = model.wv.vectors_for_all(words_by_freq)  # create word-vectors for words in your corpus

        Parameters
        ----------
        keys : iterable
            The keys that will be vectorized.
        allow_inference : bool, optional
            In subclasses such as :class:`~gensim.models.fasttext.FastTextKeyedVectors`,
            vectors for out-of-vocabulary keys (words) may be inferred. Default is True.
        copy_vecattrs : bool, optional
            Additional attributes set via the :meth:`KeyedVectors.set_vecattr` method
            will be preserved in the produced :class:`KeyedVectors` object. Default is False.
            To ensure that *all* the produced vectors will have vector attributes assigned,
            you should set `allow_inference=False`.

        Returns
        -------
        keyedvectors : :class:`~gensim.models.keyedvectors.KeyedVectors`
            Vectors for all the given keys.

        """
        # Pick only the keys that actually exist & deduplicate them.
        # We keep the original key order, to improve cache locality, for performance.
        vocab, seen = [], set()
        for key in keys:
            if key not in seen:
                seen.add(key)
                if key in (self if allow_inference else self.key_to_index):
                    vocab.append(key)

        kv = KeyedVectors(self.vector_size, len(vocab), dtype=self.vectors.dtype)

        for key in vocab:  # produce and index vectors for all the given keys
            weights = self[key]
            _add_word_to_kv(kv, None, key, weights, len(vocab))
            if copy_vecattrs:
                for attr in self.expandos:
                    try:
                        kv.set_vecattr(key, attr, self.get_vecattr(key, attr))
                    except KeyError:
                        pass
        return kv

    def _upconvert_old_d2vkv(self):
        """Convert a deserialized older Doc2VecKeyedVectors instance to latest generic KeyedVectors"""
        self.vocab = self.doctags
        self._upconvert_old_vocab()  # destroys 'vocab', fills 'key_to_index' & 'extras'
        for k in self.key_to_index.keys():
            old_offset = self.get_vecattr(k, 'offset')
            true_index = old_offset + self.max_rawint + 1
            self.key_to_index[k] = true_index
        del self.expandos['offset']  # no longer needed
        if self.max_rawint > -1:
            self.index_to_key = list(range(0, self.max_rawint + 1)) + self.offset2doctag
        else:
            self.index_to_key = self.offset2doctag
        self.vectors = self.vectors_docs
        del self.doctags
        del self.vectors_docs
        del self.count
        del self.max_rawint
        del self.offset2doctag

    def similarity_unseen_docs(self, *args, **kwargs):
        raise NotImplementedError("Call similarity_unseen_docs on a Doc2Vec model instead.")


# to help 3.8.1 & older pickles load properly
Word2VecKeyedVectors = KeyedVectors
Doc2VecKeyedVectors = KeyedVectors
EuclideanKeyedVectors = KeyedVectors


class CompatVocab:

    def __init__(self, **kwargs):
        """A single vocabulary item, used internally for collecting per-word frequency/sampling info,
        and for constructing binary trees (incl. both word leaves and inner nodes).

        Retained for now to ease the loading of older models.
        """
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s<%s>" % (self.__class__.__name__, ', '.join(vals))


# compatibility alias, allowing older pickle-based `.save()`s to load
Vocab = CompatVocab


# Functions for internal use by _load_word2vec_format function

def _add_word_to_kv(kv, counts, word, weights, vocab_size):

    if kv.has_index_for(word):
        logger.warning("duplicate word '%s' in word2vec file, ignoring all but first", word)
        return
    word_id = kv.add_vector(word, weights)

    if counts is None:
        # Most common scenario: no vocab file given. Just make up some bogus counts, in descending order.
        # TODO (someday): make this faking optional, include more realistic (Zipf-based) fake numbers.
        word_count = vocab_size - word_id
    elif word in counts:
        # use count from the vocab file
        word_count = counts[word]
    else:
        logger.warning("vocabulary file is incomplete: '%s' is missing", word)
        word_count = None
    kv.set_vecattr(word, 'count', word_count)


def _add_bytes_to_kv(kv, counts, chunk, vocab_size, vector_size, datatype, unicode_errors, encoding):
    start = 0
    processed_words = 0
    bytes_per_vector = vector_size * dtype(REAL).itemsize
    max_words = vocab_size - kv.next_index  # don't read more than kv preallocated to hold
    assert max_words > 0
    for _ in range(max_words):
        i_space = chunk.find(b' ', start)
        i_vector = i_space + 1

        if i_space == -1 or (len(chunk) - i_vector) < bytes_per_vector:
            break

        word = chunk[start:i_space].decode(encoding, errors=unicode_errors)
        # Some binary files are reported to have obsolete new line in the beginning of word, remove it
        word = word.lstrip('\n')
        vector = frombuffer(chunk, offset=i_vector, count=vector_size, dtype=REAL).astype(datatype)
        _add_word_to_kv(kv, counts, word, vector, vocab_size)
        start = i_vector + bytes_per_vector
        processed_words += 1

    return processed_words, chunk[start:]


def _word2vec_read_binary(
        fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, binary_chunk_size,
        encoding="utf-8",
    ):
    chunk = b''
    tot_processed_words = 0

    while tot_processed_words < vocab_size:
        new_chunk = fin.read(binary_chunk_size)
        chunk += new_chunk
        processed_words, chunk = _add_bytes_to_kv(
            kv, counts, chunk, vocab_size, vector_size, datatype, unicode_errors, encoding)
        tot_processed_words += processed_words
        if len(new_chunk) < binary_chunk_size:
            break
    if tot_processed_words != vocab_size:
        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")


def _word2vec_read_text(fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, encoding):
    for line_no in range(vocab_size):
        line = fin.readline()
        if line == b'':
            raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
        word, weights = _word2vec_line_to_vector(line, datatype, unicode_errors, encoding)
        _add_word_to_kv(kv, counts, word, weights, vocab_size)


def _word2vec_line_to_vector(line, datatype, unicode_errors, encoding):
    parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
    word, weights = parts[0], [datatype(x) for x in parts[1:]]
    return word, weights


def _word2vec_detect_sizes_text(fin, limit, datatype, unicode_errors, encoding):
    vector_size = None
    for vocab_size in itertools.count():
        line = fin.readline()
        if line == b'' or vocab_size == limit:  # EOF/max: return what we've got
            break
        if vector_size:
            continue  # don't bother parsing lines past the 1st
        word, weights = _word2vec_line_to_vector(line, datatype, unicode_errors, encoding)
        vector_size = len(weights)
    return vocab_size, vector_size


def _load_word2vec_format(
        cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
        limit=sys.maxsize, datatype=REAL, no_header=False, binary_chunk_size=100 * 1024,
    ):
    """Load the input-hidden weight matrix from the original C word2vec-tool format.

    Note that the information stored in the file is incomplete (the binary tree is missing),
    so while you can query for word similarity etc., you cannot continue training
    with a model loaded this way.

    Parameters
    ----------
    fname : str
        The file path to the saved word2vec-format file.
    fvocab : str, optional
        File path to the vocabulary. Word counts are read from `fvocab` filename, if set
        (this is the file generated by `-save-vocab` flag of the original C tool).
    binary : bool, optional
        If True, indicates whether the data is in binary word2vec format.
    encoding : str, optional
        If you trained the C model using non-utf8 encoding for words, specify that encoding in `encoding`.
    unicode_errors : str, optional
        default 'strict', is a string suitable to be passed as the `errors`
        argument to the unicode() (Python 2.x) or str() (Python 3.x) function. If your source
        file may include word tokens truncated in the middle of a multibyte unicode character
        (as is common from the original word2vec.c tool), 'ignore' or 'replace' may help.
    limit : int, optional
        Sets a maximum number of word-vectors to read from the file. The default,
        None, means read all.
    datatype : type, optional
        (Experimental) Can coerce dimensions to a non-default float type (such as `np.float16`) to save memory.
        Such types may result in much slower bulk operations or incompatibility with optimized routines.)
    binary_chunk_size : int, optional
        Read input file in chunks of this many bytes for performance reasons.

    Returns
    -------
    object
        Returns the loaded model as an instance of :class:`cls`.

    """
    counts = None
    if fvocab is not None:
        logger.info("loading word counts from %s", fvocab)
        counts = {}
        with utils.open(fvocab, 'rb') as fin:
            for line in fin:
                word, count = utils.to_unicode(line, errors=unicode_errors).strip().split()
                counts[word] = int(count)

    logger.info("loading projection weights from %s", fname)
    with utils.open(fname, 'rb') as fin:
        if no_header:
            # deduce both vocab_size & vector_size from 1st pass over file
            if binary:
                raise NotImplementedError("no_header only available for text-format files")
            else:  # text
                vocab_size, vector_size = _word2vec_detect_sizes_text(fin, limit, datatype, unicode_errors, encoding)
            fin.close()
            fin = utils.open(fname, 'rb')
        else:
            header = utils.to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = [int(x) for x in header.split()]  # throws for invalid file format
        if limit:
            vocab_size = min(vocab_size, limit)
        kv = cls(vector_size, vocab_size, dtype=datatype)

        if binary:
            _word2vec_read_binary(
                fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, binary_chunk_size, encoding
            )
        else:
            _word2vec_read_text(fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, encoding)
    if kv.vectors.shape[0] != len(kv):
        logger.info(
            "duplicate words detected, shrinking matrix size from %i to %i",
            kv.vectors.shape[0], len(kv),
        )
        kv.vectors = ascontiguousarray(kv.vectors[: len(kv)])
    assert (len(kv), vector_size) == kv.vectors.shape

    kv.add_lifecycle_event(
        "load_word2vec_format",
        msg=f"loaded {kv.vectors.shape} matrix of type {kv.vectors.dtype} from {fname}",
        binary=binary, encoding=encoding,
    )
    return kv


def load_word2vec_format(*args, **kwargs):
    """Alias for :meth:`~gensim.models.keyedvectors.KeyedVectors.load_word2vec_format`."""
    return KeyedVectors.load_word2vec_format(*args, **kwargs)


def pseudorandom_weak_vector(size, seed_string=None, hashfxn=hash):
    """Get a random vector, derived deterministically from `seed_string` if supplied.

    Useful for initializing KeyedVectors that will be the starting projection/input layers of _2Vec models.

    """
    if seed_string:
        once = np.random.Generator(np.random.SFC64(hashfxn(seed_string) & 0xffffffff))
    else:
        once = utils.default_prng
    return (once.random(size).astype(REAL) - 0.5) / size


def prep_vectors(target_shape, prior_vectors=None, seed=0, dtype=REAL):
    """Return a numpy array of the given shape. Reuse prior_vectors object or values
    to extent possible. Initialize new values randomly if requested.

    """
    if prior_vectors is None:
        prior_vectors = np.zeros((0, 0))
    if prior_vectors.shape == target_shape:
        return prior_vectors
    target_count, vector_size = target_shape
    rng = np.random.default_rng(seed=seed)  # use new instance of numpy's recommended generator/algorithm
    new_vectors = rng.random(target_shape, dtype=dtype)  # [0.0, 1.0)
    new_vectors *= 2.0  # [0.0, 2.0)
    new_vectors -= 1.0  # [-1.0, 1.0)
    new_vectors /= vector_size
    new_vectors[0:prior_vectors.shape[0], 0:prior_vectors.shape[1]] = prior_vectors
    return new_vectors
