#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>
# Copyright (C) 2018 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module implements word vectors and their similarity look-ups.

Since trained word vectors are independent from the way they were trained (:class:`~gensim.models.word2vec.Word2Vec`,
:class:`~gensim.models.fasttext.FastText`, :class:`~gensim.models.wrappers.wordrank.WordRank`,
:class:`~gensim.models.wrappers.varembed.VarEmbed` etc), they can be represented by a standalone structure,
as implemented in this module.

The structure is called "KeyedVectors" and is essentially a mapping between *entities*
and *vectors*. Each entity is identified by its string id, so this is a mapping between {str => 1D numpy array}.

The entity typically corresponds to a word (so the mapping maps words to 1D vectors),
but for some models, they key can also correspond to a document, a graph node etc. To generalize
over different use-cases, this module calls the keys **entities**. Each entity is
always represented by its string id, no matter whether the entity is a word, a document or a graph node.

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
| append new vectors        | ✅           | ✅         | Add new entity-vector entries to the mapping dynamically.   |
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

>>> from gensim.test.utils import common_texts
>>> from gensim.models import Word2Vec
>>>
>>> model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
>>> word_vectors = model.wv

Persist the word vectors to disk with

>>> from gensim.test.utils import get_tmpfile
>>> from gensim.models import KeyedVectors
>>>
>>> fname = get_tmpfile("vectors.kv")
>>> word_vectors.save(fname)
>>> word_vectors = KeyedVectors.load(fname, mmap='r')

The vectors can also be instantiated from an existing file on disk
in the original Google's word2vec C format as a KeyedVectors instance

>>> from gensim.test.utils import datapath
>>>
>>> wv_from_text = KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)  # C text format
>>> wv_from_bin = KeyedVectors.load_word2vec_format(datapath("euclidean_vectors.bin"), binary=True)  # C binary format

What can I do with word vectors?
================================

You can perform various syntactic/semantic NLP word tasks with the trained vectors.
Some of them are already built-in

>>> import gensim.downloader as api
>>>
>>> word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data
>>>
>>> result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
>>> print("{}: {:.4f}".format(*result[0]))
queen: 0.7699
>>>
>>> result = word_vectors.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
>>> print("{}: {:.4f}".format(*result[0]))
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
>>> print("{}: {:.4f}".format(*result[0]))
dog: 0.8798
>>>
>>> sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
>>> sentence_president = 'The president greets the press in Chicago'.lower().split()
>>>
>>> similarity = word_vectors.wmdistance(sentence_obama, sentence_president)
>>> print("{:.4f}".format(similarity))
3.4893
>>>
>>> distance = word_vectors.distance("media", "media")
>>> print("{:.1f}".format(distance))
0.0
>>>
>>> sim = word_vectors.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
>>> print("{:.4f}".format(sim))
0.7067
>>>
>>> vector = word_vectors['computer']  # numpy vector of a word
>>> vector.shape
(100,)
>>>
>>> vector = word_vectors.wv.word_vec('office', use_norm=True)
>>> vector.shape
(100,)

Correlation with human opinion on word similarity

>>> from gensim.test.utils import datapath
>>>
>>> similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))

And on word analogies

>>> analogy_scores = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))

and so on.

"""

from __future__ import division  # py3 "true division"

from collections import deque
import logging

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty  # noqa:F401

# If pyemd C extension is available, import it.
# If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance
try:
    from pyemd import emd
    PYEMD_EXT = True
except ImportError:
    PYEMD_EXT = False

from numpy import dot, float32 as REAL, empty, memmap as np_memmap, \
    double, array, zeros, vstack, sqrt, newaxis, integer, \
    ndarray, sum as np_sum, prod, argmax, divide as np_divide
import numpy as np
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from six import string_types, integer_types
from six.moves import xrange, zip
from scipy import sparse, stats
from gensim.utils import deprecated
from gensim.models.utils_any2vec import _save_word2vec_format, _load_word2vec_format, _compute_ngrams, _ft_hash

logger = logging.getLogger(__name__)


class Vocab(object):
    """A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).

    """
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class BaseKeyedVectors(utils.SaveLoad):
    """Abstract base class / interface for various types of word vectors."""
    def __init__(self, vector_size):
        self.vectors = zeros((0, vector_size))
        self.vocab = {}
        self.vector_size = vector_size
        self.index2entity = []

    def save(self, fname_or_handle, **kwargs):
        super(BaseKeyedVectors, self).save(fname_or_handle, **kwargs)

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        return super(BaseKeyedVectors, cls).load(fname_or_handle, **kwargs)

    def similarity(self, entity1, entity2):
        """Compute cosine similarity between two entities, specified by their string id."""
        raise NotImplementedError()

    def most_similar(self, **kwargs):
        """Find the top-N most similar entities.
        Possibly have `positive` and `negative` list of entities in `**kwargs`.

        """
        return NotImplementedError()

    def distance(self, entity1, entity2):
        """Compute distance between vectors of two input entities, specified by their string id."""
        raise NotImplementedError()

    def distances(self, entity1, other_entities=()):
        """Compute distances from a given entity (its string id) to all entities in `other_entity`.
        If `other_entities` is empty, return the distance between `entity1` and all entities in vocab.

        """
        raise NotImplementedError()

    def get_vector(self, entity):
        """Get the entity's representations in vector space, as a 1D numpy array.

        Parameters
        ----------
        entity : str
            Identifier of the entity to return the vector for.

        Returns
        -------
        numpy.ndarray
            Vector for the specified entity.

        Raises
        ------
        KeyError
            If the given entity identifier doesn't exist.

        """
        if entity in self.vocab:
            result = self.vectors[self.vocab[entity].index]
            result.setflags(write=False)
            return result
        else:
            raise KeyError("'%s' not in vocabulary" % entity)

    def add(self, entities, weights, replace=False):
        """Append entities and theirs vectors in a manual way.
        If some entity is already in the vocabulary, the old vector is kept unless `replace` flag is True.

        Parameters
        ----------
        entities : list of str
            Entities specified by string ids.
        weights: {list of numpy.ndarray, numpy.ndarray}
            List of 1D np.array vectors or a 2D np.array of vectors.
        replace: bool, optional
            Flag indicating whether to replace vectors for entities which already exist in the vocabulary,
            if True - replace vectors, otherwise - keep old vectors.

        """
        if isinstance(entities, string_types):
            entities = [entities]
            weights = np.array(weights).reshape(1, -1)
        elif isinstance(weights, list):
            weights = np.array(weights)

        in_vocab_mask = np.zeros(len(entities), dtype=np.bool)
        for idx, entity in enumerate(entities):
            if entity in self.vocab:
                in_vocab_mask[idx] = True

        # add new entities to the vocab
        for idx in np.nonzero(~in_vocab_mask)[0]:
            entity = entities[idx]
            self.vocab[entity] = Vocab(index=len(self.vocab), count=1)
            self.index2entity.append(entity)

        # add vectors for new entities
        self.vectors = vstack((self.vectors, weights[~in_vocab_mask]))

        # change vectors for in_vocab entities if `replace` flag is specified
        if replace:
            in_vocab_idxs = [self.vocab[entities[idx]].index for idx in np.nonzero(in_vocab_mask)[0]]
            self.vectors[in_vocab_idxs] = weights[in_vocab_mask]

    def __setitem__(self, entities, weights):
        """Add entities and theirs vectors in a manual way.
        If some entity is already in the vocabulary, old vector is replaced with the new one.
        This method is alias for :meth:`~gensim.models.keyedvectors.BaseKeyedVectors.add` with `replace=True`.

        Parameters
        ----------
        entities : {str, list of str}
            Entities specified by their string ids.
        weights: {list of numpy.ndarray, numpy.ndarray}
            List of 1D np.array vectors or 2D np.array of vectors.

        """
        if not isinstance(entities, list):
            entities = [entities]
            weights = weights.reshape(1, -1)

        self.add(entities, weights, replace=True)

    def __getitem__(self, entities):
        """Get vector representation of `entities`.

        Parameters
        ----------
        entities : {str, list of str}
            Input entity/entities.

        Returns
        -------
        numpy.ndarray
            Vector representation for `entities` (1D if `entities` is string, otherwise - 2D).

        """
        if isinstance(entities, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.get_vector(entities)

        return vstack([self.get_vector(entity) for entity in entities])

    def __contains__(self, entity):
        return entity in self.vocab

    def most_similar_to_given(self, entity1, entities_list):
        """Get the `entity` from `entities_list` most similar to `entity1`."""
        return entities_list[argmax([self.similarity(entity1, entity) for entity in entities_list])]

    def closer_than(self, entity1, entity2):
        """Get all entities that are closer to `entity1` than `entity2` is to `entity1`."""
        all_distances = self.distances(entity1)
        e1_index = self.vocab[entity1].index
        e2_index = self.vocab[entity2].index
        closer_node_indices = np.where(all_distances < all_distances[e2_index])[0]
        return [self.index2entity[index] for index in closer_node_indices if index != e1_index]

    def rank(self, entity1, entity2):
        """Rank of the distance of `entity2` from `entity1`, in relation to distances of all entities from `entity1`."""
        return len(self.closer_than(entity1, entity2)) + 1


class WordEmbeddingsKeyedVectors(BaseKeyedVectors):
    """Class containing common methods for operations over word vectors."""
    def __init__(self, vector_size):
        super(WordEmbeddingsKeyedVectors, self).__init__(vector_size=vector_size)
        self.vectors_norm = None
        self.index2word = []

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self instead")
    def wv(self):
        return self

    @property
    def index2entity(self):
        return self.index2word

    @index2entity.setter
    def index2entity(self, value):
        self.index2word = value

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors instead")
    def syn0(self):
        return self.vectors

    @syn0.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors instead")
    def syn0(self, value):
        self.vectors = value

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_norm instead")
    def syn0norm(self):
        return self.vectors_norm

    @syn0norm.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_norm instead")
    def syn0norm(self, value):
        self.vectors_norm = value

    def __contains__(self, word):
        return word in self.vocab

    def save(self, *args, **kwargs):
        """Save KeyedVectors.

        Parameters
        ----------
        fname : str
            Path to the output file.

        See Also
        --------
        :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.load`
            Load saved model.

        """
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_norm'])
        super(WordEmbeddingsKeyedVectors, self).save(*args, **kwargs)

    def word_vec(self, word, use_norm=False):
        """Get `word` representations in vector space, as a 1D numpy array.

        Parameters
        ----------
        word : str
            Input word
        use_norm : bool, optional
            If True - resulting vector will be L2-normalized (unit euclidean length).

        Returns
        -------
        numpy.ndarray
            Vector representation of `word`.

        Raises
        ------
        KeyError
            If word not in vocabulary.

        """
        if word in self.vocab:
            if use_norm:
                result = self.vectors_norm[self.vocab[word].index]
            else:
                result = self.vectors[self.vocab[word].index]

            result.setflags(write=False)
            return result
        else:
            raise KeyError("word '%s' not in vocabulary" % word)

    def get_vector(self, word):
        return self.word_vec(word)

    def words_closer_than(self, w1, w2):
        """Get all words that are closer to `w1` than `w2` is to `w1`.

        Parameters
        ----------
        w1 : str
            Input word.
        w2 : str
            Input word.

        Returns
        -------
        list (str)
            List of words that are closer to `w1` than `w2` is to `w1`.

        """
        return super(WordEmbeddingsKeyedVectors, self).closer_than(w1, w2)

    def most_similar(self, positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None):
        """Find the top-N most similar words.
        Positive words contribute positively towards the similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words and the vectors for each word in the model.
        The method corresponds to the `word-analogy` and `distance` scripts in the original
        word2vec implementation.

        Parameters
        ----------
        positive : list of str, optional
            List of words that contribute positively.
        negative : list of str, optional
            List of words that contribute negatively.
        topn : int, optional
            Number of top-N similar words to return.
        restrict_vocab : int, optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        list of (str, float)
            Sequence of (word, similarity).

        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [
            (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in positive
        ]
        negative = [
            (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in negative
        ]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            else:
                mean.append(weight * self.word_vec(word, use_norm=True))
                if word in self.vocab:
                    all_words.add(self.vocab[word].index)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        limited = self.vectors_norm if restrict_vocab is None else self.vectors_norm[:restrict_vocab]
        dists = dot(limited, mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def similar_by_word(self, word, topn=10, restrict_vocab=None):
        """Find the top-N most similar words.

        Parameters
        ----------
        word : str
            Word
        topn : {int, False}, optional
            Number of top-N similar words to return. If topn is False, similar_by_word returns
            the vector of similarity scores.
        restrict_vocab : int, optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        list of (str, float)
            Sequence of (word, similarity).

        """
        return self.most_similar(positive=[word], topn=topn, restrict_vocab=restrict_vocab)

    def similar_by_vector(self, vector, topn=10, restrict_vocab=None):
        """Find the top-N most similar words by vector.

        Parameters
        ----------
        vector : numpy.array
            Vector from which similarities are to be computed.
        topn : {int, False}, optional
            Number of top-N similar words to return. If topn is False, similar_by_vector returns
            the vector of similarity scores.
        restrict_vocab : int, optional
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        list of (str, float)
            Sequence of (word, similarity).

        """
        return self.most_similar(positive=[vector], topn=topn, restrict_vocab=restrict_vocab)

    def similarity_matrix(self, dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100, dtype=REAL):
        """Construct a term similarity matrix for computing Soft Cosine Measure.

        This creates a sparse term similarity matrix in the :class:`scipy.sparse.csc_matrix` format for computing
        Soft Cosine Measure between documents.

        Parameters
        ----------
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
            A dictionary that specifies a mapping between words and the indices of rows and columns
            of the resulting term similarity matrix.
        tfidf : :class:`gensim.models.tfidfmodel.TfidfModel`, optional
            A model that specifies the relative importance of the terms in the dictionary. The rows
            of the term similarity matrix will be build in a decreasing order of importance of terms,
            or in the order of term identifiers if None.
        threshold : float, optional
            Only pairs of words whose embeddings are more similar than `threshold` are considered
            when building the sparse term similarity matrix.
        exponent : float, optional
            The exponent applied to the similarity between two word embeddings when building the term similarity matrix.
        nonzero_limit : int, optional
            The maximum number of non-zero elements outside the diagonal in a single row or column
            of the term similarity matrix. Setting `nonzero_limit` to a constant ensures that the
            time complexity of computing the Soft Cosine Measure will be linear in the document
            length rather than quadratic.
        dtype : numpy.dtype, optional
            Data-type of the term similarity matrix.

        Returns
        -------
        :class:`scipy.sparse.csc_matrix`
            Term similarity matrix.

        See Also
        --------
        :func:`gensim.matutils.softcossim`
            The Soft Cosine Measure.
        :class:`~gensim.similarities.docsim.SoftCosineSimilarity`
            A class for performing corpus-based similarity queries with Soft Cosine Measure.

        Notes
        -----
        The constructed matrix corresponds to the matrix Mrel defined in section 2.1 of
        `Delphine Charlet and Geraldine Damnati, "SimBow at SemEval-2017 Task 3: Soft-Cosine Semantic Similarity
        between Questions for Community Question Answering", 2017
        <http://www.aclweb.org/anthology/S/S17/S17-2051.pdf>`_.

        """
        logger.info("constructing a term similarity matrix")
        matrix_order = len(dictionary)
        matrix_nonzero = [1] * matrix_order
        matrix = sparse.identity(matrix_order, dtype=dtype, format="dok")
        num_skipped = 0
        # Decide the order of rows.
        if tfidf is None:
            word_indices = deque(sorted(dictionary.keys()))
        else:
            assert max(tfidf.idfs) < matrix_order
            word_indices = deque([
                index for index, _
                in sorted(tfidf.idfs.items(), key=lambda x: (x[1], -x[0]), reverse=True)
            ])

        # Traverse rows.
        for row_number, w1_index in enumerate(list(word_indices)):
            word_indices.popleft()
            if row_number % 1000 == 0:
                logger.info(
                    "PROGRESS: at %.02f%% rows (%d / %d, %d skipped, %.06f%% density)",
                    100.0 * (row_number + 1) / matrix_order, row_number + 1, matrix_order,
                    num_skipped, 100.0 * matrix.getnnz() / matrix_order**2)
            w1 = dictionary[w1_index]
            if w1 not in self.vocab:
                num_skipped += 1
                continue  # A word from the dictionary is not present in the word2vec model.

            # Traverse upper triangle columns.
            if matrix_order <= nonzero_limit + 1:  # Traverse all columns.
                columns = (
                    (w2_index, self.similarity(w1, dictionary[w2_index]))
                    for w2_index in word_indices
                    if dictionary[w2_index] in self.vocab)
            else:  # Traverse only columns corresponding to the embeddings closest to w1.
                num_nonzero = matrix_nonzero[w1_index] - 1
                columns = (
                    (dictionary.token2id[w2], similarity)
                    for _, (w2, similarity)
                    in zip(
                        range(nonzero_limit - num_nonzero),
                        self.most_similar(positive=[w1], topn=nonzero_limit - num_nonzero)
                    )
                    if w2 in dictionary.token2id
                )
                columns = sorted(columns, key=lambda x: x[0])

            for w2_index, similarity in columns:
                # Ensure that we don't exceed `nonzero_limit` by mirroring the upper triangle.
                if similarity > threshold and matrix_nonzero[w2_index] <= nonzero_limit:
                    element = similarity**exponent
                    matrix[w1_index, w2_index] = element
                    matrix_nonzero[w1_index] += 1
                    matrix[w2_index, w1_index] = element
                    matrix_nonzero[w2_index] += 1
        logger.info(
            "constructed a term similarity matrix with %0.6f %% nonzero elements",
            100.0 * matrix.getnnz() / matrix_order**2
        )
        return matrix.tocsc()

    def wmdistance(self, document1, document2):
        """Compute the Word Mover's Distance between two documents.

        When using this code, please consider citing the following papers:

        * `Ofir Pele and Michael Werman "A linear time histogram metric for improved SIFT matching"
          <http://www.cs.huji.ac.il/~werman/Papers/ECCV2008.pdf>`_
        * `Ofir Pele and Michael Werman "Fast and robust earth mover's distances"
          <https://ieeexplore.ieee.org/document/5459199/>`_
        * `Matt Kusner et al. "From Word Embeddings To Document Distances"
          <http://proceedings.mlr.press/v37/kusnerb15.pdf>`_.

        Parameters
        ----------
        document1 : list of str
            Input document.
        document2 : list of str
            Input document.

        Returns
        -------
        float
            Word Mover's distance between `document1` and `document2`.

        Warnings
        --------
        This method only works if `pyemd <https://pypi.org/project/pyemd/>`_ is installed.

        If one of the documents have no words that exist in the vocab, `float('inf')` (i.e. infinity)
        will be returned.

        Raises
        ------
        ImportError
            If `pyemd <https://pypi.org/project/pyemd/>`_  isn't installed.

        """
        if not PYEMD_EXT:
            raise ImportError("Please install pyemd Python package to compute WMD.")

        # Remove out-of-vocabulary words.
        len_pre_oov1 = len(document1)
        len_pre_oov2 = len(document2)
        document1 = [token for token in document1 if token in self]
        document2 = [token for token in document2 if token in self]
        diff1 = len_pre_oov1 - len(document1)
        diff2 = len_pre_oov2 - len(document2)
        if diff1 > 0 or diff2 > 0:
            logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)

        if len(document1) == 0 or len(document2) == 0:
            logger.info(
                "At least one of the documents had no words that werein the vocabulary. "
                "Aborting (returning inf)."
            )
            return float('inf')

        dictionary = Dictionary(documents=[document1, document2])
        vocab_len = len(dictionary)

        if vocab_len == 1:
            # Both documents are composed by a single unique token
            return 0.0

        # Sets for faster look-up.
        docset1 = set(document1)
        docset2 = set(document2)

        # Compute distance matrix.
        distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
        for i, t1 in dictionary.items():
            for j, t2 in dictionary.items():
                if t1 not in docset1 or t2 not in docset2:
                    continue
                # Compute Euclidean distance between word vectors.
                distance_matrix[i, j] = sqrt(np_sum((self[t1] - self[t2])**2))

        if np_sum(distance_matrix) == 0.0:
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

        # Compute nBOW representation of documents.
        d1 = nbow(document1)
        d2 = nbow(document2)

        # Compute WMD.
        return emd(d1, d2, distance_matrix)

    def most_similar_cosmul(self, positive=None, negative=None, topn=10):
        """Find the top-N most similar words, using the multiplicative combination objective,
        proposed by `Omer Levy and Yoav Goldberg "Linguistic Regularities in Sparse and Explicit Word Representations"
        <http://www.aclweb.org/anthology/W14-1618>`_. Positive words still contribute positively towards the similarity,
        negative words negatively, but with less susceptibility to one large distance dominating the calculation.
        In the common analogy-solving case, of two positive and one negative examples,
        this method is equivalent to the "3CosMul" objective (equation (4)) of Levy and Goldberg.

        Additional positive or negative examples contribute to the numerator or denominator,
        respectively - a potentially sensible but untested extension of the method.
        With a single positive example, rankings will be the same as in the default
        :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.most_similar`.

        Parameters
        ----------
        positive : list of str, optional
            List of words that contribute positively.
        negative : list of str, optional
            List of words that contribute negatively.
        topn : int, optional
            Number of top-N similar words to return.

        Returns
        -------
        list of (str, float)
            Sequence of (word, similarity).

        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar_cosmul('dog'), as a shorthand for most_similar_cosmul(['dog'])
            positive = [positive]

        all_words = {
            self.vocab[word].index for word in positive + negative
            if not isinstance(word, ndarray) and word in self.vocab
            }

        positive = [
            self.word_vec(word, use_norm=True) if isinstance(word, string_types) else word
            for word in positive
        ]
        negative = [
            self.word_vec(word, use_norm=True) if isinstance(word, string_types) else word
            for word in negative
        ]

        if not positive:
            raise ValueError("cannot compute similarity with no input")

        # equation (4) of Levy & Goldberg "Linguistic Regularities...",
        # with distances shifted to [0,1] per footnote (7)
        pos_dists = [((1 + dot(self.vectors_norm, term)) / 2) for term in positive]
        neg_dists = [((1 + dot(self.vectors_norm, term)) / 2) for term in negative]
        dists = prod(pos_dists, axis=0) / (prod(neg_dists, axis=0) + 0.000001)

        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def doesnt_match(self, words):
        """Which word from the given list doesn't go with the others?

        Parameters
        ----------
        words : list of str
            List of words.

        Returns
        -------
        str
            The word further away from the mean of all words.

        """
        self.init_sims()

        used_words = [word for word in words if word in self]
        if len(used_words) != len(words):
            ignored_words = set(words) - set(used_words)
            logger.warning("vectors for words %s are not present in the model, ignoring these words", ignored_words)
        if not used_words:
            raise ValueError("cannot select a word from an empty list")
        vectors = vstack(self.word_vec(word, use_norm=True) for word in used_words).astype(REAL)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = dot(vectors, mean)
        return sorted(zip(dists, used_words))[0][1]

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
        If `other_words` is empty, return distance between `word_or_vectors` and all words in vocab.

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
        if isinstance(word_or_vector, string_types):
            input_vector = self.word_vec(word_or_vector)
        else:
            input_vector = word_or_vector
        if not other_words:
            other_vectors = self.vectors
        else:
            other_indices = [self.vocab[word].index for word in other_words]
            other_vectors = self.vectors[other_indices]
        return 1 - self.cosine_similarities(input_vector, other_vectors)

    def distance(self, w1, w2):
        """Compute cosine distance between two words.
        Calculate 1 - :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similarity`.

        Parameters
        ----------
        w1 : str
            Input word.
        w2 : str
            Input word.

        Returns
        -------
        float
            Distance between `w1` and `w2`.

        """
        return 1 - self.similarity(w1, w2)

    def similarity(self, w1, w2):
        """Compute cosine similarity between two words.

        Parameters
        ----------
        w1 : str
            Input word.
        w2 : str
            Input word.

        Returns
        -------
        float
            Cosine similarity between `w1` and `w2`.

        """
        return dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))

    def n_similarity(self, ws1, ws2):
        """Compute cosine similarity between two sets of words.

        Parameters
        ----------
        ws1 : list of str
            Sequence of words.
        ws2: list of str
            Sequence of words.

        Returns
        -------
        numpy.ndarray
            Similarities between `ws1` and `ws2`.

        """
        if not(len(ws1) and len(ws2)):
            raise ZeroDivisionError('At least one of the passed list is empty.')
        v1 = [self[word] for word in ws1]
        v2 = [self[word] for word in ws2]
        return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))

    @staticmethod
    def _log_evaluate_word_analogies(section):
        """Calculate score by section, helper for
        :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.evaluate_word_analogies`.

        Parameters
        ----------
        section : dict of (str, (str, str, str, str))
            Section given from evaluation.

        Returns
        -------
        float
            Accuracy score.

        """
        correct, incorrect = len(section['correct']), len(section['incorrect'])
        if correct + incorrect > 0:
            score = correct / (correct + incorrect)
            logger.info("%s: %.1f%% (%i/%i)", section['section'], 100.0 * score, correct, correct + incorrect)
            return score

    def evaluate_word_analogies(self, analogies, restrict_vocab=300000, case_insensitive=True, dummy4unknown=False):
        """Compute performance of the model on an analogy test set.

        This is modern variant of :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.accuracy`, see
        `discussion on GitHub #1935 <https://github.com/RaRe-Technologies/gensim/pull/1935>`_.

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

        Returns
        -------
        (float, list of dict of (str, (str, str, str))
            Overall evaluation score and full lists of correct and incorrect predictions divided by sections.

        """
        ok_vocab = [(w, self.vocab[w]) for w in self.index2word[:restrict_vocab]]
        ok_vocab = {w.upper(): v for w, v in reversed(ok_vocab)} if case_insensitive else dict(ok_vocab)
        oov = 0
        logger.info("Evaluating word analogies for top %i words in the model on %s", restrict_vocab, analogies)
        sections, section = [], None
        quadruplets_no = 0
        for line_no, line in enumerate(utils.smart_open(analogies)):
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
                original_vocab = self.vocab
                self.vocab = ok_vocab
                ignore = {a, b, c}  # input words to be ignored
                predicted = None
                # find the most likely prediction using 3CosAdd (vector offset) method
                # TODO: implement 3CosMul and set-based methods for solving analogies
                sims = self.most_similar(positive=[b, c], negative=[a], topn=5, restrict_vocab=restrict_vocab)
                self.vocab = original_vocab
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
            'correct': sum((s['correct'] for s in sections), []),
            'incorrect': sum((s['incorrect'] for s in sections), []),
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
                section['section'], 100.0 * correct / (correct + incorrect), correct, correct + incorrect
            )

    @deprecated("Method will be removed in 4.0.0, use self.evaluate_word_analogies() instead")
    def accuracy(self, questions, restrict_vocab=30000, most_similar=most_similar, case_insensitive=True):
        """Compute accuracy of the model.

        The accuracy is reported (=printed to log and returned as a list) for each
        section separately, plus there's one aggregate summary at the end.

        Parameters
        ----------
        questions : str
            Path to file, where lines are 4-tuples of words, split into sections by ": SECTION NAME" lines.
            See `gensim/test/test_data/questions-words.txt` as example.
        restrict_vocab : int, optional
            Ignore all 4-tuples containing a word not in the first `restrict_vocab` words.
            This may be meaningful if you've sorted the model vocabulary by descending frequency (which is standard
            in modern word embedding models).
        most_similar : function, optional
            Function used for similarity calculation.
        case_insensitive : bool, optional
            If True - convert all words to their uppercase form before evaluating the performance.
            Useful to handle case-mismatch between training tokens and words in the test set.
            In case of multiple case variants of a single word, the vector for the first occurrence
            (also the most frequent if vocabulary is sorted) is taken.

        Returns
        -------
        list of dict of (str, (str, str, str)
            Full lists of correct and incorrect predictions divided by sections.

        """
        ok_vocab = [(w, self.vocab[w]) for w in self.index2word[:restrict_vocab]]
        ok_vocab = {w.upper(): v for w, v in reversed(ok_vocab)} if case_insensitive else dict(ok_vocab)

        sections, section = [], None
        for line_no, line in enumerate(utils.smart_open(questions)):
            # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
            line = utils.to_unicode(line)
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    self.log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
            else:
                if not section:
                    raise ValueError("Missing section header before line #%i in %s" % (line_no, questions))
                try:
                    if case_insensitive:
                        a, b, c, expected = [word.upper() for word in line.split()]
                    else:
                        a, b, c, expected = [word for word in line.split()]
                except ValueError:
                    logger.info("Skipping invalid line #%i in %s", line_no, questions)
                    continue
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    logger.debug("Skipping line #%i with OOV words: %s", line_no, line.strip())
                    continue
                original_vocab = self.vocab
                self.vocab = ok_vocab
                ignore = {a, b, c}  # input words to be ignored
                predicted = None
                # find the most likely prediction, ignoring OOV words and input words
                sims = most_similar(self, positive=[b, c], negative=[a], topn=False, restrict_vocab=restrict_vocab)
                self.vocab = original_vocab
                for index in matutils.argsort(sims, reverse=True):
                    predicted = self.index2word[index].upper() if case_insensitive else self.index2word[index]
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
            self.log_accuracy(section)

        total = {
            'section': 'total',
            'correct': sum((s['correct'] for s in sections), []),
            'incorrect': sum((s['incorrect'] for s in sections), []),
        }
        self.log_accuracy(total)
        sections.append(total)
        return sections

    @staticmethod
    def log_evaluate_word_pairs(pearson, spearman, oov, pairs):
        logger.info('Pearson correlation coefficient against %s: %.4f', pairs, pearson[0])
        logger.info('Spearman rank-order correlation coefficient against %s: %.4f', pairs, spearman[0])
        logger.info('Pairs with unknown words ratio: %.1f%%', oov)

    def evaluate_word_pairs(self, pairs, delimiter='\t', restrict_vocab=300000,
                            case_insensitive=True, dummy4unknown=False):
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
        (float, float, float)
            Pearson correlation coefficient, Spearman rank-order correlation coefficient between the similarities
            from the dataset and the similarities produced by the model itself, ratio of pairs with unknown words.

        """
        ok_vocab = [(w, self.vocab[w]) for w in self.index2word[:restrict_vocab]]
        ok_vocab = {w.upper(): v for w, v in reversed(ok_vocab)} if case_insensitive else dict(ok_vocab)

        similarity_gold = []
        similarity_model = []
        oov = 0

        original_vocab = self.vocab
        self.vocab = ok_vocab

        for line_no, line in enumerate(utils.smart_open(pairs)):
            line = utils.to_unicode(line)
            if line.startswith('#'):
                # May be a comment
                continue
            else:
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
                        continue
                    else:
                        logger.debug('Skipping line #%d with OOV words: %s', line_no, line.strip())
                        continue
                similarity_gold.append(sim)  # Similarity from the dataset
                similarity_model.append(self.similarity(a, b))  # Similarity from the model
        self.vocab = original_vocab
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

    def init_sims(self, replace=False):
        """Precompute L2-normalized vectors.

        Parameters
        ----------
        replace : bool, optional
            If True - forget the original vectors and only keep the normalized ones = saves lots of memory!

        Warnings
        --------
        You **cannot continue training** after doing a replace.
        The model becomes effectively read-only: you can call
        :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.most_similar`,
        :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similarity`, etc., but not train.

        """
        if getattr(self, 'vectors_norm', None) is None or replace:
            logger.info("precomputing L2-norms of word weight vectors")
            if replace:
                for i in xrange(self.vectors.shape[0]):
                    self.vectors[i, :] /= sqrt((self.vectors[i, :] ** 2).sum(-1))
                self.vectors_norm = self.vectors
            else:
                self.vectors_norm = (self.vectors / sqrt((self.vectors ** 2).sum(-1))[..., newaxis]).astype(REAL)


class Word2VecKeyedVectors(WordEmbeddingsKeyedVectors):
    """Mapping between words and vectors for the :class:`~gensim.models.Word2Vec` model.
    Used to perform operations on the vectors such as vector lookup, distance, similarity etc.

    """
    def save_word2vec_format(self, fname, fvocab=None, binary=False, total_vec=None):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in
        fvocab : str, optional
            Optional file path used to save the vocabulary
        binary : bool, optional
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec : int, optional
            Optional parameter to explicitly specify total no. of vectors
            (in case word vectors are appended with document vectors afterwards).

        """
        # from gensim.models.word2vec import save_word2vec_format
        _save_word2vec_format(
            fname, self.vocab, self.vectors, fvocab=fvocab, binary=binary, total_vec=total_vec)

    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
                             limit=None, datatype=REAL):
        """Load the input-hidden weight matrix from the original C word2vec-tool format.

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

        Returns
        -------
        :class:`~gensim.models.keyedvectors.Word2VecKeyedVectors`
            Loaded model.

        """
        # from gensim.models.word2vec import load_word2vec_format
        return _load_word2vec_format(
            cls, fname, fvocab=fvocab, binary=binary, encoding=encoding, unicode_errors=unicode_errors,
            limit=limit, datatype=datatype)

    def get_keras_embedding(self, train_embeddings=False):
        """Get a Keras 'Embedding' layer with weights set as the Word2Vec model's learned word embeddings.

        Parameters
        ----------
        train_embeddings : bool
            If False, the weights are frozen and stopped from being updated.
            If True, the weights can/will be further trained/updated.

        Returns
        -------
        `keras.layers.Embedding`
            Embedding layer.

        Raises
        ------
        ImportError
            If `Keras <https://pypi.org/project/Keras/>`_ not installed.

        Warnings
        --------
        Current method work only if `Keras <https://pypi.org/project/Keras/>`_ installed.

        """
        try:
            from keras.layers import Embedding
        except ImportError:
            raise ImportError("Please install Keras to use this function")
        weights = self.vectors

        # set `trainable` as `False` to use the pretrained word embedding
        # No extra mem usage here as `Embedding` layer doesn't create any new matrix for weights
        layer = Embedding(
            input_dim=weights.shape[0], output_dim=weights.shape[1],
            weights=[weights], trainable=train_embeddings
        )
        return layer


KeyedVectors = Word2VecKeyedVectors  # alias for backward compatibility


class Doc2VecKeyedVectors(BaseKeyedVectors):

    def __init__(self, vector_size, mapfile_path):
        super(Doc2VecKeyedVectors, self).__init__(vector_size=vector_size)
        self.doctags = {}  # string -> Doctag (only filled if necessary)
        self.max_rawint = -1  # highest rawint-indexed doctag
        self.offset2doctag = []  # int offset-past-(max_rawint+1) -> String (only filled if necessary)
        self.count = 0
        self.vectors_docs = []
        self.mapfile_path = mapfile_path
        self.vector_size = vector_size
        self.vectors_docs_norm = None

    @property
    def index2entity(self):
        return self.offset2doctag

    @index2entity.setter
    def index2entity(self, value):
        self.offset2doctag = value

    @property
    @deprecated("Attribute will be removed in 4.0.0, use docvecs.vectors_docs instead")
    def doctag_syn0(self):
        return self.vectors_docs

    @property
    @deprecated("Attribute will be removed in 4.0.0, use docvecs.vectors_docs_norm instead")
    def doctag_syn0norm(self):
        return self.vectors_docs_norm

    def __getitem__(self, index):
        """Get vector representation of `index`.

        Parameters
        ----------
        index : {str, list of str}
            Doctag or sequence of doctags.

        Returns
        -------
        numpy.ndarray
            Vector representation for `index` (1D if `index` is string, otherwise - 2D).

        """
        if index in self:
            if isinstance(index, string_types + integer_types + (integer,)):
                return self.vectors_docs[self._int_index(index, self.doctags, self.max_rawint)]
            return vstack([self[i] for i in index])
        raise KeyError("tag '%s' not seen in training corpus/invalid" % index)

    def __contains__(self, index):
        if isinstance(index, integer_types + (integer,)):
            return index < self.count
        else:
            return index in self.doctags

    def __len__(self):
        return self.count

    def save(self, *args, **kwargs):
        """Save object.

        Parameters
        ----------
        fname : str
            Path to the output file.

        See Also
        --------
        :meth:`~gensim.models.keyedvectors.Doc2VecKeyedVectors.load`
            Load object.

        """
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_docs_norm'])
        super(Doc2VecKeyedVectors, self).save(*args, **kwargs)

    def init_sims(self, replace=False):
        """Precompute L2-normalized vectors.

        Parameters
        ----------
        replace : bool, optional
            If True - forget the original vectors and only keep the normalized ones = saves lots of memory!

        Warnings
        --------
        You **cannot continue training** after doing a replace.
        The model becomes effectively read-only: you can call
        :meth:`~gensim.models.keyedvectors.Doc2VecKeyedVectors.most_similar`,
        :meth:`~gensim.models.keyedvectors.Doc2VecKeyedVectors.similarity`, etc., but not train and infer_vector.

        """
        if getattr(self, 'vectors_docs_norm', None) is None or replace:
            logger.info("precomputing L2-norms of doc weight vectors")
            if replace:
                for i in xrange(self.vectors_docs.shape[0]):
                    self.vectors_docs[i, :] /= sqrt((self.vectors_docs[i, :] ** 2).sum(-1))
                self.vectors_docs_norm = self.vectors_docs
            else:
                if self.mapfile_path:
                    self.vectors_docs_norm = np_memmap(
                        self.mapfile_path + '.vectors_docs_norm', dtype=REAL,
                        mode='w+', shape=self.vectors_docs.shape)
                else:
                    self.vectors_docs_norm = empty(self.vectors_docs.shape, dtype=REAL)
                np_divide(
                    self.vectors_docs, sqrt((self.vectors_docs ** 2).sum(-1))[..., newaxis], self.vectors_docs_norm)

    def most_similar(self, positive=None, negative=None, topn=10, clip_start=0, clip_end=None, indexer=None):
        """Find the top-N most similar docvecs from the training set.
        Positive docvecs contribute positively towards the similarity, negative docvecs negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given docs. Docs may be specified as vectors, integer indexes
        of trained docvecs, or if the documents were originally presented with string tags,
        by the corresponding tags.

        TODO: Accept vectors of out-of-training-set docs, as if from inference.

        Parameters
        ----------
        positive : list of {str, int}, optional
            List of doctags/indexes that contribute positively.
        negative : list of {str, int}, optional
            List of doctags/indexes that contribute negatively.
        topn : int, optional
            Number of top-N similar docvecs to return.
        clip_start : int
            Start clipping index.
        clip_end : int
            End clipping index.

        Returns
        -------
        list of ({str, int}, float)
            Sequence of (doctag/index, similarity).

        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.init_sims()
        clip_end = clip_end or len(self.vectors_docs_norm)

        if isinstance(positive, string_types + integer_types + (integer,)) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each doc, if not already present; default to 1.0 for positive and -1.0 for negative docs
        positive = [
            (doc, 1.0) if isinstance(doc, string_types + integer_types + (ndarray, integer))
            else doc for doc in positive
        ]
        negative = [
            (doc, -1.0) if isinstance(doc, string_types + integer_types + (ndarray, integer))
            else doc for doc in negative
        ]

        # compute the weighted average of all docs
        all_docs, mean = set(), []
        for doc, weight in positive + negative:
            if isinstance(doc, ndarray):
                mean.append(weight * doc)
            elif doc in self.doctags or doc < self.count:
                mean.append(weight * self.vectors_docs_norm[self._int_index(doc, self.doctags, self.max_rawint)])
                all_docs.add(self._int_index(doc, self.doctags, self.max_rawint))
            else:
                raise KeyError("doc '%s' not in trained set" % doc)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        dists = dot(self.vectors_docs_norm[clip_start:clip_end], mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_docs), reverse=True)
        # ignore (don't return) docs from the input
        result = [
            (self._index_to_doctag(sim + clip_start, self.offset2doctag, self.max_rawint), float(dists[sim]))
            for sim in best
            if (sim + clip_start) not in all_docs
        ]
        return result[:topn]

    def doesnt_match(self, docs):
        """Which document from the given list doesn't go with the others from the training set?

        TODO: Accept vectors of out-of-training-set docs, as if from inference.

        Parameters
        ----------
        docs : list of {str, int}
            Sequence of doctags/indexes.

        Returns
        -------
        {str, int}
            Doctag/index of the document farthest away from the mean of all the documents.

        """
        self.init_sims()

        docs = [doc for doc in docs if doc in self.doctags or 0 <= doc < self.count]  # filter out unknowns
        logger.debug("using docs %s", docs)
        if not docs:
            raise ValueError("cannot select a doc from an empty list")
        vectors = vstack(
            self.vectors_docs_norm[self._int_index(doc, self.doctags, self.max_rawint)] for doc in docs).astype(REAL)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = dot(vectors, mean)
        return sorted(zip(dists, docs))[0][1]

    def similarity(self, d1, d2):
        """Compute cosine similarity between two docvecs from the training set.

        TODO: Accept vectors of out-of-training-set docs, as if from inference.

        Parameters
        ----------
        d1 : {int, str}
            Doctag/index of document.
        d2 : {int, str}
            Doctag/index of document.

        Returns
        -------
        float
            The cosine similarity between the vectors of the two documents.

        """
        return dot(matutils.unitvec(self[d1]), matutils.unitvec(self[d2]))

    def n_similarity(self, ds1, ds2):
        """Compute cosine similarity between two sets of docvecs from the trained set.

        TODO: Accept vectors of out-of-training-set docs, as if from inference.

        Parameters
        ----------
        ds1 : list of {str, int}
            Set of document as sequence of doctags/indexes.
        ds2 : list of {str, int}
            Set of document as sequence of doctags/indexes.

        Returns
        -------
        float
            The cosine similarity between the means of the documents in each of the two sets.

        """
        v1 = [self[doc] for doc in ds1]
        v2 = [self[doc] for doc in ds2]
        return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))

    def distance(self, d1, d2):
        """
        Compute cosine distance between two documents.

        """
        return 1 - self.similarity(d1, d2)

    # required by base keyed vectors class
    def distances(self, d1, other_docs=()):
        """Compute cosine distances from given `d1` to all documents in `other_docs`.

        TODO: Accept vectors of out-of-training-set docs, as if from inference.

        Parameters
        ----------
        d1 : {str, numpy.ndarray}
            Doctag/index of document.
        other_docs : iterable of {str, int}
            Sequence of doctags/indexes.
            If None or empty, distance of `d1` from all doctags in vocab is computed (including itself).

        Returns
        -------
        numpy.array
            Array containing distances to all documents in `other_docs` from input `d1`.

        """
        input_vector = self[d1]
        if not other_docs:
            other_vectors = self.vectors_docs
        else:
            other_vectors = self[other_docs]
        return 1 - WordEmbeddingsKeyedVectors.cosine_similarities(input_vector, other_vectors)

    def similarity_unseen_docs(self, model, doc_words1, doc_words2, alpha=0.1, min_alpha=0.0001, steps=5):
        """Compute cosine similarity between two post-bulk out of training documents.

        Parameters
        ----------
        model : :class:`~gensim.models.doc2vec.Doc2Vec`
            An instance of a trained `Doc2Vec` model.
        doc_words1 : list of str
            Input document.
        doc_words2 : list of str
            Input document.
        alpha : float, optional
            The initial learning rate.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        steps : int, optional
            Number of epoch to train the new document.

        Returns
        -------
        float
            The cosine similarity between `doc_words1` and `doc_words2`.

        """
        d1 = model.infer_vector(doc_words=doc_words1, alpha=alpha, min_alpha=min_alpha, steps=steps)
        d2 = model.infer_vector(doc_words=doc_words2, alpha=alpha, min_alpha=min_alpha, steps=steps)
        return dot(matutils.unitvec(d1), matutils.unitvec(d2))

    def save_word2vec_format(self, fname, prefix='*dt_', fvocab=None,
                             total_vec=None, binary=False, write_first_line=True):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        prefix : str, optional
            Uniquely identifies doctags from word vocab, and avoids collision
            in case of repeated string in doctag and word vocab.
        fvocab : str, optional
            UNUSED.
        total_vec : int, optional
            Explicitly specify total no. of vectors
            (in case word vectors are appended with document vectors afterwards)
        binary : bool, optional
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        write_first_line : bool, optional
            Whether to print the first line in the file. Useful when saving doc-vectors after word-vectors.

        """
        total_vec = total_vec or len(self)
        with utils.smart_open(fname, 'ab') as fout:
            if write_first_line:
                logger.info("storing %sx%s projection weights into %s", total_vec, self.vectors_docs.shape[1], fname)
                fout.write(utils.to_utf8("%s %s\n" % (total_vec, self.vectors_docs.shape[1])))
            # store as in input order
            for i in range(len(self)):
                doctag = u"%s%s" % (prefix, self._index_to_doctag(i, self.offset2doctag, self.max_rawint))
                row = self.vectors_docs[i]
                if binary:
                    fout.write(utils.to_utf8(doctag) + b" " + row.tostring())
                else:
                    fout.write(utils.to_utf8("%s %s\n" % (doctag, ' '.join("%f" % val for val in row))))

    @staticmethod
    def _int_index(index, doctags, max_rawint):
        """Get int index for either string or int index."""
        if isinstance(index, integer_types + (integer,)):
            return index
        else:
            return max_rawint + 1 + doctags[index].offset

    @staticmethod
    def _index_to_doctag(i_index, offset2doctag, max_rawint):
        """Get string key for given `i_index`, if available. Otherwise return raw int doctag (same int)."""
        candidate_offset = i_index - max_rawint - 1
        if 0 <= candidate_offset < len(offset2doctag):
            return offset2doctag[candidate_offset]
        else:
            return i_index

    # for backward compatibility
    def index_to_doctag(self, i_index):
        """Get string key for given `i_index`, if available. Otherwise return raw int doctag (same int)."""
        candidate_offset = i_index - self.max_rawint - 1
        if 0 <= candidate_offset < len(self.offset2doctag):
            return self.offset2doctag[candidate_offset]
        else:
            return i_index

    # for backward compatibility
    def int_index(self, index, doctags, max_rawint):
        """Get int index for either string or int index"""
        if isinstance(index, integer_types + (integer,)):
            return index
        else:
            return max_rawint + 1 + doctags[index].offset


class FastTextKeyedVectors(WordEmbeddingsKeyedVectors):
    """Vectors and vocab for :class:`~gensim.models.fasttext.FastText`."""
    def __init__(self, vector_size, min_n, max_n):
        super(FastTextKeyedVectors, self).__init__(vector_size=vector_size)
        self.vectors_vocab = None
        self.vectors_vocab_norm = None
        self.vectors_ngrams = None
        self.vectors_ngrams_norm = None
        self.buckets_word = None
        self.hash2index = {}
        self.min_n = min_n
        self.max_n = max_n
        self.num_ngram_vectors = 0

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_vocab instead")
    def syn0_vocab(self):
        return self.vectors_vocab

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_vocab_norm instead")
    def syn0_vocab_norm(self):
        return self.vectors_vocab_norm

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_ngrams instead")
    def syn0_ngrams(self):
        return self.vectors_ngrams

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_ngrams_norm instead")
    def syn0_ngrams_norm(self):
        return self.vectors_ngrams_norm

    def __contains__(self, word):
        """Check if `word` or any character ngrams in `word` are present in the vocabulary.
        A vector for the word is guaranteed to exist if current method returns True.

        Parameters
        ----------
        word : str
            Input word.

        Returns
        -------
        bool
            True if `word` or any character ngrams in `word` are present in the vocabulary, False otherwise.

        """
        if word in self.vocab:
            return True
        else:
            char_ngrams = _compute_ngrams(word, self.min_n, self.max_n)
            return any(_ft_hash(ng) % self.bucket in self.hash2index for ng in char_ngrams)

    def save(self, *args, **kwargs):
        """Save object.

        Parameters
        ----------
        fname : str
            Path to the output file.

        See Also
        --------
        :meth:`~gensim.models.keyedvectors.FastTextKeyedVectors.load`
            Load object.

        """
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get(
            'ignore', ['vectors_norm', 'vectors_vocab_norm', 'vectors_ngrams_norm', 'buckets_word'])
        super(FastTextKeyedVectors, self).save(*args, **kwargs)

    def word_vec(self, word, use_norm=False):
        """Get `word` representations in vector space, as a 1D numpy array.

        Parameters
        ----------
        word : str
            Input word
        use_norm : bool, optional
            If True - resulting vector will be L2-normalized (unit euclidean length).

        Returns
        -------
        numpy.ndarray
            Vector representation of `word`.

        Raises
        ------
        KeyError
            If word and all ngrams not in vocabulary.

        """
        if word in self.vocab:
            return super(FastTextKeyedVectors, self).word_vec(word, use_norm)
        else:
            # from gensim.models.fasttext import compute_ngrams
            word_vec = np.zeros(self.vectors_ngrams.shape[1], dtype=np.float32)
            ngrams = _compute_ngrams(word, self.min_n, self.max_n)
            if use_norm:
                ngram_weights = self.vectors_ngrams_norm
            else:
                ngram_weights = self.vectors_ngrams
            ngrams_found = 0
            for ngram in ngrams:
                ngram_hash = _ft_hash(ngram) % self.bucket
                if ngram_hash in self.hash2index:
                    word_vec += ngram_weights[self.hash2index[ngram_hash]]
                    ngrams_found += 1
            if word_vec.any():
                return word_vec / max(1, ngrams_found)
            else:  # No ngrams of the word are present in self.ngrams
                raise KeyError('all ngrams for word %s absent from model' % word)

    def init_sims(self, replace=False):
        """Precompute L2-normalized vectors.

        Parameters
        ----------
        replace : bool, optional
            If True - forget the original vectors and only keep the normalized ones = saves lots of memory!

        Warnings
        --------
        You **cannot continue training** after doing a replace.
        The model becomes effectively read-only: you can call
        :meth:`~gensim.models.keyedvectors.FastTextKeyedVectors.most_similar`,
        :meth:`~gensim.models.keyedvectors.FastTextKeyedVectors.similarity`, etc., but not train.

        """
        super(FastTextKeyedVectors, self).init_sims(replace)
        if getattr(self, 'vectors_ngrams_norm', None) is None or replace:
            logger.info("precomputing L2-norms of ngram weight vectors")
            if replace:
                for i in range(self.vectors_ngrams.shape[0]):
                    self.vectors_ngrams[i, :] /= sqrt((self.vectors_ngrams[i, :] ** 2).sum(-1))
                self.vectors_ngrams_norm = self.vectors_ngrams
            else:
                self.vectors_ngrams_norm = \
                    (self.vectors_ngrams / sqrt((self.vectors_ngrams ** 2).sum(-1))[..., newaxis]).astype(REAL)

    def save_word2vec_format(self, fname, fvocab=None, binary=False, total_vec=None):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in
        fvocab : str, optional
            Optional file path used to save the vocabulary
        binary : bool, optional
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec : int, optional
            Optional parameter to explicitly specify total no. of vectors
            (in case word vectors are appended with document vectors afterwards).

        """
        # from gensim.models.word2vec import save_word2vec_format
        _save_word2vec_format(
            fname, self.vocab, self.vectors, fvocab=fvocab, binary=binary, total_vec=total_vec)
