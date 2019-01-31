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
but for some models, the key can also correspond to a document, a graph node etc. To generalize
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

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_texts
    >>> from gensim.models import Word2Vec
    >>>
    >>> model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
    >>> word_vectors = model.wv

Persist the word vectors to disk with

.. sourcecode:: pycon

    >>> from gensim.test.utils import get_tmpfile
    >>> from gensim.models import KeyedVectors
    >>>
    >>> fname = get_tmpfile("vectors.kv")
    >>> word_vectors.save(fname)
    >>> word_vectors = KeyedVectors.load(fname, mmap='r')

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

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))

And on word analogies

.. sourcecode:: pycon

    >>> analogy_scores = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))

and so on.

"""

from __future__ import division  # py3 "true division"

from itertools import chain
import logging

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty  # noqa:F401

from numpy import dot, float32 as REAL, memmap as np_memmap, \
    double, array, zeros, vstack, sqrt, newaxis, integer, \
    ndarray, sum as np_sum, prod, argmax
import numpy as np

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from six import string_types, integer_types
from six.moves import zip, range
from scipy import stats
from gensim.utils import deprecated
from gensim.models.utils_any2vec import (
    _save_word2vec_format,
    _load_word2vec_format,
    ft_ngram_hashes,
)
from gensim.similarities.termsim import TermSimilarityIndex, SparseTermSimilarityMatrix

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
    @deprecated("Attribute will be removed in 4.0.0, use self.vectors instead")
    def syn0(self):
        return self.vectors

    @syn0.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.vectors instead")
    def syn0(self, value):
        self.vectors = value

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.vectors_norm instead")
    def syn0norm(self):
        return self.vectors_norm

    @syn0norm.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.vectors_norm instead")
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
        if topn is not None and topn < 1:
            return []

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
        if topn is None:
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

    @deprecated(
        "Method will be removed in 4.0.0, use "
        "gensim.models.keyedvectors.WordEmbeddingSimilarityIndex instead")
    def similarity_matrix(self, dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100, dtype=REAL):
        """Construct a term similarity matrix for computing Soft Cosine Measure.

        This creates a sparse term similarity matrix in the :class:`scipy.sparse.csc_matrix` format for computing
        Soft Cosine Measure between documents.

        Parameters
        ----------
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
            A dictionary that specifies the considered terms.
        tfidf : :class:`gensim.models.tfidfmodel.TfidfModel` or None, optional
            A model that specifies the relative importance of the terms in the dictionary. The
            columns of the term similarity matrix will be build in a decreasing order of importance
            of terms, or in the order of term identifiers if None.
        threshold : float, optional
            Only embeddings more similar than `threshold` are considered when retrieving word
            embeddings closest to a given word embedding.
        exponent : float, optional
            Take the word embedding similarities larger than `threshold` to the power of `exponent`.
        nonzero_limit : int, optional
            The maximum number of non-zero elements outside the diagonal in a single column of the
            sparse term similarity matrix.
        dtype : numpy.dtype, optional
            Data-type of the sparse term similarity matrix.

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
        index = WordEmbeddingSimilarityIndex(self, threshold=threshold, exponent=exponent)
        similarity_matrix = SparseTermSimilarityMatrix(
            index, dictionary, tfidf=tfidf, nonzero_limit=nonzero_limit, dtype=dtype)
        return similarity_matrix.matrix

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

        # If pyemd C extension is available, import it.
        # If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance
        from pyemd import emd

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
            logger.info(
                "At least one of the documents had no words that were in the vocabulary. "
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
            if t1 not in docset1:
                continue

            for j, t2 in dictionary.items():
                if t2 not in docset2 or distance_matrix[i, j] != 0.0:
                    continue

                # Compute Euclidean distance between word vectors.
                distance_matrix[i, j] = distance_matrix[j, i] = sqrt(np_sum((self[t1] - self[t2])**2))

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
        score : float
            The overall evaluation score on the entire evaluation set
        sections : list of dict of {str : str or list of tuple of (str, str, str, str)}
            Results broken down by each section of the evaluation set. Each dict contains the name of the section
            under the key 'section', and lists of correctly and incorrectly predicted 4-tuples of words under the
            keys 'correct' and 'incorrect'.

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
            'correct': list(chain.from_iterable(s['correct'] for s in sections)),
            'incorrect': list(chain.from_iterable(s['incorrect'] for s in sections)),
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
            'correct': list(chain.from_iterable(s['correct'] for s in sections)),
            'incorrect': list(chain.from_iterable(s['incorrect'] for s in sections)),
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
        pearson : tuple of (float, float)
            Pearson correlation coefficient with 2-tailed p-value.
        spearman : tuple of (float, float)
            Spearman rank-order correlation coefficient between the similarities from the dataset and the
            similarities produced by the model itself, with 2-tailed p-value.
        oov_ratio : float
            The ratio of pairs with unknown words.

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
            self.vectors_norm = _l2_norm(self.vectors, replace=replace)

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
        assert sims, "Failed code invariant: list of similar words must never be empty."
        rcs = float(self.similarity(wa, wb)) / (sum(sim for _, sim in sims))

        return rcs


class WordEmbeddingSimilarityIndex(TermSimilarityIndex):
    """
    Computes cosine similarities between word embeddings and retrieves the closest word embeddings
    by cosine similarity for a given word embedding.

    Parameters
    ----------
    keyedvectors : :class:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors`
        The word embeddings.
    threshold : float, optional
        Only embeddings more similar than `threshold` are considered when retrieving word embeddings
        closest to a given word embedding.
    exponent : float, optional
        Take the word embedding similarities larger than `threshold` to the power of `exponent`.
    kwargs : dict or None
        A dict with keyword arguments that will be passed to the `keyedvectors.most_similar` method
        when retrieving the word embeddings closest to a given word embedding.

    See Also
    --------
    :class:`~gensim.similarities.termsim.SparseTermSimilarityMatrix`
        Build a term similarity matrix and compute the Soft Cosine Measure.

    """
    def __init__(self, keyedvectors, threshold=0.0, exponent=2.0, kwargs=None):
        assert isinstance(keyedvectors, WordEmbeddingsKeyedVectors)
        self.keyedvectors = keyedvectors
        self.threshold = threshold
        self.exponent = exponent
        self.kwargs = kwargs or {}
        super(WordEmbeddingSimilarityIndex, self).__init__()

    def most_similar(self, t1, topn=10):
        if t1 not in self.keyedvectors.vocab:
            logger.debug('an out-of-dictionary term "%s"', t1)
        else:
            most_similar = self.keyedvectors.most_similar(positive=[t1], topn=topn, **self.kwargs)
            for t2, similarity in most_similar:
                if similarity > self.threshold:
                    yield (t2, similarity**self.exponent)


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
            If True, the data will be saved in binary word2vec format, else it will be saved in plain text.
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

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        model = super(WordEmbeddingsKeyedVectors, cls).load(fname_or_handle, **kwargs)
        if isinstance(model, FastTextKeyedVectors):
            if not hasattr(model, 'compatible_hash'):
                model.compatible_hash = False

        return model


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
            if not replace and self.mapfile_path:
                self.vectors_docs_norm = np_memmap(
                    self.mapfile_path + '.vectors_docs_norm', dtype=REAL,
                    mode='w+', shape=self.vectors_docs.shape)
            else:
                self.vectors_docs_norm = _l2_norm(self.vectors_docs, replace=replace)

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
            If True, the data will be saved in binary word2vec format, else it will be saved in plain text.
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
    """Vectors and vocab for :class:`~gensim.models.fasttext.FastText`.

    Implements significant parts of the FastText algorithm.  For example,
    the :func:`word_vec` calculates vectors for out-of-vocabulary (OOV)
    entities.  FastText achieves this by keeping vectors for ngrams:
    adding the vectors for the ngrams of an entity yields the vector for the
    entity.

    Similar to a hashmap, this class keeps a fixed number of buckets, and
    maps all ngrams to buckets using a hash function.

    This class also provides an abstraction over the hash functions used by
    Gensim's FastText implementation over time.  The hash function connects
    ngrams to buckets.  Originally, the hash function was broken and
    incompatible with Facebook's implementation.  The current hash is fully
    compatible.

    Parameters
    ----------
    vector_size : int
        The dimensionality of all vectors.
    min_n : int
        The minimum number of characters in an ngram
    max_n : int
        The maximum number of characters in an ngram
    bucket : int
        The number of buckets.
    compatible_hash : boolean
        If True, uses the Facebook-compatible hash function instead of the
        Gensim backwards-compatible hash function.

    Attributes
    ----------
    vectors_vocab : np.array
        Each row corresponds to a vector for an entity in the vocabulary.
        Columns correspond to vector dimensions.
    vectors_vocab_norm : np.array
        Same as vectors_vocab, but the vectors are L2 normalized.
    vectors_ngrams : np.array
        A vector for each ngram across all entities in the vocabulary.
        Each row is a vector that corresponds to a bucket.
        Columns correspond to vector dimensions.
    vectors_ngrams_norm : np.array
        Same as vectors_ngrams, but the vectors are L2 normalized.
        Under some conditions, may actually be the same matrix as
        vectors_ngrams, e.g. if :func:`init_sims` was called with
        replace=True.
    buckets_word : dict
        Maps vocabulary items (by their index) to the buckets they occur in.
    hash2index : dict
        Maps bucket numbers to an index within vectors_ngrams.  So, given an
        ngram, you can get its vector by determining its bucket, mapping the
        bucket to an index, and then indexing into vectors_ngrams (in other
        words, vectors_ngrams[hash2index[hash_fn(ngram) % bucket]].
    num_ngram_vectors : int
        The number of vectors that correspond to ngrams, as opposed to terms
        (full words).

    """
    def __init__(self, vector_size, min_n, max_n, bucket, compatible_hash):
        super(FastTextKeyedVectors, self).__init__(vector_size=vector_size)
        self.vectors_vocab = None
        self.vectors_vocab_norm = None
        self.vectors_ngrams = None
        self.vectors_ngrams_norm = None
        self.buckets_word = None
        self.hash2index = {}
        self.min_n = min_n
        self.max_n = max_n
        self.bucket = bucket
        self.num_ngram_vectors = 0
        self.compatible_hash = compatible_hash

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        model = super(WordEmbeddingsKeyedVectors, cls).load(fname_or_handle, **kwargs)
        if not hasattr(model, 'compatible_hash'):
            model.compatible_hash = False

        return model

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.vectors_vocab instead")
    def syn0_vocab(self):
        return self.vectors_vocab

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.vectors_vocab_norm instead")
    def syn0_vocab_norm(self):
        return self.vectors_vocab_norm

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.vectors_ngrams instead")
    def syn0_ngrams(self):
        return self.vectors_ngrams

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.vectors_ngrams_norm instead")
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
            hashes = ft_ngram_hashes(word, self.min_n, self.max_n, self.bucket, self.compatible_hash)
            return any(h in self.hash2index for h in hashes)

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
        elif self.bucket == 0:
            raise KeyError('cannot calculate vector for OOV word without ngrams')
        else:
            word_vec = np.zeros(self.vectors_ngrams.shape[1], dtype=np.float32)
            if use_norm:
                ngram_weights = self.vectors_ngrams_norm
            else:
                ngram_weights = self.vectors_ngrams
            ngrams_found = 0
            for ngram_hash in ft_ngram_hashes(word, self.min_n, self.max_n, self.bucket, self.compatible_hash):
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
            self.vectors_ngrams_norm = _l2_norm(self.vectors_ngrams, replace=replace)

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

    def init_ngrams_weights(self, seed):
        self.hash2index = {}
        ngram_indices, self.buckets_word = _process_fasttext_vocab(
            self.vocab.items(),
            self.min_n,
            self.max_n,
            self.bucket,
            self.compatible_hash,
            self.hash2index
        )
        self.num_ngram_vectors = len(ngram_indices)
        logger.info("Total number of ngrams is %d", self.num_ngram_vectors)

        rand_obj = np.random
        rand_obj.seed(seed)

        lo, hi = -1.0 / self.vector_size, 1.0 / self.vector_size
        vocab_shape = (len(self.vocab), self.vector_size)
        ngrams_shape = (len(ngram_indices), self.vector_size)
        self.vectors_vocab = rand_obj.uniform(lo, hi, vocab_shape).astype(REAL)
        self.vectors_ngrams = rand_obj.uniform(lo, hi, ngrams_shape).astype(REAL)

    def update_ngrams_weights(self, seed, old_vocab_len):
        old_hash2index_len = len(self.hash2index)

        new_ngram_hashes, self.buckets_word = _process_fasttext_vocab(
            self.vocab.items(),
            self.min_n,
            self.max_n,
            self.bucket,
            self.compatible_hash,
            self.hash2index
        )
        num_new_ngrams = len(new_ngram_hashes)
        self.num_ngram_vectors += num_new_ngrams
        logger.info("Number of new ngrams is %d", num_new_ngrams)

        rand_obj = np.random
        rand_obj.seed(seed)

        new_vocab = len(self.vocab) - old_vocab_len
        self.vectors_vocab = _pad_random(self.vectors_vocab, new_vocab, rand_obj)

        new_ngrams = len(self.hash2index) - old_hash2index_len
        self.vectors_ngrams = _pad_random(self.vectors_ngrams, new_ngrams, rand_obj)

    def init_post_load(self, vectors, match_gensim=False):
        """Perform initialization after loading a native Facebook model.

        Expects that the vocabulary (self.vocab) has already been initialized.

        Parameters
        ----------
        vectors : np.array
            A matrix containing vectors for all the entities, including words
            and ngrams.  This comes directly from the binary model.
            The order of the vectors must correspond to the indices in
            the vocabulary.
        match_gensim : boolean, optional
            Match the behavior of gensim's FastText implementation and take a
            subset of vectors_ngrams.  This behavior appears to be incompatible
            with Facebook's implementation.

        """
        vocab_words = len(self.vocab)
        assert vectors.shape[0] == vocab_words + self.bucket, 'unexpected number of vectors'
        assert vectors.shape[1] == self.vector_size, 'unexpected vector dimensionality'

        #
        # The incoming vectors contain vectors for both words AND
        # ngrams.  We split them into two separate matrices, because our
        # implementation treats them differently.
        #
        self.vectors = np.array(vectors[:vocab_words, :])
        self.vectors_vocab = np.array(vectors[:vocab_words, :])
        self.vectors_ngrams = np.array(vectors[vocab_words:, :])
        self.hash2index = {i: i for i in range(self.bucket)}
        self.buckets_word = None  # This can get initialized later
        self.num_ngram_vectors = self.bucket

        if match_gensim:
            #
            # This gives us the same shape for vectors_ngrams, and we can
            # satisfy our unit tests when running gensim vs native comparisons,
            # but because we're discarding some ngrams, the accuracy of the
            # model suffers.
            #
            ngram_hashes, _ = _process_fasttext_vocab(
                self.vocab.items(),
                self.min_n,
                self.max_n,
                self.bucket,
                self.compatible_hash,
                dict(),  # we don't care what goes here in this case
            )
            ngram_hashes = sorted(set(ngram_hashes))

            keep_indices = [self.hash2index[h] for h in self.hash2index if h in ngram_hashes]
            self.num_ngram_vectors = len(keep_indices)
            self.vectors_ngrams = self.vectors_ngrams.take(keep_indices, axis=0)
            self.hash2index = {hsh: idx for (idx, hsh) in enumerate(ngram_hashes)}

        self.adjust_vectors()

    def adjust_vectors(self):
        """Adjust the vectors for words in the vocabulary.

        The adjustment relies on the vectors of the ngrams making up each
        individual word.

        """
        if self.bucket == 0:
            return

        for w, v in self.vocab.items():
            word_vec = np.copy(self.vectors_vocab[v.index])
            ngram_hashes = ft_ngram_hashes(w, self.min_n, self.max_n, self.bucket, self.compatible_hash)
            for nh in ngram_hashes:
                word_vec += self.vectors_ngrams[self.hash2index[nh]]
            word_vec /= len(ngram_hashes) + 1
            self.vectors[v.index] = word_vec


def _process_fasttext_vocab(iterable, min_n, max_n, num_buckets, compatible_hash, hash2index):
    """
    Performs a common operation for FastText weight initialization and
    updates: scan the vocabulary, calculate ngrams and their hashes, keep
    track of new ngrams, the buckets that each word relates to via its
    ngrams, etc.

    Parameters
    ----------
    iterable : list
        A list of (word, :class:`Vocab`) tuples.
    min_n : int
        The minimum length of ngrams.
    max_n : int
        The maximum length of ngrams.
    num_buckets : int
        The number of buckets used by the model.
    compatible_hash : boolean
        True for compatibility with the Facebook implementation.
        False for compatibility with the old Gensim implementation.
    hash2index : dict
        Updated in-place.

    Returns
    -------
    A tuple of two elements.

    word_indices : dict
        Keys are indices of entities in the vocabulary (words).  Values are
        arrays containing indices into vectors_ngrams for each ngram of the
        word.
    new_ngram_hashes : list
        A list of hashes for newly encountered ngrams.  Each hash is modulo
        num_buckets.

    """
    old_hash2index_len = len(hash2index)
    word_indices = {}
    new_ngram_hashes = []

    if num_buckets == 0:
        return [], {v.index: np.array([], dtype=np.uint32) for w, v in iterable}

    for word, vocab in iterable:
        wi = []
        for ngram_hash in ft_ngram_hashes(word, min_n, max_n, num_buckets, compatible_hash):
            if ngram_hash not in hash2index:
                #
                # This is a new ngram.  Reserve a new index in hash2index.
                #
                hash2index[ngram_hash] = old_hash2index_len + len(new_ngram_hashes)
                new_ngram_hashes.append(ngram_hash)
            wi.append(hash2index[ngram_hash])
        word_indices[vocab.index] = np.array(wi, dtype=np.uint32)

    return new_ngram_hashes, word_indices


def _pad_random(m, new_rows, rand):
    """Pad a matrix with additional rows filled with random values."""
    rows, columns = m.shape
    low, high = -1.0 / columns, 1.0 / columns
    suffix = rand.uniform(low, high, (new_rows, columns)).astype(REAL)
    return vstack([m, suffix])


def _l2_norm(m, replace=False):
    """Return an L2-normalized version of a matrix.

    Parameters
    ----------
    m : np.array
        The matrix to normalize.
    replace : boolean, optional
        If True, modifies the existing matrix.

    Returns
    -------
    The normalized matrix.  If replace=True, this will be the same as m.

    """
    dist = sqrt((m ** 2).sum(-1))[..., newaxis]
    if replace:
        m /= dist
        return m
    else:
        return (m / dist).astype(REAL)
