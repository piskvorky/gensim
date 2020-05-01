# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Radim Rehurek <me@radimrehurek.com>
# Copyright (C) 2019 Masahiro Kazama <kazama.masa@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Intro
-----

This module contains integration NMSLIB with :class:`~gensim.models.word2vec.Word2Vec`,
:class:`~gensim.models.doc2vec.Doc2Vec`, :class:`~gensim.models.fasttext.FastText` and
:class:`~gensim.models.keyedvectors.KeyedVectors`.
To use NMSLIB, instantiate a :class:`~gensim.similarities.nmslib.NmslibIndexer` class
and pass the instance as the indexer parameter to your model's most_similar method
(e.g. :py:func:`~gensim.models.doc2vec.most_similar`).

Example usage
-------------

.. sourcecode:: pycon

    >>> from gensim.similarities.nmslib import NmslibIndexer
    >>> from gensim.models import Word2Vec
    >>>
    >>> sentences = [['cute', 'cat', 'say', 'meow'], ['cute', 'dog', 'say', 'woof']]
    >>> model = Word2Vec(sentences, min_count=1, seed=1)
    >>>
    >>> indexer = NmslibIndexer(model)
    >>> model.most_similar("cat", topn=2, indexer=indexer)
    [('cat', 1.0), ('meow', 0.5595494508743286)]

Load and save example
---------------------

.. sourcecode:: pycon

    >>> from gensim.similarities.nmslib import NmslibIndexer
    >>> from gensim.models import Word2Vec
    >>> from tempfile import mkstemp
    >>>
    >>> sentences = [['cute', 'cat', 'say', 'meow'], ['cute', 'dog', 'say', 'woof']]
    >>> model = Word2Vec(sentences, min_count=1, seed=1, iter=10)
    >>>
    >>> indexer = NmslibIndexer(model)
    >>> _, temp_fn = mkstemp()
    >>> indexer.save(temp_fn)
    >>>
    >>> new_indexer = NmslibIndexer.load(temp_fn)
    >>> model.most_similar("cat", topn=2, indexer=new_indexer)
    [('cat', 1.0), ('meow', 0.5595494508743286)]

What is NMSLIB
--------------

Non-Metric Space Library (NMSLIB) is an efficient cross-platform similarity search library and a toolkit
for evaluation of similarity search methods. The core-library does not have any third-party dependencies.
More information about NMSLIB: `github repository <https://github.com/nmslib/nmslib>`_.

Why use NMSIB?
--------------

The current implementation for finding k nearest neighbors in a vector space in gensim has linear complexity
via brute force in the number of indexed documents, although with extremely low constant factors.
The retrieved results are exact, which is an overkill in many applications:
approximate results retrieved in sub-linear time may be enough.
NMSLIB can find approximate nearest neighbors much faster.
Compared to Annoy, NMSLIB has more parameters to control the build and query time and accuracy.
NMSLIB can achieve faster and more accurate nearest neighbors search than annoy.
"""
from __future__ import absolute_import  # avoid import collision on py2 (nmslib.py - "bad" name)

from smart_open import open
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
try:
    import nmslib
except ImportError:
    raise ImportError(
        "NMSLIB not installed. To use the NMSLIB indexer, please run `pip install nmslib`."
    )


class NmslibIndexer(object):
    """This class allows to use `NMSLIB <https://github.com/nmslib/nmslib>`_ as indexer for `most_similar` method
    from :class:`~gensim.models.word2vec.Word2Vec`, :class:`~gensim.models.doc2vec.Doc2Vec`,
    :class:`~gensim.models.fasttext.FastText` and :class:`~gensim.models.keyedvectors.Word2VecKeyedVectors` classes.

    """

    def __init__(self, model, index_params=None, query_time_params=None):
        """
        Parameters
        ----------
        model : :class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            Model, that will be used as source for index.
        index_params : dict, optional
            index_params for NMSLIB indexer.
        query_time_params : dict, optional
            query_time_params for NMSLIB indexer.

        """
        if index_params is None:
            index_params = {'M': 100, 'indexThreadQty': 1, 'efConstruction': 100, 'post': 0}
        if query_time_params is None:
            query_time_params = {'efSearch': 100}

        self.index = None
        self.labels = None
        self.model = model
        self.index_params = index_params
        self.query_time_params = query_time_params

        #
        # In the main use case, the user will pass us a non-None model, and we use that model
        # to initialize the index and labels.  In a separate (completely internal) use case, the
        # NsmlibIndexer.load function handles the index and label initialization separately,
        # so it passes us None as the model.
        #
        if model:
            if isinstance(self.model, Doc2Vec):
                self._build_from_doc2vec()
            elif isinstance(self.model, (Word2Vec, FastText)):
                self._build_from_word2vec()
            elif isinstance(self.model, (WordEmbeddingsKeyedVectors, KeyedVectors)):
                self._build_from_keyedvectors()
            else:
                raise ValueError("model must be a Word2Vec, Doc2Vec, FastText or KeyedVectors instance")

    def save(self, fname, protocol=2):
        """Save this NmslibIndexer instance to a file.

        Parameters
        ----------
        fname : str
            Path to the output file,
            will produce 2 files: `fname` - parameters and `fname`.d - :class:`~nmslib.NmslibIndex`.
        protocol : int, optional
            Protocol for pickle.

        Notes
        -----
        This method saves **only** the index (**the model isn't preserved**).

        """
        fname_dict = fname + '.d'
        self.index.saveIndex(fname)
        d = {'index_params': self.index_params, 'query_time_params': self.query_time_params, 'labels': self.labels}
        with open(fname_dict, 'wb') as fout:
            _pickle.dump(d, fout, protocol=protocol)

    @classmethod
    def load(cls, fname):
        """Load a NmslibIndexer instance from a file

        Parameters
        ----------
        fname : str
            Path to dump with NmslibIndexer.

        """
        fname_dict = fname + '.d'
        with open(fname_dict, 'rb') as f:
            d = _pickle.load(f)
        index_params = d['index_params']
        query_time_params = d['query_time_params']
        nmslib_instance = cls(model=None, index_params=index_params, query_time_params=query_time_params)
        index = nmslib.init()
        index.loadIndex(fname)
        nmslib_instance.index = index
        nmslib_instance.labels = d['labels']
        return nmslib_instance

    def _build_from_word2vec(self):
        """Build an NMSLIB index using word vectors from a Word2Vec model."""

        self.model.init_sims()
        self._build_from_model(self.model.wv.vectors_norm, self.model.wv.index2word)

    def _build_from_doc2vec(self):
        """Build an NMSLIB index using document vectors from a Doc2Vec model."""

        docvecs = self.model.docvecs
        docvecs.init_sims()
        labels = [docvecs.index_to_doctag(i) for i in range(0, docvecs.count)]
        self._build_from_model(docvecs.vectors_docs_norm, labels)

    def _build_from_keyedvectors(self):
        """Build an NMSLIB index using word vectors from a KeyedVectors model."""

        self.model.init_sims()
        self._build_from_model(self.model.vectors_norm, self.model.index2word)

    def _build_from_model(self, vectors, labels):
        index = nmslib.init()
        index.addDataPointBatch(vectors)

        index.createIndex(self.index_params, print_progress=True)
        nmslib.setQueryTimeParams(index, self.query_time_params)

        self.index = index
        self.labels = labels

    def most_similar(self, vector, num_neighbors):
        """Find the approximate `num_neighbors` most similar items.

        Parameters
        ----------
        vector : numpy.array
            Vector for word/document.
        num_neighbors : int
            Number of most similar items

        Returns
        -------
        list of (str, float)
            List of most similar items in format [(`item`, `cosine_distance`), ... ]

        """
        ids, distances = self.index.knnQueryBatch(vector.reshape(1, -1), k=num_neighbors)[0]

        return [(self.labels[ids[i]], 1 - distances[i] / 2) for i in range(len(ids))]
