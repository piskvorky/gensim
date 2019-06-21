# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Radim Rehurek <me@radimrehurek.com>
# Copyright (C) 2019 Masahiro Kazama <kazama.masa@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Intro
-----
This module contains integration Nmslib with :class:`~gensim.models.word2vec.Word2Vec`,
:class:`~gensim.models.doc2vec.Doc2Vec`, :class:`~gensim.models.fasttext.FastText` and
:class:`~gensim.models.keyedvectors.KeyedVectors`.


What is Nmslib
-------------
Non-Metric Space Library (NMSLIB) is an efficient cross-platform similarity search library and a toolkit
for evaluation of similarity search methods. The core-library does not have any third-party dependencies.


How it works
------------
Searching in generic non-metric space.

More information about Nmslib: `github repository <https://github.com/nmslib/nmslib>`_.

"""

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
        "Nmslib has not been installed, if you wish to use the nmslib indexer, please run `pip install nmslib`"
    )


class NmslibIndexer(object):
    """This class allows to use `Nmslib <https://github.com/nmslib/nmslib>`_ as indexer for `most_similar` method
    from :class:`~gensim.models.word2vec.Word2Vec`, :class:`~gensim.models.doc2vec.Doc2Vec`,
    :class:`~gensim.models.fasttext.FastText` and :class:`~gensim.models.keyedvectors.Word2VecKeyedVectors` classes.

    """

    def __init__(self, model=None, index_params=None, query_time_params=None):
        """
        Parameters
        ----------
        model : :class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`, optional
            Model, that will be used as source for index.
        index_params : dict, optional
            index_params for Nmslib indexer.
        query_time_params : dict, optional
            query_time_params for Nmslib indexer.

        Examples
        --------
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

        """
        if index_params is None:
            index_params = {'M': 10, 'indexThreadQty': 1, 'efConstruction': 100, 'post': 0}
        if query_time_params is None:
            query_time_params = {'efSearch': 100}

        self.index = None
        self.labels = None
        self.model = model
        self.index_params = index_params
        self.query_time_params = query_time_params

        if model:
            if isinstance(self.model, Doc2Vec):
                self.build_from_doc2vec()
            elif isinstance(self.model, (Word2Vec, FastText)):
                self.build_from_word2vec()
            elif isinstance(self.model, (WordEmbeddingsKeyedVectors, KeyedVectors)):
                self.build_from_keyedvectors()
            else:
                raise ValueError("Only a Word2Vec, Doc2Vec, FastText or KeyedVectors instance can be used")

    def save(self, fname, protocol=2):
        """Save NmslibIndexer instance.

        Parameters
        ----------
        fname : str
            Path to output file,
            will produce 2 files: `fname` - parameters and `fname`.d - :class:`~nmslib.NmslibIndex`.
        protocol : int, optional
            Protocol for pickle.

        Notes
        -----
        This method save **only** index (**model isn't preserved**).

        """
        fname_dict = fname + '.d'
        self.index.saveIndex(fname)
        d = {'index_params': self.index_params, 'query_time_params': self.query_time_params, 'labels': self.labels}
        with open(fname_dict, 'wb') as fout:
            _pickle.dump(d, fout, protocol=protocol)

    def load(self, fname):
        """Load NmslibIndexer instance

        Parameters
        ----------
        fname : str
            Path to dump with NmslibIndexer.

        Examples
        --------
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
            >>> new_indexer = NmslibIndexer()
            >>> new_indexer.load(temp_fn)
            >>> new_indexer.model = model

        """
        fname_dict = fname + '.d'
        with open(fname_dict, 'rb') as f:
            d = _pickle.load(f)
        self.index_params = d['index_params']
        self.query_time_params = d['query_time_params']
        self.index = nmslib.init()
        self.index.loadIndex(fname)
        self.labels = d['labels']

    def build_from_word2vec(self):
        """Build an Nmslib index using word vectors from a Word2Vec model."""

        self.model.init_sims()
        return self._build_from_model(self.model.wv.vectors_norm, self.model.wv.index2word)

    def build_from_doc2vec(self):
        """Build an Nmslib index using document vectors from a Doc2Vec model."""

        docvecs = self.model.docvecs
        docvecs.init_sims()
        labels = [docvecs.index_to_doctag(i) for i in range(0, docvecs.count)]
        return self._build_from_model(docvecs.vectors_docs_norm, labels)

    def build_from_keyedvectors(self):
        """Build an Nmslib index using word vectors from a KeyedVectors model."""

        self.model.init_sims()
        return self._build_from_model(self.model.vectors_norm, self.model.index2word)

    def _build_from_model(self, vectors, labels):
        index = nmslib.init()
        index.addDataPointBatch(vectors)

        index.createIndex(self.index_params, print_progress=True)
        nmslib.setQueryTimeParams(index, self.query_time_params)

        self.index = index
        self.labels = labels
        print("build index")

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
