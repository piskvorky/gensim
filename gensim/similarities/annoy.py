#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""
This module integrates Spotify's `Annoy <https://github.com/spotify/annoy>`_ (Approximate Nearest Neighbors Oh Yeah)
library with Gensim's :class:`~gensim.models.word2vec.Word2Vec`, :class:`~gensim.models.doc2vec.Doc2Vec`,
:class:`~gensim.models.fasttext.FastText` and :class:`~gensim.models.keyedvectors.KeyedVectors` word embeddings.

.. Important::
    To use this module, you must have the ``annoy`` library installed.
    To install it, run ``pip install annoy``.

"""

# Avoid import collisions on py2: this module has the same name as the actual Annoy library.
from __future__ import absolute_import

import os

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

from gensim import utils
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors


_NOANNOY = ImportError("Annoy not installed. To use the Annoy indexer, please run `pip install annoy`.")


class AnnoyIndexer():
    """This class allows the use of `Annoy <https://github.com/spotify/annoy>`_ for fast (approximate)
    vector retrieval in `most_similar()` calls of
    :class:`~gensim.models.word2vec.Word2Vec`, :class:`~gensim.models.doc2vec.Doc2Vec`,
    :class:`~gensim.models.fasttext.FastText` and :class:`~gensim.models.keyedvectors.Word2VecKeyedVectors` models.

    """

    def __init__(self, model=None, num_trees=None):
        """
        Parameters
        ----------
        model : trained model, optional
            Use vectors from this model as the source for the index.
        num_trees : int, optional
            Number of trees for Annoy indexer.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.similarities.annoy import AnnoyIndexer
            >>> from gensim.models import Word2Vec
            >>>
            >>> sentences = [['cute', 'cat', 'say', 'meow'], ['cute', 'dog', 'say', 'woof']]
            >>> model = Word2Vec(sentences, min_count=1, seed=1)
            >>>
            >>> indexer = AnnoyIndexer(model, 2)
            >>> model.most_similar("cat", topn=2, indexer=indexer)
            [('cat', 1.0), ('dog', 0.32011348009109497)]

        """
        self.index = None
        self.labels = None
        self.model = model
        self.num_trees = num_trees

        if model and num_trees:
            # Extract the KeyedVectors object from whatever model we were given.
            if isinstance(self.model, Doc2Vec):
                kv = self.model.dv
            elif isinstance(self.model, (Word2Vec, FastText)):
                kv = self.model.wv
            elif isinstance(self.model, (KeyedVectors,)):
                kv = self.model
            else:
                raise ValueError("Only a Word2Vec, Doc2Vec, FastText or KeyedVectors instance can be used")
            self._build_from_model(kv.get_normed_vectors(), kv.index_to_key, kv.vector_size)

    def save(self, fname, protocol=utils.PICKLE_PROTOCOL):
        """Save AnnoyIndexer instance to disk.

        Parameters
        ----------
        fname : str
            Path to output. Save will produce 2 files:
            `fname`: Annoy index itself.
            `fname.dict`: Index metadata.
        protocol : int, optional
            Protocol for pickle.

        Notes
        -----
        This method saves **only the index**. The trained model isn't preserved.

        """
        self.index.save(fname)
        d = {'f': self.model.vector_size, 'num_trees': self.num_trees, 'labels': self.labels}
        with utils.open(fname + '.dict', 'wb') as fout:
            _pickle.dump(d, fout, protocol=protocol)

    def load(self, fname):
        """Load an AnnoyIndexer instance from disk.

        Parameters
        ----------
        fname : str
            The path as previously used by ``save()``.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.similarities.index import AnnoyIndexer
            >>> from gensim.models import Word2Vec
            >>> from tempfile import mkstemp
            >>>
            >>> sentences = [['cute', 'cat', 'say', 'meow'], ['cute', 'dog', 'say', 'woof']]
            >>> model = Word2Vec(sentences, min_count=1, seed=1, epochs=10)
            >>>
            >>> indexer = AnnoyIndexer(model, 2)
            >>> _, temp_fn = mkstemp()
            >>> indexer.save(temp_fn)
            >>>
            >>> new_indexer = AnnoyIndexer()
            >>> new_indexer.load(temp_fn)
            >>> new_indexer.model = model

        """
        fname_dict = fname + '.dict'
        if not (os.path.exists(fname) and os.path.exists(fname_dict)):
            raise IOError(
                f"Can't find index files '{fname}' and '{fname_dict}' - unable to restore AnnoyIndexer state."
            )
        try:
            from annoy import AnnoyIndex
        except ImportError:
            raise _NOANNOY

        with utils.open(fname_dict, 'rb') as f:
            d = _pickle.loads(f.read())
        self.num_trees = d['num_trees']
        self.index = AnnoyIndex(d['f'], metric='angular')
        self.index.load(fname)
        self.labels = d['labels']

    def _build_from_model(self, vectors, labels, num_features):
        try:
            from annoy import AnnoyIndex
        except ImportError:
            raise _NOANNOY

        index = AnnoyIndex(num_features, metric='angular')

        for vector_num, vector in enumerate(vectors):
            index.add_item(vector_num, vector)

        index.build(self.num_trees)
        self.index = index
        self.labels = labels

    def most_similar(self, vector, num_neighbors):
        """Find `num_neighbors` most similar items.

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
        ids, distances = self.index.get_nns_by_vector(
            vector, num_neighbors, include_distances=True)

        return [(self.labels[ids[i]], 1 - distances[i] ** 2 / 2) for i in range(len(ids))]
