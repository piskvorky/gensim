#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Intro
-----
This module contains integration Annoy with :class:`~gensim.models.word2vec.Word2Vec`,
:class:`~gensim.models.doc2vec.Doc2Vec`, :class:`~gensim.models.fasttext.FastText` and
:class:`~gensim.models.keyedvectors.KeyedVectors`.

.. Important::
    To use this module, you must have the ``annoy`` library install.
    To install it, run ``pip install annoy``.


What is Annoy
-------------
Annoy (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings to search for points in space
that are close to a given query point. It also creates large read-only file-based data structures that are mmapped
into memory so that many processes may share the same data.


How it works
------------
Using `random projections <https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Random_projection>`_
and by building up a tree. At every intermediate node in the tree, a random hyperplane is chosen,
which divides the space into two subspaces. This hyperplane is chosen by sampling two points from the subset
and taking the hyperplane equidistant from them.

More information about Annoy: `github repository <https://github.com/spotify/annoy>`_,
`author in twitter <https://twitter.com/fulhack>`_
and `annoy-user maillist <https://groups.google.com/forum/#!forum/annoy-user>`_.

"""
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
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors


_NOANNOY = ImportError(
    "Annoy is not installed, if you wish to use the annoy "
    "indexer, please run `pip install annoy`"
)


class AnnoyIndexer(object):
    """This class allows to use `Annoy <https://github.com/spotify/annoy>`_ as indexer for `most_similar` method
    from :class:`~gensim.models.word2vec.Word2Vec`, :class:`~gensim.models.doc2vec.Doc2Vec`,
    :class:`~gensim.models.fasttext.FastText` and :class:`~gensim.models.keyedvectors.Word2VecKeyedVectors` classes.

    """

    def __init__(self, model=None, num_trees=None):
        """
        Parameters
        ----------
        model : :class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`, optional
            Model, that will be used as source for index.
        num_trees : int, optional
            Number of trees for Annoy indexer.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.similarities.index import AnnoyIndexer
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
            if isinstance(self.model, Doc2Vec):
                self.build_from_doc2vec()
            elif isinstance(self.model, (Word2Vec, FastText)):
                self.build_from_word2vec()
            elif isinstance(self.model, (WordEmbeddingsKeyedVectors, KeyedVectors)):
                self.build_from_keyedvectors()
            else:
                raise ValueError("Only a Word2Vec, Doc2Vec, FastText or KeyedVectors instance can be used")

    def save(self, fname, protocol=2):
        """Save AnnoyIndexer instance.

        Parameters
        ----------
        fname : str
            Path to output file, will produce 2 files: `fname` - parameters and `fname`.d - :class:`~annoy.AnnoyIndex`.
        protocol : int, optional
            Protocol for pickle.

        Notes
        -----
        This method save **only** index (**model isn't preserved**).

        """
        fname_dict = fname + '.d'
        self.index.save(fname)
        d = {'f': self.model.vector_size, 'num_trees': self.num_trees, 'labels': self.labels}
        with utils.open(fname_dict, 'wb') as fout:
            _pickle.dump(d, fout, protocol=protocol)

    def load(self, fname):
        """Load AnnoyIndexer instance

        Parameters
        ----------
        fname : str
            Path to dump with AnnoyIndexer.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.similarities.index import AnnoyIndexer
            >>> from gensim.models import Word2Vec
            >>> from tempfile import mkstemp
            >>>
            >>> sentences = [['cute', 'cat', 'say', 'meow'], ['cute', 'dog', 'say', 'woof']]
            >>> model = Word2Vec(sentences, min_count=1, seed=1, iter=10)
            >>>
            >>> indexer = AnnoyIndexer(model, 2)
            >>> _, temp_fn = mkstemp()
            >>> indexer.save(temp_fn)
            >>>
            >>> new_indexer = AnnoyIndexer()
            >>> new_indexer.load(temp_fn)
            >>> new_indexer.model = model

        """
        fname_dict = fname + '.d'
        if not (os.path.exists(fname) and os.path.exists(fname_dict)):
            raise IOError(
                "Can't find index files '%s' and '%s' - Unable to restore AnnoyIndexer state." % (fname, fname_dict)
            )
        else:
            try:
                from annoy import AnnoyIndex
            except ImportError:
                raise _NOANNOY

            with utils.open(fname_dict, 'rb') as f:
                d = _pickle.loads(f.read())
            self.num_trees = d['num_trees']
            self.index = AnnoyIndex(d['f'])
            self.index.load(fname)
            self.labels = d['labels']

    def build_from_word2vec(self):
        """Build an Annoy index using word vectors from a Word2Vec model."""

        self.model.init_sims()
        return self._build_from_model(self.model.wv.vectors_norm, self.model.wv.index2word, self.model.vector_size)

    def build_from_doc2vec(self):
        """Build an Annoy index using document vectors from a Doc2Vec model."""

        docvecs = self.model.docvecs
        docvecs.init_sims()
        labels = [docvecs.index_to_doctag(i) for i in range(0, docvecs.count)]
        return self._build_from_model(docvecs.vectors_docs_norm, labels, self.model.vector_size)

    def build_from_keyedvectors(self):
        """Build an Annoy index using word vectors from a KeyedVectors model."""

        self.model.init_sims()
        return self._build_from_model(self.model.vectors_norm, self.model.index2word, self.model.vector_size)

    def _build_from_model(self, vectors, labels, num_features):
        try:
            from annoy import AnnoyIndex
        except ImportError:
            raise _NOANNOY

        index = AnnoyIndex(num_features)

        for vector_num, vector in enumerate(vectors):
            index.add_item(vector_num, vector)

        index.build(self.num_trees)
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

        ids, distances = self.index.get_nns_by_vector(
            vector, num_neighbors, include_distances=True)

        return [(self.labels[ids[i]], 1 - distances[i] / 2) for i in range(len(ids))]
