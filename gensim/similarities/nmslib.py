# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Radim Rehurek <me@radimrehurek.com>
# Copyright (C) 2019 Masahiro Kazama <kazama.masa@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module integrates `NMSLIB <https://github.com/nmslib/nmslib>`_ fast similarity
search with Gensim's :class:`~gensim.models.word2vec.Word2Vec`, :class:`~gensim.models.doc2vec.Doc2Vec`,
:class:`~gensim.models.fasttext.FastText` and :class:`~gensim.models.keyedvectors.KeyedVectors`
vector embeddings.

.. Important::
    To use this module, you must have the external ``nmslib`` library installed.
    To install it, run ``pip install nmslib``.

To use the integration, instantiate a :class:`~gensim.similarities.nmslib.NmslibIndexer` class
and pass the instance as the `indexer` parameter to your model's `model.most_similar()` method.

Example usage
-------------

.. sourcecode:: pycon

    >>> from gensim.similarities.nmslib import NmslibIndexer
    >>> from gensim.models import Word2Vec
    >>>
    >>> sentences = [['cute', 'cat', 'say', 'meow'], ['cute', 'dog', 'say', 'woof']]
    >>> model = Word2Vec(sentences, min_count=1, epochs=10, seed=2)
    >>>
    >>> indexer = NmslibIndexer(model)
    >>> model.wv.most_similar("cat", topn=2, indexer=indexer)
    [('cat', 1.0), ('meow', 0.16398882865905762)]

Load and save example
---------------------

.. sourcecode:: pycon

    >>> from gensim.similarities.nmslib import NmslibIndexer
    >>> from gensim.models import Word2Vec
    >>> from tempfile import mkstemp
    >>>
    >>> sentences = [['cute', 'cat', 'say', 'meow'], ['cute', 'dog', 'say', 'woof']]
    >>> model = Word2Vec(sentences, min_count=1, seed=2, epochs=10)
    >>>
    >>> indexer = NmslibIndexer(model)
    >>> _, temp_fn = mkstemp()
    >>> indexer.save(temp_fn)
    >>>
    >>> new_indexer = NmslibIndexer.load(temp_fn)
    >>> model.wv.most_similar("cat", topn=2, indexer=new_indexer)
    [('cat', 1.0), ('meow', 0.5595494508743286)]

What is NMSLIB
--------------

Non-Metric Space Library (NMSLIB) is an efficient cross-platform similarity search library and a toolkit
for evaluation of similarity search methods. The core-library does not have any third-party dependencies.
More information about NMSLIB: `github repository <https://github.com/nmslib/nmslib>`_.

Why use NMSIB?
--------------

Gensim's native :py:class:`~gensim.similarities.Similarity` for finding the `k` nearest neighbors to a vector
uses brute force and has linear complexity, albeit with extremely low constant factors.

The retrieved results are exact, which is an overkill in many applications:
approximate results retrieved in sub-linear time may be enough.

NMSLIB can find approximate nearest neighbors much faster, similar to Spotify's Annoy library.
Compared to :py:class:`~gensim.similarities.annoy.Annoy`, NMSLIB has more parameters to
control the build and query time and accuracy. NMSLIB often achieves faster and more accurate
nearest neighbors search than Annoy.

"""

# Avoid import collisions on py2: this module has the same name as the actual NMSLIB library.
from __future__ import absolute_import
import pickle as _pickle

from smart_open import open
try:
    import nmslib
except ImportError:
    raise ImportError("NMSLIB not installed. To use the NMSLIB indexer, please run `pip install nmslib`.")

from gensim import utils
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors


class NmslibIndexer():
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
            Indexing parameters passed through to NMSLIB:
            https://github.com/nmslib/nmslib/blob/master/manual/methods.md#graph-based-search-methods-sw-graph-and-hnsw

            If not specified, defaults to `{'M': 100, 'indexThreadQty': 1, 'efConstruction': 100, 'post': 0}`.
        query_time_params : dict, optional
            query_time_params for NMSLIB indexer.
            If not specified, defaults to `{'efSearch': 100}`.

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
            elif isinstance(self.model, (KeyedVectors,)):
                self._build_from_keyedvectors()
            else:
                raise ValueError("model must be a Word2Vec, Doc2Vec, FastText or KeyedVectors instance")

    def save(self, fname, protocol=utils.PICKLE_PROTOCOL):
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
        """Load a NmslibIndexer instance from a file.

        Parameters
        ----------
        fname : str
            Path previously used in `save()`.

        """
        fname_dict = fname + '.d'
        with open(fname_dict, 'rb') as f:
            d = _pickle.load(f)
        index_params = d['index_params']
        query_time_params = d['query_time_params']
        nmslib_instance = cls(model=None, index_params=index_params, query_time_params=query_time_params)
        index = nmslib.init(method='hnsw', space='cosinesimil')
        index.loadIndex(fname)
        nmslib_instance.index = index
        nmslib_instance.labels = d['labels']
        return nmslib_instance

    def _build_from_word2vec(self):
        """Build an NMSLIB index using word vectors from a Word2Vec model."""
        self._build_from_model(self.model.wv.get_normed_vectors(), self.model.wv.index_to_key)

    def _build_from_doc2vec(self):
        """Build an NMSLIB index using document vectors from a Doc2Vec model."""
        docvecs = self.model.dv
        labels = docvecs.index_to_key
        self._build_from_model(docvecs.get_normed_vectors(), labels)

    def _build_from_keyedvectors(self):
        """Build an NMSLIB index using word vectors from a KeyedVectors model."""
        self._build_from_model(self.model.get_normed_vectors(), self.model.index_to_key)

    def _build_from_model(self, vectors, labels):
        index = nmslib.init(method='hnsw', space='cosinesimil')
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
            Vector for a word or document.
        num_neighbors : int
            How many most similar items to look for?

        Returns
        -------
        list of (str, float)
            List of most similar items in the format `[(item, cosine_similarity), ... ]`.

        """
        ids, distances = self.index.knnQueryBatch(vector.reshape(1, -1), k=num_neighbors)[0]

        # NMSLIB returns cosine distance (not similarity), which is simply `dist = 1 - cossim`.
        # So, convert back to similarities here.
        return [(self.labels[id_], 1.0 - distance) for id_, distance in zip(ids, distances)]
