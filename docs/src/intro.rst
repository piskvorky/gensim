.. _intro:

============
Introduction
============

Gensim is a :ref:`free <availability>` Python library designed to automatically extract semantic
topics from documents, as efficiently (computer-wise) and painlessly (human-wise) as possible.

Gensim is designed to process raw, unstructured digital texts ("*plain text*").

The algorithms in Gensim, such as :class:`~gensim.models.word2vec.Word2Vec`, :class:`~gensim.models.fasttext.FastText`,
Latent Semantic Analysis (LSI, LSA, see :class:`~gensim.models.lsimodel.LsiModel`), Latent Dirichlet
Allocation (LDA, see :class:`~gensim.models.ldamodel.LdaModel`) etc, automatically discover the semantic structure of documents by examining statistical
co-occurrence patterns within a corpus of training documents. These algorithms are **unsupervised**,
which means no human input is necessary -- you only need a corpus of plain text documents.

Once these statistical patterns are found, any plain text documents (sentence, phrase, word…) can be succinctly expressed in the new, semantic representation and queried for topical similarity against other documents (words, phrases…).

.. note::
   If the previous paragraphs left you confused, you can read more about the `Vector
   Space Model <http://en.wikipedia.org/wiki/Vector_space_model>`_ and `unsupervised
   document analysis <http://en.wikipedia.org/wiki/Latent_semantic_indexing>`_ on Wikipedia.


.. _design:

Features
--------

* **Memory independence** -- there is no need for the whole training corpus to
  reside fully in RAM at any one time (can process large, web-scale corpora).
* **Memory sharing** -- trained models can be persisted to disk and loaded back via mmap. Multiple processes can share the same data, cutting down RAM footprint.
* Efficient implementations for several popular vector space algorithms,
  including :class:`~gensim.models.word2vec.Word2Vec`, :class:`~gensim.models.doc2vec.Doc2Vec`, :class:`~gensim.models.fasttext.FastText`,
  TF-IDF, Latent Semantic Analysis (LSI, LSA, see :class:`~gensim.models.lsimodel.LsiModel`),
  Latent Dirichlet Allocation (LDA, see :class:`~gensim.models.ldamodel.LdaModel`) or Random Projection (see :class:`~gensim.models.rpmodel.RpModel`).
* I/O wrappers and readers from several popular data formats.
* Fast similarity queries for documents in their semantic representation.

The **principal design objectives** behind Gensim are:

1. Straightforward interfaces and low API learning curve for developers. Good for prototyping.
2. Memory independence with respect to the size of the input corpus; all intermediate
   steps and algorithms operate in a streaming fashion, accessing one document
   at a time.

.. seealso::

    We also built a high performance commercial server for NLP, document analysis, indexing, search and clustering: https://scaletext.ai. ScaleText is available both on-prem and as SaaS.

    Reach out at info@scaletext.com if you need an industry-grade NLP tool with professional support.


.. _availability:

Availability
------------

Gensim is licensed under the OSI-approved `GNU LGPLv2.1 license <http://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html>`_ and can be downloaded either from its `Github repository <https://github.com/piskvorky/gensim/>`_
or from the `Python Package Index <http://pypi.python.org/pypi/gensim>`_.

.. seealso::

    See the :doc:`install <install>` page for more info on Gensim deployment.


Core concepts
-------------

The whole Gensim package revolves around the concepts of :term:`corpus`, :term:`vector` and :term:`model`.

.. glossary::

    Corpus
        A collection of digital documents. Corpora serve two roles in Gensim:

        1. Input for model training
           The corpus is used to automatically train a machine learning model, such as
           :class:`~gensim.models.lsimodel.LsiModel` or :class:`~gensim.models.ldamodel.LdaModel`.

           The models use this *training corpus* to look for common themes and topics, initializing
           their internal model parameters.

           Gensim in unique in its focus on *unsupervised* models so that no human intervention,
           such as costly annotations or tagging documents by hand, is required.

        2. Documents to organize.
           After training, a topic model can be used to extract topics from new documents (documents
           not seen in the training corpus).

           Such corpora can be :doc:`indexed <tut3>`, queried by semantic similarity, clustered etc.

    Vector space model
        In a Vector Space Model (VSM), each document is represented by an
        array of features. For example, a single feature may be thought of as a
        question-answer pair:

        1. How many times does the word *splonge* appear in the document? Zero.
        2. How many paragraphs does the document consist of? Two.
        3. How many fonts does the document use? Five.

        The question is usually represented only by its integer id (such as `1`, `2` and `3` here),
        so that the
        representation of this document becomes a series of pairs like ``(1, 0.0), (2, 2.0), (3, 5.0)``.

        If we know all the questions in advance, we may leave them implicit
        and simply write ``(0.0, 2.0, 5.0)``.

        This sequence of answers can be thought of as a **vector** (in this case a 3-dimensional dense vector).
        For practical purposes, only questions to which the answer is (or
        can be converted to) a *single floating point number* are allowed in Gensim.

        The questions are the same for each document, so that looking at two
        vectors (representing two documents), we will hopefully be able to make
        conclusions such as "The numbers in these two vectors are very similar, and
        therefore the original documents must be similar, too". Of course, whether
        such conclusions correspond to reality depends on how well we picked our questions.

    Gensim Sparse Vector, Bag-of-words Vector
        To save space, in Gensim we omit all vector elements with value 0.0. For example, instead of the
        3-dimensional dense vector ``(0.0, 2.0, 5.0)``, we write only ``[(2, 2.0), (3, 5.0)]`` (note the missing ``(1, 0.0)``). Each vector element is a pair (2-tuple) of ``(feature_id, feature_value)``. The values of all missing features in this sparse representation can be unambiguously resolved to zero, ``0.0``.

        Documents in Gensim are represented by such sparse vectors (sometimes called bag-of-words vectors).

    Gensim streamed corpus
        Gensim does not prescribe any specific corpus format. A corpus is simply a sequence
        of sparse vector (see above).

        For example, ``[ [(2, 2.0), (3, 5.0)], [(3, 1.0)] ]``
        is a simple corpus of two documents = two sparse vectors: the first with two non-zero elements,
        the second with one non-zero element. This particular corpus is represented as a plain Python ``list``.

        However, the full power of Gensim comes from the fact that a corpus doesn't have to be a ``list``,
        or a ``NumPy`` array, or a ``Pandas`` dataframe, or whatever. Gensim *accepts any object that,
        when iterated over, successively yields these sparse bag-of-word vectors*.

        This flexibility allows you to create your own corpus classes that stream the sparse vectors directly from disk, network, database, dataframes…. The models in Gensim are implemented such that they don't require all vectors to reside in RAM at once. You can even create the sparse vectors on the fly!

        See our `tutorial on streamed data processing in Python <https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/>`_.

        For a built-in example of an efficient corpus format streamed directly from disk, see
        the Matrix Market format in :mod:`~gensim.corpora.mmcorpus`. For a minimal blueprint example on
        how to create your own streamed corpora, check out the `source code of CSV corpus <https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/corpora/csvcorpus.py>`_.

    Model, Transformation
        Gensim uses **model** to refer to the code and associated data (model parameters)
        required to transform one document representation to another.

        In Gensim, documents are represented as vectors (see above) so a model can be thought of as a transformation
        from one vector space to another. The parameters of this transformation are learned from the training corpus.

        Trained models (the data parameters) can be persisted to disk and later loaded back, either to continue
        training on new training documents or to transform new documents.

        Gensim implements multiple models, such as :class:`~gensim.models.word2vec.Word2Vec`,
        :class:`~gensim.models.lsimodel.LsiModel`, :class:`~gensim.models.ldamodel.LdaModel`,
        :class:`~gensim.models.fasttext.FastText` etc. See the :doc:`API reference <apiref>` for a full list.

.. seealso::

    For some examples on how all this works out in code, go to :doc:`Tutorials <tutorial>`.
