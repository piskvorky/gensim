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
* **Memory sharing** -- trained models can be persisted to disk and loaded back via `mmap <https://en.wikipedia.org/wiki/Mmap>`_. Multiple processes can share the same data, cutting down RAM footprint.
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

Gensim is licensed under the OSI-approved `GNU LGPLv2.1 license <http://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html>`_ and can be downloaded either from its `Github repository <https://github.com/RARE-Technologies/gensim/>`_
or from the `Python Package Index <http://pypi.python.org/pypi/gensim>`_.


Core concepts
-------------

See the :ref:`sphx_glr_auto_examples_core_run_core_concepts.py` tutorial.
