
:github_url: https://github.com/RaRe-Technologies/gensim

Gensim documentation
===================================

============
Introduction
============

Gensim is a free Python library designed to automatically extract semantic
topics from documents, as efficiently (computer-wise) and painlessly (human-wise) as possible.

Gensim is designed to process raw, unstructured digital texts ("plain text").

The algorithms in Gensim, such as **Word2Vec**, **FastText**, **Latent Semantic Analysis**, **Latent Dirichlet Allocation** and **Random Projections**, discover semantic structure of documents by examining statistical co-occurrence patterns within a corpus of training documents. These algorithms are **unsupervised**, which means no human input is necessary -- you only need a corpus of plain text documents.

Once these statistical patterns are found, any plain text documents can be succinctly
expressed in the new, semantic representation and queried for topical similarity
against other documents, words or phrases.

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
  including Word2Vec, Doc2Vec, FastText, TF-IDF, Latent Semantic Analysis (LSI, LSA),
  Latent Dirichlet Allocation (LDA) or Random Projection.
* I/O wrappers and readers from several popular data formats.
* Fast similarity queries for documents in their semantic representation.

The **principal design objectives** behind Gensim are:

1. Straightforward interfaces and low API learning curve for developers. Good for prototyping.
2. Memory independence with respect to the size of the input corpus; all intermediate
   steps and algorithms operate in a streaming fashion, accessing one document
   at a time.

.. seealso::

    We built a high performance server for NLP, document analysis, indexing, search and clustering: https://scaletext.ai.
    ScaleText is a commercial product, available both on-prem or as SaaS.
    Reach out at info@scaletext.com if you need an industry-grade tool with professional support.

.. _availability:

Availability
------------

Gensim is licensed under the OSI-approved `GNU LGPLv2.1 license <http://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html>`_ and can be downloaded either from its `github repository <https://github.com/piskvorky/gensim/>`_ or from the `Python Package Index <http://pypi.python.org/pypi/gensim>`_.

.. seealso::

    See the :doc:`install <install>` page for more info on Gensim deployment.


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Getting started

   install
   intro
   support
   about
   license
   citing


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial
   tut1
   tut2
   tut3


.. toctree::
   :maxdepth: 1
   :caption: API Reference

   apiref

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
