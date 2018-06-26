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
        A collection of digital documents. This collection is used to automatically
        infer the vector structure of the documents, their topics, etc. For
        this reason, the collection is also called a *training corpus*. This inferred
        latent structure can be later used to assign topics to new documents, which did
        not appear in the *training corpus*.

        This inferred latent structure can be later used to discovert topics for new documents, which did
        not appear in the training corpus. No human intervention (such as annotating or tagging documents by hand, or creating other metadata) is required.

    Vector
        In the Vector Space Model (VSM), each document is represented by an
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

        This sequence of answers can be thought of as a *vector* (in this case a 3-dimensional vector). For practical purposes, only questions to which the answer is (or
        can be converted to) a single real number are allowed.

        The questions are the same for each document, so that looking at two
        vectors (representing two documents), we will hopefully be able to make
        conclusions such as "The numbers in these two vectors are very similar, and
        therefore the original documents must be similar, too". Of course, whether
        such conclusions correspond to reality depends on how well we picked our questions.

    Sparse Vector
        Typically, the answer to most questions will be ``0.0``. To save space,
        we omit them from the document's representation, and write only ``(2, 2.0),
        (3, 5.0)`` (note the missing ``(1, 0.0)``).
        Since the set of all questions is known in advance, all the missing features
        in a sparse representation of a document can be unambiguously resolved to zero, ``0.0``.

        Gensim does not prescribe any specific corpus format;
        a corpus is anything that, when iterated over, successively yields these sparse vectors.

        For example, ``[ [(2, 2.0), (3, 5.0)], [(0, 1.0), (3, 1.0)] ]``
        is a simple corpus of two documents, each with two non-zero `feature-answer` pairs.

    Model
        We use **model** as an abstract term referring to the code and associated data
        required to transform one document representation to another. In Gensim, documents are
        represented as vectors so a model can be thought of as a transformation
        between two vector spaces. The parameters of this transformation are learned from the training corpus. Gensim
        implements multiple models, such as :class:`~gensim.models.word2vec.Word2Vec`,
        :class:`~gensim.models.lsimodel.LsiModel`, :class:`~gensim.models.ldamodel.LdaModel`,
        :class:`~gensim.models.fasttext.FastText` etc.

.. seealso::

    For some examples on how this works out in code, go to :doc:`Tutorials <tutorial>`.
