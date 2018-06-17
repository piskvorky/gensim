.. _intro:

Core concepts
-------------

The whole Gensim package revolves around the concepts of :term:`corpus`, :term:`vector` and
:term:`model`.

.. glossary::

    Corpus
        A collection of digital documents. This collection is used to automatically
        infer the vector structure of the documents, their topics, etc. For
        this reason, the collection is also called a *training corpus*.

        This inferred latent structure can be later used to assign topics to new documents, which did
        not appear in the training corpus. No human intervention (such as tagging the documents by hand, or creating
        other metadata) is required.

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
        between two vector spaces.

        The details of this transformation are learned from the training corpus. Gensim
        implements multiple models, such as Word2Vec, Latent Semantic Indexing or FastText.

.. seealso::

    For some examples on how this works out in code, go to :doc:`Tutorials <tutorial>`.
