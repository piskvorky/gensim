.. _intro:

============
Introduction 
============

Gensim is a Python framework designed to help make
the conversion of natural language texts to the Vector Space Model as simple and 
natural as possible.

Gensim contains algorithms for unsupervised learning from raw, unstructured digital texts,
such as Latent Semantic Analysis and Latent Dirichlet Allocation.
These algorithms discover hidden (*latent*) corpus structure.
Once found, documents can be succinctly expressed in terms of this structure, queried for topical similarity and so on.

If the previous paragraphs left you confused, you can read more about the `Vector 
Space Model <http://en.wikipedia.org/wiki/Vector_space_model>`_ and `unsupervised 
document analysis <http://en.wikipedia.org/wiki/Latent_semantic_indexing>`_ at Wikipedia.

.. note::
  Gensim's target audience is the NLP research community and interested general public; 
  gensim is not meant to be a production tool for commercial environments.


Design
------
 
Gensim includes the following features:

* Memory independence -- there is no need for the whole text corpus (or any 
  intermediate term-document matrices) to reside fully in RAM at any one time.
* Provides implementations for several popular topic inference algorithms, 
  including Latent Semantic Analysis (LSA/LSI via `SVD <http://en.wikipedia.org/wiki/Singular_value_decomposition>`_) 
  and Latent Dirichlet Allocation (LDA), and makes adding new ones simple.
* Contains I/O wrappers and converters around several popular data formats.
* Allows similarity queries across documents in their latent, topical representation.
 
Creation of gensim was motivated by a perceived lack of available, scalable software 
frameworks that realize topic modeling, and/or their overwhelming internal complexity. 
You can read more about the motivation in our `LREC 2010 workshop paper <http://nlp.fi.muni.cz/projekty/gensim/lrec2010_final.pdf>`_.

The principal design objectives behind gensim are:

1. Straightforward interfaces and low API learning curve for developers, 
   facilitating modifications and rapid prototyping.
2. Memory independence with respect to the size of the input corpus; all intermediate 
   steps and algorithms operate in a streaming fashion, processing one document 
   at a time.


Availability
------------
Gensim is licensed under the OSI-approved `GNU LPGL license <http://www.gnu.org/licenses/lgpl.html>`_ 
and can be downloaded either from its `SVN repository <http://my-trac.assembla.com/gensim>`_
or from the `Python Package Index <http://pypi.python.org/pypi/gensim>`_. 

.. http://my-trac.assembla.com/gensim/browser/trunk/COPYING.LESSER

.. seealso::

    See the :doc:`install <install>` page for more info on package deployment. 


Core concepts   
-------------

The whole gensim package revolves around the concepts of :term:`corpus`, :term:`vector` and 
:term:`model`.

.. glossary::

    Corpus
        A collection of digital documents. This collection is used to automatically 
        infer structure of the documents, their topics etc. For
        this reason, the collection is also called a *training corpus*. The inferred 
        latent structure can be later used to assign topics to new documents, which did 
        not appear in the training corpus.
        No human intervention (such as tagging the documents by hand, or creating 
        other metadata) is required.

    Vector
        In the Vector Space Model (VSM), each document is represented by an 
        array of features. For example, a single feature may be thought of as a 
        question-answer pair:
        
        1. How many times does the word *splonge* appear in the document? Zero.
        2. How many paragraphs does the document consist of? Two.
        3. How many fonts does the document use? Five.
        
        The question is usually represented only by its integer id, so that the
        representation of a document becomes a series of pairs like ``(1, 0.0), (2, 2.0), (3, 5.0)``.
        If we know all the questions in advance, we may leave them implicit 
        and simply write ``(0.0, 2.0, 5.0)``.
        This sequence of answers can be thought of as a high-dimensional (in our case 3-dimensional)
        *vector*. For practical purposes, only questions to which the answer is (or
        can be converted to) a single real number are allowed. 
        
        The questions are the same for each document, so that looking at two 
        vectors (representing two documents), we will hopefully be able to make
        conclusions such as "The numbers in these two vectors are very similar, and 
        therefore the original documents must be similar, too". Of course, whether 
        such conclusions correspond to reality depends on how well we picked our questions.
        
    Sparse vector
        Typically, the answer to most questions will be ``0.0``. To save space,
        we omit them from the document's representation, and write only ``(2, 2.0), 
        (3, 5.0)`` (note the missing ``(1, 0.0)``).
        Since the set of all questions is known in advance, all the missing features
        in sparse representation of a document can be unambiguously resolved to zero, ``0.0``.
    
    Model
        For our purposes, a model is a transformation from one document representation
        to another (or, in other words, from one vector space to another). 
        Both the initial and target representations are
        still vectors -- they only differ in what the questions and answers are.
        The transformation is automatically learned from the traning :term:`corpus`, without human
        supervision, and in hopes that the final document representation will be more compact
        and more useful (with similar documents having similar representations) 
        than the initial one. The transformation process is also sometimes called 
        *clustering* in machine learning terminology, or *noise reduction*, from signal
        processing.
 
.. seealso::

    For some examples on how this works out in code, go to :doc:`tutorials <tutorial>`.
