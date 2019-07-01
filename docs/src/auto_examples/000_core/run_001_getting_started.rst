.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_000_core_run_001_getting_started.py>` to download the full example code or run this example in your browser via Binder
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_000_core_run_001_getting_started.py:


.. _intro_tutorial:

Getting Started with Gensim
===========================

A gentle introduction to Gensim.

This example demonstrates using gensim to:

    1. Create a toy corpus and store it in memory
    2. Create a Tf-Idf transformation of the corpus
    3. Calculate the similarity between all the documents in the corpus

Let's begin.

.. code-block:: default


    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)







First, let’s create a small corpus of nine documents and twelve features [1]_:


.. code-block:: default


    corpus = [
        [(0, 1.0), (1, 1.0), (2, 1.0)],
        [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
        [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
        [(0, 1.0), (4, 2.0), (7, 1.0)],
        [(3, 1.0), (5, 1.0), (6, 1.0)],
        [(9, 1.0)],
        [(9, 1.0), (10, 1.0)],
        [(9, 1.0), (10, 1.0), (11, 1.0)],
        [(8, 1.0), (10, 1.0), (11, 1.0)],
    ]







Each of the elements in `corpus` corresponds to a document.
A document consists of `features`.
In the above representation, we use tuples to represent features.
So, the first document included features 0, 1, and 2 only.
This representation is known as the `Vector Space Model <http://en.wikipedia.org/wiki/Vector_space_model>`_.

If you’re not familiar with the vector space model, we’ll bridge the gap between raw strings, corpora and sparse vectors in the next tutorial on :ref:`tut1`.
If you’re familiar with the vector space model, you’ll probably know that the way you parse your documents and convert them to vectors has major impact on the quality of any subsequent applications.

In `gensim`, a :dfn:`corpus` is simply an object which, when iterated over, returns its documents represented as sparse vectors. In this case we’re using a list of list of tuples.

.. note::
  In this example, the whole corpus is stored in memory, as a Python list.
  However, the corpus interface only dictates that a corpus must support
  iteration over its constituent documents. For very large corpora, it is
  advantageous to keep the corpus on disk, and access its documents
  sequentially, one at a time. All the operations and transformations are
  implemented in such a way that makes them independent of the size of the
  corpus, memory-wise.

Next, let’s import gensim and initialize a :dfn:`transformation`:



.. code-block:: default

    from gensim import models

    tfidf = models.TfidfModel(corpus)







A transformation is used to convert documents from one vector representation into another.
Here, we used `Tf-Idf <http://en.wikipedia.org/wiki/Tf%E2%80%93idf>`_, a simple transformation which takes documents represented as bag-of-words counts and applies a weighting which discounts common terms (or, equivalently, promotes rare terms). It also scales the resulting vector to unit length (in the `Euclidean norm <http://en.wikipedia.org/wiki/Norm_%28mathematics%29#Euclidean_norm>`_).

Transformations are covered in detail in the tutorial on Topics and Transformations.

So, given a new vector corresponding to another document (*not* in the original corpus), we can get its representation via Tf-Idf as:


.. code-block:: default


    vec = [(0, 1), (4, 1)]
    print(tfidf[vec])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [(0, 0.8075244024440723), (4, 0.5898341626740045)]


To transform the whole corpus via TfIdf and index it, in preparation for similarity queries:



.. code-block:: default

    from gensim import similarities

    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)







and to query the similarity of our query vector vec against every document in the corpus:


.. code-block:: default


    sims = index[tfidf[vec]]
    print(list(enumerate(sims)))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [(0, 0.4662244), (1, 0.19139354), (2, 0.2460055), (3, 0.82094586), (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0), (8, 0.0)]


How to read this output? Document number zero (the first document) has a similarity score of 0.466=46.6%, the second document has a similarity score of 19.1% etc.
We can make this slightly more readable by sorting:


.. code-block:: default


    for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
        print(document_number, score)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    3 0.82094586
    0 0.4662244
    2 0.2460055
    1 0.19139354
    4 0.0
    5 0.0
    6 0.0
    7 0.0
    8 0.0


Thus, according to TfIdf document representation and cosine similarity measure, the most similar to our query document vec is document no. 3, with a similarity score of 82.1%. Note that in the TfIdf representation, any documents which do not share any common features with vec at all (documents no. 4–8) get a similarity score of 0.0. See the Similarity Queries tutorial for more detail.

.. [1] This is the same corpus as used in
       `Deerwester et al. (1990): Indexing by Latent Semantic Analysis <http://www.cs.bham.ac.uk/~pxt/IDA/lsa_ind.pdf>`_, Table 2.

Review
------
In this tutorial, we took our first steps with gensim.
We created a toy corpus in the vector space representation and transformed it using Tf-Idf.
Finally, for each document in the corpus, we measured its similarity with a query document.

Next, read the tutorials on :ref:`tut1` and :ref:`tut2`.
We will also revisit document similarity in :ref:`tut3`.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.446 seconds)

**Estimated memory usage:**  9 MB


.. _sphx_glr_download_auto_examples_000_core_run_001_getting_started.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example


  .. container:: binder-badge

    .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/mpenkov/gensim/numfocus?filepath=notebooks/auto_examples/000_core/run_001_getting_started.ipynb
      :width: 150 px


  .. container:: sphx-glr-download

     :download:`Download Python source code: run_001_getting_started.py <run_001_getting_started.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: run_001_getting_started.ipynb <run_001_getting_started.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
