.. _tutorial:

Tutorials
=========


The tutorials are organized as a series of examples that highlight various features
of `gensim`. It is assumed that the reader is familiar with the `Python language <http://www.python.org/>`_, has :doc:`installed gensim <install>`
and read the :doc:`introduction <intro>`.

The examples are divided into parts on:

.. toctree::
   :maxdepth: 2

   tut1
   tut2
   tut3
   wiki
   distributed

Preliminaries
--------------

All the examples can be directly copied to your Python interpreter shell. `IPython <http://ipython.scipy.org>`_'s ``cpaste`` command is especially handy for copypasting code fragments, including the leading ``>>>`` characters.

Gensim uses Python's standard :mod:`logging` module to log various stuff at various
priority levels; to activate logging (this is optional), run

>>> import logging
>>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


.. _first-example:

Quick Example
-------------

First, let's import gensim and create a small corpus of nine documents and twelve features [1]_:

>>> from gensim import corpora, models, similarities
>>>
>>> corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
>>>           [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
>>>           [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
>>>           [(0, 1.0), (4, 2.0), (7, 1.0)],
>>>           [(3, 1.0), (5, 1.0), (6, 1.0)],
>>>           [(9, 1.0)],
>>>           [(9, 1.0), (10, 1.0)],
>>>           [(9, 1.0), (10, 1.0), (11, 1.0)],
>>>           [(8, 1.0), (10, 1.0), (11, 1.0)]]

In `gensim` a :dfn:`corpus` is simply an object which, when iterated over, returns its documents represented
as sparse vectors. In this case we're using a list of list of tuples. If you're not familiar with the `vector space model <http://en.wikipedia.org/wiki/Vector_space_model>`_, we'll bridge the gap between **raw strings**, **corpora** and **sparse vectors** in the next tutorial on :doc:`tut1`.

If you're familiar with the vector space model, you'll probably know that the way you parse your documents and convert them to vectors
has major impact on the quality of any subsequent applications.

.. note::
    In this example, the whole corpus is stored in memory, as a Python list. However,
    the corpus interface only dictates that a corpus must support iteration over its
    constituent documents. For very large corpora, it is advantageous to keep the
    corpus on disk, and access its documents sequentially, one at a time. All the
    operations and transformations are implemented in such a way that makes
    them independent of the size of the corpus, memory-wise.


Next, let's initialize a :dfn:`transformation`:

>>> tfidf = models.TfidfModel(corpus)

A transformation is used to convert documents from one vector representation into another:

>>> vec = [(0, 1), (4, 1)]
>>> print(tfidf[vec])
[(0, 0.8075244), (4, 0.5898342)]

Here, we used `Tf-Idf <http://en.wikipedia.org/wiki/Tf%E2%80%93idf>`_, a simple
transformation which takes documents represented as bag-of-words counts and applies
a weighting which discounts common terms (or, equivalently, promotes rare terms).
It also scales the resulting vector to unit length (in the `Euclidean norm <http://en.wikipedia.org/wiki/Norm_%28mathematics%29#Euclidean_norm>`_).

Transformations are covered in detail in the tutorial on :doc:`tut2`.

To transform the whole corpus via TfIdf and index it, in preparation for similarity queries:

>>> index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)

and to query the similarity of our query vector ``vec`` against every document in the corpus:

>>> sims = index[tfidf[vec]]
>>> print(list(enumerate(sims)))
[(0, 0.4662244), (1, 0.19139354), (2, 0.24600551), (3, 0.82094586), (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0), (8, 0.0)]

How to read this output? Document number zero (the first document) has a similarity score of 0.466=46.6\%,
the second document has a similarity score of 19.1\% etc.

Thus, according to TfIdf document representation and cosine similarity measure,
the most similar to our query document `vec` is document no. 3, with a similarity score of 82.1%.
Note that in the TfIdf representation, any documents which do not share any common features
with ``vec`` at all (documents no. 4--8) get a similarity score of 0.0. See the :doc:`tut3` tutorial for more detail.

------

.. [1]  This is the same corpus as used in
        `Deerwester et al. (1990): Indexing by Latent Semantic Analysis <http://www.cs.bham.ac.uk/~pxt/IDA/lsa_ind.pdf>`_, Table 2.


