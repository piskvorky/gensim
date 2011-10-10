.. gensim documentation master file, created by
   sphinx-quickstart on Tue Mar 16 19:45:41 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gensim -- Vector Space Modelling for Humans
=====================================================

.. admonition:: What's new?

   * 05/10/2011: version 0.8.1 `released <http://pypi.python.org/pypi/gensim>`_: new `document similarity server <http://radimrehurek.com/gensim/simserver.html>`_ and several speed-ups and fixes.
   * 09/09/2011: moved website to `radimrehurek.com <http://radimrehurek.com/gensim>`_.
   * 19/06/2011: version 0.8.0 is `out <http://pypi.python.org/pypi/gensim>`_! Faster & better: :doc:`walkthrough of the changes<changes_080>`.
   * 12/02/2011: faster and leaner **Latent Semantic Indexing (LSI)** and **Latent Dirichlet Allocation (LDA)**:

     * :doc:`Processing the English Wikipedia <wiki>`, 3.2 million documents (`NIPS workshop paper <http://arxiv.org/abs/1102.5597>`_)
     * :doc:`dist_lsi` & :doc:`dist_lda`

For an **overview** of what you can (or cannot) do with `gensim`, go to the :doc:`introduction <intro>`.

For **installation** and **troubleshooting**, see the :doc:`installation <install>` page and the `gensim discussion group <http://groups.google.com/group/gensim/>`_.

For **examples** on how to convert text to vectors and work with the result, try the :doc:`tutorials <tutorial>`.

When **citing** `gensim` in academic papers, use
`this BibTeX entry <bibtex_gensim.bib>`_.


Quick Reference Example
------------------------

>>> from gensim import corpora, models, similarities
>>>
>>> # Load corpus iterator from a Matrix Market file on disk.
>>> # See Tutorial 1 on text corpora and vectors.
>>> corpus = corpora.MmCorpus('/path/to/corpus.mm')
>>>
>>> # Initialize a transformation (Latent Semantic Indexing with 200 latent dimensions).
>>> # See Tutorial 2 on semantic models.
>>> lsi = models.LsiModel(corpus, num_topics=200)
>>>
>>> # Convert another corpus to the latent space and index it.
>>> # See Tutorial 3 on similarity queries.
>>> index = similarities.MatrixSimilarity(lsi[another_corpus])
>>>
>>> # determine similarity of a query document against each document in the index
>>> sims = index[query]



.. toctree::
   :hidden:
   :maxdepth: 1

   intro
   install
   tutorial
   distributed
   simserver
   apiref
