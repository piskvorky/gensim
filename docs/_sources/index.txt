.. gensim documentation master file, created by
   sphinx-quickstart on Tue Mar 16 19:45:41 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gensim -- Vector Space Modelling for Humans
=====================================================

.. admonition:: What's new?

   * 19/06/2011: version 0.8.0 is out! :doc:`CHANGELOG<changes_080>`
   * 12/02/2011: faster and leaner **Latent Semantic Indexing (LSI)** and **Latent Dirichlet Allocation (LDA)**:

     * :doc:`Processing the English Wikipedia <wiki>`, 3.2 million documents (`NIPS workshop paper <http://arxiv.org/abs/1102.5597>`_)
     * :doc:`dist_lsi` & :doc:`dist_lda`

For an **overview** of what you can (or cannot) do with `gensim`, go to the :doc:`introduction <intro>`.

For **installation** and **troubleshooting**, see the :doc:`installation <install>` page and the `gensim discussion group <http://groups.google.com/group/gensim/>`_.

For **examples** on how to use it, try the :doc:`tutorials <tutorial>`.

When **citing** `gensim` in academic papers, please use
`this BibTeX entry <http://nlp.fi.muni.cz/projekty/gensim/bibtex_gensim.bib>`_.


Quick Reference Example
------------------------

>>> from gensim import corpora, models, similarities
>>>
>>> # load corpus iterator from a Matrix Market file on disk
>>> corpus = corpora.MmCorpus('/path/to/corpus.mm')
>>>
>>> # initialize a transformation (Latent Semantic Indexing with 200 latent dimensions)
>>> lsi = models.LsiModel(corpus, num_topics=200)
>>>
>>> # convert another corpus to the latent space and index it
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
   apiref
