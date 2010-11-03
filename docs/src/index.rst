.. gensim documentation master file, created by
   sphinx-quickstart on Tue Mar 16 19:45:41 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gensim -- Python Framework for Vector Space Modelling
=====================================================

.. admonition:: What's new?

   Version |version| is `out <http://pypi.python.org/pypi/gensim>`_!
   
   * *Latent Semantic Indexing (LSI)* and *Latent Dirichlet Allocation (LDA)* are now **faster**, consume **less memory** and can be run in :doc:`distributed mode <distributed>`.
   * Optimizations to vocabulary generation.
   * Input corpus iterator can come from a compressed file (**bzip2**, **gzip**, ...), to save disk space when dealing with
     very large corpora.
   
   `gensim` now contains two algorithms for Latent Semantic Indexing:
   
   1. streamed **stochastic algorithm**: takes :doc:`2h30m hours on the English Wikipedia <wiki>` (3.2 mil. documents), on a Macbook Pro.
   2. streamed **single pass algorithm**: slower (takes 5h14m), but only accesses each document 
      once; use this if your input comes streaming in and you cannot store it
      persistently.

..
   See the `draft paper <http://nlp.fi.muni.cz/~xrehurek/nips/rehurek_nips.pdf>`_ for more info.

For an overview on what gensim does (or does not do), go to the :doc:`introduction <intro>`.

To download and install `gensim`, consult the :doc:`install <install>` page.

For examples on how to use it, try the :doc:`tutorials <tutorial>`.

Quick Reference Example
------------------------

>>> from gensim import corpora, models, similarities
>>>
>>> # load corpus iterator from a Matrix Market file on disk
>>> corpus = corpora.MmCorpus('/path/to/corpus.mm')
>>>
>>> # initialize a transformation (Latent Semantic Indexing with 200 latent dimensions)
>>> lsi = models.LsiModel(corpus, numTopics=200)
>>>
>>> # convert the same corpus to latent space and index it
>>> index = similarities.MatrixSimilarity(lsi[corpus])
>>> 
>>> # perform similarity query of another vector in LSI space against the whole corpus
>>> sims = index[query]


Contents
---------

.. toctree::
   :maxdepth: 1
   
   intro
   install
   tutorial
   apiref    
