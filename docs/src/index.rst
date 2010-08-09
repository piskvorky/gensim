.. gensim documentation master file, created by
   sphinx-quickstart on Tue Mar 16 19:45:41 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gensim -- Python Framework for Vector Space Modeling
=====================================================

.. admonition:: What's new?

   Version |version| is `out <http://pypi.python.org/pypi/gensim>`_!
   
   It contains several minor bug fixes and a **fully online** implementation of Latent Semanting Indexing! 
   You can now :ref:`update LSI with new documents <transformations>` at will, and 
   use the resulting LSI transformation at each step. The training document stream may even be infinite!

For an introduction on what gensim does (or does not do), go to the :doc:`introduction <intro>`.

To download and install `gensim`, consult the :doc:`install <install>` page.

For examples on how to use it, try the :doc:`tutorials <tutorial>`.

Quick Reference Example
------------------------

>>> from gensim import corpora, models, similarities
>>>
>>> # load corpus iterator from a Matrix Market file on disk
>>> corpus = corpora.MmCorpus('/path/to/corpus.mm')
>>>
>>> # initialize a transformation (Latent Semantic Indexing with twenty latent dimensions)
>>> lsi = models.LsiModel(corpus, numTopics = 200)
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
   distributed
   apiref    
