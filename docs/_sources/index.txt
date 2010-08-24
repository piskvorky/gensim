.. gensim documentation master file, created by
   sphinx-quickstart on Tue Mar 16 19:45:41 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gensim -- Python Framework for Vector Space Modeling
=====================================================

.. admonition:: What's new?

   Version |version| is `out <http://pypi.python.org/pypi/gensim>`_! There are big changes to *Latent Semantic Indexing*:
   
   * LSI is now about 30x faster, consumes less memory, **and** can be run in :doc:`distributed mode <distributed>`!
   * Optimizations to vocabulary generation.
   * Input corpus iterator can come from a compressed file (bzip2, gzip, ...), to save disk space when dealing with
     very large corpora.
   
   If you have a cluster of computers, the time taken to process a given corpus 
   with our distributed LSA algorithm drops almost
   linearly with the number of machines. Of course, the option of incrementally adding 
   new documents to an existing decomposition, without the need to recompute everything 
   from scratch, remains from the previous version. This means that your document
   input stream may even be infinite in size, with new documents coming in asynchronously.

..
   To read more about the theoretical side of things, check out our new `draft paper <http://todo>`_.

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
>>> # initialize a transformation (Latent Semantic Indexing with 200 latent dimensions)
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
