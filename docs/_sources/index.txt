.. gensim documentation master file, created by
   sphinx-quickstart on Tue Mar 16 19:45:41 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gensim -- Python Framework for Vector Space Modelling
=====================================================

.. admonition:: What's new in version |version|?

   * faster and leaner **Latent Semantic Indexing (LSI)** and **Latent Dirichlet Allocation (LDA)**:
   
     * :doc:`Processing the English Wikipedia <wiki>` (`NIPS workshop paper <http://nlp.fi.muni.cz/~xrehurek/nips/rehurek_nips.pdf>`_)
     * :doc:`dist_lsi` & :doc:`dist_lda`
     
   * Input corpus iterators can come from a compressed file (**bzip2**, **gzip**, ...), to save disk space when dealing with
     very large corpora.
   * `gensim` code now resides on `github <https://github.com/piskvorky/gensim/>`_.

.. ::
 
   `gensim` now completes :doc:`LSI of the English Wikipedia <wiki>` 
   (3.2 million documents, 2 billion words) in **5 hours 14 minutes**, using a one-pass incremental 
   SVD algorithm, on a Macbook Pro laptop (`NIPS workshop paper <http://nlp.fi.muni.cz/~xrehurek/nips/rehurek_nips.pdf>`_). 
   Be sure to check out the :doc:`distributed mode <distributed>`, too.

For an **overview** of what you can (or cannot) do with `gensim`, go to the :doc:`introduction <intro>`.

For **examples** on how to use it, try the :doc:`tutorials <tutorial>`.

For **troubleshooting** and further questions, see the :doc:`installation <install>` page and the `gensim discussion group <http://groups.google.com/group/gensim/>`_.


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



.. toctree::
   :hidden:
   :maxdepth: 1
   
   intro
   install
   tutorial
   distributed
   apiref
