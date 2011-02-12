.. _dist_lda:

Distributed Latent Dirichlet Allocation
============================================


.. note::
  See :doc:`distributed` for an introduction to distributed computing in `gensim`.


Setting up the cluster
_______________________

See the tutorial on :doc:`dist_lsi`; setting up a cluster for LDA is completely
analogous, except you want to run `lda_worker` and `lda_dispatcher` scripts instead 
of `lsi_worker` and `lsi_dispatcher`.

Running LDA
____________

Run LDA like you normally would, but turn on the `distributed=True` constructor 
parameter. 

FIXME add some experiments here, ala LSA: one trivial, one wiki::

    >>> # extract 100 LDA topics, using 1 pass and updating every 1 chunk (10,000 documents)
    >>> lda = gensim.models.ldamodel.LdaModel(corpus=wiki, id2word=id2word, numTopics=100, distributed=True)
    
    >>> # print the most contributing words (both positively and negatively) for the topics
    >>> lda.printTopics(-1) # -1 to print 'em all
