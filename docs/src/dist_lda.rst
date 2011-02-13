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
parameter::

    >>> # extract 100 LDA topics, using default parameters
    >>> lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, numTopics=100, distributed=True)
    using distributed version with 4 workers
    running online LDA training, 100 topics, 1 passes over the supplied corpus of 3199665 documets, updating model once every 40000 documents
    ..
    
    

In serial mode (no distribution), creating this LDA :doc:`model of Wikipedia <wiki>`
takes 10h56m on my laptop (OS X, C2D 2.53GHz, 4GB RAM with `libVec`). 
In distributed mode with four workers (Linux, Xeons of 2Ghz, 4GB RAM 
with `ATLAS`), the wallclock time taken drops to 3h20m.

If you used this distributed LDA implementation `gensim`, please let me know (my
email is at the bottom of this page). I would like to hear about your application and
the possible (inevitable?) issues that you encountered, to make `gensim` better.
