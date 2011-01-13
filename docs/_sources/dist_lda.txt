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

FIXME add some experiments here.