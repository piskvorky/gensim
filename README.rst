==============================================
gensim -- Topic Modelling in Python
==============================================

|Travis|_
|Downloads|_
|License|_

.. |Travis| image:: https://api.travis-ci.org/piskvorky/gensim.png?branch=develop
.. |Downloads| image:: https://pypip.in/d/gensim/badge.png
.. |License| image:: https://pypip.in/license/gensim/badge.png
.. _Travis: https://travis-ci.org/piskvorky/gensim
.. _Downloads: https://pypi.python.org/pypi/gensim
.. _License: http://radimrehurek.com/gensim/about.html

Gensim is a Python library for *topic modelling*, *document indexing* and *similarity retrieval* with large corpora.
Target audience is the *natural language processing* (NLP) and *information retrieval* (IR) community.

Features
---------

* All algorithms are **memory-independent** w.r.t. the corpus size (can process input larger than RAM),
* **Intuitive interfaces**

  * easy to plug in your own input corpus/datastream (trivial streaming API)
  * easy to extend with other Vector Space algorithms (trivial transformation API)

* Efficient implementations of popular algorithms, such as online **Latent Semantic Analysis (LSA/LSI)**,
  **Latent Dirichlet Allocation (LDA)**, **Random Projections (RP)**, **Hierarchical Dirichlet Process (HDP)**  or **word2vec deep learning**.
* **Distributed computing**: can run *Latent Semantic Analysis* and *Latent Dirichlet Allocation* on a cluster of computers, and *word2vec* on multiple cores.
* Extensive `HTML documentation and tutorials <http://radimrehurek.com/gensim/>`_.


If this feature list left you scratching your head, you can first read more about the `Vector
Space Model <http://en.wikipedia.org/wiki/Vector_space_model>`_ and `unsupervised
document analysis <http://en.wikipedia.org/wiki/Latent_semantic_indexing>`_ on Wikipedia.

Installation
------------

This software depends on `NumPy and Scipy <http://www.scipy.org/Download>`_, two Python packages for scientific computing.
You must have them installed prior to installing `gensim`.

It is also recommended you install a fast BLAS library before installing NumPy. This is optional, but using an optimized BLAS such as `ATLAS <http://math-atlas.sourceforge.net/>`_ or `OpenBLAS <http://xianyi.github.io/OpenBLAS/>`_ is known to improve performance by as much as an order of magnitude. On OS X, NumPy picks up the BLAS that comes with it automatically, so you don't need to do anything special.

The simple way to install `gensim` is::

    sudo easy_install gensim

Or, if you have instead downloaded and unzipped the `source tar.gz <http://pypi.python.org/pypi/gensim>`_ package,
you'll need to run::

    python setup.py test
    sudo python setup.py install


For alternative modes of installation (without root privileges, development
installation, optional install features), see the `documentation <http://radimrehurek.com/gensim/install.html>`_.

This version has been tested under Python 2.6, 2.7 and 3.3.

Documentation
-------------

Manual for the gensim package is available in `HTML <http://radimrehurek.com/gensim/>`_. It
contains a walk-through of all its features and a complete reference section.
It is also included in the source distribution package.

----------------

Gensim is open source software released under the `GNU LGPL license <http://www.gnu.org/licenses/lgpl.html>`_.
Copyright (c) 2009-2014 Radim Rehurek
