==============================================
gensim -- Topic Modelling in Python
==============================================

|Travis|_
|Downloads|_
|License|_

.. |Travis| image:: https://img.shields.io/travis/piskvorky/gensim/develop.svg
.. |Downloads| image:: https://img.shields.io/pypi/dm/gensim.svg
.. |License| image:: https://img.shields.io/pypi/l/gensim.svg
.. _Travis: https://travis-ci.org/piskvorky/gensim
.. _Downloads: https://pypi.python.org/pypi/gensim
.. _License: http://radimrehurek.com/gensim/about.html

Gensim is a Python library for *topic modelling*, *document indexing* and *similarity retrieval* with large corpora.
Target audience is the *natural language processing* (NLP) and *information retrieval* (IR) community.

Features
---------

* All algorithms are **memory-independent** w.r.t. the corpus size (can process input larger than RAM, streamed, out-of-core),
* **Intuitive interfaces**

  * easy to plug in your own input corpus/datastream (trivial streaming API)
  * easy to extend with other Vector Space algorithms (trivial transformation API)

* Efficient multicore implementations of popular algorithms, such as online **Latent Semantic Analysis (LSA/LSI/SVD)**,
  **Latent Dirichlet Allocation (LDA)**, **Random Projections (RP)**, **Hierarchical Dirichlet Process (HDP)**  or **word2vec deep learning**.
* **Distributed computing**: can run *Latent Semantic Analysis* and *Latent Dirichlet Allocation* on a cluster of computers.
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

    pip install -U gensim

Or, if you have instead downloaded and unzipped the `source tar.gz <http://pypi.python.org/pypi/gensim>`_ package,
you'd run::

    python setup.py test
    python setup.py install


For alternative modes of installation (without root privileges, development
installation, optional install features), see the `documentation <http://radimrehurek.com/gensim/install.html>`_.

This version has been tested under Python 2.6, 2.7, 3.3, 3.4 and 3.5 (support for Python 2.5 was dropped in gensim 0.10.0; install gensim 0.9.1 if you *must* use Python 2.5). Gensim's github repo is hooked to `Travis CI for automated testing <https://travis-ci.org/piskvorky/gensim>`_ on every commit push and pull request.

How come gensim is so fast and memory efficient? Isn't it pure Python, and isn't Python slow and greedy?
--------------------------------------------------------------------------------------------------------

Many scientific algorithms can be expressed in terms of large matrix operations (see the BLAS note above). Gensim taps into these low-level BLAS libraries, by means of its dependency on NumPy. So while gensim-the-top-level-code is pure Python, it actually executes highly optimized Fortran/C under the hood, including multithreading (if your BLAS is so configured).

Memory-wise, gensim makes heavy use of Python's built-in generators and iterators for streamed data processing. Memory efficiency was one of gensim's `design goals <http://radimrehurek.com/gensim/about.html>`_, and is a central feature of gensim, rather than something bolted on as an afterthought.

Documentation
-------------

Manual for the gensim package is available in `HTML <http://radimrehurek.com/gensim/>`_. It
contains a walk-through of all its features and a complete reference section.
It is also included in the source distribution package.

----------------

Gensim is open source software released under the `GNU LGPL license <http://www.gnu.org/licenses/lgpl.html>`_.
Copyright (c) 2009-now Radim Rehurek

|Analytics|_

.. |Analytics| image:: https://ga-beacon.appspot.com/UA-24066335-5/your-repo/page-name
.. _Analytics: https://github.com/igrigorik/ga-beacon
