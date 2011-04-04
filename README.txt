==============================================
gensim -- Python Framework for Topic Modelling
==============================================



Gensim is a Python library for *Vector Space Modelling* with very large corpora.
Target audience is the *Natural Language Processing* (NLP) community.


Features
---------

* All algorithms are **memory-independent** w.r.t. the corpus size (can process input larger than RAM),
* **Intuitive interfaces**

  * easy to plug in your own input corpus/datastream (trivial streaming API)
  * easy to extend with other Vector Space algorithms (trivial transformation API)

* Efficient implementations of popular algorithms, such as online **Latent Semantic Analysis**,
  **Latent Dirichlet Allocation** or **Random Projections**
* **Distributed computing**: can run *Latent Semantic Analysis* and *Latent Dirichlet Allocation* on a cluster of computers.
* Extensive `HTML documentation and tutorials <http://nlp.fi.muni.cz/projekty/gensim/>`_.


If this feature list left you scratching your head, you can first read more about the `Vector
Space Model <http://en.wikipedia.org/wiki/Vector_space_model>`_ and `unsupervised
document analysis <http://en.wikipedia.org/wiki/Latent_semantic_indexing>`_ on Wikipedia.

Installation
------------

This software depends on `NumPy and Scipy <http://www.scipy.org/Download>`_, two Python packages for scientific computing.
You must have them installed prior to installing `gensim`.

The simple way to install `gensim` is::

    sudo easy_install gensim

Or, if you have instead downloaded and unzipped the `source tar.gz <http://pypi.python.org/pypi/gensim>`_ package,
you'll need to run::

    python setup.py test
    sudo python setup.py install


For alternative modes of installation (without root priviledges, development
installation, optional install features), see the `documentation <http://nlp.fi.muni.cz/projekty/gensim/install.html>`_.

This version has been tested under Python 2.5 and 2.6, but should run on any 2.5 <= Python < 3.0.

Documentation
-------------

Manual for the gensim package is available in `HTML <http://nlp.fi.muni.cz/projekty/gensim/>`_. It
contains a walk-through of all its features and a complete reference section.
It is also included in the source distribution package.

Citations
---------

There is one automated test which uses data from Lee et al. (2005) which
compares different models for semantic similarity and verifies the results
with similarity judgements from humans. Many thanks to
Michael D. Lee (michael.lee@adelaide.edu.au) who provideded us
with his corpus and similarity data.

If you need to reference this dataset, please cite:

Lee, M., Pincombe, B., & Welsh, M. (2005).
An empirical evaluation of models of text document similarity.
Proceedings of the 27th Annual Conference of the Cognitive Science Society

----------------

Gensim is open source software, and has been released under the
`GNU LPGL license <http://www.gnu.org/licenses/lgpl.html>`_.
Copyright (c) 2010 Radim Rehurek
