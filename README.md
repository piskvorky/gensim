gensim – Topic Modelling in Python
==================================

![Travis]\_ ![Wheel]\_

Gensim is a Python library for *topic modelling*, *document indexing*
and *similarity retrieval* with large corpora. Target audience is the
*natural language processing* (NLP) and *information retrieval* (IR)
community.

Features
--------

-   All algorithms are **memory-independent** w.r.t. the corpus size
    (can process input larger than RAM, streamed, out-of-core),
-   **Intuitive interfaces**
    -   easy to plug in your own input corpus/datastream (trivial
        streaming API)
    -   easy to extend with other Vector Space algorithms (trivial
        transformation API)
-   Efficient multicore implementations of popular algorithms, such as
    online **Latent Semantic Analysis (LSA/LSI/SVD)**, **Latent
    Dirichlet Allocation (LDA)**, **Random Projections (RP)**,
    **Hierarchical Dirichlet Process (HDP)** or **word2vec deep
    learning**.
-   **Distributed computing**: can run *Latent Semantic Analysis* and
    *Latent Dirichlet Allocation* on a cluster of computers.
-   Extensive [documentation and Jupyter Notebook tutorials].

If this feature list left you scratching your head, you can first read
more about the [Vector Space Model] and [unsupervised document analysis]
on Wikipedia.

Installation
------------

This software depends on [NumPy and Scipy], two Python packages for
scientific computing. You must have them installed prior to installing
gensim.

It is also recommended you install a fast BLAS library before installing
NumPy. This is optional, but using an optimized BLAS such as [ATLAS] or
[OpenBLAS] is known to improve performance by as much as an order of
magnitude. On OS X, NumPy picks up the BLAS that comes with it
automatically, so you don’t need to do anything special.

The simple way to install gensim is:

    pip install -U gensim

Or, if you have instead downloaded and unzipped the [source tar.gz]
package, you’d run:

    python setup.py test
    python setup.py install

For alternative modes of installation (without root privileges,
development installation, optional install features), see the
[documentation].

This version has been teste

  [Travis]: https://img.shields.io/travis/RaRe-Technologies/gensim/develop.svg
  [Wheel]: https://img.shields.io/pypi/wheel/gensim.svg
  [documentation and Jupyter Notebook tutorials]: https://github.com/RaRe-Technologies/gensim/#documentation
  [Vector Space Model]: http://en.wikipedia.org/wiki/Vector_space_model
  [unsupervised document analysis]: http://en.wikipedia.org/wiki/Latent_semantic_indexing
  [NumPy and Scipy]: http://www.scipy.org/Download
  [ATLAS]: http://math-atlas.sourceforge.net/
  [OpenBLAS]: http://xianyi.github.io/OpenBLAS/
  [source tar.gz]: http://pypi.python.org/pypi/gensim
  [documentation]: http://radimrehurek.com/gensim/install.html
