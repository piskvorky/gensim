==============================================
gensim -- Python Framework for Topic Modeling
==============================================

**Gensim** is a Python framework designed to help make
the conversion of natural language texts to the Vector Space Model as simple and 
natural as possible.

Gensim contains algorithms for unsupervised learning from raw, unstructured digital texts,
such as **Latent Semantic Analysis** and **Latent Dirichlet Allocation**.
These algorithms discover hidden (*latent*) corpus structure.
Once found, documents can be succinctly expressed in terms of this structure, queried for topical similarity and so on.

If the previous paragraphs left you confused, you can first read more about the `Vector 
Space Model <http://en.wikipedia.org/wiki/Vector_space_model>`_ and `unsupervised 
document analysis <http://en.wikipedia.org/wiki/Latent_semantic_indexing>`_ at Wikipedia.

.. note::

  Gensim's target audience is the NLP research community and interested general public; 
  gensim is not meant to be a production tool for commercial environments.

This version has been tested under Python 2.5, but should run on any 2.5 <= Python < 3.0.

Installation
------------

gensim depends on NumPy and Scipy, two Python packages for scientific computing.
You need to have them installed prior to using gensim; if you don't have them yet, 
you can get them from <http://www.scipy.org/Download>.

1. The simple way to install `gensim` is::

    sudo easy_install gensim

2. If you have instead downloaded and unzipped the `source tar.gz package <http://pypi.python.org/pypi/gensim>`_, 
   you'll need to run::

    python setup.py test
    sudo python setup.py install


For alternative modes of installation (without root priviledges, development 
installation), see the `documentation <http://nlp/fi.muni.cz/projekty/gensim/install.html>`_.


Documentation
-------------

Manual for the gensim package is available as `HTML <http://nlp/fi.muni.cz/projekty/gensim/>'_
and as `PDF <http://nlp/fi.muni.cz/projekty/gensim/gensim_manual.pdf>`_. It
contains a walk-through of all the features and a complete reference section.
It is included in the source package.

-------

Gensim is open source software, and has been released under 
`GNU LPGL license <http://www.gnu.org/licenses/lgpl.html>`_.
Copyright (c) 2010 Radim Rehurek
