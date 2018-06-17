.. _install:

=============
Installation
=============

Quick install
--------------

Run in your terminal::

  easy_install -U gensim

or, alternatively::

  pip install --upgrade gensim

In case that fails, make sure you're installing into a writeable location (or use `sudo`), or read on.

-----

Dependencies
-------------
Gensim is known to run on Linux, Windows and Mac OS X and should run on any other
platform that supports Python 2.6+ and NumPy. Gensim depends on the following software:

* `Python <http://www.python.org>`_ >= 2.6. Tested with versions 2.6, 2.7, 3.3, 3.4 and 3.5. Support for Python 2.5 was discontinued starting gensim 0.10.0; if you *must* use Python 2.5, install gensim 0.9.1.
* `NumPy <http://www.numpy.org>`_ >= 1.3. Tested with version 1.9.0, 1.7.1, 1.7.0, 1.6.2, 1.6.1rc2, 1.5.0rc1, 1.4.0, 1.3.0, 1.3.0rc2.
* `SciPy <http://www.scipy.org>`_ >= 0.7. Tested with version 0.14.0, 0.12.0, 0.11.0, 0.10.1, 0.9.0, 0.8.0, 0.8.0b1, 0.7.1, 0.7.0.


Install Python and `easy_install`
---------------------------------

Check what version of Python you have with::

    python --version

You can download Python from http://python.org/download.

.. note:: Gensim requires Python 2.6 / 3.3 or greater, and will not run under earlier versions.

Next, install the `easy_install utility <http://pypi.python.org/pypi/setuptools>`_,
which will make installing other Python programs easier.

Install SciPy & NumPy
----------------------

These are quite popular Python packages, so chances are there are pre-built binary
distributions available for your platform. You can try installing from source using `easy_install`::

    easy_install numpy
    easy_install scipy

If that doesn't work or if you'd rather install using a binary package, consult
http://www.scipy.org/Download.

Install Gensim
-----------------

You can now install (or upgrade) Gensim with::

    easy_install --upgrade gensim

That's it! Congratulations, you can proceed to the :doc:`tutorials <tutorial>`.

-----

If you also want to run the algorithms over a cluster
of computers, in :doc:`distributed`, you should install with::

    easy_install gensim[distributed]

The optional `distributed` feature installs `Pyro (PYthon Remote Objects) <http://pypi.python.org/pypi/Pyro>`_.
If you don't know what distributed computing means, you can ignore it:
Gensim will work fine for you anyway.

This optional extension can also be installed separately later with::

    easy_install Pyro4

-----

There are also alternative routes to install:

1. If you have downloaded and unzipped the `tar.gz source <http://pypi.python.org/pypi/gensim>`_
   for Gensim (or you're installing Gensim from `github <https://github.com/piskvorky/gensim/>`_),
   you can run::

     python setup.py install

   to install Gensim into your ``site-packages`` folder.

2. If you wish to make local changes to the Gensim code, a preferred
   way may be installing with::

     python setup.py develop

   or

     pip install -e .

   This will only place a symlink into your ``site-packages`` directory. The actual
   files will stay wherever you unpacked them.


Testing Gensim
----------------

To test the package, unzip the `tar.gz source <http://pypi.python.org/pypi/gensim>`_ and run::

    python setup.py test

Gensim uses Travis CI for continuous integration, automatically running the full test suite on each pull request and commit: |Travis|_

.. |Travis| image:: https://api.travis-ci.org/piskvorky/gensim.png?branch=develop
.. _Travis: https://travis-ci.org/piskvorky/gensim


Problems?
---------

Use the `Gensim discussion group <http://groups.google.com/group/gensim/>`_ for
questions and troubleshooting. See the :doc:`support page <support>` for commercial support.
