.. _install:

=============
Installation
=============

Quick install
--------------

Run in your terminal (recommended)::

  pip install --upgrade gensim

or, alternatively for `conda` environments::

  conda install -c conda-forge gensim

In case that fails, make sure you're installing into a writeable location (or use `sudo`), or keep reading.

-----

Dependencies
-------------

Gensim runs on Linux, Windows and Mac OS X, and should run on any other
platform that supports Python 2.6+ and NumPy. Gensim depends on the following software:

* `Python <http://www.python.org>`_ >= 2.6. Tested with versions 2.6, 2.7, 3.3, 3.4 and 3.5. Support for Python 2.5 was discontinued starting Gensim 0.10.0; if you *must* use Python 2.5, install Gensim version 0.9.1.
* `NumPy <http://www.numpy.org>`_ >= 1.3. Tested with version 1.9.0, 1.7.1, 1.7.0, 1.6.2, 1.6.1rc2, 1.5.0rc1, 1.4.0, 1.3.0, 1.3.0rc2.
* `SciPy <http://www.scipy.org>`_ >= 0.7. Tested with version 0.14.0, 0.12.0, 0.11.0, 0.10.1, 0.9.0, 0.8.0, 0.8.0b1, 0.7.1, 0.7.0.


Install Python and `pip`
------------------------

Check what version of Python you have with::

    python --version

You can download Python from http://python.org/download.

.. note:: Gensim requires Python 2.6 / 3.3 or greater, and will not run under earlier versions.

Make sure you have `pip`, Python's recommended tool for installing and managing Python dependencies::

    pip --version

Pip typically comes pre-installed with Python. If not, refer to `Installing pip <https://pip.pypa.io/en/stable/installing/>`_.


Install SciPy & NumPy
----------------------

These are popular Python packages, so chances are there are pre-built binary
distributions available for your platform. Install them using `pip`::

    pip install numpy
    pip install scipy

If that doesn't work or if you'd rather install using a binary package, consult http://www.scipy.org/Download.

Install Gensim
--------------

You can now install (or upgrade) Gensim with::

    pip install --upgrade gensim

That's it! Congratulations, you can proceed to the :doc:`tutorials <tutorial>`.

-----

If you also want to run the algorithms over a cluster of computers, in :doc:`distributed`, you should install with::

    pip install 'gensim[distributed]'

The optional ``distributed`` feature installs `Pyro (PYthon Remote Objects) <http://pypi.python.org/pypi/Pyro>`_.
If you don't know what distributed computing means, you can ignore it: Gensim will work fine for you anyway.

This optional extension can also be installed separately later with::

    pip install Pyro4

-----

There are also alternative routes to install:

1. If you have downloaded and unzipped the `tar.gz source <http://pypi.python.org/pypi/gensim>`_
   for Gensim (or you're installing Gensim from `Github <https://github.com/piskvorky/gensim/>`_),
   you can run::

     pip install .

   to install Gensim into your ``site-packages`` folder.
2. If you wish to make local changes to the Gensim code, a preferred way may be installing with::

     pip install --editable .

   This will only place a symlink into your ``site-packages`` directory. The actual
   files will stay wherever you unpacked them, ready for editing.


Testing Gensim
--------------

Gensim uses continuous integration, automatically running a full test suite on each pull request: |Travis|_

.. |Travis| image:: https://travis-ci.org/RaRe-Technologies/gensim.svg?branch=develop
.. _Travis: https://travis-ci.org/RaRe-Technologies/gensim


Problems?
---------

Use the `Gensim discussion group <http://groups.google.com/group/gensim/>`_ for
questions and troubleshooting. See the :doc:`support page <support>` for commercial support.
