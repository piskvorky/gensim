.. _install:

=============
Installation
=============

Quick install
--------------

Run in your terminal::

  sudo easy_install -U gensim

In case that fails, or you don't know what "terminal" means, read on.

-----

Dependencies
-------------
Gensim is known to run on Linux, Windows and Mac OS X and should run on any other
platform that supports Python 2.5 and NumPy. Gensim depends on the following software:

* 3.0 > `Python <http://www.python.org>`_ >= 2.5. Tested with versions 2.5, 2.6 and 2.7.
* `NumPy <http://www.numpy.org>`_ >= 1.3. Tested with version 1.6.1rc2, 1.5.0rc1, 1.4.0, 1.3.0, 1.3.0rc2.
* `SciPy <http://www.scipy.org>`_ >= 0.7. Tested with version 0.9.0, 0.8.0, 0.8.0b1, 0.7.1, 0.7.0.

**Windows users** are well advised to try the `Enthought distribution <http://www.enthought.com/products/epd.php>`_,
which conveniently includes Python&NumPy&SciPy in a single bundle, and is free for academic use.


Install Python and `easy_install`
---------------------------------

Check what version of Python you have with::

    python --version

You can download Python from http://python.org/download.

.. note:: Gensim requires Python 2.5 or greater and will not run under earlier versions.

Next, install the `easy_install utility <http://pypi.python.org/pypi/setuptools>`_,
which will make installing other Python programs easier.

Install SciPy & NumPy
----------------------

These are quite popular Python packages, so chances are there are pre-built binary
distributions available for your platform. You can try installing from source using easy_install::

    sudo easy_install numpy
    sudo easy_install scipy

If that doesn't work or if you'd rather install using a binary package, consult
http://www.scipy.org/Download.

Install `gensim`
-----------------

You can now install (or upgrade) `gensim` with::

    sudo easy_install --upgrade gensim

That's it! Congratulations, you can proceed to the :doc:`tutorials <tutorial>`.

-----

If you also want to run the algorithms over a cluster
of computers, in :doc:`distributed`, you should install with::

    sudo easy_install gensim[distributed]

The optional `distributed` feature installs `Pyro (PYthon Remote Objects) <http://pypi.python.org/pypi/Pyro>`_.
If you don't know what distributed computing means, you can ignore it:
`gensim` will work fine for you anyway.
This optional extension can also be installed separately later with::

    sudo easy_install Pyro4

-----

There are also alternative routes to install:

1. If you have downloaded and unzipped the `tar.gz source <http://pypi.python.org/pypi/gensim>`_
   for `gensim` (or you're installing `gensim` from `github <https://github.com/piskvorky/gensim/>`_),
   you can run::

     sudo python setup.py install

   to install `gensim` into your ``site-packages`` folder.
2. If you wish to make local changes to the `gensim` code (`gensim` is, after all, a
   package which targets research prototyping and modifications), a preferred
   way may be installing with::

     sudo python setup.py develop

   This will only place a symlink into your ``site-packages`` directory. The actual
   files will stay wherever you unpacked them.
3. If you don't have root priviledges (or just don't want to put the package into
   your ``site-packages``), simply unpack the source package somewhere and that's it! No
   compilation or installation needed. Just don't forget to set your PYTHONPATH
   (or modify ``sys.path``), so that Python can find the unpacked package when importing.


Testing `gensim`
----------------

To test the package, unzip the `tar.gz source <http://pypi.python.org/pypi/gensim>`_ and run::

    python setup.py test


Contact
--------

Use the `gensim discussion group <http://groups.google.com/group/gensim/>`_ for
any questions and troubleshooting. For private enquiries, you can also send
me an email to the address at the bottom of this page.
