.. _intro: 

=============
Installation
=============

Gensim is known to run on Linux and Mac OS X and should also run on Windows and
any platform that supports Python 2.5 and NumPy. Gensim depends on the following
software:

* 3.0 > `Python <http://www.python.org>`_ >= 2.5. Tested with version 2.5.
* `NumPy <http://www.numpy.org>`_ >= 1.2. Tested with version 1.3.0rc2.
* `SciPy <http://www.scipy.org>`_ >= 0.7. Tested with version 0.7.1.


Install Python
---------------

Check what version of Python you have with::

    python --version

You can download Python 2.5 from http://python.org/download.

.. note:: Gensim requires Python 2.5 or greater and will not run under earlier versions.

Install SciPy & NumPy
----------------------

These are quite popular Python packages, so chances are there are pre-built binary 
distributions available for your platform. You can try installing from source using easy_install::

    sudo easy_install numpy
    sudo easy_install scipy

If that doesn't work or if you'd rather install using a binary package, consult
http://www.scipy.org/Download.

Install gensim
---------------

You can now install (or upgrade) gensim with::

    sudo easy_install gensim

That's it!

There are also alternative routes:

1. If you have downloaded and unzipped the `tar.gz source <http://pypi.python.org/pypi/gensim>`_
   for gensim (or you're installing gensim from `svn <http://my-trac.assembla.com/gensim>`_), 
   you can run::
   
     sudo python setup.py install 
   
   to install gensim into your ``site-packages`` folder.
2. If you wish to make local changes to gensim code (gensim is, after all, a 
   package which targets research prototyping and modifications), a preferred 
   way may be installing with::
   
     sudo python setup.py develop
   
   This will only place a symlink into your ``site-packages`` directory. The actual
   files will stay wherever you unpacked them.
3. If you don't have root priviledges (or just don't want to put the package into
   your ``site-packages``), simply unpack the package somewhere and that's it! No
   compilation or installation needed. Just don't forget to set your PYTHONPATH
   (or modify ``sys.path``), so that Python can find the package when importing.


Testing gensim
----------------

To test the package, run::

    python setup.py test

from the unzipped source directory.

Contact
--------

If you encounter problems or have any questions regarding gensim, please let us 
know by emailing <radimrehurek(at)seznam.cz>.
