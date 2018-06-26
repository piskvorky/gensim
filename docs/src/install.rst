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

That's it! Congratulations, you can proceed to the :doc:`tutorials <tutorial>`.

In case that failed, make sure you're installing into a writeable location (or use `sudo`).

-----

Code dependencies
-----------------

Gensim runs on Linux, Windows and Mac OS X, and should run on any other
platform that supports Python 2.7+ and NumPy. Gensim depends on the following software:

* `Python <http://www.python.org>`_ >= 2.7 (tested with versions 2.7, 3.5 and 3.6)
* `NumPy <http://www.numpy.org>`_ >= 1.11.3
* `SciPy <http://www.scipy.org>`_ >= 0.18.1
* `Six <https://pypi.org/project/six/>`_ >= 1.5.0
* `smart_open <https://pypi.org/project/smart_open/>`_ >= 1.2.1

Testing Gensim
--------------

Gensim uses continuous integration, automatically running a full test suite on each pull request with

+------------+-----------------------------------------------------------------------------------------+--------------+
| CI service | Task                                                                                    | Build badge  |
+============+=========================================================================================+==============+
| Travis     | Run tests on Linux and check `code-style <https://www.python.org/dev/peps/pep-0008/?>`_ | |Travis|_    |
+------------+-----------------------------------------------------------------------------------------+--------------+
| AppVeyor   | Run tests on Windows                                                                    | |AppVeyor|_  |
+------------+-----------------------------------------------------------------------------------------+--------------+
| CircleCI    | Build documentation                                                                     | |CircleCI|_  |
+------------+-----------------------------------------------------------------------------------------+--------------+

.. |Travis| image:: https://travis-ci.org/RaRe-Technologies/gensim.svg?branch=develop
.. _Travis: https://travis-ci.org/RaRe-Technologies/gensim

.. |CircleCI| image:: https://circleci.com/gh/RaRe-Technologies/gensim/tree/develop.svg?style=shield
.. _CircleCI: https://circleci.com/gh/RaRe-Technologies/gensim

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/r2au32ucpn8gr0tl/branch/develop?svg=true
.. _AppVeyor: https://ci.appveyor.com/api/projects/status/r2au32ucpn8gr0tl/branch/develop?svg=true


Problems?
---------

Use the `Gensim discussion group <http://groups.google.com/group/gensim/>`_ for
questions and troubleshooting. See the :doc:`support page <support>` for commercial support.
