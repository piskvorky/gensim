Bz2file is a Python library for reading and writing bzip2-compressed files.

It contains a drop-in replacement for the file interface in the standard
library's ``bz2`` module, including features from the latest development
version of CPython that are not available in older releases.

Bz2file is compatible with CPython 2.6, 2.7, and 3.0 through 3.4, as well as
PyPy 2.0.


Features
--------

- Supports multi-stream files.

- Can read from or write to any file-like object.

- Can open files in either text or binary mode.

- Added methods: ``peek()``, ``read1()``, ``readinto()``, ``fileno()``,
  ``readable()``, ``writable()``, ``seekable()``.


Installation
------------

To install bz2file, run: ::

   $ pip install bz2file


Documentation
-------------

The ``open()`` function and ``BZ2File`` class in this module provide the same
features and interface as the ones in the standard library's ``bz2`` module in
the current development version of CPython, `documented here
<http://docs.python.org/dev/library/bz2.html>`_.


Version History
---------------

0.98: 19 January 2014

- Added support for the 'x' family of modes.
- Ignore non-bz2 data at the end of a file, rather than raising an exception.
- Tests now pass on PyPy.

0.95: 08 October 2012

- Added the ``open()`` function.
- Improved performance when reading in small chunks.
- Removed the ``fileobj`` argument to ``BZ2File()``. To wrap an existing file
  object, pass it as the first argument (``filename``).

0.9: 04 February 2012

- Initial release.


