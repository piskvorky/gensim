#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyEMD
=====

PyEMD is a Python wrapper for Ofir Pele and Michael Werman's implementation of
the Earth Mover's Distance (http://www.seas.upenn.edu/~ofirpele/FastEMD/code/)
that integrates it with NumPy.

    >>> from pyemd import emd
    >>> import numpy as np
    >>> first_signature = np.array([0.0, 1.0])
    >>> second_signature = np.array([5.0, 3.0])
    >>> distance_matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
    >>> emd(first_signature, second_signature, distance_matrix)
    3.5


Limitations and Caveats:
------------------------

- ``distance_matrix`` must be symmetric.
- ``distance_matrix`` is assumed to represent a true metric. This must be
  enforced by the caller. See the documentation in lib/emd_hat.hpp.
- The signatures and distance matrix must be numpy arrays of ``np.float``. The
  original C++ template function can accept any numerical C++ type, but this
  wrapper only instantiates the template with ``double`` (Cython converts
  ``np.float`` to ``double``). If there's demand, I can add support for other
  types.
- The original C++ functions have optional parameters ``extra_mass_penalty``
  and ``F`` (for flows); this wrapper does not expose those parameters. See
  the documentation in lib/emd_hat.hpp.


Credits
-------

- All credit for the actual algorithm and implementation goes to Ofir Pele and
  Michael Werman. See the `relevant paper
  <http://www.seas.upenn.edu/~ofirpele/publications/ICCV2009.pdf>`.
- Thanks to the Cython devlopers for making this kind of wrapper relatively
  easy to write.


:copyright: (c) 2014 by Will Mayner.
:license: MIT. See the LICENSE file.

"""

__title__ = 'pyemd'
__version__ = '0.2.0'
__author__ = 'Will Mayner'
__author_email__ = 'wmayner@gmail.com'
__author_website__ = 'http://willmayner.com'
__license__ = 'MIT'
__copyright__ = 'Copyright 2014 Will Mayner'

from .emd import emd
