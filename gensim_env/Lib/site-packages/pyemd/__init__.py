#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __init__.py

"""
PyEMD
=====

PyEMD is a Python wrapper for `Ofir Pele and Michael Werman's implementation of
the Earth Mover's Distance
<http://www.ariel.ac.il/sites/ofirpele/fastemd/code/>`_ that allows it to be
used with NumPy.

**If you use this code, please cite the papers listed at the end of the
README.**

Use PyEMD like so:

Usage
~~~~~

    >>> from pyemd import emd
    >>> import numpy as np
    >>> first_signature = np.array([0.0, 1.0])
    >>> second_signature = np.array([5.0, 3.0])
    >>> distance_matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
    >>> emd(first_signature, second_signature, distance_matrix)
    3.5

You can also get the associated minimum-cost flow:

    >>> from pyemd import emd_with_flow
    >>> emd_with_flow(first_signature, second_signature, distance_matrix)
    (3.5, [[0.0, 0.0], [0.0, 1.0]])

You can also calculate the EMD directly from two arrays of observations:

    >>> from pyemd import emd_samples
    >>> first_array = [1,2,3,4]
    >>> second_array = [2,3,4,5]
    >>> emd_samples(first_array, second_array, bins=2)
    0.5


Limitations and Caveats
~~~~~~~~~~~~~~~~~~~~~~~

- ``distance_matrix`` must be symmetric.
- ``distance_matrix`` is assumed to represent a true metric. This must be
  enforced by the caller. See the documentation in ``pyemd/lib/emd_hat.hpp``.
- The flow matrix does not contain the flows to/from the extra mass bin.
- The signatures and distance matrix must be numpy arrays of ``np.float``. The
  original C++ template function can accept any numerical C++ type, but this
  wrapper only instantiates the template with ``double`` (Cython converts
  ``np.float`` to ``double``). If there's demand, I can add support for other
  types.


Credit
~~~~~~

- All credit for the actual algorithm and implementation goes to `Ofir Pele
  <http://www.ariel.ac.il/sites/ofirpele/>`_ and `Michael Werman
  <http://www.cs.huji.ac.il/~werman/>`_. See the `relevant paper
  <http://www.seas.upenn.edu/~ofirpele/publications/ICCV2009.pdf>`_.
- Thanks to the Cython devlopers for making this kind of wrapper relatively
  easy to write.


:copyright: Copyright (c) 2014-2018 Will Mayner.
:license: See the LICENSE file.
"""

from .__about__ import *
from .emd import emd, emd_with_flow, emd_samples
