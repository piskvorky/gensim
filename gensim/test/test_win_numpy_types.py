#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate Azure build Windows failure
"""

import logging
import unittest
import numpy as np
from gensim.models import KeyedVectors
import struct


class TestWinNumpying(unittest.TestCase):
    def testTiny(self):
        a = np.empty(1, dtype=type(2**32))
        a[0] = 2**32

    def oneliner0(self):
        np.array([2**32])

    def oneliner1(self):
        np.array([2**32], dtype=type(2**32))

    def testAscending(self):
        print("platform: %i" % (struct.calcsize("P") * 8))
        print(type(0))
        print(type(2**32))
        kv = KeyedVectors(10, 3)
        kv.add_one('zero', np.arange(10))
        kv.add_one('one', np.arange(10))
        kv.add_one('tons', np.arange(10))
        kv.set_vecattr('zero', 'probe_int', 0)
        kv.set_vecattr('one', 'probe_int', 1)
        print(kv.expandos['probe_int'].dtype)
        print(kv.expandos['probe_int'])
        kv.set_vecattr('tons', 'probe_int', 2**32)
        print(kv.expandos['probe_int'])

    def testDescending(self):
        print("platform: %i" % (struct.calcsize("P") * 8))
        print(type(0))
        print(type(2**32))
        kv = KeyedVectors(10, 3)
        kv.add_one('zero', np.arange(10))
        kv.add_one('one', np.arange(10))
        kv.add_one('tons', np.arange(10))
        kv.set_vecattr('tons', 'probe_int', 2**32)
        kv.set_vecattr('one', 'probe_int', 1)
        kv.set_vecattr('zero', 'probe_int', 0)
        print(kv.expandos['probe_int'].dtype)
        print(kv.expandos['probe_int'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
