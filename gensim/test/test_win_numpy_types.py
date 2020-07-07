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
    def testStuff(self):
        print("platform: %i" % (struct.calcsize("P") * 8))
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
