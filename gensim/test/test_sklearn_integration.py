#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Tests for sklearn_integration interface
"""


import logging
import unittest
import os
import os.path
import numpy as np

from gensim.sklearn_integration import base


class TestSklearn(unittest.TestCase):
    """
    write test script
    """
    #for now
    def testRun(self):
        model=base.BaseClass()
        self.assertTrue(np.array_equal(model.run(),np.array([0,0,0])))

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
