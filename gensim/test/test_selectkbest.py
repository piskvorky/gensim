#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 04.12.2012

@author: karsten
'''
import unittest

from itertools import izip
import numpy as np
from gensim.models.selectkbest import iSelectKBest, if_classif

class TestiSelectKBest(unittest.TestCase):


    def setUp(self):
        self.X = np.array([[5,   3.5, 8  ],
                           [4.5, 7,   4  ],
                           [4,   2,   3.5],
                           [3,   4.5, 2  ]]) 
            
        self.y = np.array([2, 1, 1, 2])

    def tearDown(self):
        pass
        
    def test_iter(self):
        #Calculate
        X_y = izip(self.X, self.y)

        selector = iSelectKBest(if_classif, k=1)
        selector.fit(X_y, self.X.shape[1])
        
        #Asserts
        np.testing.assert_array_almost_equal([0.05882353, 0.03846154, 0.17241379], 
                                             selector.scores_, 8)
        np.testing.assert_array_almost_equal([0.83096915, 0.86263944, 0.71828192], 
                                             selector.pvalues_, 8)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()