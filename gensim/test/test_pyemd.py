#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import unittest
import logging

from pyemd import emd

class TestPyemd(unittest.TestCase):
    def test_case_1(self):
        first_signature = np.array([0.0, 1.0])
        second_signature = np.array([5.0, 3.0])
        distance_matrix = np.array([[0.0, 0.5],
                                    [0.5, 0.0]])
        self.assertEqual(3.5, emd(first_signature, second_signature, distance_matrix))

    def test_case_2(self):
        first_signature = np.array([1.0, 1.0])
        second_signature = np.array([1.0, 1.0])
        distance_matrix = np.array([[0.0, 1.0],
                                    [1.0, 0.0]])
        self.assertEqual(0.0, emd(first_signature, second_signature, distance_matrix))

    def test_case_3(self):
        first_signature = np.array([6.0, 1.0])
        second_signature = np.array([1.0, 7.0])
        distance_matrix = np.array([[0.0, 0.0],
                                    [0.0, 0.0]])
        self.assertEqual(0.0, emd(first_signature, second_signature, distance_matrix))

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
