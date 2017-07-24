#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking various utils functions.
"""


import logging
import unittest
import tempfile
import os
from gensim import utils
from six import iteritems
import numpy as np

try:
    FileNotFoundError
    PermissionError
except NameError:
    FileNotFoundError = IOError
    PermissionError = OSError


class TestIsCorpus(unittest.TestCase):
    def test_None(self):
        # test None
        result = utils.is_corpus(None)
        expected = (False, None)
        self.assertEqual(expected, result)

    def test_simple_lists_of_tuples(self):
        # test list words

        # one document, one word
        potentialCorpus = [[(0, 4.)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)

        # one document, several words
        potentialCorpus = [[(0, 4.), (1, 2.)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)

        potentialCorpus = [[(0, 4.), (1, 2.), (2, 5.), (3, 8.)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)

        # several documents, one word
        potentialCorpus = [[(0, 4.)], [(1, 2.)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)

        potentialCorpus = [[(0, 4.)], [(1, 2.)], [(2, 5.)], [(3, 8.)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)

    def test_int_tuples(self):
        potentialCorpus = [[(0, 4)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)

    def test_invalid_formats(self):
        # test invalid formats
        # these are no corpus, because they do not consists of 2-tuples with
        # the form(int, float).
        potentials = list()
        potentials.append(["human"])
        potentials.append("human")
        potentials.append(["human", "star"])
        potentials.append([1, 2, 3, 4, 5, 5])
        potentials.append([[(0, 'string')]])
        for noCorpus in potentials:
            result = utils.is_corpus(noCorpus)
            expected = (False, noCorpus)
            self.assertEqual(expected, result)


class TestUtils(unittest.TestCase):
    def test_decode_entities(self):
        # create a string that fails to decode with unichr on narrow python builds
        body = u'It&#146;s the Year of the Horse. YES VIN DIESEL &#128588; &#128175;'
        expected = u'It\x92s the Year of the Horse. YES VIN DIESEL \U0001f64c \U0001f4af'
        self.assertEquals(utils.decode_htmlentities(body), expected)

    def test_check_output(self):
        if os.name == 'posix':
            self.assertTrue(utils.check_output(args=['/bin/sh', '-c', 'echo', '0']))
            self.assertRaises(FileNotFoundError, utils.check_output, args=['nonexistentFile'])

class TestSampleDict(unittest.TestCase):
    def test_sample_dict(self):
        d = {1:2,2:3,3:4,4:5}
        expected_dict = [(1,2),(2,3)]
        expected_dict_random = [(k,v) for k,v in iteritems(d)]
        sampled_dict = utils.sample_dict(d,2,False)
        self.assertEqual(sampled_dict,expected_dict)
        sampled_dict_random = utils.sample_dict(d,2)
        if sampled_dict_random in expected_dict_random:
            self.assertTrue(True)


class TestWindowing(unittest.TestCase):

    arr10_5 = np.array([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9]
    ])

    def _assert_arrays_equal(self, expected, actual):
        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue((actual == expected).all())

    def test_strided_windows1(self):
        out = utils.strided_windows(range(5), 2)
        expected = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4]
        ])
        self._assert_arrays_equal(expected, out)

    def test_strided_windows2(self):
        input_arr = np.arange(10)
        out = utils.strided_windows(input_arr, 5)
        expected = self.arr10_5.copy()
        self._assert_arrays_equal(expected, out)
        out[0, 0] = 10
        self.assertEqual(10, input_arr[0], "should make view rather than copy")

    def test_strided_windows_window_size_exceeds_size(self):
        input_arr = np.array(['this', 'is', 'test'], dtype='object')
        out = utils.strided_windows(input_arr, 4)
        expected = np.ndarray((0, 0))
        self._assert_arrays_equal(expected, out)

    def test_strided_windows_window_size_equals_size(self):
        input_arr = np.array(['this', 'is', 'test'], dtype='object')
        out = utils.strided_windows(input_arr, 3)
        expected = np.array([input_arr.copy()])
        self._assert_arrays_equal(expected, out)

    def test_iter_windows_include_below_window_size(self):
        texts = [['this', 'is', 'a'], ['test', 'document']]
        out = utils.iter_windows(texts, 3, ignore_below_size=False)
        windows = [list(w) for w in out]
        self.assertEqual(texts, windows)

        out = utils.iter_windows(texts, 3)
        windows = [list(w) for w in out]
        self.assertEqual([texts[0]], windows)

    def test_iter_windows_list_texts(self):
        texts = [['this', 'is', 'a'], ['test', 'document']]
        windows = list(utils.iter_windows(texts, 2))
        list_windows = [list(iterable) for iterable in windows]
        expected = [['this', 'is'], ['is', 'a'], ['test', 'document']]
        self.assertListEqual(list_windows, expected)

    def test_iter_windows_uses_views(self):
        texts = [np.array(['this', 'is', 'a'], dtype='object'), ['test', 'document']]
        windows = list(utils.iter_windows(texts, 2))
        list_windows = [list(iterable) for iterable in windows]
        expected = [['this', 'is'], ['is', 'a'], ['test', 'document']]
        self.assertListEqual(list_windows, expected)
        windows[0][0] = 'modified'
        self.assertEqual('modified', texts[0][0])

    def test_iter_windows_with_copy(self):
        texts = [
            np.array(['this', 'is', 'a'], dtype='object'),
            np.array(['test', 'document'], dtype='object')
        ]
        windows = list(utils.iter_windows(texts, 2, copy=True))

        windows[0][0] = 'modified'
        self.assertEqual('this', texts[0][0])

        windows[2][0] = 'modified'
        self.assertEqual('test', texts[1][0])


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
