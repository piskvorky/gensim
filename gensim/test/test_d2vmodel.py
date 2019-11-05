#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking D2VTransformer class.
"""

import unittest
import logging
from gensim.sklearn_api import D2VTransformer
from gensim.test.utils import common_texts


class IteratorForIterable:
    """Iterator capable of folding into list."""
    def __init__(self, iterable):
        self._data = iterable
        self._index = 0

    def __next__(self):
        if len(self._data) > self._index:
            result = self._data[self._index]
            self._index += 1
            return result
        raise StopIteration


class IterableWithoutZeroElement:
    """
    Iterable, emulating pandas.Series behaviour without 0-th element.
    Equivalent to calling `series.index += 1`.
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        if key == 0:
            raise KeyError("Emulation of absence of item with key 0.")
        return self.data[key]

    def __iter__(self):
        return IteratorForIterable(self.data)


class TestD2VTransformer(unittest.TestCase):
    def TestWorksWithIterableNotHavingElementWithZeroIndex(self):
        a = IterableWithoutZeroElement(common_texts)
        transformer = D2VTransformer(min_count=1, size=5)
        transformer.fit(a)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
