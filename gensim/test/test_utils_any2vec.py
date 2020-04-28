#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking utils_any2vec functionality.
"""

import logging
import unittest

import numpy as np

import gensim.utils
import gensim.test.utils

import gensim.models.utils_any2vec


logger = logging.getLogger(__name__)


def save_dict_to_word2vec_formated_file(fname, word2vec_dict):

    with gensim.utils.open(fname, "wb") as f:

        num_words = len(word2vec_dict)
        vector_length = len(list(word2vec_dict.values())[0])

        header = "%d %d\n" % (num_words, vector_length)
        f.write(header.encode(encoding="ascii"))

        for word, vector in word2vec_dict.items():
            f.write(word.encode())
            f.write(' '.encode())
            f.write(np.array(vector).astype(np.float32).tobytes())


class LoadWord2VecFormatTest(unittest.TestCase):

    def assert_dict_equal_to_model(self, d, m):
        self.assertEqual(len(d), len(m.vocab))

        for word in d.keys():
            self.assertSequenceEqual(list(d[word]), list(m[word]))

    def verify_load2vec_binary_result(self, w2v_dict, binary_chunk_size, limit):
        tmpfile = gensim.test.utils.get_tmpfile("tmp_w2v")
        save_dict_to_word2vec_formated_file(tmpfile, w2v_dict)
        w2v_model = \
            gensim.models.utils_any2vec._load_word2vec_format(
                cls=gensim.models.KeyedVectors,
                fname=tmpfile,
                binary=True,
                limit=limit,
                binary_chunk_size=binary_chunk_size)
        if limit is None:
            limit = len(w2v_dict)

        w2v_keys_postprocessed = list(w2v_dict.keys())[:limit]
        w2v_dict_postprocessed = {k.lstrip(): w2v_dict[k] for k in w2v_keys_postprocessed}

        self.assert_dict_equal_to_model(w2v_dict_postprocessed, w2v_model)

    def test_load_word2vec_format_basic(self):
        w2v_dict = {"abc": [1, 2, 3],
                    "cde": [4, 5, 6],
                    "def": [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=None)

        w2v_dict = {"abc": [1, 2, 3],
                    "cdefg": [4, 5, 6],
                    "d": [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=None)

    def test_load_word2vec_format_limit(self):
        w2v_dict = {"abc": [1, 2, 3],
                    "cde": [4, 5, 6],
                    "def": [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=1)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=1)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=1)

        w2v_dict = {"abc": [1, 2, 3],
                    "cde": [4, 5, 6],
                    "def": [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=2)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=2)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=2)

        w2v_dict = {"abc": [1, 2, 3],
                    "cdefg": [4, 5, 6],
                    "d": [7, 8, 9]}

        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=1)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=1)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=1)

        w2v_dict = {"abc": [1, 2, 3],
                    "cdefg": [4, 5, 6],
                    "d": [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=2)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=16, limit=2)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=1024, limit=2)

    def test_load_word2vec_format_space_stripping(self):
        w2v_dict = {"\nabc": [1, 2, 3],
                    "cdefdg": [4, 5, 6],
                    "\n\ndef": [7, 8, 9]}
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=None)
        self.verify_load2vec_binary_result(w2v_dict, binary_chunk_size=5, limit=1)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
