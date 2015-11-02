#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""


import logging
import unittest
import os
import sys
import random
import itertools

from gensim.models.phrases import Phrases


if sys.version_info[0] >= 3:
    unicode = str

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


WORDS = ['PHRASE%i' % i for i in range(10)]  # selected words for phrases


class TestPhrasesModel(unittest.TestCase):

    @staticmethod
    def get_word():
        """Generate random word from letters A-Z."""
        word_len = random.randint(1, 12)
        return ''.join(chr(random.randint(65, 80)) for i in range(word_len))

    @staticmethod
    def get_sentence(size=10000):
        """Generator for random sentences.
        10% probability to return sentence containing only preselected words"""
        for i in range(size):
            if random.random() > 0.9:
                yield [WORDS[random.randint(0, len(WORDS) -1)] for i in range(random.randint(2, 10))] + ["."]
            else:
                yield [TestPhrasesModel.get_word() for i in range(random.randint(2, 10))] + ["."]

    def testUpdate(self):
        """Test adding one token.
        """
        special_token = 'non_present_token'
        phrases = Phrases(TestPhrasesModel.get_sentence(), min_count=1)

        present = special_token in phrases.vocab
        freq = phrases.vocab[special_token]

        phrases.add_vocab([[special_token]])

        freq_after_change = phrases.vocab[special_token]
        present_after_change = special_token in phrases.vocab

        self.assertEqual(present, False, msg="Non-present token is marked as present.")
        self.assertEqual(present_after_change, True, msg="Present token is marked as non-present.")
        self.assertEqual(freq, 0, msg="Predicted non-zero freq for non-present token.")
        self.assertEqual(freq_after_change, 1, msg="Predicted non 1 freq for token inserted once.")

    def testFreqCount(self):
        """Test adding one token.
        """
        special_token = 'non_present_token'
        phrases = Phrases(None, min_count=1)

        current = iter([])
        for i in range(100):
            current = itertools.chain(current, iter([[special_token]]), TestPhrasesModel.get_sentence(i))
        phrases.add_vocab(current)

        freq = phrases.vocab[special_token]
        self.assertGreaterEqual(freq, 100)

        current = iter([])
        for i in range(100):
            current = itertools.chain(current, iter([[special_token]]), TestPhrasesModel.get_sentence(i))
        phrases.add_vocab(current)

        freq = phrases.vocab[special_token]
        self.assertGreaterEqual(freq, 200)


#endclass TestPhrasesModel


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
