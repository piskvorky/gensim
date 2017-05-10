#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking various utils functions.
"""


import logging
import unittest
import subprocess

from gensim import utils
from six import iteritems

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
        self.assertEqual(utils.decode_htmlentities(body), expected)

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

class TestCheckOutput(unittest.TestCase):
    def test_check_output(self):
        res = utils.check_output(args=["echo", "hello"])
        self.assertEqual(res, b'hello\n')

    def test_check_output_exception(self):
        self.assertRaises(subprocess.CalledProcessError, lambda : utils.check_output(args=["ldfs"]))



if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
