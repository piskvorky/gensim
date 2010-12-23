#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for the dictionary class in corpora
"""


import logging
import unittest
import os
import os.path
import tempfile

import numpy

from gensim.corpora import dictionary


# sample data files are located in the same folder
module_path = os.path.dirname(__file__)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.WARNING)

class TestDictionary(unittest.TestCase):
    def setUp(self):
        self.texts = [
                ['human', 'interface', 'computer'],
                ['survey', 'user', 'computer', 'system', 'response', 'time'],
                ['eps', 'user', 'interface', 'system'],
                ['system', 'human', 'system', 'eps'],
                ['user', 'response', 'time'],
                ['trees'],
                ['graph', 'trees'],
                ['graph', 'minors', 'trees'],
                ['graph', 'minors', 'survey']]

    def testBuild(self):
        d = dictionary.Dictionary(self.texts)
        expected = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 2, 7: 3, 8: 2,
                9: 3, 10: 3, 11: 2}
        self.assertEqual(d.docFreq, expected)
        expected = {'computer': 0, 'eps': 8, 'graph': 10, 'human': 1,
                'interface': 2, 'minors': 11, 'response': 3, 'survey': 4,
                'system': 5, 'time': 6, 'trees': 9, 'user': 7}
        self.assertEqual(d.token2id, expected)
        expected = dict((v, k) for k, v in expected.iteritems())
        self.assertEqual(d.id2token, expected)

    def testFilter(self):
        d = dictionary.Dictionary(self.texts)
        d.filterExtremes(noBelow=2, noAbove=1.0, keepN=4)
        expected = {0: 3, 1: 3, 2: 3, 3: 3}
        self.assertEqual(d.docFreq, expected)
#endclass TestDictionary


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
