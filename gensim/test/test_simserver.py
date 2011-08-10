#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking similarity server. Assumes simserver.py is already running.
"""

from __future__ import with_statement

import logging
import os
import os.path
import unittest
import tempfile


import gensim


module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
corpus = list(gensim.corpora.MmCorpus(os.path.join(module_path, 'test_data/testcorpus.mm')))


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'simserver')


class SimServerTester(unittest.TestCase):
    """Test a running SimServer"""
    def setUp(self):
        import Pyro4
        try:
            with Pyro4.locateNS() as ns:
                self.server = Pyro4.Proxy(ns.lookup('gensim.simserver'))
        except Pyro4.errors.PyroError, e:
            logging.error("could not locate running SimServer: %s" % e)
            raise

    def tearDown(self):
        self.server


    def test_model(self):
        self.server.add_documents(docs[:2])
        self.server.add_documents(docs[2:])
        self.server.train()
        logging.debug(self.server.status())



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
