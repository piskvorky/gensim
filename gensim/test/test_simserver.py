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

import Pyro4

import gensim


def mock_documents(language, category):
    """Create a few SimilarityDocuments, for testing."""
    documents = ["Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"]

    # create SimilarityDocument-like objects from the texts
    # these are the object that the gensim server expects as input, and translate
    # directly into the java's SimilarityDocument class.
    docs = [{'id': '_'.join((language, category, str(num))),
             'text': document, 'values': range(num), 'language': language, 'category': category}
            for num, document in enumerate(documents)]
    return docs


class SimServerTester(unittest.TestCase):
    """Test a running SimServer"""
    def setUp(self):
        self.docs = mock_documents('en', '')
        try:
            with Pyro4.locateNS() as ns:
                self.server = Pyro4.Proxy(ns.lookup('gensim.testserver'))
        except Pyro4.errors.PyroError, e:
            logging.error("could not locate running SimServer: %s" % e)
            raise

    def tearDown(self):
        self.server


    def test_model(self):
        self.server.add_documents(self.docs[:2])
        self.server.add_documents(self.docs[2:])
        self.server.train()
        logging.debug(self.server.status())


    def test_index(self):
        self.server.drop_index()
        self.server.train(self.docs)
        self.server.add_documents(self.docs[:3])
        self.server.index()
        self.server.index(self.docs[3:])
        logging.debug(self.server.status())




if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
