#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking similarity server. Assumes run_simserver.py is already running.
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
    """Create a few SimServer documents, for testing."""
    documents = ["Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"]

    # Create SimServer dicts from the texts. These are the object that the gensim
    # server expects as input. They must contain doc['id'] and doc['text'] attributes.
    docs = [{'id': '_'.join((language, category, str(num))),
             'text': document, 'values': range(num), 'language': language, 'category': category}
            for num, document in enumerate(documents)]
    return docs


class SimServerTester(unittest.TestCase):
    """Test a running SimServer"""
    def setUp(self):
        self.docs = mock_documents('en', '')
        # locate the running server on the network, using a Pyro nameserver
        try:
            with Pyro4.locateNS() as ns:
                self.server = Pyro4.Proxy(ns.lookup('gensim.testserver'))
        except Pyro4.errors.PyroError, e:
            logging.error("could not locate running SimServer: %s" % e)
            raise

    def tearDown(self):
        self.server


    def test_model(self):
        # First, upload training documents to the server. Tthe documents will be
        # stored server-side in an Sqlite db, not in memory, so the training corpus
        # may be larger than RAM
        self.server.buffer(self.docs[:2]) # upload 2 documents to server
        self.server.buffer(self.docs[2:]) # upload the rest of the documents
        # Train a model
        self.server.train(method='lsi')
        logging.debug(self.server.status())


    def test_index(self):
        # delete any existing model and indexes first
        self.server.drop_index(keep_model=False)
        # the following sends the entire training collection at once -- RAM warning!
        # see server.buffer() example above, or utils.upload_chunked() for memory-friendly uploads
        self.server.train(self.docs)
        logging.debug(self.server.status())

        # use incremental indexing -- start by indexing the first three documents
        self.server.buffer(self.docs[:3]) # upload the documents
        self.server.index() # index uploaded documents & clear upload buffer
        self.server.index(self.docs[3:]) # upload and index the rest -- same RAM warning as above
        logging.debug(self.server.status())

        # re-index documents. just index documents with the same id -- the old document
        # will be replaced by the new one, so that only the latest indexing counts.
        self.server.index(self.docs[2:5])
        logging.debug(self.server.status())

        # delete documents. pass it a collection of ids to be removed from the index
        to_delete = [doc['id'] for doc in self.docs[-3:]] # delete the last 3 documents
        self.server.drop_documents(to_delete)
        logging.debug(self.server.status())


    def test_optimize(self):
        # to speed up queries by id, call server.optimize()
        # it will precompute the most similar documents, for all documents in the index,
        # and store them to Sqlite db for lightning-fast querying.
        self.server.optimize()
        logging.debug(self.server.status())


    def test_query_id(self):
        # index some docs first
        self.test_index()

        # query index by id: return the most similar documents to an already indexed document
        docid = self.docs[0]['id']
        sims = self.server.find_similar(docid)
        logging.debug(sims)

        # same thing, but get only the 5 most similar + ignoring docs with similarity < 0.5
        sims = self.server.find_similar(docid, max_results=10, min_score=0.5)
        logging.debug(sims)


    def test_query_document(self):
        # index some docs first
        self.test_index()

        # query index by arbitrary document: document is converted to vector first
        doc = self.docs[0].copy()
        del doc['id'] # id is irrelevant when querying by fulltext
        sims = self.server.find_similar(doc)
        logging.debug(sims)

        # same thing, but get only the 5 most similar + ignoring docs with similarity < 0.5
        sims = self.server.find_similar(doc, max_results=10, min_score=0.5)
        logging.debug(sims)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
