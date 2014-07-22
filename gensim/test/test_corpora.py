#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking corpus I/O formats (the corpora package).
"""

import logging
import os.path
import unittest
import tempfile
import itertools

from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
                            ucicorpus, malletcorpus, textcorpus)

# needed because sample data files are located in the same folder
module_path = os.path.dirname(__file__)
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_corpus.tst')


class CorpusTestCase(unittest.TestCase):
    TEST_CORPUS = [[(1, 1.0)], [], [(0, 0.5), (2, 1.0)], []]

    def run(self, result=None):
        if type(self) is not CorpusTestCase:
            super(CorpusTestCase, self).run(result)

    def tearDown(self):
        # remove all temporary test files
        fname = testfile()
        extensions = ['', '', '.bz2', '.gz', '.index', '.vocab']
        for ext in itertools.permutations(extensions, 2):
            try:
                os.remove(fname + ext[0] + ext[1])
            except OSError:
                pass

    def test_load(self):
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        corpus = self.corpus_class(fname)
        docs = list(corpus)
        # the deerwester corpus always has nine documents
        self.assertEqual(len(docs), 9)

    def test_empty_input(self):
        with open(testfile(), 'w') as f:
            f.write('')

        with open(testfile() + '.vocab', 'w') as f:
            f.write('')

        corpus = self.corpus_class(testfile())
        self.assertEqual(len(corpus), 0)

        docs = list(corpus)
        self.assertEqual(len(docs), 0)

    def test_save(self):
        corpus = self.TEST_CORPUS

        # make sure the corpus can be saved
        self.corpus_class.save_corpus(testfile(), corpus)

        # and loaded back, resulting in exactly the same corpus
        corpus2 = list(self.corpus_class(testfile()))
        self.assertEqual(corpus, corpus2)

    def test_serialize(self):
        corpus = self.TEST_CORPUS

        # make sure the corpus can be saved
        self.corpus_class.serialize(testfile(), corpus)

        # and loaded back, resulting in exactly the same corpus
        corpus2 = self.corpus_class(testfile())
        self.assertEqual(corpus, list(corpus2))

        # make sure the indexing corpus[i] works
        for i in range(len(corpus)):
            self.assertEqual(corpus[i], corpus2[i])

    def test_serialize_compressed(self):
        corpus = self.TEST_CORPUS

        for extension in ['.gz', '.bz2']:
            fname = testfile() + extension
            # make sure the corpus can be saved
            self.corpus_class.serialize(fname, corpus)

            # and loaded back, resulting in exactly the same corpus
            corpus2 = self.corpus_class(fname)
            self.assertEqual(corpus, list(corpus2))

            # make sure the indexing `corpus[i]` syntax works
            for i in range(len(corpus)):
                self.assertEqual(corpus[i], corpus2[i])

# endclass CorpusTestCase


class TestMmCorpus(CorpusTestCase):
    def setUp(self):
        self.corpus_class = mmcorpus.MmCorpus
        self.file_extension = '.mm'

    def test_serialize_compressed(self):
        # MmCorpus needs file write with seek => doesn't support compressed output (only input)
        pass

# endclass TestMmCorpus


class TestSvmLightCorpus(CorpusTestCase):
    def setUp(self):
        self.corpus_class = svmlightcorpus.SvmLightCorpus
        self.file_extension = '.svmlight'

# endclass TestSvmLightCorpus


class TestBleiCorpus(CorpusTestCase):
    def setUp(self):
        self.corpus_class = bleicorpus.BleiCorpus
        self.file_extension = '.blei'

# endclass TestBleiCorpus


class TestLowCorpus(CorpusTestCase):
    TEST_CORPUS = [[(1, 1)], [], [(0, 2), (2, 1)], []]

    def setUp(self):
        self.corpus_class = lowcorpus.LowCorpus
        self.file_extension = '.low'

# endclass TestLowCorpus


class TestUciCorpus(CorpusTestCase):
    TEST_CORPUS = [[(1, 1)], [], [(0, 2), (2, 1)], []]

    def setUp(self):
        self.corpus_class = ucicorpus.UciCorpus
        self.file_extension = '.uci'

    def test_serialize_compressed(self):
        # UciCorpus needs file write with seek => doesn't support compressed output (only input)
        pass

# endclass TestUciCorpus


class TestMalletCorpus(CorpusTestCase):
    TEST_CORPUS = [[(1, 1)], [], [(0, 2), (2, 1)], []]

    def setUp(self):
        self.corpus_class = malletcorpus.MalletCorpus
        self.file_extension = '.mallet'

    def test_load_with_metadata(self):
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        corpus = self.corpus_class(fname)
        corpus.metadata = True
        self.assertEqual(len(corpus), 9)

        docs = list(corpus)
        self.assertEqual(len(docs), 9)

        for i, docmeta in enumerate(docs):
            doc, metadata = docmeta
            self.assertEqual(metadata[0], str(i + 1))
            self.assertEqual(metadata[1], 'en')

# endclass TestMalletCorpus


class TestTextCorpus(CorpusTestCase):

    def setUp(self):
        self.corpus_class = textcorpus.TextCorpus
        self.file_extension = '.txt'

    def test_load_with_metadata(self):
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        corpus = self.corpus_class(fname)
        corpus.metadata = True
        self.assertEqual(len(corpus), 9)

        docs = list(corpus)
        self.assertEqual(len(docs), 9)

        for i, docmeta in enumerate(docs):
            doc, metadata = docmeta
            self.assertEqual(metadata[0], i)

    def test_save(self):
        pass

    def test_serialize(self):
        pass

    def test_serialize_compressed(self):
        pass

# endclass TestTextCorpus


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
