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

import numpy as np

from gensim.utils import to_unicode
from gensim.interfaces import TransformedCorpus
from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
                            ucicorpus, malletcorpus, textcorpus, indexedcorpus)

# needed because sample data files are located in the same folder
module_path = os.path.dirname(__file__)


def datapath(fname):
    return os.path.join(module_path, 'test_data', fname)


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_corpus.tst')


class DummyTransformer(object):
    def __getitem__(self, bow):
        if len(next(iter(bow))) == 2:
            # single bag of words
            transformed = [(termid, count + 1) for termid, count in bow]
        else:
            # sliced corpus
            transformed = [[(termid, count + 1) for termid, count in doc] for doc in bow]
        return transformed


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

    def test_len(self):
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        corpus = self.corpus_class(fname)

        # make sure corpus.index works, too
        corpus = self.corpus_class(fname)
        self.assertEqual(len(corpus), 9)

        # for subclasses of IndexedCorpus, we need to nuke this so we don't
        # test length on the index, but just testcorpus contents
        if hasattr(corpus, 'index'):
            corpus.index = None

        self.assertEqual(len(corpus), 9)

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

        # make sure that subclasses of IndexedCorpus support fancy indexing
        # after deserialisation
        if isinstance(corpus, indexedcorpus.IndexedCorpus):
            idx = [1, 3, 5, 7]
            self.assertEquals(corpus[idx], corpus2[idx])

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

    def test_switch_id2word(self):
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        corpus = self.corpus_class(fname)
        if hasattr(corpus, 'id2word'):
            firstdoc = next(iter(corpus))
            testdoc = set((to_unicode(corpus.id2word[x]), y) for x, y in firstdoc)

            self.assertEqual(testdoc, set([('computer', 1), ('human', 1), ('interface', 1)]))

            d = corpus.id2word
            d[0], d[1] = d[1], d[0]
            corpus.id2word = d

            firstdoc2 = next(iter(corpus))
            testdoc2 = set((to_unicode(corpus.id2word[x]), y) for x, y in firstdoc2)
            self.assertEqual(testdoc2, set([('computer', 1), ('human', 1), ('interface', 1)]))

    def test_indexing(self):
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        corpus = self.corpus_class(fname)
        docs = list(corpus)

        for idx, doc in enumerate(docs):
            self.assertEqual(doc, corpus[idx])
            self.assertEqual(doc, corpus[np.int64(idx)])

        self.assertEqual(docs, list(corpus[:]))
        self.assertEqual(docs[0:], list(corpus[0:]))
        self.assertEqual(docs[0:-1], list(corpus[0:-1]))
        self.assertEqual(docs[2:4], list(corpus[2:4]))
        self.assertEqual(docs[::2], list(corpus[::2]))
        self.assertEqual(docs[::-1], list(corpus[::-1]))

        # make sure sliced corpora can be iterated over multiple times
        c = corpus[:]
        self.assertEqual(docs, list(c))
        self.assertEqual(docs, list(c))
        self.assertEqual(len(docs), len(corpus))
        self.assertEqual(len(docs), len(corpus[:]))
        self.assertEqual(len(docs[::2]), len(corpus[::2]))

        def _get_slice(corpus, slice_):
            # assertRaises for python 2.6 takes a callable
            return corpus[slice_]

        # make sure proper input validation for sliced corpora is done
        self.assertRaises(ValueError, _get_slice, corpus, set([1]))
        self.assertRaises(ValueError, _get_slice, corpus, 1.0)

        # check sliced corpora that use fancy indexing
        c = corpus[[1, 3, 4]]
        self.assertEquals([d for i, d in enumerate(docs) if i in [1, 3, 4]], list(c))
        self.assertEquals([d for i, d in enumerate(docs) if i in [1, 3, 4]], list(c))
        self.assertEquals(len(corpus[[0, 1, -1]]), 3)
        self.assertEquals(len(corpus[np.asarray([0, 1, -1])]), 3)

        # check that TransformedCorpus supports indexing when the underlying
        # corpus does, and throws an error otherwise
        if hasattr(corpus, 'index') and corpus.index is not None:
            corpus_ = TransformedCorpus(DummyTransformer(), corpus)
            self.assertEqual(corpus_[0][0][1], docs[0][0][1] + 1)
            self.assertRaises(ValueError, _get_slice, corpus_, set([1]))
            transformed_docs = [val + 1 for i, d in enumerate(docs) for _, val in d if i in [1, 3, 4]]
            self.assertEquals(transformed_docs, list(v for doc in corpus_[[1, 3, 4]] for _, v in doc))
            self.assertEqual(3, len(corpus_[[1, 3, 4]]))
        else:
            self.assertRaises(RuntimeError, _get_slice, corpus_, [1, 3, 4])
            self.assertRaises(RuntimeError, _get_slice, corpus_, set([1]))
            self.assertRaises(RuntimeError, _get_slice, corpus_, 1.0)


class TestMmCorpus(CorpusTestCase):
    def setUp(self):
        self.corpus_class = mmcorpus.MmCorpus
        self.corpus = self.corpus_class(datapath('testcorpus.mm'))
        self.file_extension = '.mm'

    def test_serialize_compressed(self):
        # MmCorpus needs file write with seek => doesn't support compressed output (only input)
        pass

    def test_load(self):
        self.assertEqual(self.corpus.num_docs, 9)
        self.assertEqual(self.corpus.num_terms, 12)
        self.assertEqual(self.corpus.num_nnz, 28)
        self.assertEqual(tuple(self.corpus.index), (97, 121, 169, 201, 225, 249, 258, 276, 303))


class TestSvmLightCorpus(CorpusTestCase):
    def setUp(self):
        self.corpus_class = svmlightcorpus.SvmLightCorpus
        self.file_extension = '.svmlight'


class TestBleiCorpus(CorpusTestCase):
    def setUp(self):
        self.corpus_class = bleicorpus.BleiCorpus
        self.file_extension = '.blei'

    def test_save_format_for_dtm(self):
        corpus = [[(1, 1.0)], [], [(0, 5.0), (2, 1.0)], []]
        test_file = testfile()
        self.corpus_class.save_corpus(test_file, corpus)
        with open(test_file) as f:
            for line in f:
                # unique_word_count index1:count1 index2:count2 ... indexn:counnt
                tokens = line.split()
                words_len = int(tokens[0])
                if words_len > 0:
                    tokens = tokens[1:]
                else:
                    tokens = []
                self.assertEqual(words_len, len(tokens))
                for token in tokens:
                    word, count = token.split(':')
                    self.assertEqual(count, str(int(count)))


class TestLowCorpus(CorpusTestCase):
    TEST_CORPUS = [[(1, 1)], [], [(0, 2), (2, 1)], []]

    def setUp(self):
        self.corpus_class = lowcorpus.LowCorpus
        self.file_extension = '.low'


class TestUciCorpus(CorpusTestCase):
    TEST_CORPUS = [[(1, 1)], [], [(0, 2), (2, 1)], []]

    def setUp(self):
        self.corpus_class = ucicorpus.UciCorpus
        self.file_extension = '.uci'

    def test_serialize_compressed(self):
        # UciCorpus needs file write with seek => doesn't support compressed output (only input)
        pass


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

    def test_indexing(self):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
