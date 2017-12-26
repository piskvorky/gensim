#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking corpus I/O formats (the corpora package).
"""

from __future__ import unicode_literals

import codecs
import bz2
import itertools
import logging
import os.path
import tempfile
import unittest

import numpy as np
from xml.etree.cElementTree import ParseError

from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
                            ucicorpus, malletcorpus, textcorpus, indexedcorpus, wikicorpus)
from gensim.interfaces import TransformedCorpus
from gensim.utils import to_unicode
from gensim.test.utils import datapath, get_tmpfile


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

    def setUp(self):
        self.corpus_class = None
        self.file_extension = None

    def run(self, result=None):
        if type(self) is not CorpusTestCase:
            super(CorpusTestCase, self).run(result)

    def tearDown(self):
        # remove all temporary test files
        fname = get_tmpfile('gensim_corpus.tst')
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
        tmpf = get_tmpfile('gensim_corpus.tst')
        with open(tmpf, 'w') as f:
            f.write('')

        with open(tmpf + '.vocab', 'w') as f:
            f.write('')

        corpus = self.corpus_class(tmpf)
        self.assertEqual(len(corpus), 0)

        docs = list(corpus)
        self.assertEqual(len(docs), 0)

    def test_save(self):
        corpus = self.TEST_CORPUS
        tmpf = get_tmpfile('gensim_corpus.tst')

        # make sure the corpus can be saved
        self.corpus_class.save_corpus(tmpf, corpus)

        # and loaded back, resulting in exactly the same corpus
        corpus2 = list(self.corpus_class(tmpf))
        self.assertEqual(corpus, corpus2)

    def test_serialize(self):
        corpus = self.TEST_CORPUS
        tmpf = get_tmpfile('gensim_corpus.tst')

        # make sure the corpus can be saved
        self.corpus_class.serialize(tmpf, corpus)

        # and loaded back, resulting in exactly the same corpus
        corpus2 = self.corpus_class(tmpf)
        self.assertEqual(corpus, list(corpus2))

        # make sure the indexing corpus[i] works
        for i in range(len(corpus)):
            self.assertEqual(corpus[i], corpus2[i])

        # make sure that subclasses of IndexedCorpus support fancy indexing
        # after deserialisation
        if isinstance(corpus, indexedcorpus.IndexedCorpus):
            idx = [1, 3, 5, 7]
            self.assertEqual(corpus[idx], corpus2[idx])

    def test_serialize_compressed(self):
        corpus = self.TEST_CORPUS
        tmpf = get_tmpfile('gensim_corpus.tst')

        for extension in ['.gz', '.bz2']:
            fname = tmpf + extension
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

            self.assertEqual(testdoc, {('computer', 1), ('human', 1), ('interface', 1)})

            d = corpus.id2word
            d[0], d[1] = d[1], d[0]
            corpus.id2word = d

            firstdoc2 = next(iter(corpus))
            testdoc2 = set((to_unicode(corpus.id2word[x]), y) for x, y in firstdoc2)
            self.assertEqual(testdoc2, {('computer', 1), ('human', 1), ('interface', 1)})

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
        self.assertRaises(ValueError, _get_slice, corpus, {1})
        self.assertRaises(ValueError, _get_slice, corpus, 1.0)

        # check sliced corpora that use fancy indexing
        c = corpus[[1, 3, 4]]
        self.assertEqual([d for i, d in enumerate(docs) if i in [1, 3, 4]], list(c))
        self.assertEqual([d for i, d in enumerate(docs) if i in [1, 3, 4]], list(c))
        self.assertEqual(len(corpus[[0, 1, -1]]), 3)
        self.assertEqual(len(corpus[np.asarray([0, 1, -1])]), 3)

        # check that TransformedCorpus supports indexing when the underlying
        # corpus does, and throws an error otherwise
        corpus_ = TransformedCorpus(DummyTransformer(), corpus)
        if hasattr(corpus, 'index') and corpus.index is not None:
            self.assertEqual(corpus_[0][0][1], docs[0][0][1] + 1)
            self.assertRaises(ValueError, _get_slice, corpus_, {1})
            transformed_docs = [val + 1 for i, d in enumerate(docs) for _, val in d if i in [1, 3, 4]]
            self.assertEqual(transformed_docs, list(v for doc in corpus_[[1, 3, 4]] for _, v in doc))
            self.assertEqual(3, len(corpus_[[1, 3, 4]]))
        else:
            self.assertRaises(RuntimeError, _get_slice, corpus_, [1, 3, 4])
            self.assertRaises(RuntimeError, _get_slice, corpus_, {1})
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
        test_file = get_tmpfile('gensim_corpus.tst')
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

    def test_default_preprocessing(self):
        lines = [
            "Šéf chomutovských komunistů dostal poštou bílý prášek",
            "this is a test for stopwords",
            "zf tooth   spaces   "
        ]
        expected = [
            ['Sef', 'chomutovskych', 'komunistu', 'dostal', 'postou', 'bily', 'prasek'],
            ['test', 'stopwords'],
            ['tooth', 'spaces']
        ]

        corpus = self.corpus_from_lines(lines)
        texts = list(corpus.get_texts())
        self.assertEqual(expected, texts)

    def corpus_from_lines(self, lines):
        fpath = tempfile.mktemp()
        with codecs.open(fpath, 'w', encoding='utf8') as f:
            f.write('\n'.join(lines))

        return self.corpus_class(fpath)

    def test_sample_text(self):
        lines = ["document%d" % i for i in range(10)]
        corpus = self.corpus_from_lines(lines)
        corpus.tokenizer = lambda text: text.split()
        docs = [doc for doc in corpus.get_texts()]

        sample1 = list(corpus.sample_texts(1))
        self.assertEqual(len(sample1), 1)
        self.assertIn(sample1[0], docs)

        sample2 = list(corpus.sample_texts(len(lines)))
        self.assertEqual(len(sample2), len(corpus))
        for i in range(len(corpus)):
            self.assertEqual(sample2[i], ["document%s" % i])

        with self.assertRaises(ValueError):
            list(corpus.sample_texts(len(corpus) + 1))

        with self.assertRaises(ValueError):
            list(corpus.sample_texts(-1))

    def test_sample_text_length(self):
        lines = ["document%d" % i for i in range(10)]
        corpus = self.corpus_from_lines(lines)
        corpus.tokenizer = lambda text: text.split()

        sample1 = list(corpus.sample_texts(1, length=1))
        self.assertEqual(sample1[0], ["document0"])

        sample2 = list(corpus.sample_texts(2, length=2))
        self.assertEqual(sample2[0], ["document0"])
        self.assertEqual(sample2[1], ["document1"])

    def test_sample_text_seed(self):
        lines = ["document%d" % i for i in range(10)]
        corpus = self.corpus_from_lines(lines)

        sample1 = list(corpus.sample_texts(5, seed=42))
        sample2 = list(corpus.sample_texts(5, seed=42))
        self.assertEqual(sample1, sample2)

    def test_save(self):
        pass

    def test_serialize(self):
        pass

    def test_serialize_compressed(self):
        pass

    def test_indexing(self):
        pass


class TestWikiCorpus(TestTextCorpus):
    def setUp(self):
        self.corpus_class = wikicorpus.WikiCorpus
        self.file_extension = '.xml.bz2'
        self.fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))

    def test_default_preprocessing(self):
        expected = ['computer', 'human', 'interface']
        corpus = self.corpus_class(self.fname, article_min_tokens=0)
        first_text = corpus.get_texts().next()
        self.assertEqual(expected, first_text)

    def test_len(self):

        def test_with_limit(article_min_tokens, expected_articles):
            corpus = self.corpus_class(self.fname, article_min_tokens=article_min_tokens)
            all_articles = corpus.get_texts()
            assert (len(list(all_articles)) == expected_articles)

        test_with_limit(0, 9)
        test_with_limit(100000, 0)

    def test_load_with_metadata(self):
        corpus = self.corpus_class(self.fname, article_min_tokens=0)
        corpus.metadata = True
        self.assertEqual(len(corpus), 9)

        docs = list(corpus)
        self.assertEqual(len(docs), 9)

        for i, docmeta in enumerate(docs):
            doc, metadata = docmeta
            article_no = i + 1  # Counting IDs from 1
            self.assertEqual(metadata[0], str(article_no))
            self.assertEqual(metadata[1], 'Article%d' % article_no)

    def test_load(self):
        corpus = self.corpus_class(self.fname, article_min_tokens=0)

        docs = list(corpus)
        # the deerwester corpus always has nine documents
        self.assertEqual(len(docs), 9)

    def test_empty_input(self):
        tmpf = get_tmpfile('emptycorpus.xml.bz2')
        content = bz2.compress(b'')  # Explicit string to byte conversion needed in python 3
        fh = open(tmpf, "wb")
        fh.write(content)
        fh.close()

        with self.assertRaises(ParseError):
            corpus = self.corpus_class(tmpf)
            del corpus  # Needed to supress tox warning

    def test_sample_text(self):
        # Cannot instantiate WikiCorpus from lines
        pass

    def test_sample_text_length(self):
        # Cannot instantiate WikiCorpus from lines
        pass

    def test_sample_text_seed(self):
        # Cannot instantiate WikiCorpus from lines
        pass


class TestTextDirectoryCorpus(unittest.TestCase):

    def write_one_level(self, *args):
        if not args:
            args = ('doc1', 'doc2')
        dirpath = tempfile.mkdtemp()
        self.write_docs_to_directory(dirpath, *args)
        return dirpath

    def write_docs_to_directory(self, dirpath, *args):
        for doc_num, name in enumerate(args):
            with open(os.path.join(dirpath, name), 'w') as f:
                f.write('document %d content' % doc_num)

    def test_one_level_directory(self):
        dirpath = self.write_one_level()

        corpus = textcorpus.TextDirectoryCorpus(dirpath)
        self.assertEqual(len(corpus), 2)
        docs = list(corpus)
        self.assertEqual(len(docs), 2)

    def write_two_levels(self):
        dirpath = self.write_one_level()
        next_level = os.path.join(dirpath, 'level_two')
        os.mkdir(next_level)
        self.write_docs_to_directory(next_level, 'doc1', 'doc2')
        return dirpath, next_level

    def test_two_level_directory(self):
        dirpath, next_level = self.write_two_levels()

        corpus = textcorpus.TextDirectoryCorpus(dirpath)
        self.assertEqual(len(corpus), 4)
        docs = list(corpus)
        self.assertEqual(len(docs), 4)

        corpus = textcorpus.TextDirectoryCorpus(dirpath, min_depth=1)
        self.assertEqual(len(corpus), 2)
        docs = list(corpus)
        self.assertEqual(len(docs), 2)

        corpus = textcorpus.TextDirectoryCorpus(dirpath, max_depth=0)
        self.assertEqual(len(corpus), 2)
        docs = list(corpus)
        self.assertEqual(len(docs), 2)

    def test_filename_filtering(self):
        dirpath = self.write_one_level('test1.log', 'test1.txt', 'test2.log', 'other1.log')
        corpus = textcorpus.TextDirectoryCorpus(dirpath, pattern="test.*\.log")
        filenames = list(corpus.iter_filepaths())
        expected = [os.path.join(dirpath, name) for name in ('test1.log', 'test2.log')]
        self.assertEqual(sorted(expected), sorted(filenames))

        corpus.pattern = ".*.txt"
        filenames = list(corpus.iter_filepaths())
        expected = [os.path.join(dirpath, 'test1.txt')]
        self.assertEqual(expected, filenames)

        corpus.pattern = None
        corpus.exclude_pattern = ".*.log"
        filenames = list(corpus.iter_filepaths())
        self.assertEqual(expected, filenames)

    def test_lines_are_documents(self):
        dirpath = tempfile.mkdtemp()
        lines = ['doc%d text' % i for i in range(5)]
        fpath = os.path.join(dirpath, 'test_file.txt')
        with open(fpath, 'w') as f:
            f.write('\n'.join(lines))

        corpus = textcorpus.TextDirectoryCorpus(dirpath, lines_are_documents=True)
        docs = [doc for doc in corpus.getstream()]
        self.assertEqual(len(lines), corpus.length)  # should have cached
        self.assertEqual(lines, docs)

        corpus.lines_are_documents = False
        docs = [doc for doc in corpus.getstream()]
        self.assertEqual(1, corpus.length)
        self.assertEqual('\n'.join(lines), docs[0])

    def test_non_trivial_structure(self):
        """Test with non-trivial directory structure, shown below:
        .
        ├── 0.txt
        ├── a_folder
        │   └── 1.txt
        └── b_folder
            ├── 2.txt
            ├── 3.txt
            └── c_folder
                └── 4.txt
        """
        dirpath = tempfile.mkdtemp()
        self.write_docs_to_directory(dirpath, '0.txt')

        a_folder = os.path.join(dirpath, 'a_folder')
        os.mkdir(a_folder)
        self.write_docs_to_directory(a_folder, '1.txt')

        b_folder = os.path.join(dirpath, 'b_folder')
        os.mkdir(b_folder)
        self.write_docs_to_directory(b_folder, '2.txt', '3.txt')

        c_folder = os.path.join(b_folder, 'c_folder')
        os.mkdir(c_folder)
        self.write_docs_to_directory(c_folder, '4.txt')

        corpus = textcorpus.TextDirectoryCorpus(dirpath)
        filenames = list(corpus.iter_filepaths())
        base_names = sorted([name[len(dirpath) + 1:] for name in filenames])
        expected = sorted([
            '0.txt',
            'a_folder/1.txt',
            'b_folder/2.txt',
            'b_folder/3.txt',
            'b_folder/c_folder/4.txt'
        ])
        expected = [os.path.normpath(path) for path in expected]
        self.assertEqual(expected, base_names)

        corpus.max_depth = 1
        self.assertEqual(expected[:-1], base_names[:-1])

        corpus.min_depth = 1
        self.assertEqual(expected[2:-1], base_names[2:-1])

        corpus.max_depth = 0
        self.assertEqual(expected[2:], base_names[2:])

        corpus.pattern = "4.*"
        self.assertEqual(expected[-1], base_names[-1])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
