#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking textcorpus I/O formats.
"""

from __future__ import unicode_literals

import codecs
import logging
import os
import tempfile
import unittest

from gensim import utils
from gensim.corpora import textcorpus
from gensim.test.test_corpora import CorpusTestCase, datapath
from gensim.test.test_word2vec import datapath


def split_tokenize(text):
    return text.split()


TEXTS = [
    ['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey']
]


def test_texts_file():
    fpath = os.path.join(tempfile.gettempdir(), 'gensim_corpus.tst')
    with open(fpath, 'w') as f:
        f.write('\n'.join([' '.join(tokens) for tokens in TEXTS]))
    return fpath


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
        corpus.tokenizer = split_tokenize
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
        corpus.tokenizer = split_tokenize

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
        self.assertEqual(expected, filenames)

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
        self.assertEqual('\n'.join(lines), docs[0])

    def test_non_trivial_structure(self):
        """Test with non-trivial directory structure, shown below:
        .
        ├── 0.txt
        ├── a_folder
        │   └── 1.txt
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
        base_names = [name[len(dirpath) + 1:] for name in filenames]
        expected = [
            '0.txt',
            'a_folder/1.txt',
            'b_folder/2.txt',
            'b_folder/3.txt',
            'b_folder/c_folder/4.txt'
        ]
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


class TestLineSentence(unittest.TestCase):
    def testLineSentenceWorksWithFilename(self):
        """Does LineSentence work with a filename argument?"""
        with utils.smart_open(datapath('lee_background.cor')) as orig:
            sentences = textcorpus.LineSentence(datapath('lee_background.cor'))
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig.readline()).split())

    def testLineSentenceWorksWithCompressedFile(self):
        """Does LineSentence work with a compressed file object argument?"""
        with utils.smart_open(datapath('head500.noblanks.cor')) as orig:
            sentences = textcorpus.LineSentence(datapath('head500.noblanks.cor.bz2'))
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig.readline()).split())

    def testLineSentenceWorksWithNormalFile(self):
        """Does LineSentence work with a file object argument, rather than filename?"""
        with utils.smart_open(datapath('head500.noblanks.cor')) as orig:
            with utils.smart_open(datapath('head500.noblanks.cor')) as fin:
                sentences = textcorpus.LineSentence(fin)
                for words in sentences:
                    self.assertEqual(words, utils.to_unicode(orig.readline()).split())

    def testPathLineSentences(self):
        """Does PathLineSentences work with a path argument?"""
        with utils.smart_open(os.path.join(datapath('PathLineSentences'), '1.txt')) as orig1, \
                utils.smart_open(os.path.join(datapath('PathLineSentences'), '2.txt.bz2')) as orig2:
            sentences = textcorpus.PathLineSentences(datapath('PathLineSentences'))
            orig = orig1.readlines() + orig2.readlines()
            orig_counter = 0  # to go through orig while matching PathLineSentences
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig[orig_counter]).split())
                orig_counter += 1

    def testPathLineSentencesOneFile(self):
        """Does PathLineSentences work with a single file argument?"""
        test_file = os.path.join(datapath('PathLineSentences'), '1.txt')
        with utils.smart_open(test_file) as orig:
            sentences = textcorpus.PathLineSentences(test_file)
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig.readline()).split())


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
