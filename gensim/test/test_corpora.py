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
import itertools
import logging
import os.path
import tempfile
import unittest

import numpy as np

from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
                            ucicorpus, malletcorpus, textcorpus, indexedcorpus, wikicorpus)
from gensim.interfaces import TransformedCorpus
from gensim.utils import to_unicode
from gensim.test.utils import datapath, get_tmpfile, common_corpus


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


class TestMmCorpusWithIndex(CorpusTestCase):
    def setUp(self):
        self.corpus_class = mmcorpus.MmCorpus
        self.corpus = self.corpus_class(datapath('test_mmcorpus_with_index.mm'))
        self.file_extension = '.mm'

    def test_serialize_compressed(self):
        # MmCorpus needs file write with seek => doesn't support compressed output (only input)
        pass

    def test_closed_file_object(self):
        file_obj = open(datapath('testcorpus.mm'))
        f = file_obj.closed
        mmcorpus.MmCorpus(file_obj)
        s = file_obj.closed
        self.assertEqual(f, 0)
        self.assertEqual(s, 0)

    def test_load(self):
        self.assertEqual(self.corpus.num_docs, 9)
        self.assertEqual(self.corpus.num_terms, 12)
        self.assertEqual(self.corpus.num_nnz, 28)

        # confirm we can iterate and that document values match expected for first three docs
        it = iter(self.corpus)
        self.assertEqual(next(it), [(0, 1.0), (1, 1.0), (2, 1.0)])
        self.assertEqual(next(it), [(0, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0)])
        self.assertEqual(next(it), [(2, 1.0), (5, 1.0), (7, 1.0), (8, 1.0)])

        # confirm that accessing document by index works
        self.assertEqual(self.corpus[3], [(1, 1.0), (5, 2.0), (8, 1.0)])
        self.assertEqual(tuple(self.corpus.index), (97, 121, 169, 201, 225, 249, 258, 276, 303))


class TestMmCorpusNoIndex(CorpusTestCase):
    def setUp(self):
        self.corpus_class = mmcorpus.MmCorpus
        self.corpus = self.corpus_class(datapath('test_mmcorpus_no_index.mm'))
        self.file_extension = '.mm'

    def test_serialize_compressed(self):
        # MmCorpus needs file write with seek => doesn't support compressed output (only input)
        pass

    def test_load(self):
        self.assertEqual(self.corpus.num_docs, 9)
        self.assertEqual(self.corpus.num_terms, 12)
        self.assertEqual(self.corpus.num_nnz, 28)

        # confirm we can iterate and that document values match expected for first three docs
        it = iter(self.corpus)
        self.assertEqual(next(it), [(0, 1.0), (1, 1.0), (2, 1.0)])
        self.assertEqual(next(it), [])
        self.assertEqual(next(it), [(2, 0.42371910849), (5, 0.6625174), (7, 1.0), (8, 1.0)])

        # confirm that accessing document by index fails
        self.assertRaises(RuntimeError, lambda: self.corpus[3])


class TestMmCorpusNoIndexGzip(CorpusTestCase):
    def setUp(self):
        self.corpus_class = mmcorpus.MmCorpus
        self.corpus = self.corpus_class(datapath('test_mmcorpus_no_index.mm.gz'))
        self.file_extension = '.mm'

    def test_serialize_compressed(self):
        # MmCorpus needs file write with seek => doesn't support compressed output (only input)
        pass

    def test_load(self):
        self.assertEqual(self.corpus.num_docs, 9)
        self.assertEqual(self.corpus.num_terms, 12)
        self.assertEqual(self.corpus.num_nnz, 28)

        # confirm we can iterate and that document values match expected for first three docs
        it = iter(self.corpus)
        self.assertEqual(next(it), [(0, 1.0), (1, 1.0), (2, 1.0)])
        self.assertEqual(next(it), [])
        self.assertEqual(next(it), [(2, 0.42371910849), (5, 0.6625174), (7, 1.0), (8, 1.0)])

        # confirm that accessing document by index fails
        self.assertRaises(RuntimeError, lambda: self.corpus[3])


class TestMmCorpusNoIndexBzip(CorpusTestCase):
    def setUp(self):
        self.corpus_class = mmcorpus.MmCorpus
        self.corpus = self.corpus_class(datapath('test_mmcorpus_no_index.mm.bz2'))
        self.file_extension = '.mm'

    def test_serialize_compressed(self):
        # MmCorpus needs file write with seek => doesn't support compressed output (only input)
        pass

    def test_load(self):
        self.assertEqual(self.corpus.num_docs, 9)
        self.assertEqual(self.corpus.num_terms, 12)
        self.assertEqual(self.corpus.num_nnz, 28)

        # confirm we can iterate and that document values match expected for first three docs
        it = iter(self.corpus)
        self.assertEqual(next(it), [(0, 1.0), (1, 1.0), (2, 1.0)])
        self.assertEqual(next(it), [])
        self.assertEqual(next(it), [(2, 0.42371910849), (5, 0.6625174), (7, 1.0), (8, 1.0)])

        # confirm that accessing document by index fails
        self.assertRaises(RuntimeError, lambda: self.corpus[3])


class TestMmCorpusCorrupt(CorpusTestCase):
    def setUp(self):
        self.corpus_class = mmcorpus.MmCorpus
        self.corpus = self.corpus_class(datapath('test_mmcorpus_corrupt.mm'))
        self.file_extension = '.mm'

    def test_serialize_compressed(self):
        # MmCorpus needs file write with seek => doesn't support compressed output (only input)
        pass

    def test_load(self):
        self.assertRaises(ValueError, lambda: [doc for doc in self.corpus])


class TestMmCorpusOverflow(CorpusTestCase):
    """
    Test to make sure cython mmreader doesn't overflow on large number of docs or terms

    """
    def setUp(self):
        self.corpus_class = mmcorpus.MmCorpus
        self.corpus = self.corpus_class(datapath('test_mmcorpus_overflow.mm'))
        self.file_extension = '.mm'

    def test_serialize_compressed(self):
        # MmCorpus needs file write with seek => doesn't support compressed output (only input)
        pass

    def test_load(self):
        self.assertEqual(self.corpus.num_docs, 44270060)
        self.assertEqual(self.corpus.num_terms, 500)
        self.assertEqual(self.corpus.num_nnz, 22134988630)

        # confirm we can iterate and that document values match expected for first three docs
        it = iter(self.corpus)
        self.assertEqual(next(it)[:3], [(0, 0.3913027376444812),
                                        (1, -0.07658791716226626),
                                        (2, -0.020870794080588395)])
        self.assertEqual(next(it), [])
        self.assertEqual(next(it), [])

        # confirm count of terms
        count = 0
        for doc in self.corpus:
            for term in doc:
                count += 1

        self.assertEqual(count, 12)

        # confirm that accessing document by index fails
        self.assertRaises(RuntimeError, lambda: self.corpus[3])


class TestSvmLightCorpus(CorpusTestCase):
    def setUp(self):
        self.corpus_class = svmlightcorpus.SvmLightCorpus
        self.file_extension = '.svmlight'

    def test_serialization(self):
        path = get_tmpfile("svml.corpus")
        labels = [1] * len(common_corpus)
        second_corpus = [(0, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0)]
        self.corpus_class.serialize(path, common_corpus, labels=labels)
        serialized_corpus = self.corpus_class(path)
        self.assertEqual(serialized_corpus[1], second_corpus)
        self.corpus_class.serialize(path, common_corpus, labels=np.array(labels))
        serialized_corpus = self.corpus_class(path)
        self.assertEqual(serialized_corpus[1], second_corpus)


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
    CORPUS_LINE = 'mom  wash  window window was washed'

    def setUp(self):
        self.corpus_class = lowcorpus.LowCorpus
        self.file_extension = '.low'

    def test_line2doc(self):
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        id2word = {1: 'mom', 2: 'window'}

        corpus = self.corpus_class(fname, id2word=id2word)

        # should return all words in doc
        corpus.use_wordids = False
        self.assertEqual(
            sorted(corpus.line2doc(self.CORPUS_LINE)),
            [('mom', 1), ('was', 1), ('wash', 1), ('washed', 1), ('window', 2)])

        # should return words in word2id
        corpus.use_wordids = True
        self.assertEqual(
            sorted(corpus.line2doc(self.CORPUS_LINE)),
            [(1, 1), (2, 2)])


class TestUciCorpus(CorpusTestCase):
    TEST_CORPUS = [[(1, 1)], [], [(0, 2), (2, 1)], []]

    def setUp(self):
        self.corpus_class = ucicorpus.UciCorpus
        self.file_extension = '.uci'

    def test_serialize_compressed(self):
        # UciCorpus needs file write with seek => doesn't support compressed output (only input)
        pass


class TestMalletCorpus(TestLowCorpus):
    TEST_CORPUS = [[(1, 1)], [], [(0, 2), (2, 1)], []]
    CORPUS_LINE = '#3  lang mom  wash  window window was washed'

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

    def test_line2doc(self):
        # case with metadata=False (by default)
        super(TestMalletCorpus, self).test_line2doc()

        # case with metadata=True
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        id2word = {1: 'mom', 2: 'window'}

        corpus = self.corpus_class(fname, id2word=id2word, metadata=True)

        # should return all words in doc
        corpus.use_wordids = False
        doc, (docid, doclang) = corpus.line2doc(self.CORPUS_LINE)
        self.assertEqual(docid, '#3')
        self.assertEqual(doclang, 'lang')
        self.assertEqual(
            sorted(doc),
            [('mom', 1), ('was', 1), ('wash', 1), ('washed', 1), ('window', 2)])

        # should return words in word2id
        corpus.use_wordids = True
        doc, (docid, doclang) = corpus.line2doc(self.CORPUS_LINE)

        self.assertEqual(docid, '#3')
        self.assertEqual(doclang, 'lang')
        self.assertEqual(
            sorted(doc),
            [(1, 1), (2, 2)])


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


# Needed for the test_custom_tokenizer is the TestWikiCorpus class.
# Cannot be nested due to serializing.
def custom_tokenizer(content, token_min_len=2, token_max_len=15, lower=True):
    return [
        to_unicode(token.lower()) if lower else to_unicode(token) for token in content.split()
        if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
    ]


class TestWikiCorpus(TestTextCorpus):
    def setUp(self):
        self.corpus_class = wikicorpus.WikiCorpus
        self.file_extension = '.xml.bz2'
        self.fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        self.enwiki = datapath('enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2')

    def test_default_preprocessing(self):
        expected = ['computer', 'human', 'interface']
        corpus = self.corpus_class(self.fname, article_min_tokens=0)
        first_text = next(corpus.get_texts())
        self.assertEqual(expected, first_text)

    def test_len(self):
        # When there is no min_token limit all 9 articles must be registered.
        corpus = self.corpus_class(self.fname, article_min_tokens=0)
        all_articles = corpus.get_texts()
        assert (len(list(all_articles)) == 9)

        # With a huge min_token limit, all articles should be filtered out.
        corpus = self.corpus_class(self.fname, article_min_tokens=100000)
        all_articles = corpus.get_texts()
        assert (len(list(all_articles)) == 0)

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

    def test_first_element(self):
        """
        First two articles in this sample are
        1) anarchism
        2) autism
        """
        corpus = self.corpus_class(self.enwiki, processes=1)

        texts = corpus.get_texts()
        self.assertTrue(u'anarchism' in next(texts))
        self.assertTrue(u'autism' in next(texts))

    def test_unicode_element(self):
        """
        First unicode article in this sample is
        1) папа
        """
        bgwiki = datapath('bgwiki-latest-pages-articles-shortened.xml.bz2')
        corpus = self.corpus_class(bgwiki)
        texts = corpus.get_texts()
        self.assertTrue(u'папа' in next(texts))

    def test_custom_tokenizer(self):
        """
        define a custom tokenizer function and use it
        """
        wc = self.corpus_class(self.enwiki, processes=1, lemmatize=False, tokenizer_func=custom_tokenizer,
                        token_max_len=16, token_min_len=1, lower=False)
        row = wc.get_texts()
        list_tokens = next(row)
        self.assertTrue(u'Anarchism' in list_tokens)
        self.assertTrue(u'collectivization' in list_tokens)
        self.assertTrue(u'a' in list_tokens)
        self.assertTrue(u'i.e.' in list_tokens)

    def test_lower_case_set_true(self):
        """
        Set the parameter lower to True and check that upper case 'Anarchism' token doesnt exist
        """
        corpus = self.corpus_class(self.enwiki, processes=1, lower=True, lemmatize=False)
        row = corpus.get_texts()
        list_tokens = next(row)
        self.assertTrue(u'Anarchism' not in list_tokens)
        self.assertTrue(u'anarchism' in list_tokens)

    def test_lower_case_set_false(self):
        """
        Set the parameter lower to False and check that upper case Anarchism' token exists
        """
        corpus = self.corpus_class(self.enwiki, processes=1, lower=False, lemmatize=False)
        row = corpus.get_texts()
        list_tokens = next(row)
        self.assertTrue(u'Anarchism' in list_tokens)
        self.assertTrue(u'anarchism' in list_tokens)

    def test_min_token_len_not_set(self):
        """
        Don't set the parameter token_min_len and check that 'a' as a token doesn't exist
        Default token_min_len=2
        """
        corpus = self.corpus_class(self.enwiki, processes=1, lemmatize=False)
        self.assertTrue(u'a' not in next(corpus.get_texts()))

    def test_min_token_len_set(self):
        """
        Set the parameter token_min_len to 1 and check that 'a' as a token exists
        """
        corpus = self.corpus_class(self.enwiki, processes=1, token_min_len=1, lemmatize=False)
        self.assertTrue(u'a' in next(corpus.get_texts()))

    def test_max_token_len_not_set(self):
        """
        Don't set the parameter token_max_len and check that 'collectivisation' as a token doesn't exist
        Default token_max_len=15
        """
        corpus = self.corpus_class(self.enwiki, processes=1, lemmatize=False)
        self.assertTrue(u'collectivization' not in next(corpus.get_texts()))

    def test_max_token_len_set(self):
        """
        Set the parameter token_max_len to 16 and check that 'collectivisation' as a token exists
        """
        corpus = self.corpus_class(self.enwiki, processes=1, token_max_len=16, lemmatize=False)
        self.assertTrue(u'collectivization' in next(corpus.get_texts()))

    def test_removed_table_markup(self):
        """
        Check if all the table markup has been removed.
        """
        enwiki_file = datapath('enwiki-table-markup.xml.bz2')
        corpus = self.corpus_class(enwiki_file)
        texts = corpus.get_texts()
        table_markup = ["style", "class", "border", "cellspacing", "cellpadding", "colspan", "rowspan"]
        for text in texts:
            for word in table_markup:
                self.assertTrue(word not in text)

    # #TODO: sporadic failure to be investigated
    # def test_get_texts_returns_generator_of_lists(self):
    #     corpus = self.corpus_class(self.enwiki)
    #     l = corpus.get_texts()
    #     self.assertEqual(type(l), types.GeneratorType)
    #     first = next(l)
    #     self.assertEqual(type(first), list)
    #     self.assertTrue(isinstance(first[0], bytes) or isinstance(first[0], str))

    def test_sample_text(self):
        # Cannot instantiate WikiCorpus from lines
        pass

    def test_sample_text_length(self):
        # Cannot instantiate WikiCorpus from lines
        pass

    def test_sample_text_seed(self):
        # Cannot instantiate WikiCorpus from lines
        pass

    def test_empty_input(self):
        # An empty file is not legit XML
        pass

    def test_custom_filterfunction(self):
        def reject_all(elem, *args, **kwargs):
            return False
        corpus = self.corpus_class(self.enwiki, filter_articles=reject_all)
        texts = corpus.get_texts()
        self.assertFalse(any(texts))

        def keep_some(elem, title, *args, **kwargs):
            return title[0] == 'C'
        corpus = self.corpus_class(self.enwiki, filter_articles=reject_all)
        corpus.metadata = True
        texts = corpus.get_texts()
        for text, (pageid, title) in texts:
            self.assertEquals(title[0], 'C')


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
        corpus = textcorpus.TextDirectoryCorpus(dirpath, pattern=r"test.*\.log")
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
        base_names = sorted(name[len(dirpath) + 1:] for name in filenames)
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
