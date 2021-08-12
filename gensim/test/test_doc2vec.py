#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""

from __future__ import with_statement, division

import logging
import unittest
import os
from collections import namedtuple

import numpy as np
from testfixtures import log_capture

from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences


class DocsLeeCorpus:
    def __init__(self, string_tags=False, unicode_tags=False):
        self.string_tags = string_tags
        self.unicode_tags = unicode_tags

    def _tag(self, i):
        if self.unicode_tags:
            return u'_\xa1_%d' % i
        elif self.string_tags:
            return '_*%d' % i
        return i

    def __iter__(self):
        with open(datapath('lee_background.cor')) as f:
            for i, line in enumerate(f):
                yield doc2vec.TaggedDocument(utils.simple_preprocess(line), [self._tag(i)])


list_corpus = list(DocsLeeCorpus())


sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(raw_sentences)]


def load_on_instance():
    # Save and load a Doc2Vec Model on instance for test
    tmpf = get_tmpfile('gensim_doc2vec.tst')
    model = doc2vec.Doc2Vec(DocsLeeCorpus(), min_count=1)
    model.save(tmpf)
    model = doc2vec.Doc2Vec()  # should fail at this point
    return model.load(tmpf)


def save_lee_corpus_as_line_sentence(corpus_file):
    utils.save_as_line_sentence((doc.words for doc in DocsLeeCorpus()), corpus_file)


class TestDoc2VecModel(unittest.TestCase):
    def test_persistence(self):
        """Test storing/loading the entire model."""
        tmpf = get_tmpfile('gensim_doc2vec.tst')
        model = doc2vec.Doc2Vec(DocsLeeCorpus(), min_count=1)
        model.save(tmpf)
        self.models_equal(model, doc2vec.Doc2Vec.load(tmpf))

    def test_persistence_fromfile(self):
        """Test storing/loading the entire model."""
        with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
            save_lee_corpus_as_line_sentence(corpus_file)

            tmpf = get_tmpfile('gensim_doc2vec.tst')
            model = doc2vec.Doc2Vec(corpus_file=corpus_file, min_count=1)
            model.save(tmpf)
            self.models_equal(model, doc2vec.Doc2Vec.load(tmpf))

    def test_persistence_word2vec_format(self):
        """Test storing the entire model in word2vec format."""
        model = doc2vec.Doc2Vec(DocsLeeCorpus(), min_count=1)
        # test saving both document and word embedding
        test_doc_word = get_tmpfile('gensim_doc2vec.dw')
        model.save_word2vec_format(test_doc_word, doctag_vec=True, word_vec=True, binary=False)
        binary_model_dv = keyedvectors.KeyedVectors.load_word2vec_format(test_doc_word, binary=False)
        self.assertEqual(len(model.wv) + len(model.dv), len(binary_model_dv))
        # test saving document embedding only
        test_doc = get_tmpfile('gensim_doc2vec.d')
        model.save_word2vec_format(test_doc, doctag_vec=True, word_vec=False, binary=True)
        binary_model_dv = keyedvectors.KeyedVectors.load_word2vec_format(test_doc, binary=True)
        self.assertEqual(len(model.dv), len(binary_model_dv))
        # test saving word embedding only
        test_word = get_tmpfile('gensim_doc2vec.w')
        model.save_word2vec_format(test_word, doctag_vec=False, word_vec=True, binary=True)
        binary_model_dv = keyedvectors.KeyedVectors.load_word2vec_format(test_word, binary=True)
        self.assertEqual(len(model.wv), len(binary_model_dv))

    def obsolete_testLoadOldModel(self):
        """Test loading an old doc2vec model from indeterminate version"""

        model_file = 'doc2vec_old'  # which version?!?
        model = doc2vec.Doc2Vec.load(datapath(model_file))
        self.assertTrue(model.wv.vectors.shape == (3955, 100))
        self.assertTrue(len(model.wv) == 3955)
        self.assertTrue(len(model.wv.index_to_key) == 3955)
        self.assertIsNone(model.corpus_total_words)
        self.assertTrue(model.syn1neg.shape == (len(model.wv), model.vector_size))
        self.assertTrue(model.wv.vectors_lockf.shape == (3955, ))
        self.assertTrue(model.cum_table.shape == (3955, ))

        self.assertTrue(model.dv.vectors.shape == (300, 100))
        self.assertTrue(model.dv.vectors_lockf.shape == (300, ))
        self.assertTrue(len(model.dv) == 300)

        self.model_sanity(model)

    def obsolete_testLoadOldModelSeparates(self):
        """Test loading an old doc2vec model from indeterminate version"""

        # Model stored in multiple files
        model_file = 'doc2vec_old_sep'
        model = doc2vec.Doc2Vec.load(datapath(model_file))
        self.assertTrue(model.wv.vectors.shape == (3955, 100))
        self.assertTrue(len(model.wv) == 3955)
        self.assertTrue(len(model.wv.index_to_key) == 3955)
        self.assertIsNone(model.corpus_total_words)
        self.assertTrue(model.syn1neg.shape == (len(model.wv), model.vector_size))
        self.assertTrue(model.wv.vectors_lockf.shape == (3955, ))
        self.assertTrue(model.cum_table.shape == (3955, ))
        self.assertTrue(model.dv.vectors.shape == (300, 100))
        self.assertTrue(model.dv.vectors_lockf.shape == (300, ))
        self.assertTrue(len(model.dv) == 300)

        self.model_sanity(model)

    def obsolete_test_load_old_models_pre_1_0(self):
        """Test loading pre-1.0 models"""
        model_file = 'd2v-lee-v0.13.0'
        model = doc2vec.Doc2Vec.load(datapath(model_file))
        self.model_sanity(model)

        old_versions = [
            '0.12.0', '0.12.1', '0.12.2', '0.12.3', '0.12.4',
            '0.13.0', '0.13.1', '0.13.2', '0.13.3', '0.13.4',
        ]
        for old_version in old_versions:
            self._check_old_version(old_version)

    def obsolete_test_load_old_models_1_x(self):
        """Test loading 1.x models"""
        old_versions = [
            '1.0.0', '1.0.1',
        ]
        for old_version in old_versions:
            self._check_old_version(old_version)

    def obsolete_test_load_old_models_2_x(self):
        """Test loading 2.x models"""
        old_versions = [
            '2.0.0', '2.1.0', '2.2.0', '2.3.0',
        ]
        for old_version in old_versions:
            self._check_old_version(old_version)

    def obsolete_test_load_old_models_pre_3_3(self):
        """Test loading 3.x models"""
        old_versions = [
            '3.2.0', '3.1.0', '3.0.0'
        ]
        for old_version in old_versions:
            self._check_old_version(old_version)

    def obsolete_test_load_old_models_post_3_2(self):
        """Test loading 3.x models"""
        old_versions = [
            '3.4.0', '3.3.0',
        ]
        for old_version in old_versions:
            self._check_old_version(old_version)

    def _check_old_version(self, old_version):
        logging.info("TESTING LOAD of %s Doc2Vec MODEL", old_version)
        saved_models_dir = datapath('old_d2v_models/d2v_{}.mdl')
        model = doc2vec.Doc2Vec.load(saved_models_dir.format(old_version))
        self.assertTrue(len(model.wv) == 3)
        self.assertIsNone(model.corpus_total_words)
        self.assertTrue(model.wv.vectors.shape == (3, 4))
        self.assertTrue(model.dv.vectors.shape == (2, 4))
        self.assertTrue(len(model.dv) == 2)
        # check if inferring vectors for new documents and similarity search works.
        doc0_inferred = model.infer_vector(list(DocsLeeCorpus())[0].words)
        sims_to_infer = model.dv.most_similar([doc0_inferred], topn=len(model.dv))
        self.assertTrue(sims_to_infer)
        # check if inferring vectors and similarity search works after saving and loading back the model
        tmpf = get_tmpfile('gensim_doc2vec.tst')
        model.save(tmpf)
        loaded_model = doc2vec.Doc2Vec.load(tmpf)
        doc0_inferred = loaded_model.infer_vector(list(DocsLeeCorpus())[0].words)
        sims_to_infer = loaded_model.dv.most_similar([doc0_inferred], topn=len(loaded_model.dv))
        self.assertTrue(sims_to_infer)

    def test_doc2vec_train_parameters(self):

        model = doc2vec.Doc2Vec(vector_size=50)
        model.build_vocab(corpus_iterable=list_corpus)

        self.assertRaises(TypeError, model.train, corpus_file=11111)
        self.assertRaises(TypeError, model.train, corpus_iterable=11111)
        self.assertRaises(TypeError, model.train, corpus_iterable=sentences, corpus_file='test')
        self.assertRaises(TypeError, model.train, corpus_iterable=None, corpus_file=None)
        self.assertRaises(TypeError, model.train, corpus_file=sentences)

    @unittest.skipIf(os.name == 'nt', "See another test for Windows below")
    def test_get_offsets_and_start_doctags(self):
        # Each line takes 6 bytes (including '\n' character)
        lines = ['line1\n', 'line2\n', 'line3\n', 'line4\n', 'line5\n']
        tmpf = get_tmpfile('gensim_doc2vec.tst')

        with utils.open(tmpf, 'wb', encoding='utf8') as fout:
            for line in lines:
                fout.write(utils.any2unicode(line))

        offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 1)
        self.assertEqual(offsets, [0])
        self.assertEqual(start_doctags, [0])

        offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 2)
        self.assertEqual(offsets, [0, 12])
        self.assertEqual(start_doctags, [0, 2])

        offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 3)
        self.assertEqual(offsets, [0, 6, 18])
        self.assertEqual(start_doctags, [0, 1, 3])

        offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 4)
        self.assertEqual(offsets, [0, 6, 12, 18])
        self.assertEqual(start_doctags, [0, 1, 2, 3])

        offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 5)
        self.assertEqual(offsets, [0, 6, 12, 18, 24])
        self.assertEqual(start_doctags, [0, 1, 2, 3, 4])

        offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 6)
        self.assertEqual(offsets, [0, 0, 6, 12, 18, 24])
        self.assertEqual(start_doctags, [0, 0, 1, 2, 3, 4])

    @unittest.skipIf(os.name != 'nt', "See another test for posix above")
    def test_get_offsets_and_start_doctags_win(self):
        # Each line takes 7 bytes (including '\n' character which is actually '\r\n' on Windows)
        lines = ['line1\n', 'line2\n', 'line3\n', 'line4\n', 'line5\n']
        tmpf = get_tmpfile('gensim_doc2vec.tst')

        with utils.open(tmpf, 'wb', encoding='utf8') as fout:
            for line in lines:
                fout.write(utils.any2unicode(line))

        offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 1)
        self.assertEqual(offsets, [0])
        self.assertEqual(start_doctags, [0])

        offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 2)
        self.assertEqual(offsets, [0, 14])
        self.assertEqual(start_doctags, [0, 2])

        offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 3)
        self.assertEqual(offsets, [0, 7, 21])
        self.assertEqual(start_doctags, [0, 1, 3])

        offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 4)
        self.assertEqual(offsets, [0, 7, 14, 21])
        self.assertEqual(start_doctags, [0, 1, 2, 3])

        offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 5)
        self.assertEqual(offsets, [0, 7, 14, 21, 28])
        self.assertEqual(start_doctags, [0, 1, 2, 3, 4])

        offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 6)
        self.assertEqual(offsets, [0, 0, 7, 14, 14, 21])
        self.assertEqual(start_doctags, [0, 0, 1, 2, 2, 3])

    def test_cython_linesentence_readline_after_getting_offsets(self):
        lines = ['line1\n', 'line2\n', 'line3\n', 'line4\n', 'line5\n']
        tmpf = get_tmpfile('gensim_doc2vec.tst')

        with utils.open(tmpf, 'wb', encoding='utf8') as fout:
            for line in lines:
                fout.write(utils.any2unicode(line))

        from gensim.models.word2vec_corpusfile import CythonLineSentence

        offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 5)
        for offset, line in zip(offsets, lines):
            ls = CythonLineSentence(tmpf, offset)
            sentence = ls.read_sentence()
            self.assertEqual(len(sentence), 1)
            self.assertEqual(sentence[0], utils.any2utf8(line.strip()))

    def test_unicode_in_doctag(self):
        """Test storing document vectors of a model with unicode titles."""
        model = doc2vec.Doc2Vec(DocsLeeCorpus(unicode_tags=True), min_count=1)
        tmpf = get_tmpfile('gensim_doc2vec.tst')
        try:
            model.save_word2vec_format(tmpf, doctag_vec=True, word_vec=True, binary=True)
        except UnicodeEncodeError:
            self.fail('Failed storing unicode title.')

    def test_load_mmap(self):
        """Test storing/loading the entire model."""
        model = doc2vec.Doc2Vec(sentences, min_count=1)
        tmpf = get_tmpfile('gensim_doc2vec.tst')

        # test storing the internal arrays into separate files
        model.save(tmpf, sep_limit=0)
        self.models_equal(model, doc2vec.Doc2Vec.load(tmpf))

        # make sure mmaping the arrays back works, too
        self.models_equal(model, doc2vec.Doc2Vec.load(tmpf, mmap='r'))

    def test_int_doctags(self):
        """Test doc2vec doctag alternatives"""
        corpus = DocsLeeCorpus()

        model = doc2vec.Doc2Vec(min_count=1)
        model.build_vocab(corpus)
        self.assertEqual(len(model.dv.vectors), 300)
        self.assertEqual(model.dv[0].shape, (100,))
        self.assertEqual(model.dv[np.int64(0)].shape, (100,))
        self.assertRaises(KeyError, model.__getitem__, '_*0')

    def test_missing_string_doctag(self):
        """Test doc2vec doctag alternatives"""
        corpus = list(DocsLeeCorpus(True))
        # force duplicated tags
        corpus = corpus[0:10] + corpus

        model = doc2vec.Doc2Vec(min_count=1)
        model.build_vocab(corpus)
        self.assertRaises(KeyError, model.dv.__getitem__, 'not_a_tag')

    def test_string_doctags(self):
        """Test doc2vec doctag alternatives"""
        corpus = list(DocsLeeCorpus(True))
        # force duplicated tags
        corpus = corpus[0:10] + corpus

        model = doc2vec.Doc2Vec(min_count=1)
        model.build_vocab(corpus)

        self.assertEqual(len(model.dv.vectors), 300)
        self.assertEqual(model.dv[0].shape, (100,))
        self.assertEqual(model.dv['_*0'].shape, (100,))
        self.assertTrue(all(model.dv['_*0'] == model.dv[0]))
        self.assertTrue(max(model.dv.key_to_index.values()) < len(model.dv.index_to_key))
        self.assertLess(
            max(model.dv.get_index(str_key) for str_key in model.dv.key_to_index.keys()),
            len(model.dv.vectors)
        )
        # verify dv.most_similar() returns string doctags rather than indexes
        self.assertEqual(model.dv.index_to_key[0], model.dv.most_similar([model.dv[0]])[0][0])

    def test_empty_errors(self):
        # no input => "RuntimeError: you must first build vocabulary before training the model"
        self.assertRaises(RuntimeError, doc2vec.Doc2Vec, [])

        # input not empty, but rather completely filtered out
        self.assertRaises(RuntimeError, doc2vec.Doc2Vec, list_corpus, min_count=10000)

    def test_similarity_unseen_docs(self):
        """Test similarity of out of training sentences"""
        rome_words = ['rome', 'italy']
        car_words = ['car']
        corpus = list(DocsLeeCorpus(True))

        model = doc2vec.Doc2Vec(min_count=1)
        model.build_vocab(corpus)
        self.assertTrue(
            model.similarity_unseen_docs(rome_words, rome_words)
            > model.similarity_unseen_docs(rome_words, car_words)
        )

    def model_sanity(self, model, keep_training=True):
        """Any non-trivial model on DocsLeeCorpus can pass these sanity checks"""
        fire1 = 0  # doc 0 sydney fires
        fire2 = np.int64(8)  # doc 8 sydney fires
        alt1 = 29  # doc 29 palestine

        # inferred vector should be top10 close to bulk-trained one
        doc0_inferred = model.infer_vector(list(DocsLeeCorpus())[0].words)
        sims_to_infer = model.dv.most_similar([doc0_inferred], topn=len(model.dv))
        sims_ids = [docid for docid, sim in sims_to_infer]
        self.assertTrue(fire1 in sims_ids, "{0} not found in {1}".format(fire1, sims_to_infer))
        f_rank = sims_ids.index(fire1)
        self.assertLess(f_rank, 10)

        # fire2 should be top30 close to fire1
        sims = model.dv.most_similar(fire1, topn=len(model.dv))
        f2_rank = [docid for docid, sim in sims].index(fire2)
        self.assertLess(f2_rank, 30)

        # same sims should appear in lookup by vec as by index
        doc0_vec = model.dv[fire1]
        sims2 = model.dv.most_similar(positive=[doc0_vec], topn=21)
        sims2 = [(id, sim) for id, sim in sims2 if id != fire1]  # ignore the doc itself
        sims = sims[:20]
        self.assertEqual(list(zip(*sims))[0], list(zip(*sims2))[0])  # same doc ids
        self.assertTrue(np.allclose(list(zip(*sims))[1], list(zip(*sims2))[1]))  # close-enough dists

        # sim results should be in clip range if given
        clip_sims = \
            model.dv.most_similar(fire1, clip_start=len(model.dv) // 2, clip_end=len(model.dv) * 2 // 3)
        sims_doc_id = [docid for docid, sim in clip_sims]
        for s_id in sims_doc_id:
            self.assertTrue(len(model.dv) // 2 <= s_id <= len(model.dv) * 2 // 3)

        # fire docs should be closer than fire-alt
        self.assertLess(model.dv.similarity(fire1, alt1), model.dv.similarity(fire1, fire2))
        self.assertLess(model.dv.similarity(fire2, alt1), model.dv.similarity(fire1, fire2))

        # alt doc should be out-of-place among fire news
        self.assertEqual(model.dv.doesnt_match([fire1, alt1, fire2]), alt1)

        # keep training after save
        if keep_training:
            tmpf = get_tmpfile('gensim_doc2vec_resave.tst')
            model.save(tmpf)
            loaded = doc2vec.Doc2Vec.load(tmpf)
            loaded.train(corpus_iterable=sentences, total_examples=loaded.corpus_count, epochs=loaded.epochs)

    def test_training(self):
        """Test doc2vec training."""
        corpus = DocsLeeCorpus()
        model = doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=20, workers=1)
        model.build_vocab(corpus)
        self.assertEqual(model.dv.vectors.shape, (300, 100))
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

        self.model_sanity(model)

        # build vocab and train in one step; must be the same as above
        model2 = doc2vec.Doc2Vec(corpus, vector_size=100, min_count=2, epochs=20, workers=1)
        self.models_equal(model, model2)

    def test_training_fromfile(self):
        """Test doc2vec training."""
        with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
            save_lee_corpus_as_line_sentence(corpus_file)

            model = doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=20, workers=1)
            model.build_vocab(corpus_file=corpus_file)
            self.assertEqual(model.dv.vectors.shape, (300, 100))
            model.train(corpus_file=corpus_file, total_words=model.corpus_total_words, epochs=model.epochs)

            self.model_sanity(model)

            model = doc2vec.Doc2Vec(corpus_file=corpus_file, vector_size=100, min_count=2, epochs=20, workers=1)
            self.model_sanity(model)

    def test_dbow_hs(self):
        """Test DBOW doc2vec training."""
        model = doc2vec.Doc2Vec(list_corpus, dm=0, hs=1, negative=0, min_count=2, epochs=20)
        self.model_sanity(model)

    def test_dbow_hs_fromfile(self):
        """Test DBOW doc2vec training."""
        with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
            save_lee_corpus_as_line_sentence(corpus_file)
            model = doc2vec.Doc2Vec(corpus_file=corpus_file, dm=0, hs=1, negative=0, min_count=2, epochs=20)
            self.model_sanity(model)

    def test_dmm_hs(self):
        """Test DM/mean doc2vec training."""
        model = doc2vec.Doc2Vec(
            list_corpus, dm=1, dm_mean=1, vector_size=24, window=4,
            hs=1, negative=0, alpha=0.05, min_count=2, epochs=20
        )
        self.model_sanity(model)

    def test_dmm_hs_fromfile(self):
        """Test DBOW doc2vec training."""
        with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
            save_lee_corpus_as_line_sentence(corpus_file)
            model = doc2vec.Doc2Vec(
                list_corpus, dm=1, dm_mean=1, vector_size=24, window=4,
                hs=1, negative=0, alpha=0.05, min_count=2, epochs=20
            )
            self.model_sanity(model)

    def test_dms_hs(self):
        """Test DM/sum doc2vec training."""
        model = doc2vec.Doc2Vec(
            list_corpus, dm=1, dm_mean=0, vector_size=24, window=4, hs=1,
            negative=0, alpha=0.05, min_count=2, epochs=20
        )
        self.model_sanity(model)

    def test_dms_hs_fromfile(self):
        """Test DBOW doc2vec training."""
        with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
            save_lee_corpus_as_line_sentence(corpus_file)
            model = doc2vec.Doc2Vec(
                list_corpus, dm=1, dm_mean=0, vector_size=24, window=4, hs=1,
                negative=0, alpha=0.05, min_count=2, epochs=20
            )
            self.model_sanity(model)

    def test_dmc_hs(self):
        """Test DM/concatenate doc2vec training."""
        model = doc2vec.Doc2Vec(
            list_corpus, dm=1, dm_concat=1, vector_size=24, window=4,
            hs=1, negative=0, alpha=0.05, min_count=2, epochs=20
        )
        self.model_sanity(model)

    def test_dmc_hs_fromfile(self):
        """Test DBOW doc2vec training."""
        with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
            save_lee_corpus_as_line_sentence(corpus_file)
            model = doc2vec.Doc2Vec(
                list_corpus, dm=1, dm_concat=1, vector_size=24, window=4,
                hs=1, negative=0, alpha=0.05, min_count=2, epochs=20
            )
            self.model_sanity(model)

    def test_dbow_neg(self):
        """Test DBOW doc2vec training."""
        model = doc2vec.Doc2Vec(list_corpus, vector_size=16, dm=0, hs=0, negative=5, min_count=2, epochs=40)
        self.model_sanity(model)

    def test_dbow_neg_fromfile(self):
        """Test DBOW doc2vec training."""
        with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
            save_lee_corpus_as_line_sentence(corpus_file)
            model = doc2vec.Doc2Vec(list_corpus, vector_size=16, dm=0, hs=0, negative=5, min_count=2, epochs=40)
            self.model_sanity(model)

    def test_dmm_neg(self):
        """Test DM/mean doc2vec training."""
        model = doc2vec.Doc2Vec(
            list_corpus, dm=1, dm_mean=1, vector_size=24, window=4, hs=0,
            negative=10, alpha=0.05, min_count=2, epochs=20
        )
        self.model_sanity(model)

    def test_dmm_neg_fromfile(self):
        """Test DBOW doc2vec training."""
        with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
            save_lee_corpus_as_line_sentence(corpus_file)
            model = doc2vec.Doc2Vec(
                list_corpus, dm=1, dm_mean=1, vector_size=24, window=4, hs=0,
                negative=10, alpha=0.05, min_count=2, epochs=20
            )
            self.model_sanity(model)

    def test_dms_neg(self):
        """Test DM/sum doc2vec training."""
        model = doc2vec.Doc2Vec(
            list_corpus, dm=1, dm_mean=0, vector_size=24, window=4, hs=0,
            negative=10, alpha=0.05, min_count=2, epochs=20
        )
        self.model_sanity(model)

    def test_dms_neg_fromfile(self):
        """Test DBOW doc2vec training."""
        with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
            save_lee_corpus_as_line_sentence(corpus_file)
            model = doc2vec.Doc2Vec(
                list_corpus, dm=1, dm_mean=0, vector_size=24, window=4, hs=0,
                negative=10, alpha=0.05, min_count=2, epochs=20
            )
            self.model_sanity(model)

    def test_dmc_neg(self):
        """Test DM/concatenate doc2vec training."""
        model = doc2vec.Doc2Vec(
            list_corpus, dm=1, dm_concat=1, vector_size=24, window=4, hs=0,
            negative=10, alpha=0.05, min_count=2, epochs=20
        )
        self.model_sanity(model)

    def test_dmc_neg_fromfile(self):
        """Test DBOW doc2vec training."""
        with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
            save_lee_corpus_as_line_sentence(corpus_file)
            model = doc2vec.Doc2Vec(
                list_corpus, dm=1, dm_concat=1, vector_size=24, window=4, hs=0,
                negative=10, alpha=0.05, min_count=2, epochs=20
            )
            self.model_sanity(model)

    def test_dmm_fixedwindowsize(self):
        """Test DMM doc2vec training with fixed window size."""
        model = doc2vec.Doc2Vec(
            list_corpus, vector_size=24,
            dm=1, dm_mean=1, window=4, shrink_windows=False,
            hs=0, negative=10, alpha=0.05, min_count=2, epochs=20
        )
        self.model_sanity(model)

    def test_dmm_fixedwindowsize_fromfile(self):
        """Test DMM doc2vec training with fixed window size, from file."""
        with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
            save_lee_corpus_as_line_sentence(corpus_file)
            model = doc2vec.Doc2Vec(
                corpus_file=corpus_file, vector_size=24,
                dm=1, dm_mean=1, window=4, shrink_windows=False,
                hs=0, negative=10, alpha=0.05, min_count=2, epochs=20
            )
            self.model_sanity(model)

    def test_dbow_fixedwindowsize(self):
        """Test DBOW doc2vec training with fixed window size."""
        model = doc2vec.Doc2Vec(
            list_corpus, vector_size=16, shrink_windows=False,
            dm=0, hs=0, negative=5, min_count=2, epochs=20
        )
        self.model_sanity(model)

    def test_dbow_fixedwindowsize_fromfile(self):
        """Test DBOW doc2vec training with fixed window size, from file."""
        with temporary_file(get_tmpfile('gensim_doc2vec.tst')) as corpus_file:
            save_lee_corpus_as_line_sentence(corpus_file)
            model = doc2vec.Doc2Vec(
                corpus_file=corpus_file, vector_size=16, shrink_windows=False,
                dm=0, hs=0, negative=5, min_count=2, epochs=20
            )
            self.model_sanity(model)

    def test_parallel(self):
        """Test doc2vec parallel training with more than default 3 threads."""
        # repeat the ~300 doc (~60000 word) Lee corpus to get 6000 docs (~1.2M words)
        corpus = utils.RepeatCorpus(DocsLeeCorpus(), 6000)

        # use smaller batches-to-workers for more contention
        model = doc2vec.Doc2Vec(corpus, workers=6, batch_words=5000)
        self.model_sanity(model)

    def test_deterministic_hs(self):
        """Test doc2vec results identical with identical RNG seed."""
        # hs
        model = doc2vec.Doc2Vec(DocsLeeCorpus(), seed=42, workers=1)
        model2 = doc2vec.Doc2Vec(DocsLeeCorpus(), seed=42, workers=1)
        self.models_equal(model, model2)

    def test_deterministic_neg(self):
        """Test doc2vec results identical with identical RNG seed."""
        # neg
        model = doc2vec.Doc2Vec(DocsLeeCorpus(), hs=0, negative=3, seed=42, workers=1)
        model2 = doc2vec.Doc2Vec(DocsLeeCorpus(), hs=0, negative=3, seed=42, workers=1)
        self.models_equal(model, model2)

    def test_deterministic_dmc(self):
        """Test doc2vec results identical with identical RNG seed."""
        # bigger, dmc
        model = doc2vec.Doc2Vec(
            DocsLeeCorpus(), dm=1, dm_concat=1, vector_size=24,
            window=4, hs=1, negative=3, seed=42, workers=1
        )
        model2 = doc2vec.Doc2Vec(
            DocsLeeCorpus(), dm=1, dm_concat=1, vector_size=24,
            window=4, hs=1, negative=3, seed=42, workers=1
        )
        self.models_equal(model, model2)

    def test_mixed_tag_types(self):
        """Ensure alternating int/string tags don't share indexes in vectors"""
        mixed_tag_corpus = [doc2vec.TaggedDocument(words, [i, words[0]]) for i, words in enumerate(raw_sentences)]
        model = doc2vec.Doc2Vec()
        model.build_vocab(mixed_tag_corpus)
        expected_length = len(sentences) + len(model.dv.key_to_index)  # 9 sentences, 7 unique first tokens
        self.assertEqual(len(model.dv.vectors), expected_length)
        # TODO: test saving in word2vec format

    def models_equal(self, model, model2):
        # check words/hidden-weights
        self.assertEqual(len(model.wv), len(model2.wv))
        self.assertTrue(np.allclose(model.wv.vectors, model2.wv.vectors))
        if model.hs:
            self.assertTrue(np.allclose(model.syn1, model2.syn1))
        if model.negative:
            self.assertTrue(np.allclose(model.syn1neg, model2.syn1neg))
        # check docvecs
        self.assertEqual(len(model.dv), len(model2.dv))
        self.assertEqual(len(model.dv.index_to_key), len(model2.dv.index_to_key))

    def test_word_vec_non_writeable(self):
        model = keyedvectors.KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'))
        vector = model['says']
        with self.assertRaises(ValueError):
            vector *= 0

    @log_capture()
    def test_build_vocab_warning(self, loglines):
        """Test if logger warning is raised on non-ideal input to a doc2vec model"""
        raw_sentences = ['human', 'machine']
        sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(raw_sentences)]
        model = doc2vec.Doc2Vec()
        model.build_vocab(sentences)
        warning = "Each 'words' should be a list of words (usually unicode strings)."
        self.assertTrue(warning in str(loglines))

    @log_capture()
    def test_train_warning(self, loglines):
        """Test if warning is raised if alpha rises during subsequent calls to train()"""
        raw_sentences = [['human'],
                         ['graph', 'trees']]
        sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(raw_sentences)]
        model = doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.025, min_count=1, workers=8, vector_size=5)
        model.build_vocab(sentences)
        for epoch in range(10):
            model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
            if epoch == 5:
                model.alpha += 0.05
        warning = "Effective 'alpha' higher than previous training cycles"
        self.assertTrue(warning in str(loglines))

    def test_load_on_class_error(self):
        """Test if exception is raised when loading doc2vec model on instance"""
        self.assertRaises(AttributeError, load_on_instance)
# endclass TestDoc2VecModel


if not hasattr(TestDoc2VecModel, 'assertLess'):
    # workaround for python 2.6
    def assertLess(self, a, b, msg=None):
        self.assertTrue(a < b, msg="%s is not less than %s" % (a, b))

    setattr(TestDoc2VecModel, 'assertLess', assertLess)


# Following code is useful for reproducing paragraph-vectors paper sentiment experiments

class ConcatenatedDoc2Vec:
    """
    Concatenation of multiple models for reproducing the Paragraph Vectors paper.
    Models must have exactly-matching vocabulary and document IDs. (Models should
    be trained separately; this wrapper just returns concatenated results.)
    """

    def __init__(self, models):
        self.models = models
        if hasattr(models[0], 'dv'):
            self.dv = ConcatenatedDocvecs([model.dv for model in models])

    def __getitem__(self, token):
        return np.concatenate([model[token] for model in self.models])

    def __str__(self):
        """Abbreviated name, built from submodels' names"""
        return "+".join(str(model) for model in self.models)

    @property
    def epochs(self):
        return self.models[0].epochs

    def infer_vector(self, document, alpha=None, min_alpha=None, epochs=None):
        return np.concatenate([model.infer_vector(document, alpha, min_alpha, epochs) for model in self.models])

    def train(self, *ignore_args, **ignore_kwargs):
        pass  # train subcomponents individually


class ConcatenatedDocvecs:
    def __init__(self, models):
        self.models = models

    def __getitem__(self, token):
        return np.concatenate([model[token] for model in self.models])


SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')


def read_su_sentiment_rotten_tomatoes(dirname, lowercase=True):
    """
    Read and return documents from the Stanford Sentiment Treebank
    corpus (Rotten Tomatoes reviews), from http://nlp.Stanford.edu/sentiment/

    Initialize the corpus from a given directory, where
    http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
    has been expanded. It's not too big, so compose entirely into memory.
    """
    logging.info("loading corpus from %s", dirname)

    # many mangled chars in sentences (datasetSentences.txt)
    chars_sst_mangled = [
        'à', 'á', 'â', 'ã', 'æ', 'ç', 'è', 'é', 'í',
        'í', 'ï', 'ñ', 'ó', 'ô', 'ö', 'û', 'ü'
    ]
    sentence_fixups = [(char.encode('utf-8').decode('latin1'), char) for char in chars_sst_mangled]
    # more junk, and the replace necessary for sentence-phrase consistency
    sentence_fixups.extend([
        ('Â', ''),
        ('\xa0', ' '),
        ('-LRB-', '('),
        ('-RRB-', ')'),
    ])
    # only this junk in phrases (dictionary.txt)
    phrase_fixups = [('\xa0', ' ')]

    # sentence_id and split are only positive for the full sentences

    # read sentences to temp {sentence -> (id,split) dict, to correlate with dictionary.txt
    info_by_sentence = {}
    with open(os.path.join(dirname, 'datasetSentences.txt'), 'r') as sentences:
        with open(os.path.join(dirname, 'datasetSplit.txt'), 'r') as splits:
            next(sentences)  # legend
            next(splits)     # legend
            for sentence_line, split_line in zip(sentences, splits):
                id, text = sentence_line.split('\t')
                id = int(id)
                text = text.rstrip()
                for junk, fix in sentence_fixups:
                    text = text.replace(junk, fix)
                (id2, split_i) = split_line.split(',')
                assert id == int(id2)
                if text not in info_by_sentence:    # discard duplicates
                    info_by_sentence[text] = (id, int(split_i))

    # read all phrase text
    phrases = [None] * 239232  # known size of phrases
    with open(os.path.join(dirname, 'dictionary.txt'), 'r') as phrase_lines:
        for line in phrase_lines:
            (text, id) = line.split('|')
            for junk, fix in phrase_fixups:
                text = text.replace(junk, fix)
            phrases[int(id)] = text.rstrip()  # for 1st pass just string

    SentimentPhrase = namedtuple('SentimentPhrase', SentimentDocument._fields + ('sentence_id',))
    # add sentiment labels, correlate with sentences
    with open(os.path.join(dirname, 'sentiment_labels.txt'), 'r') as sentiments:
        next(sentiments)  # legend
        for line in sentiments:
            (id, sentiment) = line.split('|')
            id = int(id)
            sentiment = float(sentiment)
            text = phrases[id]
            words = text.split()
            if lowercase:
                words = [word.lower() for word in words]
            (sentence_id, split_i) = info_by_sentence.get(text, (None, 0))
            split = [None, 'train', 'test', 'dev'][split_i]
            phrases[id] = SentimentPhrase(words, [id], split, sentiment, sentence_id)

    assert sum(1 for phrase in phrases if phrase.sentence_id is not None) == len(info_by_sentence)  # all
    # counts don't match 8544, 2210, 1101 because 13 TRAIN and 1 DEV sentences are duplicates
    assert sum(1 for phrase in phrases if phrase.split == 'train') == 8531  # 'train'
    assert sum(1 for phrase in phrases if phrase.split == 'test') == 2210  # 'test'
    assert sum(1 for phrase in phrases if phrase.split == 'dev') == 1100  # 'dev'

    logging.info(
        "loaded corpus with %i sentences and %i phrases from %s",
        len(info_by_sentence), len(phrases), dirname
    )

    return phrases


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main(module='gensim.test.test_doc2vec')
