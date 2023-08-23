#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""
Automated tests for checking transformation algorithms (the models package).
"""

import logging
import unittest
import os
import bz2
import sys
import tempfile
import subprocess

import numpy as np

from testfixtures import log_capture

try:
    from ot import emd2  # noqa:F401
    POT_EXT = True
except (ImportError, ValueError):
    POT_EXT = False

from gensim import utils
from gensim.models import word2vec, keyedvectors
from gensim.utils import check_output
from gensim.test.utils import (
    datapath, get_tmpfile, temporary_file, common_texts as sentences,
    LeeCorpus, lee_corpus_list,
)


new_sentences = [
    ['computer', 'artificial', 'intelligence'],
    ['artificial', 'trees'],
    ['human', 'intelligence'],
    ['artificial', 'graph'],
    ['intelligence'],
    ['artificial', 'intelligence', 'system']
]


def _rule(word, count, min_count):
    if word == "human":
        return utils.RULE_DISCARD  # throw out
    else:
        return utils.RULE_DEFAULT  # apply default rule, i.e. min_count


def load_on_instance():
    # Save and load a Word2Vec Model on instance for test
    tmpf = get_tmpfile('gensim_word2vec.tst')
    model = word2vec.Word2Vec(sentences, min_count=1)
    model.save(tmpf)
    model = word2vec.Word2Vec()  # should fail at this point
    return model.load(tmpf)


class TestWord2VecModel(unittest.TestCase):
    def test_build_vocab_from_freq(self):
        """Test that the algorithm is able to build vocabulary from given
        frequency table"""
        freq_dict = {
        'minors': 2, 'graph': 3, 'system': 4,
        'trees': 3, 'eps': 2, 'computer': 2,
        'survey': 2, 'user': 3, 'human': 2,
        'time': 2, 'interface': 2, 'response': 2
        }
        freq_dict_orig = freq_dict.copy()
        model_hs = word2vec.Word2Vec(vector_size=10, min_count=0, seed=42, hs=1, negative=0)
        model_neg = word2vec.Word2Vec(vector_size=10, min_count=0, seed=42, hs=0, negative=5)
        model_hs.build_vocab_from_freq(freq_dict)
        model_neg.build_vocab_from_freq(freq_dict)
        self.assertEqual(len(model_hs.wv), 12)
        self.assertEqual(len(model_neg.wv), 12)
        for k in freq_dict_orig.keys():
            self.assertEqual(model_hs.wv.get_vecattr(k, 'count'), freq_dict_orig[k])
            self.assertEqual(model_neg.wv.get_vecattr(k, 'count'), freq_dict_orig[k])

        new_freq_dict = {
            'computer': 1, 'artificial': 4, 'human': 1, 'graph': 1, 'intelligence': 4, 'system': 1, 'trees': 1
        }
        model_hs.build_vocab_from_freq(new_freq_dict, update=True)
        model_neg.build_vocab_from_freq(new_freq_dict, update=True)
        self.assertEqual(model_hs.wv.get_vecattr('graph', 'count'), 4)
        self.assertEqual(model_hs.wv.get_vecattr('artificial', 'count'), 4)
        self.assertEqual(len(model_hs.wv), 14)
        self.assertEqual(len(model_neg.wv), 14)

    def test_prune_vocab(self):
        """Test Prune vocab while scanning sentences"""
        sentences = [
            ["graph", "system"],
            ["graph", "system"],
            ["system", "eps"],
            ["graph", "system"]
        ]
        model = word2vec.Word2Vec(sentences, vector_size=10, min_count=0, max_vocab_size=2, seed=42, hs=1, negative=0)
        self.assertEqual(len(model.wv), 2)
        self.assertEqual(model.wv.get_vecattr('graph', 'count'), 3)
        self.assertEqual(model.wv.get_vecattr('system', 'count'), 4)

        sentences = [
            ["graph", "system"],
            ["graph", "system"],
            ["system", "eps"],
            ["graph", "system"],
            ["minors", "survey", "minors", "survey", "minors"]
        ]
        model = word2vec.Word2Vec(sentences, vector_size=10, min_count=0, max_vocab_size=2, seed=42, hs=1, negative=0)
        self.assertEqual(len(model.wv), 3)
        self.assertEqual(model.wv.get_vecattr('graph', 'count'), 3)
        self.assertEqual(model.wv.get_vecattr('minors', 'count'), 3)
        self.assertEqual(model.wv.get_vecattr('system', 'count'), 4)

    def test_total_word_count(self):
        model = word2vec.Word2Vec(vector_size=10, min_count=0, seed=42)
        total_words = model.scan_vocab(sentences)[0]
        self.assertEqual(total_words, 29)

    def test_max_final_vocab(self):
        # Test for less restricting effect of max_final_vocab
        # max_final_vocab is specified but has no effect
        model = word2vec.Word2Vec(vector_size=10, max_final_vocab=4, min_count=4, sample=0)
        model.scan_vocab(sentences)
        reported_values = model.prepare_vocab()
        self.assertEqual(reported_values['drop_unique'], 11)
        self.assertEqual(reported_values['retain_total'], 4)
        self.assertEqual(reported_values['num_retained_words'], 1)
        self.assertEqual(model.effective_min_count, 4)

        # Test for more restricting effect of max_final_vocab
        # results in setting a min_count more restricting than specified min_count
        model = word2vec.Word2Vec(vector_size=10, max_final_vocab=4, min_count=2, sample=0)
        model.scan_vocab(sentences)
        reported_values = model.prepare_vocab()
        self.assertEqual(reported_values['drop_unique'], 8)
        self.assertEqual(reported_values['retain_total'], 13)
        self.assertEqual(reported_values['num_retained_words'], 4)
        self.assertEqual(model.effective_min_count, 3)

    def test_online_learning(self):
        """Test that the algorithm is able to add new words to the
        vocabulary and to a trained model when using a sorted vocabulary"""
        model_hs = word2vec.Word2Vec(sentences, vector_size=10, min_count=0, seed=42, hs=1, negative=0)
        model_neg = word2vec.Word2Vec(sentences, vector_size=10, min_count=0, seed=42, hs=0, negative=5)
        self.assertTrue(len(model_hs.wv), 12)
        self.assertTrue(model_hs.wv.get_vecattr('graph', 'count'), 3)
        model_hs.build_vocab(new_sentences, update=True)
        model_neg.build_vocab(new_sentences, update=True)
        self.assertTrue(model_hs.wv.get_vecattr('graph', 'count'), 4)
        self.assertTrue(model_hs.wv.get_vecattr('artificial', 'count'), 4)
        self.assertEqual(len(model_hs.wv), 14)
        self.assertEqual(len(model_neg.wv), 14)

    def test_online_learning_after_save(self):
        """Test that the algorithm is able to add new words to the
        vocabulary and to a trained model when using a sorted vocabulary"""
        tmpf = get_tmpfile('gensim_word2vec.tst')
        model_neg = word2vec.Word2Vec(sentences, vector_size=10, min_count=0, seed=42, hs=0, negative=5)
        model_neg.save(tmpf)
        model_neg = word2vec.Word2Vec.load(tmpf)
        self.assertTrue(len(model_neg.wv), 12)
        model_neg.build_vocab(new_sentences, update=True)
        model_neg.train(new_sentences, total_examples=model_neg.corpus_count, epochs=model_neg.epochs)
        self.assertEqual(len(model_neg.wv), 14)

    def test_online_learning_from_file(self):
        """Test that the algorithm is able to add new words to the
        vocabulary and to a trained model when using a sorted vocabulary"""
        with temporary_file(get_tmpfile('gensim_word2vec1.tst')) as corpus_file, \
                temporary_file(get_tmpfile('gensim_word2vec2.tst')) as new_corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)
            utils.save_as_line_sentence(new_sentences, new_corpus_file)

            model_hs = word2vec.Word2Vec(corpus_file=corpus_file, vector_size=10, min_count=0, seed=42,
                                         hs=1, negative=0)
            model_neg = word2vec.Word2Vec(corpus_file=corpus_file, vector_size=10, min_count=0, seed=42,
                                          hs=0, negative=5)
            self.assertTrue(len(model_hs.wv), 12)
            self.assertTrue(model_hs.wv.get_vecattr('graph', 'count'), 3)
            model_hs.build_vocab(corpus_file=new_corpus_file, update=True)
            model_hs.train(corpus_file=new_corpus_file, total_words=model_hs.corpus_total_words, epochs=model_hs.epochs)

            model_neg.build_vocab(corpus_file=new_corpus_file, update=True)
            model_neg.train(
                corpus_file=new_corpus_file, total_words=model_hs.corpus_total_words, epochs=model_hs.epochs)
            self.assertTrue(model_hs.wv.get_vecattr('graph', 'count'), 4)
            self.assertTrue(model_hs.wv.get_vecattr('artificial', 'count'), 4)
            self.assertEqual(len(model_hs.wv), 14)
            self.assertEqual(len(model_neg.wv), 14)

    def test_online_learning_after_save_from_file(self):
        """Test that the algorithm is able to add new words to the
        vocabulary and to a trained model when using a sorted vocabulary"""
        with temporary_file(get_tmpfile('gensim_word2vec1.tst')) as corpus_file, \
                temporary_file(get_tmpfile('gensim_word2vec2.tst')) as new_corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)
            utils.save_as_line_sentence(new_sentences, new_corpus_file)

            tmpf = get_tmpfile('gensim_word2vec.tst')
            model_neg = word2vec.Word2Vec(corpus_file=corpus_file, vector_size=10, min_count=0, seed=42,
                                          hs=0, negative=5)
            model_neg.save(tmpf)
            model_neg = word2vec.Word2Vec.load(tmpf)
            self.assertTrue(len(model_neg.wv), 12)
            # Check that training works on the same data after load without calling build_vocab
            model_neg.train(corpus_file=corpus_file, total_words=model_neg.corpus_total_words, epochs=model_neg.epochs)
            # Train on new corpus file
            model_neg.build_vocab(corpus_file=new_corpus_file, update=True)
            model_neg.train(corpus_file=new_corpus_file, total_words=model_neg.corpus_total_words,
                            epochs=model_neg.epochs)
            self.assertEqual(len(model_neg.wv), 14)

    def onlineSanity(self, model, trained_model=False):
        terro, others = [], []
        for line in lee_corpus_list:
            if 'terrorism' in line:
                terro.append(line)
            else:
                others.append(line)
        self.assertTrue(all('terrorism' not in line for line in others))
        model.build_vocab(others, update=trained_model)
        model.train(others, total_examples=model.corpus_count, epochs=model.epochs)
        self.assertFalse('terrorism' in model.wv)
        model.build_vocab(terro, update=True)
        self.assertTrue('terrorism' in model.wv)
        orig0 = np.copy(model.wv.vectors)
        model.train(terro, total_examples=len(terro), epochs=model.epochs)
        self.assertFalse(np.allclose(model.wv.vectors, orig0))
        sim = model.wv.n_similarity(['war'], ['terrorism'])
        self.assertLess(0., sim)

    def test_sg_hs_online(self):
        """Test skipgram w/ hierarchical softmax"""
        model = word2vec.Word2Vec(sg=1, window=5, hs=1, negative=0, min_count=3, epochs=10, seed=42, workers=2)
        self.onlineSanity(model)

    def test_sg_neg_online(self):
        """Test skipgram w/ negative sampling"""
        model = word2vec.Word2Vec(sg=1, window=4, hs=0, negative=15, min_count=3, epochs=10, seed=42, workers=2)
        self.onlineSanity(model)

    def test_cbow_hs_online(self):
        """Test CBOW w/ hierarchical softmax"""
        model = word2vec.Word2Vec(
            sg=0, cbow_mean=1, alpha=0.05, window=5, hs=1, negative=0,
            min_count=3, epochs=20, seed=42, workers=2
        )
        self.onlineSanity(model)

    def test_cbow_neg_online(self):
        """Test CBOW w/ negative sampling"""
        model = word2vec.Word2Vec(
            sg=0, cbow_mean=1, alpha=0.05, window=5, hs=0, negative=15,
            min_count=5, epochs=10, seed=42, workers=2, sample=0
        )
        self.onlineSanity(model)

    def test_persistence(self):
        """Test storing/loading the entire model."""
        tmpf = get_tmpfile('gensim_word2vec.tst')
        model = word2vec.Word2Vec(sentences, min_count=1)
        model.save(tmpf)
        self.models_equal(model, word2vec.Word2Vec.load(tmpf))
        #  test persistence of the KeyedVectors of a model
        wv = model.wv
        wv.save(tmpf)
        loaded_wv = keyedvectors.KeyedVectors.load(tmpf)
        self.assertTrue(np.allclose(wv.vectors, loaded_wv.vectors))
        self.assertEqual(len(wv), len(loaded_wv))

    def test_persistence_backwards_compatible(self):
        """Can we still load a model created with an older gensim version?"""
        path = datapath('model-from-gensim-3.8.0.w2v')
        model = word2vec.Word2Vec.load(path)
        x = model.score(['test'])
        assert x is not None

    def test_persistence_from_file(self):
        """Test storing/loading the entire model trained with corpus_file argument."""
        with temporary_file(get_tmpfile('gensim_word2vec.tst')) as corpus_file:
            utils.save_as_line_sentence(sentences, corpus_file)

            tmpf = get_tmpfile('gensim_word2vec.tst')
            model = word2vec.Word2Vec(corpus_file=corpus_file, min_count=1)
            model.save(tmpf)
            self.models_equal(model, word2vec.Word2Vec.load(tmpf))
            #  test persistence of the KeyedVectors of a model
            wv = model.wv
            wv.save(tmpf)
            loaded_wv = keyedvectors.KeyedVectors.load(tmpf)
            self.assertTrue(np.allclose(wv.vectors, loaded_wv.vectors))
            self.assertEqual(len(wv), len(loaded_wv))

    def test_persistence_with_constructor_rule(self):
        """Test storing/loading the entire model with a vocab trimming rule passed in the constructor."""
        tmpf = get_tmpfile('gensim_word2vec.tst')
        model = word2vec.Word2Vec(sentences, min_count=1, trim_rule=_rule)
        model.save(tmpf)
        self.models_equal(model, word2vec.Word2Vec.load(tmpf))

    def test_rule_with_min_count(self):
        """Test that returning RULE_DEFAULT from trim_rule triggers min_count."""
        model = word2vec.Word2Vec(sentences + [["occurs_only_once"]], min_count=2, trim_rule=_rule)
        self.assertTrue("human" not in model.wv)
        self.assertTrue("occurs_only_once" not in model.wv)
        self.assertTrue("interface" in model.wv)

    def test_rule(self):
        """Test applying vocab trim_rule to build_vocab instead of constructor."""
        model = word2vec.Word2Vec(min_count=1)
        model.build_vocab(sentences, trim_rule=_rule)
        self.assertTrue("human" not in model.wv)

    def test_lambda_rule(self):
        """Test that lambda trim_rule works."""
        def rule(word, count, min_count):
            return utils.RULE_DISCARD if word == "human" else utils.RULE_DEFAULT

        model = word2vec.Word2Vec(sentences, min_count=1, trim_rule=rule)
        self.assertTrue("human" not in model.wv)

    def obsolete_testLoadPreKeyedVectorModel(self):
        """Test loading pre-KeyedVectors word2vec model"""

        if sys.version_info[:2] == (3, 4):
            model_file_suffix = '_py3_4'
        elif sys.version_info < (3,):
            model_file_suffix = '_py2'
        else:
            model_file_suffix = '_py3'

        # Model stored in one file
        model_file = 'word2vec_pre_kv%s' % model_file_suffix
        model = word2vec.Word2Vec.load(datapath(model_file))
        self.assertTrue(model.wv.vectors.shape == (len(model.wv), model.vector_size))
        self.assertTrue(model.syn1neg.shape == (len(model.wv), model.vector_size))

        # Model stored in multiple files
        model_file = 'word2vec_pre_kv_sep%s' % model_file_suffix
        model = word2vec.Word2Vec.load(datapath(model_file))
        self.assertTrue(model.wv.vectors.shape == (len(model.wv), model.vector_size))
        self.assertTrue(model.syn1neg.shape == (len(model.wv), model.vector_size))

    def test_load_pre_keyed_vector_model_c_format(self):
        """Test loading pre-KeyedVectors word2vec model saved in word2vec format"""
        model = keyedvectors.KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'))
        self.assertTrue(model.vectors.shape[0] == len(model))

    def test_persistence_word2vec_format(self):
        """Test storing/loading the entire model in word2vec format."""
        tmpf = get_tmpfile('gensim_word2vec.tst')
        model = word2vec.Word2Vec(sentences, min_count=1)
        model.wv.save_word2vec_format(tmpf, binary=True)
        binary_model_kv = keyedvectors.KeyedVectors.load_word2vec_format(tmpf, binary=True)
        self.assertTrue(np.allclose(model.wv['human'], binary_model_kv['human']))
        norm_only_model = keyedvectors.KeyedVectors.load_word2vec_format(tmpf, binary=True)
        norm_only_model.unit_normalize_all()
        self.assertFalse(np.allclose(model.wv['human'], norm_only_model['human']))
        self.assertTrue(np.allclose(model.wv.get_vector('human', norm=True), norm_only_model['human']))
        limited_model_kv = keyedvectors.KeyedVectors.load_word2vec_format(tmpf, binary=True, limit=3)
        self.assertEqual(len(limited_model_kv.vectors), 3)
        half_precision_model_kv = keyedvectors.KeyedVectors.load_word2vec_format(
            tmpf, binary=True, datatype=np.float16
        )
        self.assertEqual(binary_model_kv.vectors.nbytes, half_precision_model_kv.vectors.nbytes * 2)

    def test_no_training_c_format(self):
        tmpf = get_tmpfile('gensim_word2vec.tst')
        model = word2vec.Word2Vec(sentences, min_count=1)
        model.wv.save_word2vec_format(tmpf, binary=True)
        kv = keyedvectors.KeyedVectors.load_word2vec_format(tmpf, binary=True)
        binary_model = word2vec.Word2Vec()
        binary_model.wv = kv
        self.assertRaises(ValueError, binary_model.train, sentences)

    def test_too_short_binary_word2vec_format(self):
        tfile = get_tmpfile('gensim_word2vec.tst')
        model = word2vec.Word2Vec(sentences, min_count=1)
        model.wv.save_word2vec_format(tfile, binary=True)
        f = open(tfile, 'r+b')
        f.write(b'13')  # write wrong (too-long) vector count
        f.close()
        self.assertRaises(EOFError, keyedvectors.KeyedVectors.load_word2vec_format, tfile, binary=True)

    def test_too_short_text_word2vec_format(self):
        tfile = get_tmpfile('gensim_word2vec.tst')
        model = word2vec.Word2Vec(sentences, min_count=1)
        model.wv.save_word2vec_format(tfile, binary=False)
        f = open(tfile, 'r+b')
        f.write(b'13')  # write wrong (too-long) vector count
        f.close()
        self.assertRaises(EOFError, keyedvectors.KeyedVectors.load_word2vec_format, tfile, binary=False)

    def test_persistence_word2vec_format_non_binary(self):
        """Test storing/loading the entire model in word2vec non-binary format."""
        tmpf = get_tmpfile('gensim_word2vec.tst')
        model = word2vec.Word2Vec(sentences, min_count=1)
        model.wv.save_word2vec_format(tmpf, binary=False)
        text_model = keyedvectors.KeyedVectors.load_word2vec_format(tmpf, binary=False)
        self.assertTrue(np.allclose(model.wv['human'], text_model['human'], atol=1e-6))
        norm_only_model = keyedvectors.KeyedVectors.load_word2vec_format(tmpf, binary=False)
        norm_only_model.unit_normalize_all()
        self.assertFalse(np.allclose(model.wv['human'], norm_only_model['human'], atol=1e-6))
        self.assertTrue(np.allclose(
            model.wv.get_vector('human', norm=True), norm_only_model['human'], atol=1e-4
        ))

    def test_persistence_word2vec_format_with_vocab(self):
        """Test storing/loading the entire model and vocabulary in word2vec format."""
        tmpf = get_tmpfile('gensim_word2vec.tst')
        model = word2vec.Word2Vec(sentences, min_count=1)
        testvocab = get_tmpfile('gensim_word2vec.vocab')
        model.wv.save_word2vec_format(tmpf, testvocab, binary=True)
        binary_model_with_vocab_kv = keyedvectors.KeyedVectors.load_word2vec_format(tmpf, testvocab, binary=True)
        self.assertEqual(
            model.wv.get_vecattr('human', 'count'),
            binary_model_with_vocab_kv.get_vecattr('human', 'count'),
        )

    def test_persistence_keyed_vectors_format_with_vocab(self):
        """Test storing/loading the entire model and vocabulary in word2vec format."""
        tmpf = get_tmpfile('gensim_word2vec.tst')
        model = word2vec.Word2Vec(sentences, min_count=1)
        testvocab = get_tmpfile('gensim_word2vec.vocab')
        model.wv.save_word2vec_format(tmpf, testvocab, binary=True)
        kv_binary_model_with_vocab = keyedvectors.KeyedVectors.load_word2vec_format(tmpf, testvocab, binary=True)
        self.assertEqual(
            model.wv.get_vecattr('human', 'count'),
            kv_binary_model_with_vocab.get_vecattr('human', 'count'),
        )

    def test_persistence_word2vec_format_combination_with_standard_persistence(self):
        """Test storing/loading the entire model and vocabulary in word2vec format chained with
         saving and loading via `save` and `load` methods`.
         It was possible prior to 1.0.0 release, now raises Exception"""
        tmpf = get_tmpfile('gensim_word2vec.tst')
        model = word2vec.Word2Vec(sentences, min_count=1)
        testvocab = get_tmpfile('gensim_word2vec.vocab')
        model.wv.save_word2vec_format(tmpf, testvocab, binary=True)
        binary_model_with_vocab_kv = keyedvectors.KeyedVectors.load_word2vec_format(tmpf, testvocab, binary=True)
        binary_model_with_vocab_kv.save(tmpf)
        self.assertRaises(AttributeError, word2vec.Word2Vec.load, tmpf)

    def test_large_mmap(self):
        """Test storing/loading the entire model."""
        tmpf = get_tmpfile('gensim_word2vec.tst')
        model = word2vec.Word2Vec(sentences, min_count=1)

        # test storing the internal arrays into separate files
        model.save(tmpf, sep_limit=0)
        self.models_equal(model, word2vec.Word2Vec.load(tmpf))

        # make sure mmaping the arrays back works, too
        self.models_equal(model, word2vec.Word2Vec.load(tmpf, mmap='r'))

    def test_vocab(self):
        """Test word2vec vocabulary building."""
        corpus = LeeCorpus()
        total_words = sum(len(sentence) for sentence in corpus)

        # try vocab building explicitly, using all words
        model = word2vec.Word2Vec(min_count=1, hs=1, negative=0)
        model.build_vocab(corpus)
        self.assertTrue(len(model.wv) == 6981)
        # with min_count=1, we're not throwing away anything,
        # so make sure the word counts add up to be the entire corpus
        self.assertEqual(sum(model.wv.get_vecattr(k, 'count') for k in model.wv.key_to_index), total_words)
        # make sure the binary codes are correct
        np.allclose(model.wv.get_vecattr('the', 'code'), [1, 1, 0, 0])

        # test building vocab with default params
        model = word2vec.Word2Vec(hs=1, negative=0)
        model.build_vocab(corpus)
        self.assertTrue(len(model.wv) == 1750)
        np.allclose(model.wv.get_vecattr('the', 'code'), [1, 1, 1, 0])

        # no input => "RuntimeError: you must first build vocabulary before training the model"
        self.assertRaises(RuntimeError, word2vec.Word2Vec, [])

        # input not empty, but rather completely filtered out
        self.assertRaises(RuntimeError, word2vec.Word2Vec, corpus, min_count=total_words + 1)

    def test_training(self):
        """Test word2vec training."""
        # build vocabulary, don't train yet
        model = word2vec.Word2Vec(vector_size=2, min_count=1, hs=1, negative=0)
        model.build_vocab(sentences)

        self.assertTrue(model.wv.vectors.shape == (len(model.wv), 2))
        self.assertTrue(model.syn1.shape == (len(model.wv), 2))

        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        sims = model.wv.most_similar('graph', topn=10)
        # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # test querying for "most similar" by vector
        graph_vector = model.wv.get_vector('graph', norm=True)
        sims2 = model.wv.most_similar(positive=[graph_vector], topn=11)
        sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
        self.assertEqual(sims, sims2)

        # build vocab and train in one step; must be the same as above
        model2 = word2vec.Word2Vec(sentences, vector_size=2, min_count=1, hs=1, negative=0)
        self.models_equal(model, model2)

    def test_training_from_file(self):
        """Test word2vec training with corpus_file argument."""
        # build vocabulary, don't train yet
        with temporary_file(get_tmpfile('gensim_word2vec.tst')) as tf:
            utils.save_as_line_sentence(sentences, tf)

            model = word2vec.Word2Vec(vector_size=2, min_count=1, hs=1, negative=0)
            model.build_vocab(corpus_file=tf)

            self.assertTrue(model.wv.vectors.shape == (len(model.wv), 2))
            self.assertTrue(model.syn1.shape == (len(model.wv), 2))

            model.train(corpus_file=tf, total_words=model.corpus_total_words, epochs=model.epochs)
            sims = model.wv.most_similar('graph', topn=10)
            # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

            # test querying for "most similar" by vector
            graph_vector = model.wv.get_vector('graph', norm=True)
            sims2 = model.wv.most_similar(positive=[graph_vector], topn=11)
            sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
            self.assertEqual(sims, sims2)

    def test_scoring(self):
        """Test word2vec scoring."""
        model = word2vec.Word2Vec(sentences, vector_size=2, min_count=1, hs=1, negative=0)

        # just score and make sure they exist
        scores = model.score(sentences, len(sentences))
        self.assertEqual(len(scores), len(sentences))

    def test_locking(self):
        """Test word2vec training doesn't change locked vectors."""
        corpus = LeeCorpus()
        # build vocabulary, don't train yet
        for sg in range(2):  # test both cbow and sg
            model = word2vec.Word2Vec(vector_size=4, hs=1, negative=5, min_count=1, sg=sg, window=5)
            model.build_vocab(corpus)

            # remember two vectors
            locked0 = np.copy(model.wv.vectors[0])
            unlocked1 = np.copy(model.wv.vectors[1])
            # alocate a full lockf array (not just default single val for all)
            model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)
            # lock the vector in slot 0 against change
            model.wv.vectors_lockf[0] = 0.0

            model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
            self.assertFalse((unlocked1 == model.wv.vectors[1]).all())  # unlocked vector should vary
            self.assertTrue((locked0 == model.wv.vectors[0]).all())  # locked vector should not vary

    def test_evaluate_word_analogies(self):
        """Test that evaluating analogies on KeyedVectors give sane results"""
        model = word2vec.Word2Vec(LeeCorpus())
        score, sections = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
        score_cosmul, sections_cosmul = model.wv.evaluate_word_analogies(
            datapath('questions-words.txt'),
            similarity_function='most_similar_cosmul'
        )
        self.assertEqual(score, score_cosmul)
        self.assertEqual(sections, sections_cosmul)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(len(sections), 0)
        # Check that dict contains the right keys
        first_section = sections[0]
        self.assertIn('section', first_section)
        self.assertIn('correct', first_section)
        self.assertIn('incorrect', first_section)

    def test_evaluate_word_pairs(self):
        """Test Spearman and Pearson correlation coefficients give sane results on similarity datasets"""
        corpus = word2vec.LineSentence(datapath('head500.noblanks.cor.bz2'))
        model = word2vec.Word2Vec(corpus, min_count=3, epochs=20)
        correlation = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
        pearson = correlation[0][0]
        spearman = correlation[1][0]
        oov = correlation[2]
        self.assertTrue(0.1 < pearson < 1.0, f"pearson {pearson} not between 0.1 & 1.0")
        self.assertTrue(0.1 < spearman < 1.0, f"spearman {spearman} not between 0.1 and 1.0")
        self.assertTrue(0.0 <= oov < 90.0, f"OOV {oov} not between 0.0 and 90.0")

    def test_evaluate_word_pairs_from_file(self):
        """Test Spearman and Pearson correlation coefficients give sane results on similarity datasets"""
        with temporary_file(get_tmpfile('gensim_word2vec.tst')) as tf:
            utils.save_as_line_sentence(word2vec.LineSentence(datapath('head500.noblanks.cor.bz2')), tf)

            model = word2vec.Word2Vec(corpus_file=tf, min_count=3, epochs=20)
            correlation = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
            pearson = correlation[0][0]
            spearman = correlation[1][0]
            oov = correlation[2]
            self.assertTrue(0.1 < pearson < 1.0, f"pearson {pearson} not between 0.1 & 1.0")
            self.assertTrue(0.1 < spearman < 1.0, f"spearman {spearman} not between 0.1 and 1.0")
            self.assertTrue(0.0 <= oov < 90.0, f"OOV {oov} not between 0.0 and 90.0")

    def model_sanity(self, model, train=True, with_corpus_file=False, ranks=None):
        """Even tiny models trained on LeeCorpus should pass these sanity checks"""
        # run extra before/after training tests if train=True
        if train:
            model.build_vocab(lee_corpus_list)
            orig0 = np.copy(model.wv.vectors[0])

            if with_corpus_file:
                tmpfile = get_tmpfile('gensim_word2vec.tst')
                utils.save_as_line_sentence(lee_corpus_list, tmpfile)
                model.train(corpus_file=tmpfile, total_words=model.corpus_total_words, epochs=model.epochs)
            else:
                model.train(lee_corpus_list, total_examples=model.corpus_count, epochs=model.epochs)
            self.assertFalse((orig0 == model.wv.vectors[1]).all())  # vector should vary after training
        query_word = 'attacks'
        expected_word = 'bombings'
        sims = model.wv.most_similar(query_word, topn=len(model.wv.index_to_key))
        t_rank = [word for word, score in sims].index(expected_word)
        # in >200 calibration runs w/ calling parameters, 'terrorism' in 50-most_sim for 'war'
        if ranks is not None:
            ranks.append(t_rank)  # tabulate trial rank if requested
        self.assertLess(t_rank, 50)
        query_vec = model.wv[query_word]
        sims2 = model.wv.most_similar([query_vec], topn=51)
        self.assertTrue(query_word in [word for word, score in sims2])
        self.assertTrue(expected_word in [word for word, score in sims2])

    def test_sg_hs(self):
        """Test skipgram w/ hierarchical softmax"""
        model = word2vec.Word2Vec(sg=1, window=4, hs=1, negative=0, min_count=5, epochs=10, workers=2)
        self.model_sanity(model)

    def test_sg_hs_fromfile(self):
        model = word2vec.Word2Vec(sg=1, window=4, hs=1, negative=0, min_count=5, epochs=10, workers=2)
        self.model_sanity(model, with_corpus_file=True)

    def test_sg_neg(self):
        """Test skipgram w/ negative sampling"""
        model = word2vec.Word2Vec(sg=1, window=4, hs=0, negative=15, min_count=5, epochs=10, workers=2)
        self.model_sanity(model)

    def test_sg_neg_fromfile(self):
        model = word2vec.Word2Vec(sg=1, window=4, hs=0, negative=15, min_count=5, epochs=10, workers=2)
        self.model_sanity(model, with_corpus_file=True)

    @unittest.skipIf('BULK_TEST_REPS' not in os.environ, reason="bulk test only occasionally run locally")
    def test_method_in_bulk(self):
        """Not run by default testing, but can be run locally to help tune stochastic aspects of tests
        to very-very-rarely fail. EG:
        % BULK_TEST_REPS=200 METHOD_NAME=test_cbow_hs pytest test_word2vec.py -k "test_method_in_bulk"
        Method must accept `ranks` keyword-argument, empty list into which salient internal result can be reported.
        """
        failures = 0
        ranks = []
        reps = int(os.environ['BULK_TEST_REPS'])
        method_name = os.environ.get('METHOD_NAME', 'test_cbow_hs')  # by default test that specially-troublesome one
        method_fn = getattr(self, method_name)
        for i in range(reps):
            try:
                method_fn(ranks=ranks)
            except Exception as ex:
                print('%s failed: %s' % (method_name, ex))
                failures += 1
        print(ranks)
        print(np.mean(ranks))
        self.assertEquals(failures, 0, "too many failures")

    def test_cbow_hs(self, ranks=None):
        """Test CBOW w/ hierarchical softmax"""
        model = word2vec.Word2Vec(
            sg=0, cbow_mean=1, alpha=0.1, window=2, hs=1, negative=0,
            min_count=5, epochs=60, workers=2, batch_words=1000
        )
        self.model_sanity(model, ranks=ranks)

    def test_cbow_hs_fromfile(self):
        model = word2vec.Word2Vec(
            sg=0, cbow_mean=1, alpha=0.1, window=2, hs=1, negative=0,
            min_count=5, epochs=60, workers=2, batch_words=1000
        )
        self.model_sanity(model, with_corpus_file=True)

    def test_cbow_neg(self, ranks=None):
        """Test CBOW w/ negative sampling"""
        model = word2vec.Word2Vec(
            sg=0, cbow_mean=1, alpha=0.05, window=5, hs=0, negative=15,
            min_count=5, epochs=10, workers=2, sample=0
        )
        self.model_sanity(model, ranks=ranks)

    def test_cbow_neg_fromfile(self):
        model = word2vec.Word2Vec(
            sg=0, cbow_mean=1, alpha=0.05, window=5, hs=0, negative=15,
            min_count=5, epochs=10, workers=2, sample=0
        )
        self.model_sanity(model, with_corpus_file=True)

    def test_sg_fixedwindowsize(self):
        """Test skipgram with fixed window size. Use NS."""
        model = word2vec.Word2Vec(
            sg=1, window=5, shrink_windows=False, hs=0,
            negative=15, min_count=5, epochs=10, workers=2
        )
        self.model_sanity(model)

    def test_sg_fixedwindowsize_fromfile(self):
        """Test skipgram with fixed window size. Use HS and train from file."""
        model = word2vec.Word2Vec(
            sg=1, window=5, shrink_windows=False, hs=1,
            negative=0, min_count=5, epochs=10, workers=2
        )
        self.model_sanity(model, with_corpus_file=True)

    def test_cbow_fixedwindowsize(self, ranks=None):
        """Test CBOW with fixed window size. Use HS."""
        model = word2vec.Word2Vec(
            sg=0, cbow_mean=1, alpha=0.1, window=5, shrink_windows=False,
            hs=1, negative=0, min_count=5, epochs=10, workers=2
        )
        self.model_sanity(model, ranks=ranks)

    def test_cbow_fixedwindowsize_fromfile(self):
        """Test CBOW with fixed window size. Use NS and train from file."""
        model = word2vec.Word2Vec(
            sg=0, cbow_mean=1, alpha=0.1, window=5, shrink_windows=False,
            hs=0, negative=15, min_count=5, epochs=10, workers=2
        )
        self.model_sanity(model, with_corpus_file=True)

    def test_cosmul(self):
        model = word2vec.Word2Vec(sentences, vector_size=2, min_count=1, hs=1, negative=0)
        sims = model.wv.most_similar_cosmul('graph', topn=10)
        # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # test querying for "most similar" by vector
        graph_vector = model.wv.get_vector('graph', norm=True)
        sims2 = model.wv.most_similar_cosmul(positive=[graph_vector], topn=11)
        sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
        self.assertEqual(sims, sims2)

    def test_training_cbow(self):
        """Test CBOW word2vec training."""
        # to test training, make the corpus larger by repeating its sentences over and over
        # build vocabulary, don't train yet
        model = word2vec.Word2Vec(vector_size=2, min_count=1, sg=0, hs=1, negative=0)
        model.build_vocab(sentences)
        self.assertTrue(model.wv.vectors.shape == (len(model.wv), 2))
        self.assertTrue(model.syn1.shape == (len(model.wv), 2))

        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        sims = model.wv.most_similar('graph', topn=10)
        # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # test querying for "most similar" by vector
        graph_vector = model.wv.get_vector('graph', norm=True)
        sims2 = model.wv.most_similar(positive=[graph_vector], topn=11)
        sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
        self.assertEqual(sims, sims2)

        # build vocab and train in one step; must be the same as above
        model2 = word2vec.Word2Vec(sentences, vector_size=2, min_count=1, sg=0, hs=1, negative=0)
        self.models_equal(model, model2)

    def test_training_sg_negative(self):
        """Test skip-gram (negative sampling) word2vec training."""
        # to test training, make the corpus larger by repeating its sentences over and over
        # build vocabulary, don't train yet
        model = word2vec.Word2Vec(vector_size=2, min_count=1, sg=1, hs=0, negative=2)
        model.build_vocab(sentences)
        self.assertTrue(model.wv.vectors.shape == (len(model.wv), 2))
        self.assertTrue(model.syn1neg.shape == (len(model.wv), 2))

        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        sims = model.wv.most_similar('graph', topn=10)
        # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # test querying for "most similar" by vector
        graph_vector = model.wv.get_vector('graph', norm=True)
        sims2 = model.wv.most_similar(positive=[graph_vector], topn=11)
        sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
        self.assertEqual(sims, sims2)

        # build vocab and train in one step; must be the same as above
        model2 = word2vec.Word2Vec(sentences, vector_size=2, min_count=1, sg=1, hs=0, negative=2)
        self.models_equal(model, model2)

    def test_training_cbow_negative(self):
        """Test CBOW (negative sampling) word2vec training."""
        # to test training, make the corpus larger by repeating its sentences over and over
        # build vocabulary, don't train yet
        model = word2vec.Word2Vec(vector_size=2, min_count=1, sg=0, hs=0, negative=2)
        model.build_vocab(sentences)
        self.assertTrue(model.wv.vectors.shape == (len(model.wv), 2))
        self.assertTrue(model.syn1neg.shape == (len(model.wv), 2))

        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        sims = model.wv.most_similar('graph', topn=10)
        # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # test querying for "most similar" by vector
        graph_vector = model.wv.get_vector('graph', norm=True)
        sims2 = model.wv.most_similar(positive=[graph_vector], topn=11)
        sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
        self.assertEqual(sims, sims2)

        # build vocab and train in one step; must be the same as above
        model2 = word2vec.Word2Vec(sentences, vector_size=2, min_count=1, sg=0, hs=0, negative=2)
        self.models_equal(model, model2)

    def test_similarities(self):
        """Test similarity and n_similarity methods."""
        # The model is trained using CBOW
        model = word2vec.Word2Vec(vector_size=2, min_count=1, sg=0, hs=0, negative=2)
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

        self.assertTrue(model.wv.n_similarity(['graph', 'trees'], ['trees', 'graph']))
        self.assertTrue(model.wv.n_similarity(['graph'], ['trees']) == model.wv.similarity('graph', 'trees'))
        self.assertRaises(ZeroDivisionError, model.wv.n_similarity, ['graph', 'trees'], [])
        self.assertRaises(ZeroDivisionError, model.wv.n_similarity, [], ['graph', 'trees'])
        self.assertRaises(ZeroDivisionError, model.wv.n_similarity, [], [])

    def test_similar_by(self):
        """Test word2vec similar_by_word and similar_by_vector."""
        model = word2vec.Word2Vec(sentences, vector_size=2, min_count=1, hs=1, negative=0)
        wordsims = model.wv.similar_by_word('graph', topn=10)
        wordsims2 = model.wv.most_similar(positive='graph', topn=10)
        vectorsims = model.wv.similar_by_vector(model.wv['graph'], topn=10)
        vectorsims2 = model.wv.most_similar([model.wv['graph']], topn=10)
        self.assertEqual(wordsims, wordsims2)
        self.assertEqual(vectorsims, vectorsims2)

    def test_parallel(self):
        """Test word2vec parallel training."""
        corpus = utils.RepeatCorpus(LeeCorpus(), 10000)  # repeats about 33 times

        for workers in [4, ]:  # [4, 2]
            model = word2vec.Word2Vec(corpus, vector_size=16, min_count=(10 * 33), workers=workers)
            origin_word = 'israeli'
            expected_neighbor = 'palestinian'
            sims = model.wv.most_similar(origin_word, topn=len(model.wv))
            # the exact vectors and therefore similarities may differ, due to different thread collisions/randomization
            # so let's test only for topN
            neighbor_rank = [word for word, sim in sims].index(expected_neighbor)
            self.assertLess(neighbor_rank, 6)

    def test_r_n_g(self):
        """Test word2vec results identical with identical RNG seed."""
        model = word2vec.Word2Vec(sentences, min_count=2, seed=42, workers=1)
        model2 = word2vec.Word2Vec(sentences, min_count=2, seed=42, workers=1)
        self.models_equal(model, model2)

    def models_equal(self, model, model2):
        self.assertEqual(len(model.wv), len(model2.wv))
        self.assertTrue(np.allclose(model.wv.vectors, model2.wv.vectors))
        if model.hs:
            self.assertTrue(np.allclose(model.syn1, model2.syn1))
        if model.negative:
            self.assertTrue(np.allclose(model.syn1neg, model2.syn1neg))
        most_common_word_index = np.argsort(model.wv.expandos['count'])[-1]
        most_common_word = model.wv.index_to_key[most_common_word_index]
        self.assertTrue(np.allclose(model.wv[most_common_word], model2.wv[most_common_word]))

    def test_predict_output_word(self):
        '''Test word2vec predict_output_word method handling for negative sampling scheme'''
        # under normal circumstances
        model_with_neg = word2vec.Word2Vec(sentences, min_count=1)
        predictions_with_neg = model_with_neg.predict_output_word(['system', 'human'], topn=5)
        self.assertTrue(len(predictions_with_neg) == 5)

        # out-of-vobaculary scenario
        predictions_out_of_vocab = model_with_neg.predict_output_word(['some', 'random', 'words'], topn=5)
        self.assertEqual(predictions_out_of_vocab, None)

        # when required model parameters have been deleted
        tmpf = get_tmpfile('gensim_word2vec.tst')
        model_with_neg.wv.save_word2vec_format(tmpf, binary=True)
        kv_model_with_neg = keyedvectors.KeyedVectors.load_word2vec_format(tmpf, binary=True)
        binary_model_with_neg = word2vec.Word2Vec()
        binary_model_with_neg.wv = kv_model_with_neg
        self.assertRaises(RuntimeError, binary_model_with_neg.predict_output_word, ['system', 'human'])

        # negative sampling scheme not used
        model_without_neg = word2vec.Word2Vec(sentences, min_count=1, hs=1, negative=0)
        self.assertRaises(RuntimeError, model_without_neg.predict_output_word, ['system', 'human'])

        # passing indices instead of words in context
        str_context = ['system', 'human']
        mixed_context = [model_with_neg.wv.get_index(str_context[0]), str_context[1]]
        idx_context = [model_with_neg.wv.get_index(w) for w in str_context]
        prediction_from_str = model_with_neg.predict_output_word(str_context, topn=5)
        prediction_from_mixed = model_with_neg.predict_output_word(mixed_context, topn=5)
        prediction_from_idx = model_with_neg.predict_output_word(idx_context, topn=5)
        self.assertEqual(prediction_from_str, prediction_from_mixed)
        self.assertEqual(prediction_from_str, prediction_from_idx)

    def test_load_old_model(self):
        """Test loading an old word2vec model of indeterminate version"""

        model_file = 'word2vec_old'  # which version?!?
        model = word2vec.Word2Vec.load(datapath(model_file))
        self.assertTrue(model.wv.vectors.shape == (12, 100))
        self.assertTrue(len(model.wv) == 12)
        self.assertTrue(len(model.wv.index_to_key) == 12)
        self.assertTrue(model.syn1neg.shape == (len(model.wv), model.wv.vector_size))
        self.assertTrue(len(model.wv.vectors_lockf.shape) > 0)
        self.assertTrue(model.cum_table.shape == (12,))

        self.onlineSanity(model, trained_model=True)

    def test_load_old_model_separates(self):
        """Test loading an old word2vec model of indeterminate version"""

        # Model stored in multiple files
        model_file = 'word2vec_old_sep'
        model = word2vec.Word2Vec.load(datapath(model_file))
        self.assertTrue(model.wv.vectors.shape == (12, 100))
        self.assertTrue(len(model.wv) == 12)
        self.assertTrue(len(model.wv.index_to_key) == 12)
        self.assertTrue(model.syn1neg.shape == (len(model.wv), model.wv.vector_size))
        self.assertTrue(len(model.wv.vectors_lockf.shape) > 0)
        self.assertTrue(model.cum_table.shape == (12,))

        self.onlineSanity(model, trained_model=True)

    def obsolete_test_load_old_models_pre_1_0(self):
        """Test loading pre-1.0 models"""
        # load really old model
        model_file = 'w2v-lee-v0.12.0'
        model = word2vec.Word2Vec.load(datapath(model_file))
        self.onlineSanity(model, trained_model=True)

        old_versions = [
            '0.12.0', '0.12.1', '0.12.2', '0.12.3', '0.12.4',
            '0.13.0', '0.13.1', '0.13.2', '0.13.3', '0.13.4',
        ]

        for old_version in old_versions:
            self._check_old_version(old_version)

    def test_load_old_models_1_x(self):
        """Test loading 1.x models"""

        old_versions = [
            '1.0.0', '1.0.1',
        ]

        for old_version in old_versions:
            self._check_old_version(old_version)

    def test_load_old_models_2_x(self):
        """Test loading 2.x models"""

        old_versions = [
            '2.0.0', '2.1.0', '2.2.0', '2.3.0',
        ]

        for old_version in old_versions:
            self._check_old_version(old_version)

    def test_load_old_models_3_x(self):
        """Test loading 3.x models"""

        # test for max_final_vocab for model saved in 3.3
        model_file = 'word2vec_3.3'
        model = word2vec.Word2Vec.load(datapath(model_file))
        self.assertEqual(model.max_final_vocab, None)
        self.assertEqual(model.max_final_vocab, None)

        old_versions = [
            '3.0.0', '3.1.0', '3.2.0', '3.3.0', '3.4.0'
        ]

        for old_version in old_versions:
            self._check_old_version(old_version)

    def _check_old_version(self, old_version):
        logging.info("TESTING LOAD of %s Word2Vec MODEL", old_version)
        saved_models_dir = datapath('old_w2v_models/w2v_{}.mdl')
        model = word2vec.Word2Vec.load(saved_models_dir.format(old_version))
        self.assertIsNone(model.corpus_total_words)
        self.assertTrue(len(model.wv) == 3)
        try:
            self.assertTrue(model.wv.vectors.shape == (3, 4))
        except AttributeError as ae:
            print("WV")
            print(model.wv)
            print(dir(model.wv))
            print(model.wv.syn0)
            raise ae
        # check if similarity search and online training works.
        self.assertTrue(len(model.wv.most_similar('sentence')) == 2)
        model.build_vocab(lee_corpus_list, update=True)
        model.train(lee_corpus_list, total_examples=model.corpus_count, epochs=model.epochs)
        # check if similarity search and online training works after saving and loading back the model.
        tmpf = get_tmpfile('gensim_word2vec.tst')
        model.save(tmpf)
        loaded_model = word2vec.Word2Vec.load(tmpf)
        loaded_model.build_vocab(lee_corpus_list, update=True)
        loaded_model.train(lee_corpus_list, total_examples=model.corpus_count, epochs=model.epochs)

    @log_capture()
    def test_build_vocab_warning(self, loglines):
        """Test if warning is raised on non-ideal input to a word2vec model"""
        sentences = ['human', 'machine']
        model = word2vec.Word2Vec()
        model.build_vocab(sentences)
        warning = "Each 'sentences' item should be a list of words (usually unicode strings)."
        self.assertTrue(warning in str(loglines))

    @log_capture()
    def test_train_warning(self, loglines):
        """Test if warning is raised if alpha rises during subsequent calls to train()"""
        sentences = [
            ['human'],
            ['graph', 'trees']
        ]
        model = word2vec.Word2Vec(min_count=1)
        model.build_vocab(sentences)
        for epoch in range(10):
            model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
            if epoch == 5:
                model.alpha += 0.05
        warning = "Effective 'alpha' higher than previous training cycles"
        self.assertTrue(warning in str(loglines))

    @log_capture()
    def test_train_hs_and_neg(self, loglines):
        """
        Test if ValueError is raised when both hs=0 and negative=0
        Test if warning is raised if both hs and negative are activated
        """
        with self.assertRaises(ValueError):
            word2vec.Word2Vec(sentences, min_count=1, hs=0, negative=0)

        word2vec.Word2Vec(sentences, min_count=1, hs=1, negative=5)
        warning = "Both hierarchical softmax and negative sampling are activated."
        self.assertTrue(warning in str(loglines))

    def test_train_with_explicit_param(self):
        model = word2vec.Word2Vec(vector_size=2, min_count=1, hs=1, negative=0)
        model.build_vocab(sentences)
        with self.assertRaises(ValueError):
            model.train(sentences, total_examples=model.corpus_count)

        with self.assertRaises(ValueError):
            model.train(sentences, epochs=model.epochs)

        with self.assertRaises(ValueError):
            model.train(sentences)

    def test_sentences_should_not_be_a_generator(self):
        """
        Is sentences a generator object?
        """
        gen = (s for s in sentences)
        self.assertRaises(TypeError, word2vec.Word2Vec, (gen,))

    def test_load_on_class_error(self):
        """Test if exception is raised when loading word2vec model on instance"""
        self.assertRaises(AttributeError, load_on_instance)

    def test_file_should_not_be_compressed(self):
        """
        Is corpus_file a compressed file?
        """
        with tempfile.NamedTemporaryFile(suffix=".bz2") as fp:
            self.assertRaises(TypeError, word2vec.Word2Vec, (None, fp.name))

    def test_reset_from(self):
        """Test if reset_from() uses pre-built structures from other model"""
        model = word2vec.Word2Vec(sentences, min_count=1)
        other_model = word2vec.Word2Vec(new_sentences, min_count=1)
        model.reset_from(other_model)
        self.assertEqual(model.wv.key_to_index, other_model.wv.key_to_index)

    def test_compute_training_loss(self):
        model = word2vec.Word2Vec(min_count=1, sg=1, negative=5, hs=1)
        model.build_vocab(sentences)
        model.train(sentences, compute_loss=True, total_examples=model.corpus_count, epochs=model.epochs)
        training_loss_val = model.get_latest_training_loss()
        self.assertTrue(training_loss_val > 0.0)

    def test_negative_ns_exp(self):
        """The model should accept a negative ns_exponent as a valid value."""
        model = word2vec.Word2Vec(sentences, ns_exponent=-1, min_count=1, workers=1)
        tmpf = get_tmpfile('w2v_negative_exp.tst')
        model.save(tmpf)
        loaded_model = word2vec.Word2Vec.load(tmpf)
        loaded_model.train(sentences, total_examples=model.corpus_count, epochs=1)
        assert loaded_model.ns_exponent == -1, loaded_model.ns_exponent


# endclass TestWord2VecModel

class TestWMD(unittest.TestCase):

    @unittest.skipIf(POT_EXT is False, "POT not installed")
    def test_nonzero(self):
        '''Test basic functionality with a test sentence.'''

        model = word2vec.Word2Vec(sentences, min_count=2, seed=42, workers=1)
        sentence1 = ['human', 'interface', 'computer']
        sentence2 = ['survey', 'user', 'computer', 'system', 'response', 'time']
        distance = model.wv.wmdistance(sentence1, sentence2)

        # Check that distance is non-zero.
        self.assertFalse(distance == 0.0)

    @unittest.skipIf(POT_EXT is False, "POT not installed")
    def test_symmetry(self):
        '''Check that distance is symmetric.'''

        model = word2vec.Word2Vec(sentences, min_count=2, seed=42, workers=1)
        sentence1 = ['human', 'interface', 'computer']
        sentence2 = ['survey', 'user', 'computer', 'system', 'response', 'time']
        distance1 = model.wv.wmdistance(sentence1, sentence2)
        distance2 = model.wv.wmdistance(sentence2, sentence1)
        self.assertTrue(np.allclose(distance1, distance2))

    @unittest.skipIf(POT_EXT is False, "POT not installed")
    def test_identical_sentences(self):
        '''Check that the distance from a sentence to itself is zero.'''

        model = word2vec.Word2Vec(sentences, min_count=1)
        sentence = ['survey', 'user', 'computer', 'system', 'response', 'time']
        distance = model.wv.wmdistance(sentence, sentence)
        self.assertEqual(0.0, distance)


class TestWord2VecSentenceIterators(unittest.TestCase):
    def test_line_sentence_works_with_filename(self):
        """Does LineSentence work with a filename argument?"""
        with utils.open(datapath('lee_background.cor'), 'rb') as orig:
            sentences = word2vec.LineSentence(datapath('lee_background.cor'))
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig.readline()).split())

    def test_cython_line_sentence_works_with_filename(self):
        """Does CythonLineSentence work with a filename argument?"""
        from gensim.models import word2vec_corpusfile
        with utils.open(datapath('lee_background.cor'), 'rb') as orig:
            sentences = word2vec_corpusfile.CythonLineSentence(datapath('lee_background.cor'))
            for words in sentences:
                self.assertEqual(words, orig.readline().split())

    def test_line_sentence_works_with_compressed_file(self):
        """Does LineSentence work with a compressed file object argument?"""
        with utils.open(datapath('head500.noblanks.cor'), 'rb') as orig:
            sentences = word2vec.LineSentence(bz2.BZ2File(datapath('head500.noblanks.cor.bz2')))
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig.readline()).split())

    def test_line_sentence_works_with_normal_file(self):
        """Does LineSentence work with a file object argument, rather than filename?"""
        with utils.open(datapath('head500.noblanks.cor'), 'rb') as orig:
            with utils.open(datapath('head500.noblanks.cor'), 'rb') as fin:
                sentences = word2vec.LineSentence(fin)
                for words in sentences:
                    self.assertEqual(words, utils.to_unicode(orig.readline()).split())

    def test_path_line_sentences(self):
        """Does PathLineSentences work with a path argument?"""
        with utils.open(os.path.join(datapath('PathLineSentences'), '1.txt'), 'rb') as orig1:
            with utils.open(os.path.join(datapath('PathLineSentences'), '2.txt.bz2'), 'rb') as orig2:
                sentences = word2vec.PathLineSentences(datapath('PathLineSentences'))
                orig = orig1.readlines() + orig2.readlines()
                orig_counter = 0  # to go through orig while matching PathLineSentences
                for words in sentences:
                    self.assertEqual(words, utils.to_unicode(orig[orig_counter]).split())
                    orig_counter += 1

    def test_path_line_sentences_one_file(self):
        """Does PathLineSentences work with a single file argument?"""
        test_file = os.path.join(datapath('PathLineSentences'), '1.txt')
        with utils.open(test_file, 'rb') as orig:
            sentences = word2vec.PathLineSentences(test_file)
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig.readline()).split())


# endclass TestWord2VecSentenceIterators


class TestWord2VecScripts(unittest.TestCase):
    def test_word2vec_stand_alone_script(self):
        """Does Word2Vec script launch standalone?"""
        cmd = [
            sys.executable, '-m', 'gensim.scripts.word2vec_standalone',
            '-train', datapath('testcorpus.txt'),
            '-output', 'vec.txt', '-size', '200', '-sample', '1e-4',
            '-binary', '0', '-iter', '3', '-min_count', '1',
        ]
        output = check_output(args=cmd, stderr=subprocess.PIPE)
        self.assertEqual(output, b'')


if not hasattr(TestWord2VecModel, 'assertLess'):
    # workaround for python 2.6
    def assertLess(self, a, b, msg=None):
        self.assertTrue(a < b, msg="%s is not less than %s" % (a, b))

    setattr(TestWord2VecModel, 'assertLess', assertLess)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.DEBUG
    )
    unittest.main(module='gensim.test.test_word2vec')
