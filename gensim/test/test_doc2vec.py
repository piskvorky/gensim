#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""


from __future__ import with_statement

import logging
import unittest
import os
import tempfile

from six.moves import zip as izip
from collections import namedtuple
from testfixtures import log_capture

import numpy as np

from gensim import utils, matutils
from gensim.models import doc2vec, keyedvectors

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


class DocsLeeCorpus(object):
    def __init__(self, string_tags=False):
        self.string_tags = string_tags

    def _tag(self, i):
        return i if not self.string_tags else '_*%d' % i

    def __iter__(self):
        with open(datapath('lee_background.cor')) as f:
            for i, line in enumerate(f):
                yield doc2vec.TaggedDocument(utils.simple_preprocess(line), [self._tag(i)])

list_corpus = list(DocsLeeCorpus())

raw_sentences = [
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

sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(raw_sentences)]


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_doc2vec.tst')

def load_on_instance():
    # Save and load a Doc2Vec Model on instance for test
    model = doc2vec.Doc2Vec(DocsLeeCorpus(), min_count=1)
    model.save(testfile())
    model = doc2vec.Doc2Vec() # should fail at this point
    return model.load(testfile())

class TestDoc2VecModel(unittest.TestCase):
    def test_persistence(self):
        """Test storing/loading the entire model."""
        model = doc2vec.Doc2Vec(DocsLeeCorpus(), min_count=1)
        model.save(testfile())
        self.models_equal(model, doc2vec.Doc2Vec.load(testfile()))

    def testPersistenceWord2VecFormat(self):
        """Test storing the entire model in word2vec format."""
        model = doc2vec.Doc2Vec(DocsLeeCorpus(), min_count=1)
        # test saving both document and word embedding
        test_doc_word = os.path.join(tempfile.gettempdir(), 'gensim_doc2vec.dw')
        model.save_word2vec_format(test_doc_word, doctag_vec=True, word_vec=True, binary=True)
        binary_model_dv = keyedvectors.KeyedVectors.load_word2vec_format(test_doc_word, binary=True)
        self.assertEqual(len(model.wv.vocab) + len(model.docvecs), len(binary_model_dv.vocab))
        # test saving document embedding only
        test_doc = os.path.join(tempfile.gettempdir(), 'gensim_doc2vec.d')
        model.save_word2vec_format(test_doc, doctag_vec=True, word_vec=False, binary=True)
        binary_model_dv = keyedvectors.KeyedVectors.load_word2vec_format(test_doc, binary=True)
        self.assertEqual(len(model.docvecs), len(binary_model_dv.vocab))
        # test saving word embedding only
        test_word = os.path.join(tempfile.gettempdir(), 'gensim_doc2vec.w')
        model.save_word2vec_format(test_word, doctag_vec=False, word_vec=True, binary=True)
        binary_model_dv = keyedvectors.KeyedVectors.load_word2vec_format(test_word, binary=True)
        self.assertEqual(len(model.wv.vocab), len(binary_model_dv.vocab))

    def test_load_mmap(self):
        """Test storing/loading the entire model."""
        model = doc2vec.Doc2Vec(sentences, min_count=1)

        # test storing the internal arrays into separate files
        model.save(testfile(), sep_limit=0)
        self.models_equal(model, doc2vec.Doc2Vec.load(testfile()))

        # make sure mmaping the arrays back works, too
        self.models_equal(model, doc2vec.Doc2Vec.load(testfile(), mmap='r'))

    def test_int_doctags(self):
        """Test doc2vec doctag alternatives"""
        corpus = DocsLeeCorpus()

        model = doc2vec.Doc2Vec(min_count=1)
        model.build_vocab(corpus)
        self.assertEqual(len(model.docvecs.doctag_syn0), 300)
        self.assertEqual(model.docvecs[0].shape, (100,))
        self.assertEqual(model.docvecs[np.int64(0)].shape, (100,))
        self.assertRaises(KeyError, model.__getitem__, '_*0')

    def test_missing_string_doctag(self):
        """Test doc2vec doctag alternatives"""
        corpus = list(DocsLeeCorpus(True))
        # force duplicated tags
        corpus = corpus[0:10] + corpus

        model = doc2vec.Doc2Vec(min_count=1)
        model.build_vocab(corpus)
        self.assertRaises(KeyError, model.docvecs.__getitem__, 'not_a_tag')

    def test_string_doctags(self):
        """Test doc2vec doctag alternatives"""
        corpus = list(DocsLeeCorpus(True))
        # force duplicated tags
        corpus = corpus[0:10] + corpus

        model = doc2vec.Doc2Vec(min_count=1)
        model.build_vocab(corpus)

        self.assertEqual(len(model.docvecs.doctag_syn0), 300)
        self.assertEqual(model.docvecs[0].shape, (100,))
        self.assertEqual(model.docvecs['_*0'].shape, (100,))
        self.assertTrue(all(model.docvecs['_*0'] == model.docvecs[0]))
        self.assertTrue(max(d.offset for d in model.docvecs.doctags.values()) < len(model.docvecs.doctags))
        self.assertTrue(max(model.docvecs._int_index(str_key) for str_key in model.docvecs.doctags.keys()) < len(model.docvecs.doctag_syn0))
        # verify docvecs.most_similar() returns string doctags rather than indexes
        self.assertEqual(model.docvecs.offset2doctag[0], model.docvecs.most_similar([model.docvecs[0]])[0][0])

    def test_empty_errors(self):
        # no input => "RuntimeError: you must first build vocabulary before training the model"
        self.assertRaises(RuntimeError, doc2vec.Doc2Vec, [])

        # input not empty, but rather completely filtered out
        self.assertRaises(RuntimeError, doc2vec.Doc2Vec, list_corpus, min_count=10000)

    def test_similarity_unseen_docs(self):
        """Test similarity of out of training sentences"""
        rome_str = ['rome', 'italy']
        car_str = ['car']
        corpus = list(DocsLeeCorpus(True))

        model = doc2vec.Doc2Vec(min_count=1)
        model.build_vocab(corpus)
        self.assertTrue(model.docvecs.similarity_unseen_docs(model, rome_str, rome_str) > model.docvecs.similarity_unseen_docs(model, rome_str, car_str))

    def model_sanity(self, model, keep_training=True):
        """Any non-trivial model on DocsLeeCorpus can pass these sanity checks"""
        fire1 = 0  # doc 0 sydney fires
        fire2 = 8  # doc 8 sydney fires
        tennis1 = 6  # doc 6 tennis

        # inferred vector should be top10 close to bulk-trained one
        doc0_inferred = model.infer_vector(list(DocsLeeCorpus())[0].words)
        sims_to_infer = model.docvecs.most_similar([doc0_inferred], topn=len(model.docvecs))
        f_rank = [docid for docid, sim in sims_to_infer].index(fire1)
        self.assertLess(f_rank, 10)

        # fire2 should be top30 close to fire1
        sims = model.docvecs.most_similar(fire1, topn=len(model.docvecs))
        f2_rank = [docid for docid, sim in sims].index(fire2)
        self.assertLess(f2_rank, 30)

        # same sims should appear in lookup by vec as by index
        doc0_vec = model.docvecs[fire1]
        sims2 = model.docvecs.most_similar(positive=[doc0_vec], topn=21)
        sims2 = [(id, sim) for id, sim in sims2 if id != fire1]  # ignore the doc itself
        sims = sims[:20]
        self.assertEqual(list(zip(*sims))[0], list(zip(*sims2))[0])  # same doc ids
        self.assertTrue(np.allclose(list(zip(*sims))[1], list(zip(*sims2))[1]))  # close-enough dists

        # sim results should be in clip range if given
        clip_sims = model.docvecs.most_similar(fire1, clip_start=len(model.docvecs) // 2, clip_end=len(model.docvecs) * 2 // 3)
        sims_doc_id = [docid for docid, sim in clip_sims]
        for s_id in sims_doc_id:
            self.assertTrue(len(model.docvecs) // 2 <= s_id <= len(model.docvecs) * 2 // 3)

        # tennis doc should be out-of-place among fire news
        self.assertEqual(model.docvecs.doesnt_match([fire1, tennis1, fire2]), tennis1)

        # fire docs should be closer than fire-tennis
        self.assertTrue(model.docvecs.similarity(fire1, fire2) > model.docvecs.similarity(fire1, tennis1))

        # keep training after save
        if keep_training:
            model.save(testfile())
            loaded = doc2vec.Doc2Vec.load(testfile())
            loaded.train(sentences, total_examples=loaded.corpus_count, epochs=loaded.iter)

    def test_training(self):
        """Test doc2vec training."""
        corpus = DocsLeeCorpus()
        model = doc2vec.Doc2Vec(size=100, min_count=2, iter=20, workers=1)
        model.build_vocab(corpus)
        self.assertEqual(model.docvecs.doctag_syn0.shape, (300, 100))
        model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)

        self.model_sanity(model)

        # build vocab and train in one step; must be the same as above
        model2 = doc2vec.Doc2Vec(corpus, size=100, min_count=2, iter=20, workers=1)
        self.models_equal(model, model2)

    def test_dbow_hs(self):
        """Test DBOW doc2vec training."""
        model = doc2vec.Doc2Vec(list_corpus, dm=0, hs=1, negative=0, min_count=2, iter=20)
        self.model_sanity(model)

    def test_dmm_hs(self):
        """Test DM/mean doc2vec training."""
        model = doc2vec.Doc2Vec(list_corpus, dm=1, dm_mean=1, size=24, window=4, hs=1, negative=0,
                                alpha=0.05, min_count=2, iter=20)
        self.model_sanity(model)

    def test_dms_hs(self):
        """Test DM/sum doc2vec training."""
        model = doc2vec.Doc2Vec(list_corpus, dm=1, dm_mean=0, size=24, window=4, hs=1, negative=0,
                                alpha=0.05, min_count=2, iter=20)
        self.model_sanity(model)

    def test_dmc_hs(self):
        """Test DM/concatenate doc2vec training."""
        model = doc2vec.Doc2Vec(list_corpus, dm=1, dm_concat=1, size=24, window=4, hs=1, negative=0,
                                alpha=0.05, min_count=2, iter=20)
        self.model_sanity(model)

    def test_dbow_neg(self):
        """Test DBOW doc2vec training."""
        model = doc2vec.Doc2Vec(list_corpus, dm=0, hs=0, negative=10, min_count=2, iter=20)
        self.model_sanity(model)

    def test_dmm_neg(self):
        """Test DM/mean doc2vec training."""
        model = doc2vec.Doc2Vec(list_corpus, dm=1, dm_mean=1, size=24, window=4, hs=0, negative=10,
                                alpha=0.05, min_count=2, iter=20)
        self.model_sanity(model)

    def test_dms_neg(self):
        """Test DM/sum doc2vec training."""
        model = doc2vec.Doc2Vec(list_corpus, dm=1, dm_mean=0, size=24, window=4, hs=0, negative=10,
                                alpha=0.05, min_count=2, iter=20)
        self.model_sanity(model)

    def test_dmc_neg(self):
        """Test DM/concatenate doc2vec training."""
        model = doc2vec.Doc2Vec(list_corpus, dm=1, dm_concat=1, size=24, window=4, hs=0, negative=10,
                                alpha=0.05, min_count=2, iter=20)
        self.model_sanity(model)

    def test_parallel(self):
        """Test doc2vec parallel training."""
        if doc2vec.FAST_VERSION < 0:  # don't test the plain NumPy version for parallelism (too slow)
            return

        corpus = utils.RepeatCorpus(DocsLeeCorpus(), 10000)

        for workers in [2, 4]:
            model = doc2vec.Doc2Vec(corpus, workers=workers)
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
        model = doc2vec.Doc2Vec(DocsLeeCorpus(), dm=1, dm_concat=1, size=24, window=4, hs=1, negative=3,
                                seed=42, workers=1)
        model2 = doc2vec.Doc2Vec(DocsLeeCorpus(), dm=1, dm_concat=1, size=24, window=4, hs=1, negative=3,
                                 seed=42, workers=1)
        self.models_equal(model, model2)

    def test_mixed_tag_types(self):
        """Ensure alternating int/string tags don't share indexes in doctag_syn0"""
        mixed_tag_corpus = [doc2vec.TaggedDocument(words, [i, words[0]]) for i, words in enumerate(raw_sentences)]
        model = doc2vec.Doc2Vec()
        model.build_vocab(mixed_tag_corpus)
        expected_length = len(sentences) + len(model.docvecs.doctags)  # 9 sentences, 7 unique first tokens
        self.assertEqual(len(model.docvecs.doctag_syn0), expected_length)

    def models_equal(self, model, model2):
        # check words/hidden-weights
        self.assertEqual(len(model.wv.vocab), len(model2.wv.vocab))
        self.assertTrue(np.allclose(model.wv.syn0, model2.wv.syn0))
        if model.hs:
            self.assertTrue(np.allclose(model.syn1, model2.syn1))
        if model.negative:
            self.assertTrue(np.allclose(model.syn1neg, model2.syn1neg))
        # check docvecs
        self.assertEqual(len(model.docvecs.doctags), len(model2.docvecs.doctags))
        self.assertEqual(len(model.docvecs.offset2doctag), len(model2.docvecs.offset2doctag))
        self.assertTrue(np.allclose(model.docvecs.doctag_syn0, model2.docvecs.doctag_syn0))

    def test_delete_temporary_training_data(self):
        """Test doc2vec model after delete_temporary_training_data"""
        for i in [0, 1]:
            for j in [0, 1]:
                model = doc2vec.Doc2Vec(sentences, size=5, min_count=1, window=4, hs=i, negative=j)
                if i:
                    self.assertTrue(hasattr(model, 'syn1'))
                if j:
                    self.assertTrue(hasattr(model, 'syn1neg'))
                self.assertTrue(hasattr(model, 'syn0_lockf'))
                model.delete_temporary_training_data(keep_doctags_vectors=False, keep_inference=False)
                self.assertTrue(len(model['human']), 10)
                self.assertTrue(model.wv.vocab['graph'].count, 5)
                self.assertTrue(not hasattr(model, 'syn1'))
                self.assertTrue(not hasattr(model, 'syn1neg'))
                self.assertTrue(not hasattr(model, 'syn0_lockf'))
                self.assertTrue(model.docvecs and not hasattr(model.docvecs, 'doctag_syn0'))
                self.assertTrue(model.docvecs and not hasattr(model.docvecs, 'doctag_syn0_lockf'))
        model = doc2vec.Doc2Vec(list_corpus, dm=1, dm_mean=1, size=24, window=4, hs=1, negative=0, alpha=0.05, min_count=2, iter=20)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        self.assertTrue(model.docvecs and hasattr(model.docvecs, 'doctag_syn0'))
        self.assertTrue(hasattr(model, 'syn1'))
        self.model_sanity(model, keep_training=False)
        model = doc2vec.Doc2Vec(list_corpus, dm=1, dm_mean=1, size=24, window=4, hs=0, negative=1, alpha=0.05, min_count=2, iter=20)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        self.model_sanity(model, keep_training=False)
        self.assertTrue(hasattr(model, 'syn1neg'))

    @log_capture()
    def testBuildVocabWarning(self, l):
        """Test if logger warning is raised on non-ideal input to a doc2vec model"""
        raw_sentences = ['human', 'machine']
        sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(raw_sentences)]
        model = doc2vec.Doc2Vec()
        model.build_vocab(sentences)
        warning = "Each 'words' should be a list of words (usually unicode strings)."
        self.assertTrue(warning in str(l))

    @log_capture()
    def testTrainWarning(self, l):
        """Test if warning is raised if alpha rises during subsequent calls to train()"""
        raw_sentences = [['human'],
                         ['graph', 'trees']]
        sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(raw_sentences)]
        model = doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.025, min_count=1, workers=8, size=5)
        model.build_vocab(sentences)
        for epoch in range(10):
            model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
            if epoch == 5:
                model.alpha += 0.05
        warning = "Effective 'alpha' higher than previous training cycles"
        self.assertTrue(warning in str(l))
        
    def testLoadOnClassError(self):
        """Test if exception is raised when loading doc2vec model on instance"""
        self.assertRaises(AttributeError, load_on_instance)
#endclass TestDoc2VecModel


if not hasattr(TestDoc2VecModel, 'assertLess'):
    # workaround for python 2.6
    def assertLess(self, a, b, msg=None):
        self.assertTrue(a < b, msg="%s is not less than %s" % (a, b))

    setattr(TestDoc2VecModel, 'assertLess', assertLess)


# following code is useful for reproducing paragraph-vectors paper sentiment experiments

class ConcatenatedDoc2Vec(object):
    """
    Concatenation of multiple models for reproducing the Paragraph Vectors paper.
    Models must have exactly-matching vocabulary and document IDs. (Models should
    be trained separately; this wrapper just returns concatenated results.)
    """
    def __init__(self, models):
        self.models = models
        if hasattr(models[0], 'docvecs'):
            self.docvecs = ConcatenatedDocvecs([model.docvecs for model in models])

    def __getitem__(self, token):
        return np.concatenate([model[token] for model in self.models])

    def infer_vector(self, document, alpha=0.1, min_alpha=0.0001, steps=5):
        return np.concatenate([model.infer_vector(document, alpha, min_alpha, steps) for model in self.models])

    def train(self, ignored):
        pass  # train subcomponents individually


class ConcatenatedDocvecs(object):
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
    logging.info("loading corpus from %s" % dirname)

    # many mangled chars in sentences (datasetSentences.txt)
    chars_sst_mangled = ['à', 'á', 'â', 'ã', 'æ', 'ç', 'è', 'é', 'í',
                         'í', 'ï', 'ñ', 'ó', 'ô', 'ö', 'û', 'ü']
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
            for sentence_line, split_line in izip(sentences, splits):
                (id, text) = sentence_line.split('\t')
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

    assert len([phrase for phrase in phrases if phrase.sentence_id is not None]) == len(info_by_sentence)  # all
    # counts don't match 8544, 2210, 1101 because 13 TRAIN and 1 DEV sentences are duplicates
    assert len([phrase for phrase in phrases if phrase.split == 'train']) == 8531  # 'train'
    assert len([phrase for phrase in phrases if phrase.split == 'test']) == 2210  # 'test'
    assert len([phrase for phrase in phrases if phrase.split == 'dev']) == 1100  # 'dev'

    logging.info("loaded corpus with %i sentences and %i phrases from %s",
                 len(info_by_sentence), len(phrases), dirname)

    return phrases


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    logging.info("using optimization %s", doc2vec.FAST_VERSION)
    unittest.main()
