#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Tests for Vowpal Wabbit LDA wrapper.

Will not be run unless the environment variable 'VOWPAL_WABBIT_PATH' is set
and points to the `vw` executable.
"""


import logging
import unittest
import os
import os.path
import tempfile
from collections import defaultdict

import six

from gensim.corpora import Dictionary

import gensim.models.wrappers.ldavowpalwabbit as ldavowpalwabbit
from gensim.models.wrappers.ldavowpalwabbit import LdaVowpalWabbit


module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder


def datapath(fname):
    return os.path.join(module_path, 'test_data', fname)


# set up vars used in testing ("Deerwester" from the web tutorial)
TOPIC_WORDS = [
    'cat lion leopard mouse jaguar lynx cheetah tiger kitten puppy'.split(),
    'engine car wheel brakes tyre motor suspension cylinder exhaust clutch'.split(),
    'alice bob robert tim sue rachel dave harry alex jim'.split(),
    'c cplusplus go python haskell scala java ruby csharp erlang'.split(),
    'eggs ham mushrooms cereal coffee beans tea juice sausages bacon'.split()
]


def get_corpus():
    text_path = datapath('ldavowpalwabbit.txt')
    dict_path = datapath('ldavowpalwabbit.dict.txt')
    dictionary = Dictionary.load_from_text(dict_path)
    with open(text_path) as fhandle:
        corpus = [dictionary.doc2bow(l.strip().split()) for l in fhandle]
    return corpus, dictionary


class TestLdaVowpalWabbit(unittest.TestCase):
    def setUp(self):
        vw_path = os.environ.get('VOWPAL_WABBIT_PATH', None)
        if not vw_path:
            msg = "Environment variable 'VOWPAL_WABBIT_PATH' not specified, skipping tests"

            try:
                raise unittest.SkipTest(msg)
            except AttributeError:
                # couldn't find a way of skipping tests in python 2.6
                self.vw_path = None

        corpus, dictionary = get_corpus()
        self.vw_path = vw_path
        self.corpus = corpus
        self.dictionary = dictionary

    def test_save_load(self):
        """Test loading/saving LdaVowpalWabbit model."""
        if not self.vw_path:  # for python 2.6
            return
        lda = LdaVowpalWabbit(
            self.vw_path, corpus=self.corpus, passes=10, chunksize=256,
            id2word=self.dictionary, cleanup_files=True, alpha=0.1,
            eta=0.1, num_topics=len(TOPIC_WORDS), random_seed=1
        )

        with tempfile.NamedTemporaryFile() as fhandle:
            lda.save(fhandle.name)
            lda2 = LdaVowpalWabbit.load(fhandle.name)

            # ensure public fields are saved/loaded correctly
            saved_fields = [
                lda.alpha, lda.chunksize, lda.cleanup_files,
                lda.decay, lda.eta, lda.gamma_threshold,
                lda.id2word, lda.num_terms, lda.num_topics,
                lda.passes, lda.random_seed, lda.vw_path
            ]
            loaded_fields = [
                lda2.alpha, lda2.chunksize, lda2.cleanup_files,
                lda2.decay, lda2.eta, lda2.gamma_threshold,
                lda2.id2word, lda2.num_terms, lda2.num_topics,
                lda2.passes, lda2.random_seed, lda2.vw_path
            ]
            self.assertEqual(saved_fields, loaded_fields)

            # ensure topic matrices are saved/loaded correctly
            saved_topics = lda.show_topics(num_topics=5, num_words=10)
            loaded_topics = lda2.show_topics(num_topics=5, num_words=10)
            self.assertEqual(loaded_topics, saved_topics)

    def test_model_update(self):
        """Test updating existing LdaVowpalWabbit model."""
        if not self.vw_path:  # for python 2.6
            return
        lda = LdaVowpalWabbit(
            self.vw_path, corpus=[self.corpus[0]], passes=10, chunksize=256,
            id2word=self.dictionary, cleanup_files=True, alpha=0.1,
            eta=0.1, num_topics=len(TOPIC_WORDS), random_seed=1
        )

        lda.update(self.corpus[1:])
        result = lda.log_perplexity(self.corpus)
        self.assertTrue(result < -1)
        self.assertTrue(result > -5)

    def test_perplexity(self):
        """Test LdaVowpalWabbit perplexity is within expected range."""
        if not self.vw_path:  # for python 2.6
            return
        lda = LdaVowpalWabbit(
            self.vw_path, corpus=self.corpus, passes=10, chunksize=256,
            id2word=self.dictionary, cleanup_files=True, alpha=0.1,
            eta=0.1, num_topics=len(TOPIC_WORDS), random_seed=1)

        # varies, but should be between -1 and -5
        result = lda.log_perplexity(self.corpus)
        self.assertTrue(result < -1)
        self.assertTrue(result > -5)

    def test_topic_coherence(self):
        """Test LdaVowpalWabbit topic coherence."""
        if not self.vw_path:  # for python 2.6
            return
        corpus, dictionary = get_corpus()
        lda = LdaVowpalWabbit(
            self.vw_path, corpus=corpus, passes=10, chunksize=256,
            id2word=dictionary, cleanup_files=True, alpha=0.1,
            eta=0.1, num_topics=len(TOPIC_WORDS), random_seed=1
        )
        lda.print_topics(5, 10)

        # map words in known topic to an ID
        topic_map = {}
        for i, words in enumerate(TOPIC_WORDS):
            topic_map[frozenset(words)] = i

        n_coherent = 0
        for topic_id in range(lda.num_topics):
            topic = lda.show_topic(topic_id, topn=20)

            # get all words from LDA topic
            topic_words = [w[1] for w in topic]

            # get list of original topics that each word actually belongs to
            ids = []
            for word in topic_words:
                for src_topic_words, src_topic_id in six.iteritems(topic_map):
                    if word in src_topic_words:
                        ids.append(src_topic_id)

            # count the number of times each original topic appears
            counts = defaultdict(int)
            for found_topic_id in ids:
                counts[found_topic_id] += 1

            # if at least 6/10 words assigned to same topic, consider it coherent
            max_count = 0
            for count in six.itervalues(counts):
                max_count = max(max_count, count)

            if max_count >= 6:
                n_coherent += 1

        # not 100% deterministic, but should always get 3+ coherent topics
        self.assertTrue(n_coherent >= 3)

    def test_corpus_to_vw(self):
        """Test corpus to Vowpal Wabbit format conversion."""
        if not self.vw_path:  # for python 2.6
            return
        corpus = [
            [(0, 5), (7, 1), (5, 3), (0, 2)],
            [(7, 2), (2, 1), (3, 11)],
            [(1, 1)],
            [],
            [(5, 2), (0, 1)]
        ]
        expected = """
| 0:5 7:1 5:3 0:2
| 7:2 2:1 3:11
| 1:1
|
| 5:2 0:1
""".strip()
        result = '\n'.join(ldavowpalwabbit.corpus_to_vw(corpus))
        self.assertEqual(result, expected)

    def testvwmodel2ldamodel(self):
        """Test copying of VWModel to LdaModel"""
        if not self.vw_path:
            return
        tm1 = LdaVowpalWabbit(vw_path=self.vw_path, corpus=self.corpus, num_topics=2, id2word=self.dictionary)
        tm2 = ldavowpalwabbit.vwmodel2ldamodel(tm1)
        for document in self.corpus:
            element1_1, element1_2 = tm1[document][0]
            element2_1, element2_2 = tm2[document][0]
            self.assertAlmostEqual(element1_1, element2_1)
            self.assertAlmostEqual(element1_2, element2_2, 5)
            logging.debug('%d %d', element1_1, element2_1)
            logging.debug('%d %d', element1_2, element2_2)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
