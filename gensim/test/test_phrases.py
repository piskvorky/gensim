#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""


import logging
import unittest
import os
import sys

from gensim import utils
from gensim.models.phrases import Phrases, Phraser

if sys.version_info[0] >= 3:
    unicode = str

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


sentences = [
    ['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey'],
    ['graph', 'minors', 'survey', 'human', 'interface']  # test bigrams within same sentence
]
unicode_sentences = [[utils.to_unicode(w) for w in sentence] for sentence in sentences]


def gen_sentences():
    return ((w for w in sentence) for sentence in sentences)


class TestPhrasesCommon(unittest.TestCase):
    """ Tests that need to be run for both Prases and Phraser classes."""

    def setUp(self):
        self.bigram = Phrases(sentences, min_count=1, threshold=1)
        self.bigram_default = Phrases(sentences)
        self.bigram_utf8 = Phrases(sentences, min_count=1, threshold=1)
        self.bigram_unicode = Phrases(unicode_sentences, min_count=1, threshold=1)

    def testEmptyInputsOnBigramConstruction(self):
        """Test that empty inputs don't throw errors and return the expected result."""
        # Empty list -> empty list
        self.assertEqual(list(self.bigram_default[[]]), [])
        # Empty iterator -> empty list
        self.assertEqual(list(self.bigram_default[iter(())]), [])
        # List of empty list -> list of empty list
        self.assertEqual(list(self.bigram_default[[[], []]]), [[], []])
        # Iterator of empty list -> list of empty list
        self.assertEqual(list(self.bigram_default[iter([[], []])]), [[], []])
        # Iterator of empty iterator -> list of empty list
        self.assertEqual(list(self.bigram_default[(iter(()) for i in range(2))]), [[], []])

    def testSentenceGeneration(self):
        """Test basic bigram using a dummy corpus."""
        # test that we generate the same amount of sentences as the input
        self.assertEqual(len(sentences), len(list(self.bigram_default[sentences])))

    def testSentenceGenerationWithGenerator(self):
        """Test basic bigram production when corpus is a generator."""
        self.assertEqual(len(list(gen_sentences())),
                         len(list(self.bigram_default[gen_sentences()])))

    def testBigramConstruction(self):
        """Test Phrases bigram construction building."""
        # with this setting we should get response_time and graph_minors
        bigram1_seen = False
        bigram2_seen = False

        for s in self.bigram[sentences]:
            if not bigram1_seen and u'response_time' in s:
                bigram1_seen = True
            if not bigram2_seen and u'graph_minors' in s:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break

        self.assertTrue(bigram1_seen and bigram2_seen)

        # check the same thing, this time using single doc transformation
        # last sentence should contain both graph_minors and human_interface
        self.assertTrue(u'response_time' in self.bigram[sentences[1]])
        self.assertTrue(u'response_time' in self.bigram[sentences[4]])
        self.assertTrue(u'graph_minors' in self.bigram[sentences[-2]])
        self.assertTrue(u'graph_minors' in self.bigram[sentences[-1]])
        self.assertTrue(u'human_interface' in self.bigram[sentences[-1]])

    def testBigramConstructionFromGenerator(self):
        """Test Phrases bigram construction building when corpus is a generator"""
        bigram1_seen = False
        bigram2_seen = False

        for s in self.bigram[gen_sentences()]:
            if not bigram1_seen and 'response_time' in s:
                bigram1_seen = True
            if not bigram2_seen and 'graph_minors' in s:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break
        self.assertTrue(bigram1_seen and bigram2_seen)

    def testEncoding(self):
        """Test that both utf8 and unicode input work; output must be unicode."""
        expected = [u'survey', u'user', u'computer', u'system', u'response_time']

        self.assertEqual(self.bigram_utf8[sentences[1]], expected)
        self.assertEqual(self.bigram_unicode[sentences[1]], expected)

        transformed = ' '.join(self.bigram_utf8[sentences[1]])
        self.assertTrue(isinstance(transformed, unicode))


class TestPhrasesModel(unittest.TestCase):
    def testExportPhrases(self):
        """Test Phrases bigram export_phrases functionality."""
        bigram = Phrases(sentences, min_count=1, threshold=1)

        seen_bigrams = set()

        for phrase, score in bigram.export_phrases(sentences):
            seen_bigrams.add(phrase)

        assert seen_bigrams == {b'response time', b'graph minors', b'human interface'}

    def testMultipleBigramsSingleEntry(self):
        """ a single entry should produce multiple bigrams. """
        bigram = Phrases(sentences, min_count=1, threshold=1)

        seen_bigrams = set()

        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface']]
        for phrase, score in bigram.export_phrases(test_sentences):
            seen_bigrams.add(phrase)

        assert seen_bigrams == {b'graph minors', b'human interface'}

    def testScoringDefault(self):
        """ test the default scoring, from the mikolov word2vec paper """
        bigram = Phrases(sentences, min_count=1, threshold=1)

        seen_scores = set()

        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface']]
        for phrase, score in bigram.export_phrases(test_sentences):
            seen_scores.add(round(score, 3))

        assert seen_scores == {
            5.167,  # score for graph minors
            3.444  # score for human interface
        }

    def testScoringNpmi(self):
        """ test normalized pointwise mutual information scoring """
        bigram = Phrases(sentences, min_count=1, threshold=.5, scoring='npmi')

        seen_scores = set()

        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface']]
        for phrase, score in bigram.export_phrases(test_sentences):
            seen_scores.add(round(score, 3))

        assert seen_scores == {
            .882,  # score for graph minors
            .714  # score for human interface
        }

    def testBadParameters(self):
        """Test the phrases module with bad parameters."""
        # should fail with something less or equal than 0
        self.assertRaises(ValueError, Phrases, sentences, min_count=0)

        # threshold should be positive
        self.assertRaises(ValueError, Phrases, sentences, threshold=-1)

    def testPruning(self):
        """Test that max_vocab_size parameter is respected."""
        bigram = Phrases(sentences, max_vocab_size=5)
        self.assertTrue(len(bigram.vocab) <= 5)
# endclass TestPhrasesModel


class TestPhraserModel(TestPhrasesCommon):
    """ Test Phraser models."""

    def setUp(self):
        """Set up Phraser models for the tests."""
        bigram_phrases = Phrases(sentences, min_count=1, threshold=1)
        self.bigram = Phraser(bigram_phrases)

        bigram_default_phrases = Phrases(sentences)
        self.bigram_default = Phraser(bigram_default_phrases)

        bigram_utf8_phrases = Phrases(sentences, min_count=1, threshold=1)
        self.bigram_utf8 = Phraser(bigram_utf8_phrases)

        bigram_unicode_phrases = Phrases(unicode_sentences, min_count=1, threshold=1)
        self.bigram_unicode = Phraser(bigram_unicode_phrases)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
