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
from gensim.models.phrases import Phrases, Phraser, CommonTermsPhrases, CommonTermsPhraser

if sys.version_info[0] >= 3:
    unicode = str

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


class PhrasesData:

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
        ['graph', 'minors', 'survey','human','interface'] #test bigrams within same sentence
    ]
    unicode_sentences = [[utils.to_unicode(w) for w in sentence] for sentence in sentences]



class TestPhrasesCommon(PhrasesData, unittest.TestCase):
    """ Tests that need to be run for both Prases and Phraser classes."""

    def gen_sentences(self):
        return ((w for w in sentence) for sentence in self.sentences)

    def setUp(self):
        self.bigram = Phrases(self.sentences, min_count=1, threshold=1)
        self.bigram_default = Phrases(self.sentences)
        self.bigram_utf8 = Phrases(self.sentences, min_count=1, threshold=1)
        self.bigram_unicode = Phrases(self.unicode_sentences, min_count=1, threshold=1)

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
        self.assertEqual(len(self.sentences), len(list(self.bigram_default[self.sentences])))

    def testSentenceGenerationWithGenerator(self):
        """Test basic bigram production when corpus is a generator."""
        self.assertEqual(len(list(self.gen_sentences())),
                         len(list(self.bigram_default[self.gen_sentences()])))

    def testBigramConstruction(self):
        """Test Phrases bigram construction building."""
        # with this setting we should get response_time and graph_minors
        bigram1_seen = False
        bigram2_seen = False

        for s in self.bigram[self.sentences]:
            if not bigram1_seen and u'response_time' in s:
                bigram1_seen = True
            if not bigram2_seen and u'graph_minors' in s:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break

        self.assertTrue(bigram1_seen and bigram2_seen)

        # check the same thing, this time using single doc transformation
        # last sentence should contain both graph_minors and human_interface
        self.assertTrue(u'response_time' in self.bigram[self.sentences[1]])
        self.assertTrue(u'response_time' in self.bigram[self.sentences[4]])
        self.assertTrue(u'graph_minors' in self.bigram[self.sentences[-2]])
        self.assertTrue(u'graph_minors' in self.bigram[self.sentences[-1]])
        self.assertTrue(u'human_interface' in self.bigram[self.sentences[-1]])

    def testBigramConstructionFromGenerator(self):
        """Test Phrases bigram construction building when corpus is a generator"""
        bigram1_seen = False
        bigram2_seen = False

        for s in self.bigram[self.gen_sentences()]:
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

        self.assertEqual(self.bigram_utf8[self.sentences[1]], expected)
        self.assertEqual(self.bigram_unicode[self.sentences[1]], expected)

        transformed = ' '.join(self.bigram_utf8[self.sentences[1]])
        self.assertTrue(isinstance(transformed, unicode))


class TestPhrasesModel(PhrasesData, unittest.TestCase):
    def testExportPhrases(self):
        """Test Phrases bigram export_phrases functionality."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1)

        # with this setting we should get response_time and graph_minors
        bigram1_seen = False
        bigram2_seen = False

        for phrase, score in bigram.export_phrases(self.sentences):
            if not bigram1_seen and b'response time' == phrase:
                bigram1_seen = True
            elif not bigram2_seen and b'graph minors' == phrase:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break

        self.assertTrue(bigram1_seen)
        self.assertTrue(bigram2_seen)

    def testBadParameters(self):
        """Test the phrases module with bad parameters."""
        # should fail with something less or equal than 0
        self.assertRaises(ValueError, Phrases, self.sentences, min_count=0)

        # threshold should be positive
        self.assertRaises(ValueError, Phrases, self.sentences, threshold=-1)

    def testPruning(self):
        """Test that max_vocab_size parameter is respected."""
        bigram = Phrases(self.sentences, max_vocab_size=5)
        self.assertTrue(len(bigram.vocab) <= 5)
#endclass TestPhrasesModel


class TestPhraserModel(TestPhrasesCommon):
    """ Test Phraser models."""
    def setUp(self):
        """Set up Phraser models for the tests."""
        bigram_phrases = Phrases(self.sentences, min_count=1, threshold=1)
        self.bigram = Phraser(bigram_phrases)

        bigram_default_phrases = Phrases(self.sentences)
        self.bigram_default = Phraser(bigram_default_phrases)

        bigram_utf8_phrases = Phrases(self.sentences, min_count=1, threshold=1)
        self.bigram_utf8 = Phraser(bigram_utf8_phrases)

        bigram_unicode_phrases = Phrases(self.unicode_sentences, min_count=1, threshold=1)
        self.bigram_unicode = Phraser(bigram_unicode_phrases)


class CommonTermsPhrasesData:

    sentences = [
        ['human', 'interface', 'with', 'computer'],
        ['survey', 'of', 'user', 'computer', 'system', 'lack', 'of', 'interest'],
        ['eps', 'user', 'interface', 'system'],
        ['system', 'and', 'human', 'system', 'eps'],
        ['user', 'lack', 'of', 'interest'],
        ['trees'],
        ['graph', 'of', 'trees'],
        ['data', 'and', 'graph', 'of', 'trees'],
        ['data', 'and', 'graph', 'survey'],
        ['data', 'and', 'graph', 'survey', 'for', 'human','interface'] #test bigrams within same sentence
    ]
    unicode_sentences = [[utils.to_unicode(w) for w in sentence] for sentence in sentences]
    common_terms = ['of', 'and', 'for']


class TestCommonTermsPhrasesCommon(CommonTermsPhrasesData, TestPhrasesCommon):
    """ Test CommonTermsPhrases models."""
    def setUp(self):
        """Set up Phraser models for the tests."""
        self.bigram = CommonTermsPhrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram_default = CommonTermsPhrases(self.sentences, common_terms=self.common_terms)
        self.bigram_utf8 = CommonTermsPhrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram_unicode = CommonTermsPhrases(
            self.unicode_sentences, min_count=1, threshold=1, common_terms=self.common_terms)

    def testBigramConstruction(self):
        """Test Phrases bigram construction building."""
        # with this setting we should get response_time and graph_minors
        bigram1_seen = False
        bigram2_seen = False

        for s in self.bigram[self.sentences]:
            if not bigram1_seen and u'lack_of_interest' in s:
                bigram1_seen = True
            if not bigram2_seen and u'data_and_graph' in s:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break

        self.assertTrue(bigram1_seen and bigram2_seen)

        # check the same thing, this time using single doc transformation
        # last sentence should contain both graph_minors and human_interface
        self.assertTrue(u'lack_of_interest' in self.bigram[self.sentences[1]])
        self.assertTrue(u'lack_of_interest' in self.bigram[self.sentences[4]])
        self.assertTrue(u'data_and_graph' in self.bigram[self.sentences[-2]])
        self.assertTrue(u'data_and_graph' in self.bigram[self.sentences[-1]])
        self.assertTrue(u'human_interface' in self.bigram[self.sentences[-1]])

    def testBigramConstructionFromGenerator(self):
        """Test Phrases bigram construction building when corpus is a generator"""
        bigram1_seen = False
        bigram2_seen = False

        for s in self.bigram[self.gen_sentences()]:
            if not bigram1_seen and 'lack_of_interest' in s:
                bigram1_seen = True
            if not bigram2_seen and 'data_and_graph' in s:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break
        self.assertTrue(bigram1_seen and bigram2_seen)

    def testEncoding(self):
        """Test that both utf8 and unicode input work; output must be unicode."""
        expected = [u'survey', u'of', u'user', u'computer', u'system', u'lack_of_interest']

        self.assertEqual(self.bigram_utf8[self.sentences[1]], expected)
        self.assertEqual(self.bigram_unicode[self.sentences[1]], expected)

        transformed = ' '.join(self.bigram_utf8[self.sentences[1]])
        self.assertTrue(isinstance(transformed, unicode))




class TestCommonTermsPhrasesModel(CommonTermsPhrasesData, unittest.TestCase):
    def testExportPhrases(self):
        """Test Phrases bigram export_phrases functionality."""
        bigram = CommonTermsPhrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)

        # with this setting we should get response_time and graph_minors
        bigram1_seen = False
        bigram2_seen = False

        for phrase, score in bigram.export_phrases(self.sentences):
            if not bigram1_seen and b'lack of interest' == phrase:
                bigram1_seen = True
            elif not bigram2_seen and b'data and graph' == phrase:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break

        self.assertTrue(bigram1_seen)
        self.assertTrue(bigram2_seen)

    def testBadParameters(self):
        """Test the phrases module with bad parameters."""
        # should fail with something less or equal than 0
        self.assertRaises(
            ValueError, CommonTermsPhrases, self.sentences, min_count=0, common_terms=self.common_terms)

        # threshold should be positive
        self.assertRaises(
            ValueError, CommonTermsPhrases, self.sentences, threshold=-1, common_terms=self.common_terms)

        # common_terms is a mandatory keyword parameter
        self.assertRaises(
            ValueError, CommonTermsPhrases, self.sentences)

    def testPruning(self):
        """Test that max_vocab_size parameter is respected."""
        bigram = CommonTermsPhrases(
            self.sentences, max_vocab_size=5, common_terms=self.common_terms)
        self.assertTrue(len(bigram.vocab) <= 5)


class TestCommonTermsPhraserModel(TestCommonTermsPhrasesCommon):
    """ Test Phraser models."""
    def setUp(self):
        """Set up Phraser models for the tests."""
        bigram_phrases = CommonTermsPhrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram = CommonTermsPhraser(bigram_phrases)

        bigram_default_phrases = CommonTermsPhrases(
            self.sentences, common_terms=self.common_terms)
        self.bigram_default = CommonTermsPhraser(bigram_default_phrases)

        bigram_utf8_phrases = CommonTermsPhrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram_utf8 = CommonTermsPhraser(bigram_utf8_phrases)

        bigram_unicode_phrases = CommonTermsPhrases(
            self.unicode_sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram_unicode = CommonTermsPhraser(bigram_unicode_phrases)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
