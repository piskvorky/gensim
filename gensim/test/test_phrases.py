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
from gensim.models.phrases import SentenceAnalyzer, Phrases, Phraser, pseudocorpus

if sys.version_info[0] >= 3:
    unicode = str

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


class TestUtils(unittest.TestCase):

    def test_pseudocorpus_no_common_terms(self):
        vocab = [
            "prime_minister",
            "gold",
            "chief_technical_officer",
            "effective"]
        result = list(pseudocorpus(vocab, "_"))
        self.assertEqual(
            result,
            [["prime", "minister"],
             ["chief", "technical_officer"],
             ["chief_technical", "officer"]])

    def test_pseudocorpus_with_common_terms(self):
        vocab = [
            "hall_of_fame",
            "gold",
            "chief_of_political_bureau",
            "effective",
            "beware_of_the_dog_in_the_yard"]
        common_terms = frozenset(["in", "the", "of"])
        result = list(pseudocorpus(vocab, "_", common_terms=common_terms))
        self.assertEqual(
            result,
            [["hall", "of", "fame"],
             ["chief", "of", "political_bureau"],
             ["chief_of_political", "bureau"],
             ["beware", "of", "the", "dog_in_the_yard"],
             ["beware_of_the_dog", "in", "the", "yard"]])


class TestPhraseAnalysis(unittest.TestCase):

    class AnalysisTester(SentenceAnalyzer):

        def __init__(self, scores):
            self.scores = scores

        def scorer(self, word_a, word_b, components):
            if word_a is not None and word_b is not None:
                bigram_word = b"_".join(components)
                return self.scores.get(bigram_word, -1)
            else:
                return -1

    def analyze(self, scores, sentence):
        analyzer = self.AnalysisTester(scores)
        return list(analyzer.analyze_sentence(
            sentence,
            threshold=1,
            common_terms={b"a", b"the", b"with", b"of"},
            scoring=analyzer.scorer))

    def analyze_words(self, scores, sentence):
        result = (
            w if isinstance(w, (tuple, list)) else [w]
            for w, score in self.analyze(scores, sentence))
        return [b"_".join(w).decode("utf-8") for w in result]

    def test_simple_analysis(self):
        s = ["simple", "sentence", "should", "pass"]
        result = self.analyze_words({}, s)
        self.assertEqual(result, s)
        s = ["a", "simple", "sentence", "with", "no", "bigram", "but", "common", "terms"]
        result = self.analyze_words({}, s)
        self.assertEqual(result, s)

    def test_analysis_bigrams(self):
        scores = {
            b"simple_sentence": 2, b"sentence_many": 2,
            b"many_possible": 2, b"possible_bigrams": 2}
        s = ["simple", "sentence", "many", "possible", "bigrams"]
        result = self.analyze_words(scores, s)
        self.assertEqual(result, ["simple_sentence", "many_possible", "bigrams"])

        s = ["some", "simple", "sentence", "many", "bigrams"]
        result = self.analyze_words(scores, s)
        self.assertEqual(result, ["some", "simple_sentence", "many", "bigrams"])

        s = ["some", "unrelated", "simple", "words"]
        result = self.analyze_words(scores, s)
        self.assertEqual(result, s)

    def test_analysis_common_terms(self):
        scores = {
            b"simple_sentence": 2, b"sentence_many": 2,
            b"many_possible": 2, b"possible_bigrams": 2}
        s = ["a", "simple", "sentence", "many", "the", "possible", "bigrams"]
        result = self.analyze_words(scores, s)
        self.assertEqual(result, ["a", "simple_sentence", "many", "the", "possible_bigrams"])

        s = ["simple", "the", "sentence", "and", "many", "possible", "bigrams", "with", "a"]
        result = self.analyze_words(scores, s)
        self.assertEqual(result, [
            "simple", "the", "sentence", "and", "many_possible", "bigrams", "with", "a"])

    def test_analysis_common_terms_in_between(self):
        scores = {
            b"simple_sentence": 2, b"sentence_with_many": 2,
            b"many_possible": 2, b"many_of_the_possible": 2, b"possible_bigrams": 2}
        s = ["sentence", "with", "many", "possible", "bigrams"]
        result = self.analyze_words(scores, s)
        self.assertEqual(result, ["sentence_with_many", "possible_bigrams"])

        s = ["a", "simple", "sentence", "with", "many", "of", "the", "possible", "bigrams", "with"]
        result = self.analyze_words(scores, s)
        self.assertEqual(
            result, ["a", "simple_sentence", "with", "many_of_the_possible", "bigrams", "with"])


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
        ['graph', 'minors', 'survey', 'human', 'interface']  # test bigrams within same sentence
    ]
    unicode_sentences = [[utils.to_unicode(w) for w in sentence] for sentence in sentences]
    common_terms = frozenset()

    bigram1 = u'response_time'
    bigram2 = u'graph_minors'
    bigram3 = u'human_interface'

    def gen_sentences(self):
        return ((w for w in sentence) for sentence in self.sentences)


class PhrasesCommon:
    """ Tests that need to be run for both Prases and Phraser classes."""

    def setUp(self):
        self.bigram = Phrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram_default = Phrases(
            self.sentences, common_terms=self.common_terms)
        self.bigram_utf8 = Phrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram_unicode = Phrases(
            self.unicode_sentences, min_count=1, threshold=1, common_terms=self.common_terms)

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
            if not bigram1_seen and self.bigram1 in s:
                bigram1_seen = True
            if not bigram2_seen and self.bigram2 in s:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break

        self.assertTrue(bigram1_seen and bigram2_seen)

        # check the same thing, this time using single doc transformation
        # last sentence should contain both graph_minors and human_interface
        self.assertTrue(self.bigram1 in self.bigram[self.sentences[1]])
        self.assertTrue(self.bigram1 in self.bigram[self.sentences[4]])
        self.assertTrue(self.bigram2 in self.bigram[self.sentences[-2]])
        self.assertTrue(self.bigram2 in self.bigram[self.sentences[-1]])
        self.assertTrue(self.bigram3 in self.bigram[self.sentences[-1]])

    def testBigramConstructionFromGenerator(self):
        """Test Phrases bigram construction building when corpus is a generator"""
        bigram1_seen = False
        bigram2_seen = False

        for s in self.bigram[self.gen_sentences()]:
            if not bigram1_seen and self.bigram1 in s:
                bigram1_seen = True
            if not bigram2_seen and self.bigram2 in s:
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


class TestPhrasesModel(PhrasesData, PhrasesCommon, unittest.TestCase):

    def testExportPhrases(self):
        """Test Phrases bigram export_phrases functionality."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1)

        seen_bigrams = set()

        for phrase, score in bigram.export_phrases(self.sentences):
            seen_bigrams.add(phrase)

        assert seen_bigrams == {
            b'response time',
            b'graph minors',
            b'human interface',
        }

    def testMultipleBigramsSingleEntry(self):
        """ a single entry should produce multiple bigrams. """
        bigram = Phrases(self.sentences, min_count=1, threshold=1)
        seen_bigrams = set()

        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface']]
        for phrase, score in bigram.export_phrases(test_sentences):
            seen_bigrams.add(phrase)

        assert seen_bigrams == {b'graph minors', b'human interface'}

    def testScoringDefault(self):
        """ test the default scoring, from the mikolov word2vec paper """
        bigram = Phrases(self.sentences, min_count=1, threshold=1)

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
        bigram = Phrases(self.sentences, min_count=1, threshold=.5, scoring='npmi')

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
        self.assertRaises(ValueError, Phrases, self.sentences, min_count=0)

        # threshold should be positive
        self.assertRaises(ValueError, Phrases, self.sentences, threshold=-1)

    def testPruning(self):
        """Test that max_vocab_size parameter is respected."""
        bigram = Phrases(self.sentences, max_vocab_size=5)
        self.assertTrue(len(bigram.vocab) <= 5)
# endclass TestPhrasesModel


class TestPhraserModel(PhrasesData, PhrasesCommon, unittest.TestCase):
    """ Test Phraser models."""

    def setUp(self):
        """Set up Phraser models for the tests."""
        bigram_phrases = Phrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram = Phraser(bigram_phrases)

        bigram_default_phrases = Phrases(self.sentences, common_terms=self.common_terms)
        self.bigram_default = Phraser(bigram_default_phrases)

        bigram_utf8_phrases = Phrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram_utf8 = Phraser(bigram_utf8_phrases)

        bigram_unicode_phrases = Phrases(
            self.unicode_sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram_unicode = Phraser(bigram_unicode_phrases)


class CommonTermsPhrasesData:
    """This mixin permits to reuse the test, using, this time the common_terms option
    """

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
        ['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']  # test bigrams within same sentence
    ]
    unicode_sentences = [[utils.to_unicode(w) for w in sentence] for sentence in sentences]
    common_terms = ['of', 'and', 'for']

    bigram1 = u'lack_of_interest'
    bigram2 = u'data_and_graph'
    bigram3 = u'human_interface'
    expression1 = u'lack of interest'
    expression2 = u'data and graph'
    expression3 = u'human interface'

    def gen_sentences(self):
        return ((w for w in sentence) for sentence in self.sentences)


class TestPhrasesModelCommonTerms(CommonTermsPhrasesData, TestPhrasesModel):
    """Test Phrases models with common terms"""

    def testEncoding(self):
        """Test that both utf8 and unicode input work; output must be unicode."""
        expected = [u'survey', u'of', u'user', u'computer', u'system', u'lack_of_interest']

        self.assertEqual(self.bigram_utf8[self.sentences[1]], expected)
        self.assertEqual(self.bigram_unicode[self.sentences[1]], expected)

        transformed = ' '.join(self.bigram_utf8[self.sentences[1]])
        self.assertTrue(isinstance(transformed, unicode))

    def testMultipleBigramsSingleEntry(self):
        """ a single entry should produce multiple bigrams. """
        bigram = Phrases(self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)

        seen_bigrams = set()
        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        for phrase, score in bigram.export_phrases(test_sentences):
            seen_bigrams.add(phrase)
        assert seen_bigrams == set([
            b'data and graph',
            b'human interface',
        ])

    def testExportPhrases(self):
        """Test Phrases bigram export_phrases functionality."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)

        seen_bigrams = set()

        for phrase, score in bigram.export_phrases(self.sentences):
            seen_bigrams.add(phrase)

        assert seen_bigrams == set([
            b'human interface',
            b'graph of trees',
            b'data and graph',
            b'lack of interest',
        ])

    def testScoringDefault(self):
        """ test the default scoring, from the mikolov word2vec paper """
        bigram = Phrases(self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)

        seen_scores = set()

        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        for phrase, score in bigram.export_phrases(test_sentences):
            seen_scores.add(round(score, 3))

        min_count = float(bigram.min_count)
        len_vocab = float(len(bigram.vocab))
        graph = float(bigram.vocab[b"graph"])
        data = float(bigram.vocab[b"data"])
        data_and_graph = float(bigram.vocab[b"data_and_graph"])
        human = float(bigram.vocab[b"human"])
        interface = float(bigram.vocab[b"interface"])
        human_interface = float(bigram.vocab[b"human_interface"])

        assert seen_scores == set([
            # score for data and graph
            round((data_and_graph - min_count) / data / graph * len_vocab, 3),
            # score for human interface
            round((human_interface - min_count) / human / interface * len_vocab, 3),
        ])

    def testScoringNpmi(self):
        """ test normalized pointwise mutual information scoring """
        bigram = Phrases(self.sentences, min_count=1, threshold=.5,
                         scoring='npmi', common_terms=self.common_terms)

        seen_scores = set()

        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        for phrase, score in bigram.export_phrases(test_sentences):
            seen_scores.add(round(score, 3))

        assert seen_scores == set([
            .74,  # score for data and graph
            .894  # score for human interface
        ])


class TestPhraserModelCommonTerms(CommonTermsPhrasesData, TestPhraserModel):

    def testEncoding(self):
        """Test that both utf8 and unicode input work; output must be unicode."""
        expected = [u'survey', u'of', u'user', u'computer', u'system', u'lack_of_interest']

        self.assertEqual(self.bigram_utf8[self.sentences[1]], expected)
        self.assertEqual(self.bigram_unicode[self.sentences[1]], expected)

        transformed = ' '.join(self.bigram_utf8[self.sentences[1]])
        self.assertTrue(isinstance(transformed, unicode))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
