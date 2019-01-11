#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""


import logging
import unittest

import six
import numpy as np

from gensim.utils import to_unicode
from gensim.models.phrases import SentenceAnalyzer, Phrases, Phraser
from gensim.models.phrases import pseudocorpus, original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath


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

        def score_item(self, worda, wordb, components, scorer):
            """Override for test purpose"""
            if worda is not None and wordb is not None:
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
            scorer=None))

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
    sentences = common_texts + [
        ['graph', 'minors', 'survey', 'human', 'interface']
    ]
    unicode_sentences = [[to_unicode(w) for w in sentence] for sentence in sentences]
    common_terms = frozenset()

    bigram1 = u'response_time'
    bigram2 = u'graph_minors'
    bigram3 = u'human_interface'

    def gen_sentences(self):
        return ((w for w in sentence) for sentence in self.sentences)


class PhrasesCommon:
    """ Tests that need to be run for both Phrases and Phraser classes."""

    def setUp(self):
        self.bigram = Phrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram_default = Phrases(
            self.sentences, common_terms=self.common_terms)
        self.bigram_utf8 = Phrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram_unicode = Phrases(
            self.unicode_sentences, min_count=1, threshold=1, common_terms=self.common_terms)

    def testEmptyPhrasifiedSentencesIterator(self):
        bigram_phrases = Phrases(self.sentences)
        bigram_phraser = Phraser(bigram_phrases)
        trigram_phrases = Phrases(bigram_phraser[self.sentences])
        trigram_phraser = Phraser(trigram_phrases)
        trigrams = trigram_phraser[bigram_phraser[self.sentences]]
        fst, snd = list(trigrams), list(trigrams)
        self.assertEqual(fst, snd)
        self.assertNotEqual(snd, [])

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

    def testBigramConstructionFromArray(self):
        """Test Phrases bigram construction building when corpus is a numpy array"""
        bigram1_seen = False
        bigram2_seen = False

        for s in self.bigram[np.array(self.sentences)]:
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
        self.assertTrue(isinstance(transformed, six.text_type))


# scorer for testCustomScorer
# function is outside of the scope of the test because for picklability of custom scorer
# Phrases tests for picklability
# all scores will be 1
def dumb_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    return 1


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

    def test__getitem__(self):
        """ test Phrases[sentences] with a single sentence"""
        bigram = Phrases(self.sentences, min_count=1, threshold=1)
        # pdb.set_trace()
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface']]
        phrased_sentence = next(bigram[test_sentences].__iter__())

        assert phrased_sentence == ['graph_minors', 'survey', 'human_interface']

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

    def testCustomScorer(self):
        """ test using a custom scoring function """

        bigram = Phrases(self.sentences, min_count=1, threshold=.001, scoring=dumb_scorer)

        seen_scores = []
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
        for phrase, score in bigram.export_phrases(test_sentences):
            seen_scores.append(score)

        assert all(seen_scores)  # all scores 1
        assert len(seen_scores) == 3  # 'graph minors' and 'survey human' and 'interface system'

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


class TestPhrasesPersistence(PhrasesData, unittest.TestCase):

    def testSaveLoadCustomScorer(self):
        """ saving and loading a Phrases object with a custom scorer """

        with temporary_file("test.pkl") as fpath:
            bigram = Phrases(self.sentences, min_count=1, threshold=.001, scoring=dumb_scorer)
            bigram.save(fpath)
            bigram_loaded = Phrases.load(fpath)
            seen_scores = []
            test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
            for phrase, score in bigram_loaded.export_phrases(test_sentences):
                seen_scores.append(score)

            assert all(seen_scores)  # all scores 1
            assert len(seen_scores) == 3  # 'graph minors' and 'survey human' and 'interface system'

    def testSaveLoad(self):
        """ Saving and loading a Phrases object."""

        with temporary_file("test.pkl") as fpath:
            bigram = Phrases(self.sentences, min_count=1, threshold=1)
            bigram.save(fpath)
            bigram_loaded = Phrases.load(fpath)
            seen_scores = set()
            test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
            for phrase, score in bigram_loaded.export_phrases(test_sentences):
                seen_scores.add(round(score, 3))

            assert seen_scores == set([
                5.167,  # score for graph minors
                3.444  # score for human interface
            ])

    def testSaveLoadStringScoring(self):
        """ Saving and loading a Phrases object with a string scoring parameter.
        This should ensure backwards compatibility with the previous version of Phrases"""
        bigram_loaded = Phrases.load(datapath("phrases-scoring-str.pkl"))
        seen_scores = set()
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
        for phrase, score in bigram_loaded.export_phrases(test_sentences):
            seen_scores.add(round(score, 3))

        assert seen_scores == set([
            5.167,  # score for graph minors
            3.444  # score for human interface
        ])

    def testSaveLoadNoScoring(self):
        """ Saving and loading a Phrases object with no scoring parameter.
        This should ensure backwards compatibility with old versions of Phrases"""

        bigram_loaded = Phrases.load(datapath("phrases-no-scoring.pkl"))
        seen_scores = set()
        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface', 'system']]
        for phrase, score in bigram_loaded.export_phrases(test_sentences):
            seen_scores.add(round(score, 3))

        assert seen_scores == set([
            5.167,  # score for graph minors
            3.444  # score for human interface
        ])

    def testSaveLoadNoCommonTerms(self):
        """ Ensure backwards compatibility with old versions of Phrases, before common_terms"""
        bigram_loaded = Phrases.load(datapath("phrases-no-common-terms.pkl"))
        self.assertEqual(bigram_loaded.common_terms, frozenset())
        # can make a phraser, cf #1751
        phraser = Phraser(bigram_loaded)  # does not raise
        phraser[["human", "interface", "survey"]]  # does not raise


class TestPhraserPersistence(PhrasesData, unittest.TestCase):

    def testSaveLoadCustomScorer(self):
        """Saving and loading a Phraser object with a custom scorer """

        with temporary_file("test.pkl") as fpath:
            bigram = Phraser(
                Phrases(self.sentences, min_count=1, threshold=.001, scoring=dumb_scorer))
            bigram.save(fpath)
            bigram_loaded = Phraser.load(fpath)
            # we do not much with scoring, just verify its the one expected
            self.assertEqual(bigram_loaded.scoring, dumb_scorer)

    def testSaveLoad(self):
        """ Saving and loading a Phraser object."""
        with temporary_file("test.pkl") as fpath:
            bigram = Phraser(Phrases(self.sentences, min_count=1, threshold=1))
            bigram.save(fpath)
            bigram_loaded = Phraser.load(fpath)
            self.assertEqual(
                bigram_loaded[['graph', 'minors', 'survey', 'human', 'interface', 'system']],
                ['graph_minors', 'survey', 'human_interface', 'system'])

    def testSaveLoadStringScoring(self):
        """ Saving and loading a Phraser object with a string scoring parameter.
        This should ensure backwards compatibility with the previous version of Phraser"""
        bigram_loaded = Phraser.load(datapath("phraser-scoring-str.pkl"))
        # we do not much with scoring, just verify its the one expected
        self.assertEqual(bigram_loaded.scoring, original_scorer)

    def testSaveLoadNoScoring(self):
        """ Saving and loading a Phraser object with no scoring parameter.
        This should ensure backwards compatibility with old versions of Phraser"""
        bigram_loaded = Phraser.load(datapath("phraser-no-scoring.pkl"))
        # we do not much with scoring, just verify its the one expected
        self.assertEqual(bigram_loaded.scoring, original_scorer)

    def testSaveLoadNoCommonTerms(self):
        """ Ensure backwards compatibility with old versions of Phraser, before common_terms"""
        bigram_loaded = Phraser.load(datapath("phraser-no-common-terms.pkl"))
        self.assertEqual(bigram_loaded.common_terms, frozenset())


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
    unicode_sentences = [[to_unicode(w) for w in sentence] for sentence in sentences]
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
        self.assertTrue(isinstance(transformed, six.text_type))

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

    def testCustomScorer(self):
        """ test using a custom scoring function """

        bigram = Phrases(self.sentences, min_count=1, threshold=.001,
                         scoring=dumb_scorer, common_terms=self.common_terms)

        seen_scores = []
        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        for phrase, score in bigram.export_phrases(test_sentences):
            seen_scores.append(score)

        assert all(seen_scores)  # all scores 1
        assert len(seen_scores) == 2  # 'data and graph' 'survey for human'

    def test__getitem__(self):
        """ test Phrases[sentences] with a single sentence"""
        bigram = Phrases(self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        # pdb.set_trace()
        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human', 'interface']]
        phrased_sentence = next(bigram[test_sentences].__iter__())

        assert phrased_sentence == ['data_and_graph', 'survey', 'for', 'human_interface']


class TestPhraserModelCommonTerms(CommonTermsPhrasesData, TestPhraserModel):

    def testEncoding(self):
        """Test that both utf8 and unicode input work; output must be unicode."""
        expected = [u'survey', u'of', u'user', u'computer', u'system', u'lack_of_interest']

        self.assertEqual(self.bigram_utf8[self.sentences[1]], expected)
        self.assertEqual(self.bigram_unicode[self.sentences[1]], expected)

        transformed = ' '.join(self.bigram_utf8[self.sentences[1]])
        self.assertTrue(isinstance(transformed, six.text_type))


class TestPhraserModelCompatibilty(unittest.TestCase):

    def testCompatibilty(self):
        phr = Phraser.load(datapath("phraser-3.6.0.model"))
        model = Phrases.load(datapath("phrases-3.6.0.model"))

        test_sentences = ['trees', 'graph', 'minors']
        expected_res = ['trees', 'graph_minors']

        phr_out = phr[test_sentences]
        model_out = model[test_sentences]

        self.assertEqual(phr_out, expected_res)
        self.assertEqual(model_out, expected_res)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
