import logging
import unittest

from gensim.corpora.dictionary import Dictionary
from gensim.topic_coherence.text_analysis import (
    InvertedIndexAccumulator, WordOccurrenceAccumulator, ParallelWordOccurrenceAccumulator,
    CorpusAccumulator)


class BaseTestCases(object):

    class TextAnalyzerTestBase(unittest.TestCase):
        texts = [
            ['this', 'is', 'a'],
            ['test', 'document'],
            ['this', 'test', 'document'],
            ['test', 'test', 'this']
        ]
        token2id = {
            'this': 10,
            'is': 15,
            'a': 20,
            'test': 21,
            'document': 17
        }
        dictionary = Dictionary(texts)
        dictionary.token2id = token2id
        dictionary.id2token = {v: k for k, v in token2id.items()}
        top_ids = set(token2id.values())

        texts2 = [
            ['human', 'interface', 'computer'],
            ['survey', 'user', 'computer', 'system', 'response', 'time'],
            ['eps', 'user', 'interface', 'system'],
            ['system', 'human', 'system', 'eps'],
            ['user', 'response', 'time'],
            ['trees'],
            ['graph', 'trees'],
            ['graph', 'minors', 'trees'],
            ['graph', 'minors', 'survey'],
            ['user', 'user']
        ]
        dictionary2 = Dictionary(texts2)
        dictionary2.id2token = {v: k for k, v in dictionary2.token2id.items()}
        top_ids2 = set(dictionary2.token2id.values())

        accumulator_cls = None

        def init_accumulator(self):
            return self.accumulator_cls(self.top_ids, self.dictionary)

        def init_accumulator2(self):
            return self.accumulator_cls(self.top_ids2, self.dictionary2)

        def test_occurrence_counting(self):
            accumulator = self.init_accumulator().accumulate(self.texts, 3)
            self.assertEqual(3, accumulator.get_occurrences("this"))
            self.assertEqual(1, accumulator.get_occurrences("is"))
            self.assertEqual(1, accumulator.get_occurrences("a"))

            self.assertEqual(2, accumulator.get_co_occurrences("test", "document"))
            self.assertEqual(2, accumulator.get_co_occurrences("test", "this"))
            self.assertEqual(1, accumulator.get_co_occurrences("is", "a"))

        def test_occurrence_counting2(self):
            accumulator = self.init_accumulator2().accumulate(self.texts2, 110)
            self.assertEqual(2, accumulator.get_occurrences("human"))
            self.assertEqual(4, accumulator.get_occurrences("user"))
            self.assertEqual(3, accumulator.get_occurrences("graph"))
            self.assertEqual(3, accumulator.get_occurrences("trees"))

            cases = [
                (1, ("human", "interface")),
                (2, ("system", "user")),
                (2, ("graph", "minors")),
                (2, ("graph", "trees")),
                (4, ("user", "user")),
                (3, ("graph", "graph")),
                (0, ("time", "eps"))
            ]
            for expected_count, (word1, word2) in cases:
                # Verify co-occurrence counts are correct, regardless of word order.
                self.assertEqual(expected_count, accumulator.get_co_occurrences(word1, word2))
                self.assertEqual(expected_count, accumulator.get_co_occurrences(word2, word1))

                # Also verify that using token ids instead of tokens works the same.
                word_id1 = self.dictionary2.token2id[word1]
                word_id2 = self.dictionary2.token2id[word2]
                self.assertEqual(expected_count, accumulator.get_co_occurrences(word_id1, word_id2))
                self.assertEqual(expected_count, accumulator.get_co_occurrences(word_id2, word_id1))

        def test_occurences_for_irrelevant_words(self):
            accumulator = self.init_accumulator().accumulate(self.texts, 2)
            with self.assertRaises(KeyError):
                accumulator.get_occurrences("irrelevant")
            with self.assertRaises(KeyError):
                accumulator.get_co_occurrences("test", "irrelevant")


class TestInvertedIndexAccumulator(BaseTestCases.TextAnalyzerTestBase):
    accumulator_cls = InvertedIndexAccumulator

    def test_accumulate1(self):
        accumulator = InvertedIndexAccumulator(self.top_ids, self.dictionary)\
            .accumulate(self.texts, 2)
        # [['this', 'is'], ['is', 'a'], ['test', 'document'], ['this', 'test'],
        #  ['test', 'document'], ['test', 'test'], ['test', 'this']]
        inverted_index = accumulator.index_to_dict()
        expected = {
            10: {0, 3, 6},
            15: {0, 1},
            20: {1},
            21: {2, 3, 4, 5, 6},
            17: {2, 4}
        }
        self.assertDictEqual(expected, inverted_index)

    def test_accumulate2(self):
        accumulator = InvertedIndexAccumulator(self.top_ids, self.dictionary)\
            .accumulate(self.texts, 3)
        # [['this', 'is', 'a'], ['test', 'document'], ['this', 'test', 'document'],
        #  ['test', 'test', 'this']
        inverted_index = accumulator.index_to_dict()
        expected = {
            10: {0, 2, 3},
            15: {0},
            20: {0},
            21: {1, 2, 3},
            17: {1, 2}
        }
        self.assertDictEqual(expected, inverted_index)


class TestWordOccurrenceAccumulator(BaseTestCases.TextAnalyzerTestBase):
    accumulator_cls = WordOccurrenceAccumulator


class TestParallelWordOccurrenceAccumulator(BaseTestCases.TextAnalyzerTestBase):
    accumulator_cls = ParallelWordOccurrenceAccumulator

    def init_accumulator(self):
        return self.accumulator_cls(2, self.top_ids, self.dictionary)

    def init_accumulator2(self):
        return self.accumulator_cls(2, self.top_ids2, self.dictionary2)


class TestCorpusAnalyzer(unittest.TestCase):

    def setUp(self):
        self.dictionary = BaseTestCases.TextAnalyzerTestBase.dictionary
        self.top_ids = BaseTestCases.TextAnalyzerTestBase.top_ids
        self.corpus = \
            [self.dictionary.doc2bow(doc) for doc in BaseTestCases.TextAnalyzerTestBase.texts]

    def test_index_accumulation(self):
        accumulator = CorpusAccumulator(self.top_ids).accumulate(self.corpus)
        inverted_index = accumulator.index_to_dict()
        expected = {
            10: {0, 2, 3},
            15: {0},
            20: {0},
            21: {1, 2, 3},
            17: {1, 2}
        }
        self.assertDictEqual(expected, inverted_index)

        self.assertEqual(3, accumulator.get_occurrences(10))
        self.assertEqual(2, accumulator.get_occurrences(17))
        self.assertEqual(2, accumulator.get_co_occurrences(10, 21))
        self.assertEqual(1, accumulator.get_co_occurrences(10, 17))


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
