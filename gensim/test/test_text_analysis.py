import logging
import unittest
from collections import namedtuple

from gensim.topic_coherence.text_analysis import \
    InvertedIndexAccumulator, WordOccurrenceAccumulator


class BaseTestCases(object):

    class TextAnalyzerTestBase(unittest.TestCase):
        texts = [
            ['this', 'is', 'a'],
            ['test', 'document'],
            ['this', 'test', 'document']
        ]
        token2id = {
            'this': 10,
            'is': 15,
            'a': 20,
            'test': 21,
            'document': 17
        }
        id2token = {v: k for k, v in token2id.items()}
        dictionary = namedtuple('Dictionary', 'token2id, id2token')(token2id, id2token)
        top_ids = set(token2id.values())

        accumulator_cls = None

        def test_occurrence_counting(self):
            accumulator = self.accumulator_cls(self.top_ids, self.dictionary) \
                .accumulate(self.texts, 3)
            self.assertEqual(2, accumulator.get_occurrences("this"))
            self.assertEqual(1, accumulator.get_occurrences("is"))
            self.assertEqual(1, accumulator.get_occurrences("a"))

            self.assertEqual(2, accumulator.get_co_occurrences("test", "document"))
            self.assertEqual(1, accumulator.get_co_occurrences("is", "a"))

        def test_occurences_for_irrelevant_words(self):
            accumulator = WordOccurrenceAccumulator(self.top_ids, self.dictionary) \
                .accumulate(self.texts, 2)
            with self.assertRaises(KeyError):
                accumulator.get_occurrences("irrelevant")
            with self.assertRaises(KeyError):
                accumulator.get_co_occurrences("test", "irrelevant")


class TestInvertedIndexAccumulator(BaseTestCases.TextAnalyzerTestBase):
    accumulator_cls = InvertedIndexAccumulator

    def test_accumulate1(self):
        accumulator = InvertedIndexAccumulator(self.top_ids, self.dictionary)\
            .accumulate(self.texts, 2)
        # [['this', 'is'], ['is', 'a'], ['test', 'document'], ['this', 'test'], ['test', 'document']]
        inverted_index = accumulator.index_to_dict()
        expected = {
            10: {0, 3},
            15: {0, 1},
            20: {1},
            21: {2, 3, 4},
            17: {2, 4}
        }
        self.assertDictEqual(expected, inverted_index)

    def test_accumulate2(self):
        accumulator = InvertedIndexAccumulator(self.top_ids, self.dictionary) \
            .accumulate(self.texts, 3)
        # [['this', 'is', 'a'], ['test', 'document'], ['this', 'test', 'document']]
        inverted_index = accumulator.index_to_dict()
        expected = {
            10: {0, 2},
            15: {0},
            20: {0},
            21: {1, 2},
            17: {1, 2}
        }
        self.assertDictEqual(expected, inverted_index)


class TestWordOccurrenceAccumulator(BaseTestCases.TextAnalyzerTestBase):
    accumulator_cls = WordOccurrenceAccumulator


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
