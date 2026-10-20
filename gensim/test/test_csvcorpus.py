"""Tests for reading numeric CSV corpora through text streams."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gensim import matutils, utils
from gensim.corpora.csvcorpus import CsvCorpus
from gensim.similarities import MatrixSimilarity


class TestCsvCorpus(unittest.TestCase):
    def test_text_and_compressed_corpora(self):
        expected = [[(0, 1.0), (1, 2.0)], [(0, 3.0), (1, 4.0)], [(0, 5.0), (1, 6.0)]]
        cases = [
            ('1,2\n3,4\n5,6\n', False),
            ('first,second\r\n"1","2"\r\n"3","4"\r\n"5","6"\r\n', False),
            ('label,first,second\n"café",1,2\n"東京",3,4\n"other",5,6\n', True),
            ('label,first,second\n"ca\nfé",1,2\n"東京",3,4\n"other",5,6\n', True),
            ('10,1,2\n20,3,4\n30,5,6\n', True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('.csv', '.csv.gz', '.csv.bz2'):
                path = str(Path(directory) / ('corpus' + suffix))
                for contents, labels in cases:
                    with self.subTest(suffix=suffix, contents=contents, labels=labels):
                        with utils.open(path, 'wt', encoding='utf-8', newline='') as stream:
                            stream.write(contents)
                        corpus = CsvCorpus(path, labels=labels)
                        self.assertEqual(len(corpus), 3)
                        self.assertEqual(list(corpus), expected)
                        self.assertEqual(len(corpus), 3)
                        self.assertEqual(list(corpus), expected)

    def test_similarity_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'corpus.csv')
            Path(path).write_text('1,2\n3,4\n5,6\n', encoding='utf-8')
            corpus = CsvCorpus(path, labels=False)
            np.testing.assert_array_equal(
                matutils.corpus2dense(corpus, num_terms=2), [[1, 3, 5], [2, 4, 6]],
            )
            index = MatrixSimilarity(corpus, num_features=2)
            scores = index[next(iter(corpus))]
            self.assertAlmostEqual(float(scores[0]), 1.0, places=6)
            self.assertTrue(np.all(scores[1:] < scores[0]))
