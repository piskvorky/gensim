import os
import tempfile
import unittest
import numpy as np

from gensim.corpora import Dictionary
from gensim.models import LdaModel


class TestLdaDiff(unittest.TestCase):
    def setUp(self):
        texts = [['human', 'interface', 'computer'],
                 ['survey', 'user', 'computer', 'system', 'response', 'time'],
                 ['eps', 'user', 'interface', 'system'],
                 ['system', 'human', 'system', 'eps'],
                 ['user', 'response', 'time'],
                 ['trees'],
                 ['graph', 'trees'],
                 ['graph', 'minors', 'trees'],
                 ['graph', 'minors', 'survey']]
        self.dictionary = Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.num_topics = 5
        self.n_ann_terms = 10

    def testBasic(self):
        model = LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=self.num_topics, passes=10)
        mdiff, annotation = model.diff(model, n_ann_terms=self.n_ann_terms)

        self.assertEqual(mdiff.shape, (self.num_topics, self.num_topics))
        self.assertEquals(len(annotation), self.num_topics)
        self.assertEquals(len(annotation[0]), self.num_topics)

    def testIdentity(self):
        model = LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=self.num_topics, passes=10)

        for dist_name in ["hellinger", "kulback_leibler", "jaccard"]:
            mdiff, annotation = model.diff(model, n_ann_terms=self.n_ann_terms, distance=dist_name)

            for row in annotation:
                for (int_tokens, diff_tokens) in row:
                    self.assertEquals(diff_tokens, [])
                    self.assertEquals(len(int_tokens), self.n_ann_terms)

            self.assertTrue(np.allclose(np.diag(mdiff), np.zeros(mdiff.shape[0], dtype=mdiff.dtype)))

            if dist_name == "jaccard":
                self.assertTrue(np.allclose(mdiff, np.zeros(mdiff.shape, dtype=mdiff.dtype)))
