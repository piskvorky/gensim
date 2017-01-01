#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""

import io
import logging
import unittest

import numpy
from gensim.models import word2vec
from gensim.models.labeledword2vec import LabeledWord2Vec, score_document_labeled_cbow
from gensim.utils import pickle

logger = logging.getLogger()
logger.level = logging.ERROR

pre_docs = (
    ('i', 'study', 'machine', 'learning', 'to', 'drive', 'a', 'car'),
    ('important', 'topics', 'of', 'this', 'week', 'are', 'today', 'cheers'),
    ('like', 'my', 'faster', 'bike', 'supervised', 'is', 'dedicated', 'to', 'me'),
)
dataset_samples = (
    ('i', 'like', 'to', 'study', 'machine', 'learning'),
    ('machine', 'learning', 'is', 'important'),
    ('my', 'car', 'is', 'faster', 'than', 'your', 'bike'),
    ('nothing', 'is', 'faster', 'than', 'my', 'truck'),
    ('supervised', 'machine', 'learning', 'is', 'the', 'topic', 'of', 'today'),
    ('my', 'study', 'is', 'dedicated', 'to', 'me')
)
dataset_targets = (('aa', 'b'), 'b', 'cc', 'cc', 'b', 'aa')


def small_model():
    model = LabeledWord2Vec(iter=1, size=30, min_count=0, loss='hs', negative=0)
    model.build_vocab(dataset_samples, frozenset(
        target for targets in dataset_targets for target in BaseClassifier._target_list(targets)))
    model.train(zip(dataset_samples, dataset_targets))
    return model


def bunch_of_models():
    models = []
    for kwarg in ({'loss': 'hs', 'negative': 0}, {'loss': 'ns', 'negative': 5}, {'loss': 'softmax', 'negative': 0}):
        for bucket in (0, 5, 100):
            kwarg['bucket'] = bucket
            models.extend([
                LabeledWord2Vec(iter=5, size=30, min_count=0, sample=0, min_alpha=0.0, **kwarg),
                LabeledWord2Vec(iter=1, alpha=1.0, size=100, min_count=0, **kwarg),
                LabeledWord2Vec(iter=3, size=50, min_count=1, **kwarg),
                LabeledWord2Vec(iter=2, size=50, min_count=1, sample=0, **kwarg),
                LabeledWord2Vec(iter=10, size=10, min_count=0, **kwarg),
                LabeledWord2Vec(iter=5, workers=1, size=50, min_count=0, **kwarg)
            ])
    targets = frozenset(
        target for targets in dataset_targets for target in BaseClassifier._target_list(targets)
    )
    for model in models:
        model.build_vocab(dataset_samples, targets)
        model.train(zip(dataset_samples, dataset_targets))
    return models


class TestLabeledWord2VecModel(unittest.TestCase):

    def test_init(self):
        model1 = LabeledWord2Vec()
        model2 = LabeledWord2Vec(iter=1, size=50)
        model3 = LabeledWord2Vec(seed=66)
        self.assertNotEqual(model1, model2)
        self.assertNotEqual(model1, model3)

    def test_vocabulary(self, small_model):
        small_model = small_model()
        assert 'to' in small_model.wv.vocab
        assert frozenset(('aa', 'b', 'cc')) == frozenset(small_model.lvocab.keys())
        assert max(v.index for v in small_model.lvocab.values()) == 2

    def test_matrices(self, small_model):
        small_model = small_model()
        assert small_model.syn1.shape[0] == 3
        assert small_model.syn1.shape[1] == 30
        assert not hasattr(small_model, 'syn1neg')

    def test_serializzation(self, small_model):
        small_model = small_model()
        with io.BytesIO() as fileobj:
            pickle.dump(small_model, fileobj)
            fileobj.seek(0)
            loaded = pickle.load(fileobj)
            assert all(str(loaded.wv.vocab[w]) == str(small_model.wv.vocab[w]) for w in small_model.wv.vocab)
            assert all(str(loaded.lvocab[w]) == str(small_model.lvocab[w]) for w in small_model.lvocab)
            assert numpy.array_equiv(loaded.syn1, small_model.syn1)
            assert numpy.array_equiv(loaded.wv.syn0, small_model.wv.syn0)

    def test_learning_functions(self, bunch_of_models):
        bunch_of_models = bunch_of_models()
        for model in bunch_of_models:
            a = list(score_document_labeled_cbow(model, ('study', 'to', 'learn'), ['aa']))[0][1]
            b = list(score_document_labeled_cbow(model, ('study', 'to', 'learn'), ['b']))[0][1]
            c = list(score_document_labeled_cbow(model, ('study', 'to', 'learn'), ['cc']))[0][1]
            assert round(a + b + c, 1) == 1.
            assert round(sum([pred[1] for pred in score_document_labeled_cbow(model, ('study', 'to', 'learn'))]),
                         1) == 1.


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.DEBUG)
    logging.info("using optimization %s", word2vec.FAST_VERSION)
    unittest.main()
