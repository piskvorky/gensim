import six
import unittest
import numpy
import os
import codecs
import pickle

from scipy import sparse
try:
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.datasets import load_files
    from sklearn import linear_model
    from sklearn.exceptions import NotFittedError
except ImportError:
    raise unittest.SkipTest("Test requires scikit-learn to be installed, which is not available")

from gensim.sklearn_integration.sklearn_wrapper_gensim_ldamodel import SklLdaModel
from gensim.sklearn_integration.sklearn_wrapper_gensim_lsimodel import SklLsiModel
from gensim.corpora import Dictionary
from gensim import matutils

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)

texts = [
    ['complier', 'system', 'computer'],
    ['eulerian', 'node', 'cycle', 'graph', 'tree', 'path'],
    ['graph', 'flow', 'network', 'graph'],
    ['loading', 'computer', 'system'],
    ['user', 'server', 'system'],
    ['tree', 'hamiltonian'],
    ['graph', 'trees'],
    ['computer', 'kernel', 'malfunction', 'computer'],
    ['server', 'system', 'computer'],
]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


class TestSklLdaModelWrapper(unittest.TestCase):
    def setUp(self):
        self.model = SklLdaModel(id2word=dictionary, num_topics=2, passes=100, minimum_probability=0, random_state=numpy.random.seed(0))
        self.model.fit(corpus)

    def testTransform(self):
        texts_new = ['graph', 'eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        matrix = self.model.transform(bow)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], self.model.num_topics)
        texts_new = [['graph', 'eulerian'], ['server', 'flow'], ['path', 'system']]
        bow = []
        for i in texts_new:
            bow.append(self.model.id2word.doc2bow(i))
        matrix = self.model.transform(bow)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.shape[1], self.model.num_topics)

    def testPartialFit(self):
        for i in range(10):
            self.model.partial_fit(X=corpus)  # fit against the model again
            doc = list(corpus)[0]  # transform only the first document
            transformed = self.model.transform(doc)
            transformed_approx = matutils.sparse2full(transformed, 2)  # better approximation
        expected = [0.87, 0.0]
        passed = numpy.allclose(sorted(transformed_approx), sorted(expected), atol=1e-1)
        self.assertTrue(passed)

    def testCSRMatrixConversion(self):
        arr = numpy.array([[1, 2, 0], [0, 0, 3], [1, 0, 0]])
        sarr = sparse.csr_matrix(arr)
        newmodel = SklLdaModel(num_topics=2, passes=100)
        newmodel.fit(sarr)
        bow = [(0, 1), (1, 2), (2, 0)]
        transformed_vec = newmodel.transform(bow)
        expected_vec = [0.35367903, 0.64632097]
        passed = numpy.allclose(sorted(transformed_vec), sorted(expected_vec), atol=1e-1)
        self.assertTrue(passed)

    def testPipeline(self):
        model = SklLdaModel(num_topics=2, passes=10, minimum_probability=0, random_state=numpy.random.seed(0))
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        id2word = Dictionary(map(lambda x: x.split(), data.data))
        corpus = [id2word.doc2bow(i.split()) for i in data.data]
        numpy.random.mtrand.RandomState(1)  # set seed for getting same result
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_lda = Pipeline((('features', model,), ('classifier', clf)))
        text_lda.fit(corpus, data.target)
        score = text_lda.score(corpus, data.target)
        self.assertGreater(score, 0.40)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)

        # updating multiple params
        param_dict = {"eval_every": 20, "decay": 0.7}
        self.model.set_params(**param_dict)
        model_params = self.model.get_params()
        for key in param_dict.keys():
            self.assertEqual(model_params[key], param_dict[key])

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        texts_new = ['graph', 'eulerian']
        loaded_bow = model_load.id2word.doc2bow(texts_new)
        loaded_matrix = model_load.transform(loaded_bow)

        # sanity check for transformation operation
        self.assertEqual(loaded_matrix.shape[0], 1)
        self.assertEqual(loaded_matrix.shape[1], model_load.num_topics)

        # comparing the original and loaded models
        original_bow = self.model.id2word.doc2bow(texts_new)
        original_matrix = self.model.transform(original_bow)
        passed = numpy.allclose(sorted(loaded_matrix), sorted(original_matrix), atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        lda_wrapper = SklLdaModel(id2word=dictionary, num_topics=2, passes=100, minimum_probability=0, random_state=numpy.random.seed(0))
        texts_new = ['graph', 'eulerian']
        bow = lda_wrapper.id2word.doc2bow(texts_new)
        self.assertRaises(NotFittedError, lda_wrapper.transform, bow)


class TestSklLsiModelWrapper(unittest.TestCase):
    def setUp(self):
        self.model = SklLsiModel(id2word=dictionary, num_topics=2)
        self.model.fit(corpus)

    def testTransform(self):
        texts_new = ['graph', 'eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        matrix = self.model.transform(bow)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], self.model.num_topics)
        texts_new = [['graph', 'eulerian'], ['server', 'flow'], ['path', 'system']]
        bow = []
        for i in texts_new:
            bow.append(self.model.id2word.doc2bow(i))
        matrix = self.model.transform(bow)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.shape[1], self.model.num_topics)

    def testPartialFit(self):
        for i in range(10):
            self.model.partial_fit(X=corpus)  # fit against the model again
            doc = list(corpus)[0]  # transform only the first document
            transformed = self.model.transform(doc)
            transformed_approx = matutils.sparse2full(transformed, 2)  # better approximation
        expected = [0, 9.99999996e-13]
        passed = numpy.allclose(sorted(transformed_approx), sorted(expected), atol=1e-1)
        self.assertTrue(passed)

    def testPipeline(self):
        model = SklLsiModel(num_topics=2)
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        id2word = Dictionary(map(lambda x: x.split(), data.data))
        corpus = [id2word.doc2bow(i.split()) for i in data.data]
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_lsi = Pipeline((('features', model,), ('classifier', clf)))
        text_lsi.fit(corpus, data.target)
        score = text_lsi.score(corpus, data.target)
        self.assertGreater(score, 0.50)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)

        # updating multiple params
        param_dict = {"chunksize": 10000, "decay": 0.9}
        self.model.set_params(**param_dict)
        model_params = self.model.get_params()
        for key in param_dict.keys():
            self.assertEqual(model_params[key], param_dict[key])

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        texts_new = ['graph', 'eulerian']
        loaded_bow = model_load.id2word.doc2bow(texts_new)
        loaded_matrix = model_load.transform(loaded_bow)

        # sanity check for transformation operation
        self.assertEqual(loaded_matrix.shape[0], 1)
        self.assertEqual(loaded_matrix.shape[1], model_load.num_topics)

        # comparing the original and loaded models
        original_bow = self.model.id2word.doc2bow(texts_new)
        original_matrix = self.model.transform(original_bow)
        passed = numpy.allclose(sorted(loaded_matrix), sorted(original_matrix), atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        lsi_wrapper = SklLsiModel(id2word=dictionary, num_topics=2)
        texts_new = ['graph', 'eulerian']
        bow = lsi_wrapper.id2word.doc2bow(texts_new)
        self.assertRaises(NotFittedError, lsi_wrapper.transform, bow)


if __name__ == '__main__':
    unittest.main()
