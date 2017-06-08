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
except ImportError:
    raise unittest.SkipTest("Test requires scikit-learn to be installed, which is not available")

from gensim.sklearn_integration.sklearn_wrapper_gensim_ldamodel import SklearnWrapperLdaModel
from gensim.sklearn_integration.sklearn_wrapper_gensim_lsimodel import SklearnWrapperLsiModel
from gensim.sklearn_integration.sklearn_wrapper_gensim_rpmodel import SklearnWrapperRpModel
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


class TestSklearnLDAWrapper(unittest.TestCase):
    def setUp(self):
        self.model = SklearnWrapperLdaModel(id2word=dictionary, num_topics=2, passes=100, minimum_probability=0, random_state=numpy.random.seed(0))
        self.model.fit(corpus)

    def testPrintTopic(self):
        topic = self.model.print_topics(2)

        for k, v in topic:
            self.assertTrue(isinstance(v, six.string_types))
            self.assertTrue(isinstance(k, int))

    def testTransform(self):
        texts_new = ['graph', 'eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        matrix = self.model.transform(bow)
        self.assertTrue(matrix.shape[0], 1)
        self.assertTrue(matrix.shape[1], self.model.num_topics)
        texts_new = [['graph', 'eulerian'], ['server', 'flow'], ['path', 'system']]
        bow = []
        for i in texts_new:
            bow.append(self.model.id2word.doc2bow(i))
        matrix = self.model.transform(bow)
        self.assertTrue(matrix.shape[0], 3)
        self.assertTrue(matrix.shape[1], self.model.num_topics)

    def testGetTopicDist(self):
        texts_new = ['graph', 'eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        doc_topics, word_topics, phi_values = self.model.get_topic_dist(bow, per_word_topics=True)

        for k, v in word_topics:
            self.assertTrue(isinstance(v, list))
            self.assertTrue(isinstance(k, int))
        for k, v in doc_topics:
            self.assertTrue(isinstance(v, float))
            self.assertTrue(isinstance(k, int))
        for k, v in phi_values:
            self.assertTrue(isinstance(v, list))
            self.assertTrue(isinstance(k, int))

    def testPartialFit(self):
        for i in range(10):
            self.model.partial_fit(X=corpus)  # fit against the model again
            doc = list(corpus)[0]  # transform only the first document
            transformed = self.model[doc]
            transformed_approx = matutils.sparse2full(transformed, 2)  # better approximation
        expected = [0.13, 0.87]
        passed = numpy.allclose(sorted(transformed_approx), sorted(expected), atol=1e-1)
        self.assertTrue(passed)

    def testCSRMatrixConversion(self):
        arr = numpy.array([[1, 2, 0], [0, 0, 3], [1, 0, 0]])
        sarr = sparse.csr_matrix(arr)
        newmodel = SklearnWrapperLdaModel(num_topics=2, passes=100)
        newmodel.fit(sarr)
        topic = newmodel.print_topics()
        for k, v in topic:
            self.assertTrue(isinstance(v, six.string_types))
            self.assertTrue(isinstance(k, int))

    def testPipeline(self):
        model = SklearnWrapperLdaModel(num_topics=2, passes=10, minimum_probability=0, random_state=numpy.random.seed(0))
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


class TestSklearnLSIWrapper(unittest.TestCase):
    def setUp(self):
        self.model = SklearnWrapperLsiModel(id2word=dictionary, num_topics=2)
        self.model.fit(corpus)

    def testModelSanity(self):
        topic = self.model.print_topics(2)
        for k, v in topic:
            self.assertTrue(isinstance(v, six.string_types))
            self.assertTrue(isinstance(k, int))

    def testTransform(self):
        texts_new = ['graph', 'eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        matrix = self.model.transform(bow)
        self.assertTrue(matrix.shape[0], 1)
        self.assertTrue(matrix.shape[1], self.model.num_topics)
        texts_new = [['graph', 'eulerian'], ['server', 'flow'], ['path', 'system']]
        bow = []
        for i in texts_new:
            bow.append(self.model.id2word.doc2bow(i))
        matrix = self.model.transform(bow)
        self.assertTrue(matrix.shape[0], 3)
        self.assertTrue(matrix.shape[1], self.model.num_topics)

    def testPartialFit(self):
        for i in range(10):
            self.model.partial_fit(X=corpus)  # fit against the model again
            doc = list(corpus)[0]  # transform only the first document
            transformed = self.model[doc]
            transformed_approx = matutils.sparse2full(transformed, 2)  # better approximation
        expected = [1.39, 0.0]
        passed = numpy.allclose(sorted(transformed_approx), sorted(expected), atol=1e-1)
        self.assertTrue(passed)

    def testPipeline(self):
        model = SklearnWrapperLsiModel(num_topics=2)
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        id2word = Dictionary(map(lambda x: x.split(), data.data))
        corpus = [id2word.doc2bow(i.split()) for i in data.data]
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_lda = Pipeline((('features', model,), ('classifier', clf)))
        text_lda.fit(corpus, data.target)
        score = text_lda.score(corpus, data.target)
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


class TestSklearnRpModelWrapper(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(13)
        self.model = SklearnWrapperRpModel(corpus, num_topics=2)
        self.model.fit(corpus)

    def testTransform(self):
        # transform one document
        doc = list(self.model.corpus)[0]
        transformed_doc = self.model.transform(doc)
        vec = matutils.sparse2full(transformed_doc, 2)  # convert to dense vector, for easier equality tests

        expected_vec = numpy.array([-0.70710677, 0.70710677])
        self.assertTrue(numpy.allclose(vec, expected_vec))  # transformed entries must be equal up to sign

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)


if __name__ == '__main__':
    unittest.main()
