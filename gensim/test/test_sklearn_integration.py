import six
import unittest
import numpy

from scipy import sparse
from gensim.sklearn_integration.sklearn_wrapper_gensim_ldamodel import SklearnWrapperLdaModel
from gensim.corpora import Dictionary
from gensim import matutils

texts = [['complier', 'system', 'computer'],
         ['eulerian', 'node', 'cycle', 'graph', 'tree', 'path'],
         ['graph', 'flow', 'network', 'graph'],
         ['loading', 'computer', 'system'],
         ['user', 'server', 'system'],
         ['tree', 'hamiltonian'],
         ['graph', 'trees'],
         ['computer', 'kernel', 'malfunction', 'computer'],
         ['server', 'system', 'computer']]
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
        texts_new = ['graph','eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        doc_topics, word_topics, phi_values = self.model.transform(bow,per_word_topics=True)

        for k,v in word_topics:
            self.assertTrue(isinstance(v, list))
            self.assertTrue(isinstance(k, int))
        for k,v in doc_topics:
            self.assertTrue(isinstance(v, float))
            self.assertTrue(isinstance(k, int))
        for k,v in phi_values:
            self.assertTrue(isinstance(v, list))
            self.assertTrue(isinstance(k, int))

    def testPartialFit(self):
        for i in range(10):
            self.model.partial_fit(X=corpus)  # fit against the model again
            doc=list(corpus)[0]  # transform only the first document
            transformed = self.model[doc]
            transformed_approx = matutils.sparse2full(transformed, 2)  # better approximation
        expected=[0.13, 0.87]
        passed = numpy.allclose(sorted(transformed_approx), sorted(expected), atol=1e-1)
        self.assertTrue(passed)

    def testCSRMatrixConversion(self):
        Arr = numpy.array([[1, 2, 0], [0, 0, 3], [1, 0, 0]])
        sArr = sparse.csr_matrix(Arr)
        newmodel = SklearnWrapperLdaModel(num_topics=2, passes=100)
        newmodel.fit(sArr)
        topic = newmodel.print_topics()
        for k, v in topic:
            self.assertTrue(isinstance(v, six.string_types))
            self.assertTrue(isinstance(k, int))

if __name__ == '__main__':
    unittest.main()
