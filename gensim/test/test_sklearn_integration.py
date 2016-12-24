import six
import unittest

from gensim.sklearn_integration import base
from gensim.corpora import Dictionary
texts = [['complier', 'system', 'computer'],
 ['eulerian', 'node', 'cycle', 'graph', 'tree', 'path'],
 ['graph', 'flow', 'network', 'graph'],
 ['loading', 'computer', 'system'],
 ['user', 'server', 'system'],
 ['tree','hamiltonian'],
 ['graph', 'trees'],
 ['computer', 'kernel', 'malfunction','computer'],
 ['server','system','computer']]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


class TestLdaModel(unittest.TestCase):
    def setUp(self):
        self.model=base.LdaModel(id2word=dictionary,num_topics=2,passes=100)
        self.model.fit(corpus)

    def testPrintTopic(self):
        topic = self.model.print_topics(2)

        for k, v in topic:
            self.assertTrue(isinstance(v, six.string_types))
            self.assertTrue(isinstance(k, int))

    def testTransform(self):
        texts_new=['graph','eulerian']
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
        texts_update=[['graph','eulerian','maxflow'],
                     ['graph','maxflow','graph'],
                     ['graph','maxflow']]
        dictionary_up = Dictionary(texts_update)
        corpus_up = [dictionary_up.doc2bow(text) for text in texts_update]
        self.model.partial_fit(corpus_up)
        self.testPrintTopic()

if __name__ == '__main__':
    unittest.main()
