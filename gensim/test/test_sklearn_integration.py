import six
import unittest

from gensim.sklearn_integration import base
from gensim.corpora import Dictionary
texts = [['human', 'interface', 'computer'],
 ['survey', 'user', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'system'],
 ['system', 'human', 'system', 'eps'],
 ['user', 'response', 'time'],
 ['trees'],
 ['graph', 'trees'],
 ['graph', 'minors', 'trees'],
 ['graph', 'minors', 'survey']]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


class TestLdaModel(object):
    def __init__(self):
        self.model=base.LdaModel(id2word=dictionary,n_topics=2,passes=100)
        self.model.fit(corpus)

    def testPrintTopic(self):
        topic = self.model.print_topics(2)

        for k, v in topic:
            self.assertTrue(isinstance(k, six.string_types))
            self.assertTrue(isinstance(v, float))

if __name__ == '__main__':
    unittest.main()

