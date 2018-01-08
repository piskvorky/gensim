import logging
import unittest

import numpy as np

from gensim.models import sldamodel

logger = logging.getLogger(__name__)

"""

The code draws directly from sLDA library by Matt Burbidge: https://github.com/Savvysherpa/slda

"""


def generate_topics(rows):
    topics = []
    base_topic = np.concatenate((np.ones((1, rows)) * (1/rows),
                                 np.zeros((rows-1, rows))), axis=0).ravel()
    for i in range(rows):
        topics.append(np.roll(base_topic, i * rows))
    topic_base = np.concatenate((np.ones((rows, 1)) * (1/rows),
                                 np.zeros((rows, rows-1))), axis=1).ravel()
    for i in range(rows):
        topics.append(np.roll(base_topic, i))
    return np.array(topics)

def gen_documents(seed, K, N, thetas, V, topics, D):
    topic_assignments = np.array([np.random.choice(range(K), size=N, p=theta)
                                  for theta in thetas])
    word_assignment = np.array([[np.random.choice(range(V), size=1, p=topics[topic_assignments[d, n]])[0] for n in range(N)] for d in range(D)])
    return np.array([np.histogram(word_assignment[d], bins=V, range=(0, V - 1))[0] for d in range(D)])


def language(document_size):
    # Generate topics
    rows = 3
    V = rows * rows
    K = rows * 2
    N = K * K
    D = document_size
    seed = 21
    topics = generate_topics(rows)

    # Generate documents from topics
    alpha = np.ones(K)
    np.random.seed(seed)
    thetas = gen_thetas(alpha, D)
    doc_term_matrix = gen_documents(seed, K, N, thetas, V, topics, D)
    return {'V': V, 'K': K, 'D': D, 'seed': seed, 'alpha': alpha,
            'topics': topics, 'thetas': thetas,
            'doc_term_matrix': doc_term_matrix}



def gen_thetas(alpha, D):
    return np.random.dirichlet(alpha, size=D)


def assert_probablity_distribution(results):
    assert (results >= 0).all()
    assert results.sum(axis=1).all()

class TestSLdaModel(unittest.TestCase):

    def test_slda(self):
        l = language(10000)
        n_iter = 2000

        nu = l['K']
        sigma2 = 1
        np.random.seed(l['seed'])
        eta = np.random.normal(scale=nu2, size=l['K'])
        y = [np.dot(eta, l['thetas'][i]) for i in range(l['D'])] + \
            np.random.normal(scale=sigma2, size=l['D'])
        beta = np.repeat(0.01, l['V'])
        mu = 0
        slda = SLDA(l['K'], l['alpha'], beta, mu, nu, sigma, n_iter,
                    seed=l['seed'])
        slda.fit(l['doc_term_matrix'], y)

        assert_probablity_distribution(slda.phi)
