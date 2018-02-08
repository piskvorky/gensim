#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module calculate PageRank [1]_ based on wordgraph.


.. [1] https://en.wikipedia.org/wiki/PageRank

Examples
--------

Calculate Pagerank for words

>>> from gensim.summarization.keywords import get_graph
>>> from gensim.summarization.pagerank_weighted import pagerank_weighted
>>> graph = get_graph("The road to hell is paved with good intentions.")
>>> # result will looks like {'good': 0.70432858653171504, 'hell': 0.051128871128006126, ...}
>>> result = pagerank_weighted(graph)

Build matrix from graph

>>> from gensim.summarization.pagerank_weighted import build_adjacency_matrix
>>> build_adjacency_matrix(graph).todense()
matrix([[ 0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  1.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.]])

"""


import numpy
from numpy import empty as empty_matrix
from scipy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from six.moves import xrange


def pagerank_weighted(graph, damping=0.85):
    """Get dictionary of `graph` nodes and its ranks.

    Parameters
    ----------
    graph : :class:`~gensim.summarization.graph.Graph`
        Given graph.
    damping : float
        Damping parameter, optional

    Returns
    -------
    dict
        Nodes of `graph` as keys, its ranks as values.

    """
    adjacency_matrix = build_adjacency_matrix(graph)
    probability_matrix = build_probability_matrix(graph)

    pagerank_matrix = damping * adjacency_matrix.todense() + (1 - damping) * probability_matrix

    vec = principal_eigenvector(pagerank_matrix.T)

    # Because pagerank_matrix is positive, vec is always real (i.e. not complex)
    return process_results(graph, vec.real)


def build_adjacency_matrix(graph):
    """Get matrix representation of given `graph`.

    Parameters
    ----------
    graph : :class:`~gensim.summarization.graph.Graph`
        Given graph.

    Returns
    -------
    :class:`scipy.sparse.csr_matrix`, shape = [n, n]
        Adjacency matrix of given `graph`, n is number of nodes.

    """
    row = []
    col = []
    data = []
    nodes = graph.nodes()
    length = len(nodes)

    for i in xrange(length):
        current_node = nodes[i]
        neighbors_sum = sum(graph.edge_weight((current_node, neighbor)) for neighbor in graph.neighbors(current_node))
        for j in xrange(length):
            edge_weight = float(graph.edge_weight((current_node, nodes[j])))
            if i != j and edge_weight != 0.0:
                row.append(i)
                col.append(j)
                data.append(edge_weight / neighbors_sum)

    return csr_matrix((data, (row, col)), shape=(length, length))


def build_probability_matrix(graph):
    """Get square matrix of shape (n, n), where n is number of nodes of the
    given `graph`.

    Parameters
    ----------
    graph : :class:`~gensim.summarization.graph.Graph`
        Given graph.

    Returns
    -------
    numpy.ndarray, shape = [n, n]
        Eigenvector of matrix `a`, n is number of nodes of `graph`.

    """
    dimension = len(graph.nodes())
    matrix = empty_matrix((dimension, dimension))

    probability = 1.0 / float(dimension)
    matrix.fill(probability)

    return matrix


def principal_eigenvector(a):
    """Get eigenvector of square matrix `a`.

    Parameters
    ----------
    a : numpy.ndarray, shape = [n, n]
        Given matrix.

    Returns
    -------
    numpy.ndarray, shape = [n, ]
        Eigenvector of matrix `a`.

    """
    # Note that we prefer to use `eigs` even for dense matrix
    # because we need only one eigenvector. See #441, #438 for discussion.

    # But it doesn't work for dim A < 3, so we just handle this special case
    if len(a) < 3:
        vals, vecs = eig(a)
        ind = numpy.abs(vals).argmax()
        return vecs[:, ind]
    else:
        vals, vecs = eigs(a, k=1)
        return vecs[:, 0]


def process_results(graph, vec):
    """Get `graph` nodes and corresponding absolute values of provided eigenvector.
    This function is helper for :func:`~gensim.summarization.pagerank_weighted.pagerank_weighted`

    Parameters
    ----------
    graph : :class:`~gensim.summarization.graph.Graph`
        Given graph.
    vec : numpy.ndarray, shape = [n, ]
        Given eigenvector, n is number of nodes of `graph`.

    Returns
    -------
    dict
        Graph nodes as keys, corresponding elements of eigenvector as values.

    """
    scores = {}
    for i, node in enumerate(graph.nodes()):
        scores[node] = abs(vec[i])

    return scores
