#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""


Example
-------
>>> from gensim.summarization.keywords import get_graph 
>>> from gensim.summarization.pagerank_weighted import pagerank_weighted
>>> text = "In graph theory and computer science, an adjacency matrix \
>>> is a square matrix used to represent a finite graph."
>>> graph = get_graph(text)
>>> pagerank_weighted(graph)
{'adjac': array([ 0.29628575]),
 'finit': array([ 0.29628575]),
 'graph': array([ 0.56766066]),
 'matrix': array([ 0.56766066]),
 'repres': array([ 0.04680678]),
 'scienc': array([ 0.04680678]),
 'squar': array([ 0.29628575]),
 'theori': array([ 0.29628575])}

"""


import numpy
from numpy import empty as empty_matrix
from scipy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from six.moves import xrange

try:
    from numpy import VisibleDeprecationWarning
    import warnings
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
except ImportError:
    pass


def pagerank_weighted(graph, damping=0.85):
    """Returns dictionary of `graph`'s nodes and its ranks.

    Parameters
    ----------
    graph : Graph
        Given graph.
    damping : float
        Damping parameter, optional

    Returns
    -------
    dict
        Keys are `graph` nodes, values are its ranks.

    """
    adjacency_matrix = build_adjacency_matrix(graph)
    probability_matrix = build_probability_matrix(graph)

    pagerank_matrix = damping * adjacency_matrix.todense() + (1 - damping) * probability_matrix

    vec = principal_eigenvector(pagerank_matrix.T)

    # Because pagerank_matrix is positive, vec is always real (i.e. not complex)
    return process_results(graph, vec.real)


def build_adjacency_matrix(graph):
    """Returns matrix representation of given `graph`. 

    Parameters
    ----------
    graph : Graph
        Given graph.

    Returns
    -------
    csr_matrix (n, n)
        Adjacency matrix of given `graph`.

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
    """Returns square matrix of shape (n, n), where n is number of nodes of the
    given `graph`.

    Parameters
    ----------
    graph : Graph
        Given graph.

    Returns
    -------
    array (n, )
        Eigenvector of matrix `a`.

    """
    dimension = len(graph.nodes())
    matrix = empty_matrix((dimension, dimension))

    probability = 1.0 / float(dimension)
    matrix.fill(probability)

    return matrix


def principal_eigenvector(a):
    """Returns eigenvector of square matrix `a`.

    Parameters
    ----------
    a : array (n, n)
        Given matrix.

    Returns
    -------
    array (n, )
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
    """Returns `graph` nodes and corresponding absolute values of provided 
    eigenvector.

    Parameters
    ----------
    graph : Graph
        Given graph.
    vec : array
        Given eigenvector.

    Returns
    -------
    dict
        Keys are graph nodes, values are elements of eigenvector.

    """
    scores = {}
    for i, node in enumerate(graph.nodes()):
        scores[node] = abs(vec[i])

    return scores
