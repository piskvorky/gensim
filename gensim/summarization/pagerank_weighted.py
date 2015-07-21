#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from numpy import empty as empty_matrix
from scipy.sparse import csr_matrix
from scipy.linalg import eig
from six.moves import xrange

try:
    from numpy import VisibleDeprecationWarning
    import warnings
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
except ImportError:
    pass


def pagerank_weighted(graph, damping=0.85):
    adjacency_matrix = build_adjacency_matrix(graph)
    probability_matrix = build_probability_matrix(graph)

    pagerank_matrix = damping * adjacency_matrix.todense() + (1 - damping) * probability_matrix
    vals, vecs = eig(pagerank_matrix, left=True, right=False)
    return process_results(graph, vecs)


def build_adjacency_matrix(graph):
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
            if i != j and edge_weight != 0:
                row.append(i)
                col.append(j)
                data.append(edge_weight / neighbors_sum)

    return csr_matrix((data, (row, col)), shape=(length, length))


def build_probability_matrix(graph):
    dimension = len(graph.nodes())
    matrix = empty_matrix((dimension, dimension))

    probability = 1 / float(dimension)
    matrix.fill(probability)

    return matrix


def process_results(graph, vecs):
    scores = {}
    for i, node in enumerate(graph.nodes()):
        scores[node] = abs(vecs[i][0])

    return scores
