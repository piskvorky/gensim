#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module provides functions of creating graph from sequence of values and removing of unreachable nodes.


Examples
--------

Create simple graph and add edges. Let's take a look at nodes.

>>> gg = build_graph(['Felidae', 'Lion', 'Tiger', 'Wolf'])
>>> gg.add_edge(("Felidae", "Lion"))
>>> gg.add_edge(("Felidae", "Tiger"))
>>> sorted(gg.nodes())
['Felidae', 'Lion', 'Tiger', 'Wolf']

Remove nodes with no edges.

>>> remove_unreachable_nodes(gg)
>>> sorted(gg.nodes())
['Felidae', 'Lion', 'Tiger']

"""

from gensim.summarization.graph import Graph


def build_graph(sequence):
    """Creates and returns undirected graph with given sequence of values.

    Parameters
    ----------
    sequence : list of hashable
        Sequence of values.

    Returns
    -------
    :class:`~gensim.summarization.graph.Graph`
        Created graph.

    """
    graph = Graph()
    for item in sequence:
        if not graph.has_node(item):
            graph.add_node(item)
    return graph


def remove_unreachable_nodes(graph):
    """Removes unreachable nodes (nodes with no edges), inplace.

    Parameters
    ----------
    graph : :class:`~gensim.summarization.graph.Graph`
        Given graph.

    """

    for node in graph.nodes():
        if sum(graph.edge_weight((node, other)) for other in graph.neighbors(node)) == 0:
            graph.del_node(node)
