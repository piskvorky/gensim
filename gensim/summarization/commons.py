#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from gensim.summarization.graph import Graph


def build_graph(sequence):
    graph = Graph()
    for item in sequence:
        if not graph.has_node(item):
            graph.add_node(item)
    return graph


def remove_unreachable_nodes(graph):
    for node in graph.nodes():
        if sum(graph.edge_weight((node, other)) for other in graph.neighbors(node)) == 0:
            graph.del_node(node)
