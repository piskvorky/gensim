from os import system as _shell

import networkx as _nx

from summarizer import get_graph as _get_sentence_graph
from keywords import get_graph as _get_word_graph
from pagerank_weighted import pagerank_weighted_scipy as _pagerank_weighted_scipy
from gensim.summarization.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from gensim.summarization.textcleaner import clean_text_by_word as _clean_text_by_word


NODE_COLOR = {'r': 239, 'g': 10, 'b': 10}


def _get_labels(text, language, by_sentence):
    syntactic_units = _clean_text_by_sentences(text, language) if by_sentence \
        else _clean_text_by_word(text, language).values()
    return {unit.token: unit.text for unit in syntactic_units}


def _write_gexf(graph, scores, path="test.gexf", labels=None):
    nx_graph = _get_nx_graph(graph)
    _set_layout(nx_graph, scores, labels)
    _nx.write_gexf(nx_graph, path)
    _shell("sed -i 's/<ns0/<viz/g' {0}".format(path))
    _shell('echo \'<?xml version="1.0" encoding="UTF-8"?>\' | cat - {0} > out.tmp && mv out.tmp {0}'.format(path))
    #_shell("mv {0} views/{0}".format(path))


def _get_nx_graph(graph):
    nx_graph = _nx.Graph()
    nx_graph.add_nodes_from(graph.nodes())
    for edge in graph.edges():
        weight = graph.edge_weight(edge)
        if weight != 0:
            nx_graph.add_edge(edge[0], edge[1], {'weight':weight})
    return nx_graph


def _set_layout(nx_graph, scores, labels):
    positions = _nx.graphviz_layout(nx_graph, prog="neato") # prog options: neato, dot, fdp, sfdp, twopi, circo
    centered_positions = _center_positions(positions)
    for node in nx_graph.nodes():
        nx_graph.node[node]['viz'] = _get_viz_data(node, centered_positions, scores)
        label = labels[node] if labels is not None and node in labels else node
        nx_graph.node[node]['label'] = " ".join(label.split()[0:2])
        nx_graph.node[node]['id'] = label


def _center_positions(positions):
    min_x = positions[min(positions, key=lambda k:positions[k][0])][0]
    min_y = positions[min(positions, key=lambda k:positions[k][1])][1]
    max_x = positions[max(positions, key=lambda k:positions[k][0])][0]
    max_y = positions[max(positions, key=lambda k:positions[k][1])][1]
    delta_x = (min_x + max_x) / 2
    delta_y = (min_y + max_y) / 2

    centered_positions = {}
    for key, position in positions.iteritems():
        new_position = (round(position[0] - delta_x, 2), round(position[1] - delta_y, 2))
        centered_positions[key] = new_position
    return centered_positions


def _get_viz_data(node, positions, scores):
    viz_data = {}
    viz_data['position'] = {'x':positions[node][0], 'y':positions[node][1]}
    viz_data['size'] = scores[node]
    viz_data['color'] = NODE_COLOR
    return viz_data


def gexf_export(text, language="english", path="test.gexf", labels=None, by_sentence=True, by_word=False):
    if (by_sentence and by_word) or (not by_sentence and not by_word):
        raise TypeError("Must select one and only one of by_sentence or by_word")
    if labels is None:
        labels = _get_labels(text, language, by_sentence)

    graph = _get_sentence_graph(text, language) if by_sentence else _get_word_graph(text, language)
    scores = _pagerank_weighted_scipy(graph)
    _write_gexf(graph, scores, path, labels)


def gexf_export_from_graph(graph, path="test.gexf", labels=None):
    scores = _pagerank_weighted_scipy(graph)
    _write_gexf(graph, scores, path, labels)