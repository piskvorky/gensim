#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from gensim.summarization.pagerank_weighted import pagerank_weighted as _pagerank
from gensim.summarization.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from gensim.summarization.commons import build_graph as _build_graph
from gensim.summarization.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from gensim.summarization.bm25 import get_bm25_weights as _bm25_weights
from gensim.corpora import Dictionary
from scipy.sparse import csr_matrix
from math import log10 as _log10
from six.moves import xrange

INPUT_MIN_LENGTH = 10


def _set_graph_edge_weights(graph):
    documents = graph.nodes()
    weights = _bm25_weights(documents)

    for i in xrange(len(documents)):
        for j in xrange(len(documents)):
            if i == j:
                continue

            sentence_1 = documents[i]
            sentence_2 = documents[j]

            edge_1 = (sentence_1, sentence_2)
            edge_2 = (sentence_2, sentence_1)

            if not graph.has_edge(edge_1):
                graph.add_edge(edge_1, weights[i][j])
            if not graph.has_edge(edge_2):
                graph.add_edge(edge_2, weights[j][i])

    # Handles the case in which all similarities are zero.
    # The resultant summary will consist of random sentences.
    if all(graph.edge_weight(edge) == 0 for edge in graph.edges()):
        _create_valid_graph(graph)


def _create_valid_graph(graph):
    nodes = graph.nodes()

    for i in xrange(len(nodes)):
        for j in xrange(len(nodes)):
            if i == j:
                continue

            edge = (nodes[i], nodes[j])

            if graph.has_edge(edge):
                graph.del_edge(edge)

            graph.add_edge(edge, 1)


def _get_doc_length(doc):
    return sum([item[1] for item in doc])


def _get_similarity(doc1, doc2, vec1, vec2):
    numerator = vec1.dot(vec2.transpose()).toarray()[0][0]
    length_1 = _get_doc_length(doc1)
    length_2 = _get_doc_length(doc2)

    denominator = _log10(length_1) + _log10(length_2) if length_1 > 0 and length_2 > 0 else 0

    return numerator / denominator if denominator != 0 else 0


def _build_corpus(sentences):
    split_tokens = [sentence.token.split() for sentence in sentences]
    dictionary = Dictionary(split_tokens)
    return [dictionary.doc2bow(token) for token in split_tokens]


def _get_important_sentences(sentences, corpus, important_docs):
    hashable_corpus = _build_hasheable_corpus(corpus)
    sentences_by_corpus = dict(zip(hashable_corpus, sentences))
    return [sentences_by_corpus[tuple(important_doc)] for important_doc in important_docs]


def _get_sentences_with_word_count(sentences, word_count):
    """ Given a list of sentences, returns a list of sentences with a
    total word count similar to the word count provided."""
    length = 0
    selected_sentences = []

    # Loops until the word count is reached.
    for sentence in sentences:
        words_in_sentence = len(sentence.text.split())

        # Checks if the inclusion of the sentence gives a better approximation
        # to the word parameter.
        if abs(word_count - length - words_in_sentence) > abs(word_count - length):
            return selected_sentences

        selected_sentences.append(sentence)
        length += words_in_sentence

    return selected_sentences


def _extract_important_sentences(sentences, corpus, important_docs, word_count):
    important_sentences = _get_important_sentences(sentences, corpus, important_docs)

    # If no "words" option is selected, the number of sentences is
    # reduced by the provided ratio. Else, the ratio is ignored.
    return important_sentences if word_count is None else _get_sentences_with_word_count(important_sentences, word_count)


def _format_results(extracted_sentences, split):
    if split:
        return [sentence.text for sentence in extracted_sentences]
    return "\n".join([sentence.text for sentence in extracted_sentences])


def _build_hasheable_corpus(corpus):
    return [tuple(doc) for doc in corpus]


def summarize_corpus(corpus, ratio=0.2):
    hashable_corpus = _build_hasheable_corpus(corpus)

    if len(corpus) < INPUT_MIN_LENGTH:
        raise RuntimeError("Input corpus must have at least " + str(INPUT_MIN_LENGTH) + " documents.")

    graph = _build_graph(hashable_corpus)
    _set_graph_edge_weights(graph)
    _remove_unreachable_nodes(graph)

    pagerank_scores = _pagerank(graph)

    hashable_corpus.sort(key=lambda doc: pagerank_scores.get(doc, 0), reverse=True)

    return [list(doc) for doc in hashable_corpus[:int(len(corpus) * ratio)]]


def summarize(text, ratio=0.2, word_count=None, split=False):
    # Gets a list of processed sentences.
    sentences = _clean_text_by_sentences(text)

    if len(sentences) < INPUT_MIN_LENGTH:
        raise RuntimeError("Input text must have at least " + str(INPUT_MIN_LENGTH) + " sentences.")

    corpus = _build_corpus(sentences)

    most_important_docs = textrank_from_corpus(corpus, ratio=ratio)

    # Extracts the most important sentences with the selected criterion.
    extracted_sentences = _extract_important_sentences(sentences, corpus, most_important_docs, word_count)

    # Sorts the extracted sentences by apparition order in the original text.
    extracted_sentences.sort(key=lambda s: s.index)

    return _format_results(extracted_sentences, split)
