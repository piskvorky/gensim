from math import log10 as _log10

from pagerank_weighted import pagerank_weighted_scipy as _pagerank
from gensim.summarization.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from gensim.corpora import Dictionary
from commons import build_graph as _build_graph
from commons import remove_unreachable_nodes as _remove_unreachable_nodes
from scipy.sparse import csr_matrix

import pdb

def _build_sparse_vectors(docs, num_features):
    vectors = []
    for doc in docs:
        freq = [1 for item in doc]
        id = [item[0] for item in doc]
        zeros = [0 for i in doc]
        vector = csr_matrix((freq, (zeros, id)), shape=(1,num_features))
        vectors.append(vector)
    return vectors


def _get_doc_length(doc):
    return sum([item[1] for item in doc])


def _get_similarity(doc1, doc2, vec1, vec2):
    numerator = vec1.dot(vec2.transpose()).toarray()[0][0]
    length_1 = _get_doc_length(doc1)
    length_2 = _get_doc_length(doc2)
    return numerator / (_log10(length_1) + _log10(length_2))


def _set_graph_edge_weights(graph, num_features):
    nodes = graph.nodes()
    sparse_vectors = _build_sparse_vectors(nodes, num_features)

    for i in xrange(len(nodes)):
        for j in xrange(len(nodes)):
            if i == j: continue

            edge = (nodes[i], nodes[j])
            if not graph.has_edge(edge):
                # pdb.set_trace()
                similarity = _get_similarity(nodes[i], nodes[j], sparse_vectors[i], sparse_vectors[j])
                if similarity != 0:
                    print edge, similarity
                    graph.add_edge(edge, similarity)


def _format_results(extracted_sentences, split, score):
    if score:
        return [(sentence.text, sentence.score) for sentence in extracted_sentences]
    if split:
        return [sentence.text for sentence in extracted_sentences]
    return "\n".join([sentence.text for sentence in extracted_sentences])


def _add_scores_to_sentences(sentences, scores):
    for sentence in sentences:
        # Adds the score to the object if it has one.
        if sentence.token in scores:
            sentence.score = scores[sentence.token]
        else:
            sentence.score = 0


def _get_sentences_with_word_count(sentences, words):
    """ Given a list of sentences, returns a list of sentences with a
    total word count similar to the word count provided.
    """
    word_count = 0
    selected_sentences = []
    # Loops until the word count is reached.
    for sentence in sentences:
        words_in_sentence = len(sentence.text.split())

        # Checks if the inclusion of the sentence gives a better approximation
        # to the word parameter.
        if abs(words - word_count - words_in_sentence) > abs(words - word_count):
            return selected_sentences

        selected_sentences.append(sentence)
        word_count += words_in_sentence

    return selected_sentences


def _extract_most_important_sentences(sentences, ratio, words):
    sentences.sort(key=lambda s: s.score, reverse=True)

    # If no "words" option is selected, the number of sentences is
    # reduced by the provided ratio.
    if words is None:
        length = len(sentences) * ratio
        return sentences[:int(length)]

    # Else, the ratio is ignored.
    else:
        return _get_sentences_with_word_count(sentences, words)


def summarize(text, ratio=0.2, words=None, split=False, scores=False):
    # Gets a list of processed sentences.
    sentences = _clean_text_by_sentences(text)

    # Creates the graph and calculates the similarity coefficient for every pair of nodes.
    graph = _build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights_REAL(graph)


    # current_node = sentences[0]
    # print [graph.edge_weight((current_node, neighbor)) for neighbor in graph.neighbors(current_node)]


    # Remove all nodes with all edges weights equal to zero.
    _remove_unreachable_nodes(graph)

    # Ranks the tokens using the PageRank algorithm. Returns dict of sentence -> score
    pagerank_scores = _pagerank(graph)
    print "PAGERANK:" + str(pagerank_scores)

    # Adds the summa scores to the sentence objects.
    _add_scores_to_sentences(sentences, pagerank_scores)

    # Extracts the most important sentences with the selected criterion.
    extracted_sentences = _extract_most_important_sentences(sentences, ratio, words)

    # Sorts the extracted sentences by apparition order in the original text.
    extracted_sentences.sort(key=lambda s: s.index)

    return _format_results(extracted_sentences, split, scores)


def textrank_from_corpus(corpus, num_features, ratio=0.2, scores=False):
    hashable_corpus = [tuple(doc) for doc in corpus]
    graph = _build_graph(hashable_corpus)
    _set_graph_edge_weights(graph, num_features)

    # current_node = hashable_corpus[0]
    # print [graph.edge_weight((current_node, neighbor)) for neighbor in graph.neighbors(current_node)]

    _remove_unreachable_nodes(graph)
    pagerank_scores = _pagerank(graph)
    print "PAGERANK:" + str(pagerank_scores)
    hashable_corpus.sort(key=lambda d:pagerank_scores[d], reverse=True)
    return [list(doc) for doc in hashable_corpus[:int(len(corpus) * ratio)]]


def get_graph(text):
    sentences = _clean_text_by_sentences(text)

    graph = _build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph)

    return graph



def build_dictionary_and_corpus(sentences):
    split_tokens = [sentence.token.split() for sentence in sentences]
    dictionary = Dictionary(split_tokens)
    corpus = [dictionary.doc2bow(token) for token in split_tokens]
    return dictionary, corpus



def _set_graph_edge_weights_REAL(graph):
    for sentence_1 in graph.nodes():
        for sentence_2 in graph.nodes():

            edge = (sentence_1, sentence_2)
            if sentence_1 != sentence_2 and not graph.has_edge(edge):
                similarity = _get_similarity_REAL(sentence_1, sentence_2)
                if similarity != 0:
                    print edge, similarity
                    graph.add_edge(edge, similarity)


def _get_similarity_REAL(s1, s2):
    words_sentence_one = s1.split()
    words_sentence_two = s2.split()

    common_word_count = _count_common_words(words_sentence_one, words_sentence_two)

    log_s1 = _log10(len(words_sentence_one))
    log_s2 = _log10(len(words_sentence_two))

    if log_s1 + log_s2 == 0:
        return 0

    return common_word_count / (log_s1 + log_s2)


def _count_common_words(words_sentence_one, words_sentence_two):
    words_set_one = set(words_sentence_one)
    words_set_two = set(words_sentence_two)
    return sum(1 for w in words_set_one if w in words_set_two)