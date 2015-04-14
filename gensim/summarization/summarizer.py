from math import log10 as _log10

from pagerank_weighted import pagerank_weighted_scipy as _pagerank
from gensim.summarization.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from gensim.corpora import Dictionary
from commons import build_graph as _build_graph
from commons import remove_unreachable_nodes as _remove_unreachable_nodes
from scipy.sparse import csr_matrix

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

    denominator = _log10(length_1) + _log10(length_2)

    return numerator if denominator != 0 else 0


def _set_graph_edge_weights(graph, num_features):
    nodes = graph.nodes()
    sparse_vectors = _build_sparse_vectors(nodes, num_features)

    for i in xrange(len(nodes)):
        for j in xrange(len(nodes)):
            if i == j: continue

            edge = (nodes[i], nodes[j])
            if not graph.has_edge(edge):
                similarity = _get_similarity(nodes[i], nodes[j], sparse_vectors[i], sparse_vectors[j])
                if similarity != 0:
                    graph.add_edge(edge, similarity)


def _build_dictionary_and_corpus(sentences):
    split_tokens = [sentence.token.split() for sentence in sentences]
    dictionary = Dictionary(split_tokens)
    corpus = [dictionary.doc2bow(token) for token in split_tokens]
    return dictionary, corpus


def _get_important_sentences(sentences, corpus, important_docs):
    hashable_corpus = _build_hashable_corpus(corpus)
    sentences_by_corpus = {hashable_corpus[i]:sentences[i] for i in xrange(len(sentences))}
    return [sentences_by_corpus[tuple(important_doc)] for important_doc in important_docs]


def _get_sentences_with_word_count(sentences, words):
    """ Given a list of sentences, returns a list of sentences with a
    total word count similar to the word count provided."""
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


def _extract_important_sentences(sentences, corpus, important_docs, words):
    important_sentences = _get_important_sentences(sentences, corpus, important_docs)

    # If no "words" option is selected, the number of sentences is
    # reduced by the provided ratio. Else, the ratio is ignored.
    return important_sentences if words is None else _get_sentences_with_word_count(important_sentences, words)


def _format_results(extracted_sentences, split):
    if split:
        return [sentence.text for sentence in extracted_sentences]
    return "\n".join([sentence.text for sentence in extracted_sentences])


def _build_hashable_corpus(corpus):
    return [tuple(doc) for doc in corpus]


def textrank_from_corpus(corpus, num_features, ratio=0.2):
    hashable_corpus = _build_hashable_corpus(corpus)

    graph = _build_graph(hashable_corpus)
    _set_graph_edge_weights(graph, num_features)
    _remove_unreachable_nodes(graph)

    pagerank_scores = _pagerank(graph)

    hashable_corpus.sort(key=lambda doc:pagerank_scores.get(doc, 0), reverse=True)

    return [list(doc) for doc in hashable_corpus[:int(len(corpus) * ratio)]]


def summarize(text, ratio=0.2, words=None, split=False):
    # Gets a list of processed sentences.
    sentences = _clean_text_by_sentences(text)
    dictionary, corpus = _build_dictionary_and_corpus(sentences)

    most_important_docs = textrank_from_corpus(corpus, len(dictionary.token2id), ratio)

    # Extracts the most important sentences with the selected criterion.
    extracted_sentences = _extract_important_sentences(sentences, corpus, most_important_docs, words)

    # Sorts the extracted sentences by apparition order in the original text.
    extracted_sentences.sort(key=lambda s: s.index)

    return _format_results(extracted_sentences, split)