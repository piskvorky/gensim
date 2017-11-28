#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains functions to find keywords of the text and building
graph on tokens from text.


Examples
--------
>>> from gensim.summarization import keywords
>>> text='''Challenges in natural language processing frequently involve
... speech recognition, natural language understanding, natural language
... generation (frequently from formal, machine-readable logical forms),
... connecting language and machine perception, dialog systems, or some
... combination thereof.'''
>>> print(gensim.summarization.keywords(text))
natural language
machine
frequently


>>> from gensim.summarization.keywords import get_graph
>>> text = '''Fly me to the moon
... Let me play among the stars
... Let me see what spring is like
... On a, Jupiter and Mars'''
>>> g = get_graph(text)
>>> print(g.nodes())
['fly', 'moon', 'let', 'plai', 'star', 'spring', 'like', 'jupit', 'mar']
>>> print(g.neighbors("let"))
['moon', 'star']



Data:
-----
.. data:: WINDOW_SIZE - Size of window, number of consequtive tokens in processing.
.. data:: INCLUDING_FILTER - including part of speech filters.
.. data:: EXCLUDING_FILTER - excluding part of speech filters.

"""

from gensim.summarization.pagerank_weighted import pagerank_weighted as _pagerank
from gensim.summarization.textcleaner import clean_text_by_word as _clean_text_by_word
from gensim.summarization.textcleaner import tokenize_by_word as _tokenize_by_word
from gensim.summarization.commons import build_graph as _build_graph
from gensim.summarization.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from gensim.utils import to_unicode
from itertools import combinations as _combinations
from six.moves.queue import Queue as _Queue
from six.moves import xrange
from six import iteritems


WINDOW_SIZE = 2

"""
Check tags in http://www.clips.ua.ac.be/pages/mbsp-tags and use only first two letters
Example: filter for nouns and adjectives:
INCLUDING_FILTER = ['NN', 'JJ']
"""
INCLUDING_FILTER = ['NN', 'JJ']
EXCLUDING_FILTER = []


def _get_pos_filters():
    """Returns default including and excluding filters as frozen sets.
    
    Returns
    -------
    tuple of frozenset
        Including and excluding filters.

    """
    return frozenset(INCLUDING_FILTER), frozenset(EXCLUDING_FILTER)


def _get_words_for_graph(tokens, pos_filter=None):
    """Filters given dictionary of tokens using provided part of speech filters
    and returns appropriate list of words.

    Parameters
    ----------
    tokens : dict
        Original units (words) as keys and processed units (tokens) as values.
    pos_filter : tuple of list
        Part of speech filters, optional.
    
    Returns
    -------
    list
        Filtered words.

    Raises
    ------
    ValueError
        If include and exclude filters ar not empty at the same time.

    """
    if pos_filter is None:
        include_filters, exclude_filters = _get_pos_filters()
    else:
        include_filters = set(pos_filter)
        exclude_filters = frozenset([])
    if include_filters and exclude_filters:
        raise ValueError("Can't use both include and exclude filters, should use only one")

    result = []
    for word, unit in iteritems(tokens):
        if exclude_filters and unit.tag in exclude_filters:
            continue
        if (include_filters and unit.tag in include_filters) or not include_filters or not unit.tag:
            result.append(unit.token)
    return result


def _get_first_window(split_text):
    """Returns first :const:`~gensim.parsing.keywords.WINDOW_SIZE` tokens from 
    given Splited text.

    Parameters
    ----------
    split_text : list
        Given Splited text.
    
    Returns
    -------
    tuple of frozenset
        Including and excluding filters.
        
    """
    return split_text[:WINDOW_SIZE]


def _set_graph_edge(graph, tokens, word_a, word_b):
    """Sets an edge between nodes named word_a and word_b if they exists in 
    `tokens` and `graph`, inplace.

    Parameters
    ----------
    graph : :class:~gensim.summarization.graph.Graph
        Given graph.
    tokens : dict
        Original units (words) as keys and processed units (tokens) as values.
    word_a : str
        First word, name of first node.
    word_b : str
        Second word, name of second node.
        
    """
    if word_a in tokens and word_b in tokens:
        lemma_a = tokens[word_a].token
        lemma_b = tokens[word_b].token
        edge = (lemma_a, lemma_b)

        if graph.has_node(lemma_a) and graph.has_node(lemma_b) and not graph.has_edge(edge):
            graph.add_edge(edge)


def _process_first_window(graph, tokens, split_text):
    """Sets an edges between nodes taken from first :const:`~gensim.parsing.keywords.WINDOW_SIZE`
    words of `split_text` if they exist in `tokens` and `graph`, inplace.

    Parameters
    ----------
    graph : :class:~gensim.summarization.graph.Graph
        Given graph.
    tokens : dict
        Original units (words) as keys and processed units (tokens) as values.
    split_text : list of str
        Splited text.
        
    """
    first_window = _get_first_window(split_text)
    for word_a, word_b in _combinations(first_window, 2):
        _set_graph_edge(graph, tokens, word_a, word_b)


def _init_queue(split_text):
    """Initializies queue by first words from `split_text`. 

    Parameters
    ----------
    split_text : list of str
        Splited text.

    Returns
    -------
    Queue
        Initialized queue.
        
    """
    queue = _Queue()
    first_window = _get_first_window(split_text)
    for word in first_window[1:]:
        queue.put(word)
    return queue


def _process_word(graph, tokens, queue, word):
    """Sets edge between `word` and each element in queue in `graph` if such nodes
    exist in `tokens` and `graph`. 

    Parameters
    ----------
    graph : :class:~gensim.summarization.graph.Graph
        Given graph.
    tokens : dict
        Original units (words) as keys and processed units (tokens) as values.
    queue : Queue
        Given queue.
    word : str
        Word, possible `node` in graph and item in `tokens`.

    """
    for word_to_compare in _queue_iterator(queue):
        _set_graph_edge(graph, tokens, word, word_to_compare)


def _update_queue(queue, word):
    """Updates given `queue` (removes last item and puts `word`).

    Parameters
    ----------
    queue : Queue
        Given queue.
    word : str
        Word to be added to queue.
    """
    queue.get()
    queue.put(word)
    assert queue.qsize() == (WINDOW_SIZE - 1)


def _process_text(graph, tokens, split_text):
    """Processes `split_text` by updating given `graph` with new eges between 
    nodes if they exists in `tokens` and `graph`. Words are taken from 
    `split_text` with window size :const:`~gensim.parsing.keywords.WINDOW_SIZE`.

    Parameters
    ----------
    graph : :class:~gensim.summarization.graph.Graph
        Given graph.
    tokens : dict
        Original units (words) as keys and processed units (tokens) as values.
    split_text : list of str
        Splited text.
    """
    queue = _init_queue(split_text)
    for i in xrange(WINDOW_SIZE, len(split_text)):
        word = split_text[i]
        _process_word(graph, tokens, queue, word)
        _update_queue(queue, word)


def _queue_iterator(queue):
    """Represents iterator of the given queue.

    Parameters
    ----------
    queue : Queue
        Given queue.

    Yields
    ------
    str
        Current item of queue.
        
    """
    iterations = queue.qsize()
    for _ in xrange(iterations):
        var = queue.get()
        yield var
        queue.put(var)


def _set_graph_edges(graph, tokens, split_text):
    """Updates given `graph` by setting eges between nodes if they exists in 
    `tokens` and `graph`. Words are taken from `split_text` with window size
    :const:`~gensim.parsing.keywords.WINDOW_SIZE`.

    Parameters
    ----------
    graph : :class:~gensim.summarization.graph.Graph
        Given graph.
    tokens : dict
        Original units (words) as keys and processed units (tokens) as values.
    split_text : list of str
        Splited text.
    """
    _process_first_window(graph, tokens, split_text)
    _process_text(graph, tokens, split_text)


def _extract_tokens(lemmas, scores, ratio, words):
    """Extracts tokens from provided lemmas. Most scored lemmas are used if 
    `words` not provided.

    Parameters
    ----------
    lemmas : list
        Given lemmas.
    scores : dict
        Dictionary with lemmas and its scores.
    ratio : float
        Proportion of lemmas used for final result. 
    words : int
        Number of used words. If no "words" option is selected, the number of 
        sentences is reduced by the provided ratio, else, the ratio is ignored.

    Returns
    -------
    list of (tuple of float and str)
        Scores and corresponded lemmas.

    """
    lemmas.sort(key=lambda s: scores[s], reverse=True)
    length = len(lemmas) * ratio if words is None else words
    return [(scores[lemmas[i]], lemmas[i],) for i in range(int(length))]


def _lemmas_to_words(tokens):
    """Returns words and lemmas from given tokens. Produces "reversed" `tokens`.

    Parameters
    ----------
    tokens : dict
        Original units (words) as keys and processed units (tokens) as values.

    Returns
    -------
    dict
        Lemmas as keys and lists corresponding words as values.
         
    """
    lemma_to_word = {}
    for word, unit in iteritems(tokens):
        lemma = unit.token
        if lemma in lemma_to_word:
            lemma_to_word[lemma].append(word)
        else:
            lemma_to_word[lemma] = [word]
    return lemma_to_word


def _get_keywords_with_score(extracted_lemmas, lemma_to_word):
    """Returns words of `extracted_lemmas` and its scores. Words contains in
    `lemma_to_word`.

    Parameters
    ----------
    extracted_lemmas : list of tuples
        Given lemmas.
    lemma_to_word : dict of {lemma:list of words}
        Lemmas and corresponding words.

    Returns
    -------
    dict
        Keywords as keys and its scores as values.
         
    """

    keywords = {}
    for score, lemma in extracted_lemmas:
        keyword_list = lemma_to_word[lemma]
        for keyword in keyword_list:
            keywords[keyword] = score
    return keywords


def _strip_word(word):
    """Return cleaned `word`.

    Parameters
    ----------
    word : str
        Given word.
    
    Returns
    -------
    str
        Cleaned word.
    """
    stripped_word_list = list(_tokenize_by_word(word))
    return stripped_word_list[0] if stripped_word_list else ""


def _get_combined_keywords(_keywords, split_text):
    """Returns most scored words (`_keywords`) contained in `split_text` and its
    combinations.

    Parameters
    ----------
    _keywords : dict
        Keywords as keys and its scores as values.
    split_text : list of str
        Splited text.
    
    Returns
    -------
    list of str
        Keywords and/or its combinations.

    """
    result = []
    _keywords = _keywords.copy()
    len_text = len(split_text)
    for i in xrange(len_text):
        word = _strip_word(split_text[i])
        if word in _keywords:
            combined_word = [word]
            if i + 1 == len_text:
                result.append(word)   # appends last word if keyword and doesn't iterate
            for j in xrange(i + 1, len_text):
                other_word = _strip_word(split_text[j])
                if other_word in _keywords and other_word == split_text[j] and other_word not in combined_word:
                    combined_word.append(other_word)
                else:
                    for keyword in combined_word:
                        _keywords.pop(keyword)
                    result.append(" ".join(combined_word))
                    break
    return result


def _get_average_score(concept, _keywords):
    """Returns average score of words in `concept`.

    Parameters
    ----------
    concept : str
        Input text.
    _keywords : dict
        Keywords as keys and its scores as values.
    
    Returns
    -------
    float
        Average score.

    """
    word_list = concept.split()
    word_counter = 0
    total = 0
    for word in word_list:
        total += _keywords[word]
        word_counter += 1
    return total / word_counter


def _format_results(_keywords, combined_keywords, split, scores):
    """Formats, sorts and returns `combined_keywords` in desired format.

    Parameters
    ----------
    _keywords : dict
        Keywords as keys and its scores as values.
    combined_keywords : list of str
        Most ranked words and/or its combinations.
    split : bool
        Split result if True or return string otherwise, optional.
    scores : bool
        Whether return `combined_keywords` with scores, optional. If True 
        `split` is ignored.

    Returns
    -------
    str or list of str or list of (tuple of str)
        Formated `combined_keywords`.

    """
    combined_keywords.sort(key=lambda w: _get_average_score(w, _keywords), reverse=True)
    if scores:
        return [(word, _get_average_score(word, _keywords)) for word in combined_keywords]
    if split:
        return combined_keywords
    return "\n".join(combined_keywords)


def keywords(text, ratio=0.2, words=None, split=False, scores=False, pos_filter=('NN', 'JJ'),
             lemmatize=False, deacc=True):
    """Returns most ranked words of provided text and/or its combinations .

    Parameters
    ----------
    text : str
        Sequence of values.
    ratio : float, optional
        If no "words" option is selected, the number of sentences is
        reduced by the provided ratio, else, the ratio is ignored.
    words : int, optional
        Number of returned words.
    split : bool, optional
        Whether split keywords if True.
    scores : bool, optional
        Whether score of keyword.
    pos_filter : tuple, optional
        Part of speech filters.
    lemmatize : bool, optional
        Lemmatize words if True.
    deacc : bool, optional
        Remove accentuation if True.

    Returns
    -------
    str or list of str or list of (tuple of str)

    """
    # Gets a dict of word -> lemma
    text = to_unicode(text)
    tokens = _clean_text_by_word(text, deacc=deacc)
    split_text = list(_tokenize_by_word(text))

    # Creates the graph and adds the edges
    graph = _build_graph(_get_words_for_graph(tokens, pos_filter))
    _set_graph_edges(graph, tokens, split_text)
    del split_text  # It's no longer used

    _remove_unreachable_nodes(graph)

    # Ranks the tokens using the PageRank algorithm. Returns dict of lemma -> score
    pagerank_scores = _pagerank(graph)

    extracted_lemmas = _extract_tokens(graph.nodes(), pagerank_scores, ratio, words)

    # The results can be polluted by many variations of the same word
    if lemmatize:
        lemmas_to_word = {}
        for word, unit in iteritems(tokens):
            lemmas_to_word[unit.token] = [word]
    else:
        lemmas_to_word = _lemmas_to_words(tokens)

    keywords = _get_keywords_with_score(extracted_lemmas, lemmas_to_word)

    # text.split() to keep numbers and punctuation marks, so separeted concepts are not combined
    combined_keywords = _get_combined_keywords(keywords, text.split())

    return _format_results(keywords, combined_keywords, split, scores)


def get_graph(text):
    """Creates and returns graph from given text. Cleans and tokenizes text 
    before building graph.

    Parameters
    ----------
    text : str
        Sequence of values.

    Returns
    -------
    :class:~gensim.summarization.graph.Graph
        Created graph.

    """
    tokens = _clean_text_by_word(text)
    split_text = list(_tokenize_by_word(text))

    graph = _build_graph(_get_words_for_graph(tokens))
    _set_graph_edges(graph, tokens, split_text)

    return graph
