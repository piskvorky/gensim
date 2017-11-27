#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains function of computing rank scores for documents in 
corpus and helper class `BM25` used in calculations. Original alhorithm 
descibed in [1]_, also you may check Wikipedia page [2]_.


.. [1] Robertson, Stephen; Zaragoza, Hugo (2009).  The Probabilistic Relevance Framework: BM25 and Beyond, http://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf
.. [2] Okapi BM25 on Wikipedia, https://en.wikipedia.org/wiki/Okapi_BM25



Example
-------
>>> import numpy as np
>>> from gensim.summarization.bm25 import get_bm25_weights
>>> corpus = [
>>>     ["black", "cat", "white", "cat"],
>>>     ["cat", "outer", "space"],
>>>     ["wag", "dog"]
>>> ]
>>> np.round(get_bm25_weights(corpus), 3)
array([[ 1.282,  0.182,  0.   ],
       [ 0.13 ,  1.113,  0.   ],
       [ 0.   ,  0.   ,  1.022]])

Data:
-----
.. data:: PARAM_K1 - free smoothing parameter for BM25.
.. data:: PARAM_B - free smoothing parameter for BM25.
.. data:: EPSILON - constant used for negative idf of document in corpus.
"""


import math
from six import iteritems
from six.moves import xrange


PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25


class BM25(object):
    """Implementation of Best Matching 25 ranking function.

    Attributes
    ----------
    corpus_size : int
        Size of corpus (number of documents).
    avgdl : float
        Average length of document in `corpus`.
    corpus : list of (list of str)
        Corpus of documents.
    f : list of dict
        Dictionary with terms frequencies for each document in `corpus`. Words 
        used as keys and frequencies as values.
    df : dict
        Dictionary with terms frequencies for whole `corpus`. Words used as keys
        and frequencies as values.
    idf : dict
        Dictionary with inversed terms frequencies for whole `corpus`. Words 
        used as keys and frequencies as values.

    """


    def __init__(self, corpus):
        """Presets atributes and runs initialize() function.

        Parameters
        ----------
        corpus : list of (list of str)
            Given corups.

        """
        self.corpus_size = len(corpus)
        self.avgdl = sum(float(len(x)) for x in corpus) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.initialize()


    def initialize(self):
        """Calculates frequencies of terms in documents and in corpus. Also
        computes inverse document frequencies.

        """
        for document in self.corpus:
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)


    def get_score(self, document, index, average_idf):
        """Computes BM25 score of given `document` in relation to item of corpus
        selected by `index`.

        Parameters
        ----------
        document : list of str
            Document to be scored.
        index : integer
            Index of document in corpus selected to score with `document`.
        average_idf : float
            Average idf in corpus.

        Returns
        -------
        float
            BM25 score.

        """
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON * average_idf
            score += (idf * self.f[index][word] * (PARAM_K1 + 1)
                      / (self.f[index][word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.corpus_size / self.avgdl)))
        return score


    def get_scores(self, document, average_idf):
        """Computes and returns BM25 scores of given `document` in relation to 
        every item in corpus. 

        Parameters
        ----------
        document : list of str
            Document to be scored.
        average_idf : float
            Average idf in corpus.

        Returns
        -------
        list of float
            BM25 scores.

        """
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(document, index, average_idf)
            scores.append(score)
        return scores


def get_bm25_weights(corpus):
    """Returns BM25 scores (weights) of documents in corpus. Each document
    has to be weighted with every document in given corpus. 

    Parameters
    ----------
    corpus : list of (list of str)
        Corpus of documents.

    Returns
    -------
    list of (list of float)
        BM25 scores.

    """
    bm25 = BM25(corpus)
    average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)

    weights = []
    for doc in corpus:
        scores = bm25.get_scores(doc, average_idf)
        weights.append(scores)

    return weights
