#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains function of computing rank scores for documents in
corpus and helper class `BM25` used in calculations. Original algorithm
descibed in [1]_, also you may check Wikipedia page [2]_.


.. [1] Robertson, Stephen; Zaragoza, Hugo (2009).  The Probabilistic Relevance Framework: BM25 and Beyond,
       http://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf
.. [2] Okapi BM25 on Wikipedia, https://en.wikipedia.org/wiki/Okapi_BM25



Examples
--------

.. sourcecode:: pycon

    >>> from gensim.summarization.bm25 import get_bm25_weights
    >>> corpus = [
    ...     ["black", "cat", "white", "cat"],
    ...     ["cat", "outer", "space"],
    ...     ["wag", "dog"]
    ... ]
    >>> result = get_bm25_weights(corpus, n_jobs=-1)


Data:
-----
.. data:: PARAM_K1 - Free smoothing parameter for BM25.
.. data:: PARAM_B - Free smoothing parameter for BM25.
.. data:: EPSILON - Constant used for negative idf of document in corpus.

"""


import logging
import math
from six import iteritems
from six.moves import range
from functools import partial
from multiprocessing import Pool
from ..utils import effective_n_jobs

PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25

logger = logging.getLogger(__name__)


class BM25(object):
    """Implementation of Best Matching 25 ranking function.

    Attributes
    ----------
    corpus_size : int
        Size of corpus (number of documents).
    avgdl : float
        Average length of document in `corpus`.
    doc_freqs : list of dicts of int
        Dictionary with terms frequencies for each document in `corpus`. Words used as keys and frequencies as values.
    idf : dict
        Dictionary with inversed documents frequencies for whole `corpus`. Words used as keys and frequencies as values.
    doc_len : list of int
        List of document lengths.
    """

    def __init__(self, corpus, k1=PARAM_K1, b=PARAM_B, epsilon=EPSILON):
        """
        Parameters
        ----------
        corpus : list of list of str
            Given corpus.
        k1 : float
            Constant used for influencing the term frequency saturation. After saturation is reached, additional
            presence for the term adds a significantly less additional score. According to [1]_, experiments suggest
            that 1.2 < k1 < 2 yields reasonably good results, although the optimal value depends on factors such as
            the type of documents or queries.
        b : float
            Constant used for influencing the effects of different document lengths relative to average document length.
            When b is bigger, lengthier documents (compared to average) have more impact on its effect. According to
            [1]_, experiments suggest that 0.5 < b < 0.8 yields reasonably good results, although the optimal value
            depends on factors such as the type of documents or queries.
        epsilon : float
            Constant used as floor value for idf of a document in the corpus. When epsilon is positive, it restricts
            negative idf values. Negative idf implies that adding a very common term to a document penalize the overall
            score (with 'very common' meaning that it is present in more than half of the documents). That can be
            undesirable as it means that an identical document would score less than an almost identical one (by
            removing the referred term). Increasing epsilon above 0 raises the sense of how rare a word has to be (among
            different documents) to receive an extra score.

        """

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self._initialize(corpus)

    def _initialize(self, corpus):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.corpus_size += 1
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = float(num_doc) / self.corpus_size
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in iteritems(nd):
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = float(idf_sum) / len(self.idf)

        if self.average_idf < 0:
            logger.warning(
                'Average inverse document frequency is less than zero. Your corpus of {} documents'
                ' is either too small or it does not originate from natural text. BM25 may produce'
                ' unintuitive results.'.format(self.corpus_size)
            )

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_score(self, document, index):
        """Computes BM25 score of given `document` in relation to item of corpus selected by `index`.

        Parameters
        ----------
        document : list of str
            Document to be scored.
        index : int
            Index of document in corpus selected to score with `document`.

        Returns
        -------
        float
            BM25 score.

        """
        score = 0.0
        doc_freqs = self.doc_freqs[index]
        numerator_constant = self.k1 + 1
        denominator_constant = self.k1 * (1 - self.b + self.b * self.doc_len[index] / self.avgdl)
        for word in document:
            if word in doc_freqs:
                df = self.doc_freqs[index][word]
                idf = self.idf[word]
                score += (idf * df * numerator_constant) / (df + denominator_constant)
        return score

    def get_scores(self, document):
        """Computes and returns BM25 scores of given `document` in relation to
        every item in corpus.

        Parameters
        ----------
        document : list of str
            Document to be scored.

        Returns
        -------
        list of float
            BM25 scores.

        """
        scores = [self.get_score(document, index) for index in range(self.corpus_size)]
        return scores

    def get_scores_bow(self, document):
        """Computes and returns BM25 scores of given `document` in relation to
        every item in corpus.

        Parameters
        ----------
        document : list of str
            Document to be scored.

        Returns
        -------
        list of float
            BM25 scores.

        """
        scores = []
        for index in range(self.corpus_size):
            score = self.get_score(document, index)
            if score > 0:
                scores.append((index, score))
        return scores


def _get_scores_bow(bm25, document):
    """Helper function for retrieving bm25 scores of given `document` in parallel
    in relation to every item in corpus.

    Parameters
    ----------
    bm25 : BM25 object
        BM25 object fitted on the corpus where documents are retrieved.
    document : list of str
        Document to be scored.

    Returns
    -------
    list of (index, float)
        BM25 scores in a bag of weights format.

    """
    return bm25.get_scores_bow(document)


def _get_scores(bm25, document):
    """Helper function for retrieving bm25 scores of given `document` in parallel
    in relation to every item in corpus.

    Parameters
    ----------
    bm25 : BM25 object
        BM25 object fitted on the corpus where documents are retrieved.
    document : list of str
        Document to be scored.

    Returns
    -------
    list of float
        BM25 scores.

    """
    return bm25.get_scores(document)


def iter_bm25_bow(corpus, n_jobs=1, k1=PARAM_K1, b=PARAM_B, epsilon=EPSILON):
    """Yield BM25 scores (weights) of documents in corpus.
    Each document has to be weighted with every document in given corpus.

    Parameters
    ----------
    corpus : list of list of str
        Corpus of documents.
    n_jobs : int
        The number of processes to use for computing bm25.
    k1 : float
        Constant used for influencing the term frequency saturation. After saturation is reached, additional
        presence for the term adds a significantly less additional score. According to [1]_, experiments suggest
        that 1.2 < k1 < 2 yields reasonably good results, although the optimal value depends on factors such as
        the type of documents or queries.
    b : float
        Constant used for influencing the effects of different document lengths relative to average document length.
        When b is bigger, lengthier documents (compared to average) have more impact on its effect. According to
        [1]_, experiments suggest that 0.5 < b < 0.8 yields reasonably good results, although the optimal value
        depends on factors such as the type of documents or queries.
    epsilon : float
        Constant used as floor value for idf of a document in the corpus. When epsilon is positive, it restricts
        negative idf values. Negative idf implies that adding a very common term to a document penalize the overall
        score (with 'very common' meaning that it is present in more than half of the documents). That can be
        undesirable as it means that an identical document would score less than an almost identical one (by
        removing the referred term). Increasing epsilon above 0 raises the sense of how rare a word has to be (among
        different documents) to receive an extra score.

    Yields
    -------
    list of (index, float)
        BM25 scores in bag of weights format.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.summarization.bm25 import iter_bm25_weights
        >>> corpus = [
        ...     ["black", "cat", "white", "cat"],
        ...     ["cat", "outer", "space"],
        ...     ["wag", "dog"]
        ... ]
        >>> result = iter_bm25_weights(corpus, n_jobs=-1)

    """
    bm25 = BM25(corpus, k1, b, epsilon)

    n_processes = effective_n_jobs(n_jobs)
    if n_processes == 1:
        for doc in corpus:
            yield bm25.get_scores_bow(doc)
        return

    get_score = partial(_get_scores_bow, bm25)
    pool = Pool(n_processes)

    for bow in pool.imap(get_score, corpus):
        yield bow
    pool.close()
    pool.join()


def get_bm25_weights(corpus, n_jobs=1, k1=PARAM_K1, b=PARAM_B, epsilon=EPSILON):
    """Returns BM25 scores (weights) of documents in corpus.
    Each document has to be weighted with every document in given corpus.

    Parameters
    ----------
    corpus : list of list of str
        Corpus of documents.
    n_jobs : int
        The number of processes to use for computing bm25.
    k1 : float
        Constant used for influencing the term frequency saturation. After saturation is reached, additional
        presence for the term adds a significantly less additional score. According to [1]_, experiments suggest
        that 1.2 < k1 < 2 yields reasonably good results, although the optimal value depends on factors such as
        the type of documents or queries.
    b : float
        Constant used for influencing the effects of different document lengths relative to average document length.
        When b is bigger, lengthier documents (compared to average) have more impact on its effect. According to
        [1]_, experiments suggest that 0.5 < b < 0.8 yields reasonably good results, although the optimal value
        depends on factors such as the type of documents or queries.
    epsilon : float
        Constant used as floor value for idf of a document in the corpus. When epsilon is positive, it restricts
        negative idf values. Negative idf implies that adding a very common term to a document penalize the overall
        score (with 'very common' meaning that it is present in more than half of the documents). That can be
        undesirable as it means that an identical document would score less than an almost identical one (by
        removing the referred term). Increasing epsilon above 0 raises the sense of how rare a word has to be (among
        different documents) to receive an extra score.

    Returns
    -------
    list of list of float
        BM25 scores.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.summarization.bm25 import get_bm25_weights
        >>> corpus = [
        ...     ["black", "cat", "white", "cat"],
        ...     ["cat", "outer", "space"],
        ...     ["wag", "dog"]
        ... ]
        >>> result = get_bm25_weights(corpus, n_jobs=-1)

    """
    bm25 = BM25(corpus, k1, b, epsilon)

    n_processes = effective_n_jobs(n_jobs)
    if n_processes == 1:
        weights = [bm25.get_scores(doc) for doc in corpus]
        return weights

    get_score = partial(_get_scores, bm25)
    pool = Pool(n_processes)
    weights = pool.map(get_score, corpus)
    pool.close()
    pool.join()
    return weights
