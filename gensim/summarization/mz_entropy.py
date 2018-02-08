#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


from gensim.summarization.textcleaner import tokenize_by_word as _tokenize_by_word
from gensim.utils import to_unicode
import numpy
import scipy


def mz_keywords(text, blocksize=1024, scores=False, split=False, weighted=True, threshold=0.0):
    """Extract keywords from text using the Montemurro and Zanette entropy algorithm. [1]_

    Parameters
    ----------
    text: str
        Document for summarization.
    blocksize: int, optional
        Size of blocks to use in analysis.
    scores: bool, optional
        Whether to return score with keywords.
    split: bool, optional
        Whether to return results as list.
    weighted: bool, optional
        Whether to weight scores by word frequency.
        False can useful for shorter texts, and allows automatic thresholding.
    threshold: float or 'auto', optional
        Minimum score for returned keywords,  'auto' calculates the threshold as n_blocks / (n_blocks + 1.0) + 1e-8,
        use 'auto' with `weighted=False`.

    Returns
    -------
    results: str
        newline separated keywords if `split` == False **OR**
    results: list(str)
        list of keywords if `scores` == False **OR**
    results: list(tuple(str, float))
        list of (keyword, score) tuples if `scores` == True

    Results are returned in descending order of score regardless of the format.

    Note
    ----
    This algorithm looks for keywords that contribute to the structure of the
    text on scales of `blocksize` words of larger. It is suitable for extracting
    keywords representing the major themes of long texts.

    References
    ----------
    .. [1] Marcello A Montemurro, Damian Zanette, "Towards the quantification of the semantic information encoded in
           written language". Advances in Complex Systems, Volume 13, Issue 2 (2010), pp. 135-153,
           DOI: 10.1142/S0219525910002530, https://arxiv.org/abs/0907.1558

    """
    text = to_unicode(text)
    words = [word for word in _tokenize_by_word(text)]
    vocab = sorted(set(words))
    word_counts = numpy.array(
        [
            [words[i:i + blocksize].count(word) for word in vocab]
            for i in range(0, len(words), blocksize)
        ]
    ).astype('d')
    n_blocks = word_counts.shape[0]
    totals = word_counts.sum(axis=0)
    n_words = totals.sum()
    p = word_counts / totals
    log_p = numpy.log2(p)
    h = numpy.nan_to_num(p * log_p).sum(axis=0)
    analytic = __analytic_entropy(blocksize, n_blocks, n_words)
    h += analytic(totals).astype('d')
    if weighted:
        h *= totals / n_words
    if threshold == 'auto':
        threshold = n_blocks / (n_blocks + 1.0) + 1.0e-8
    weights = [(word, score) for (word, score) in zip(vocab, h) if score > threshold]
    weights.sort(key=lambda x: -x[1])
    result = weights if scores else [word for (word, score) in weights]
    if not (scores or split):
        result = '\n'.join(result)
    return result


def __log_combinations_inner(n, m):
    """Calculates the logarithm of n!/m!(n-m)!"""
    return -(numpy.log(n + 1) + scipy.special.betaln(n - m + 1, m + 1))


__log_combinations = numpy.frompyfunc(__log_combinations_inner, 2, 1)


def __marginal_prob(blocksize, n_words):

    def marginal_prob(n, m):
        """Marginal probability of a word that occurs n times in the document
           occurring m times in a given block"""

        return numpy.exp(
            __log_combinations(n, m) +
            __log_combinations(n_words - n, blocksize - m) -
            __log_combinations(n_words, blocksize)
        )

    return numpy.frompyfunc(marginal_prob, 2, 1)


def __analytic_entropy(blocksize, n_blocks, n_words):
    marginal = __marginal_prob(blocksize, n_words)

    def analytic_entropy(n):
        """Predicted entropy for a word that occurs n times in the document"""
        m = numpy.arange(1, min(blocksize, n) + 1).astype('d')
        p = m / n
        elements = numpy.nan_to_num(p * numpy.log2(p)) * marginal(n, m)
        return -n_blocks * elements.sum()

    return numpy.frompyfunc(analytic_entropy, 1, 1)
