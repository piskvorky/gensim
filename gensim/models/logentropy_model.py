#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module allows simple Bag of Words (BoW) represented corpus to be transformed into log entropy space.
It implements Log Entropy Model that produces entropy-weighted logarithmic term frequency representation.

Empirical study by Lee et al. 2015 [1]_ suggests log entropy-weighted model yields better results among other forms of
representation.

References
----------
.. [1] Lee et al. 2005. An Empirical Evaluation of Models of Text Document Similarity.
       https://escholarship.org/uc/item/48g155nq

"""

import logging
import math

from gensim import interfaces, matutils, utils

logger = logging.getLogger(__name__)


class LogEntropyModel(interfaces.TransformationABC):
    r"""Objects of this class realize the transformation between word-document co-occurrence matrix (int)
    into a locally/globally weighted matrix (positive floats).

    This is done by a log entropy normalization, optionally normalizing the resulting documents to unit length.
    The following formulas explain how o compute the log entropy weight for term :math:`i` in document :math:`j`:

    .. math::

        local\_weight_{i,j} = log(frequency_{i,j} + 1)

        P_{i,j} = \frac{frequency_{i,j}}{\sum_j frequency_{i,j}}

        global\_weight_i = 1 + \frac{\sum_j P_{i,j} * log(P_{i,j})}{log(number\_of\_documents + 1)}

        final\_weight_{i,j} = local\_weight_{i,j} * global\_weight_i

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.models import LogEntropyModel
        >>> from gensim.test.utils import common_texts
        >>> from gensim.corpora import Dictionary
        >>>
        >>> dct = Dictionary(common_texts)  # fit dictionary
        >>> corpus = [dct.doc2bow(row) for row in common_texts]  # convert to BoW format
        >>> model = LogEntropyModel(corpus)  # fit model
        >>> vector = model[corpus[1]]  # apply model to document

    """

    def __init__(self, corpus, normalize=True):
        """

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Input corpus in BoW format.
        normalize : bool, optional
            If True, the resulted log entropy weighted vector will be normalized to length of 1,
            If False - do nothing.

        """
        self.normalize = normalize
        self.n_docs = 0
        self.n_words = 0
        self.entr = {}
        if corpus is not None:
            self.initialize(corpus)

    def __str__(self):
        return "%s<n_docs=%s, n_words=%s>" % (self.__class__.__name__, self.n_docs, self.n_words)

    def initialize(self, corpus):
        """Calculates the global weighting for all terms in a given corpus and transforms the simple
        count representation into the log entropy normalized space.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Corpus is BoW format

        """
        logger.info("calculating counts")
        glob_freq = {}
        glob_num_words, doc_no = 0, -1
        for doc_no, bow in enumerate(corpus):
            if doc_no % 10000 == 0:
                logger.info("PROGRESS: processing document #%i", doc_no)
            glob_num_words += len(bow)
            for term_id, term_count in bow:
                glob_freq[term_id] = glob_freq.get(term_id, 0) + term_count

        # keep some stats about the training corpus
        self.n_docs = doc_no + 1
        self.n_words = glob_num_words

        # and finally compute the global weights
        logger.info(
            "calculating global log entropy weights for %i documents and %i features (%i matrix non-zeros)",
            self.n_docs, len(glob_freq), self.n_words
        )
        logger.debug('iterating over corpus')

        # initialize doc_no2 index in case corpus is empty
        doc_no2 = 0
        for doc_no2, bow in enumerate(corpus):
            for key, freq in bow:
                p = (float(freq) / glob_freq[key]) * math.log(float(freq) / glob_freq[key])
                self.entr[key] = self.entr.get(key, 0.0) + p
        if doc_no2 != doc_no:
            raise ValueError("LogEntropyModel doesn't support generators as training data")

        logger.debug('iterating over keys')
        for key in self.entr:
            self.entr[key] = 1 + self.entr[key] / math.log(self.n_docs + 1)

    def __getitem__(self, bow):
        """Get log entropy representation of the input vector and/or corpus.

        Parameters
        ----------
        bow : list of (int, int)
            Document in BoW format.

        Returns
        -------
        list of (int, float)
            Log-entropy vector for passed `bow`.

        """
        # if the input vector is in fact a corpus, return a transformed corpus
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)

        # unknown (new) terms will be given zero weight (NOT infinity/huge)
        vector = [
            (term_id, math.log(tf + 1) * self.entr.get(term_id))
            for term_id, tf in bow
            if term_id in self.entr
        ]
        if self.normalize:
            vector = matutils.unitvec(vector)
        return vector
