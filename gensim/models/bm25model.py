#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements functionality related to the `Okapi Best Matching
<https://en.wikipedia.org/wiki/Okapi_BM25>`_ class of bag-of-words vector space models.

"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
import logging
import math

from gensim import interfaces, utils
import numpy as np


logger = logging.getLogger(__name__)


class BM25ABC(interfaces.TransformationABC, metaclass=ABCMeta):
    def __init__(self, corpus=None, dictionary=None):
        self.avgdl, self.idfs = None, None
        if dictionary:
            if corpus:
                logger.warning("constructor received both corpus and dictionary; ignoring the corpus")
            self.initialize_from_dictionary(dictionary)
        elif corpus:
            self.initialize_from_corpus(corpus)
        else:
            pass

    def initialize_from_dictionary(self, dictionary):
        num_tokens = sum(dictionary.cfs.values())
        self.avgdl = num_tokens / dictionary.num_docs
        self.idfs = self.precompute_idfs(dictionary.dfs, dictionary.num_docs)

    def initialize_from_corpus(self, corpus):
        dfs = defaultdict(lambda: 0)
        num_tokens = 0
        num_docs = 0
        for bow in corpus:
            num_tokens += len(bow)
            for term_id in set(term_id for term_id, _ in bow):
                dfs[term_id] += 1
            num_docs += 1
        self.avgdl = num_tokens / num_docs
        self.idfs = self.precompute_idfs(dfs, num_docs)

    @abstractmethod
    def precompute_idfs(self, dfs, num_docs):
        pass


class OkapiBM25Model(BM25ABC):
    def __init__(self, corpus=None, dictionary=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        super().__init__(corpus, dictionary)

    def precompute_idfs(self, dfs, num_docs):
        idf_sum = 0
        idfs = dict()
        negative_idfs = []
        for term_id, freq in dfs.items():
            idf = math.log(num_docs - freq + 0.5) - math.log(freq + 0.5)
            idfs[term_id] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(term_id)
        average_idf = idf_sum / len(idfs)

        eps = self.epsilon * average_idf
        for term_id in negative_idfs:
            idfs[term_id] = eps

        return idfs

    def __getitem__(self, bow):
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)

        num_tokens = sum(freq for term_id, freq in bow)

        term_ids, term_frequencies, idfs = [], [], []
        for term_id, term_frequency in bow:
            term_ids.append(term_id)
            term_frequencies.append(term_frequency)
            idfs.append(self.idfs.get(term_id) or 0.0)
        term_frequencies, idfs = np.array(term_frequencies), np.array(idfs)

        term_weights = idfs * (term_frequencies * (self.k1 + 1)
                              / (term_frequencies + self.k1 * (1 - self.b + self.b
                                                              * num_tokens / self.avgdl)))

        vector = [
            (term_id, float(weight))
            for term_id, weight
            in zip(term_ids, term_weights)
        ]
        return vector
