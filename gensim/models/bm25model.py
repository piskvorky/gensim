#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements functionality related to the `Okapi Best Matching
<https://en.wikipedia.org/wiki/Okapi_BM25>`_ class of bag-of-words vector space models.

Robertson and Zaragoza [1]_ describe the original algorithm and its modifications.

.. [1] Robertson S., Zaragoza H. (2015). `The Probabilistic Relevance Framework: BM25 and
   Beyond, <http://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf>`_.

"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
import logging
import math

from gensim import interfaces, utils
import numpy as np


logger = logging.getLogger(__name__)


class BM25ABC(interfaces.TransformationABC, metaclass=ABCMeta):
    """Objects of this abstract class realize the transformation between word-document co-occurrence
    matrix (int) into a BM25 matrix (positive floats). Concrete subclasses of this abstract class
    implement different BM25 scoring functions.

    """
    def __init__(self, corpus=None, dictionary=None):
        r"""Pre-compute the average length of a document and inverse term document frequencies,
        which will be used to weight term frequencies for the documents.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int) or None, optional
            An input corpus, which will be used to compute the average length of a document and
            inverse term document frequencies. If None, then `dictionary` will be used to compute
            the statistics. If both `corpus` and `dictionary` are None, the statistics will be left
            unintialized. Default is None.
        dictionary : :class:`~gensim.corpora.Dictionary`
            An input dictionary, which will be used to compute the average length of a document and
            inverse term document frequencies.  If None, then `corpus` will be used to compute the
            statistics. If both `corpus` and `dictionary` are None, the statistics will be left
            unintialized. Default is None.

        Attributes
        ----------
        avgdl : float
            The average length of a document.
        idfs : dict of (int, float)
            A mapping from term ids to inverse term document frequencies.

        """
        self.avgdl, self.idfs = None, None
        if dictionary:
            if corpus:
                logger.warning("constructor received both corpus and dictionary; ignoring the corpus")
            num_tokens = sum(dictionary.cfs.values())
            self.avgdl = num_tokens / dictionary.num_docs
            self.idfs = self.precompute_idfs(dictionary.dfs, dictionary.num_docs)
        elif corpus:
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
        else:
            pass

    @abstractmethod
    def precompute_idfs(self, dfs, num_docs):
        """Precompute inverse term document frequencies, which will be used to weight term frequencies
        for the documents.

        Parameters
        ----------
        dfs : dict of (int, int)
            A mapping from term ids to term document frequencies.
        num_docs : int
            The total number of documents in the training corpus.

        Returns
        -------
        idfs : dict of (int, float)
            A mapping from term ids to inverse term document frequencies.

        """
        pass

    @abstractmethod
    def get_term_weights(self, num_tokens, term_frequencies, idfs):
        """Compute vector space weights for a set of terms in a document.

        Parameters
        ----------
        num_tokens : int
            The number of tokens in the document.
        term_frequencies : ndarray
            1D array of term frequencies.
        idfs : ndarray
            1D array of inverse term document frequencies.

        Returns
        -------
        term_weights : ndarray
            1D array of vector space weights.

        """
        pass

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

        term_weights = self.get_term_weights(num_tokens, term_frequencies, idfs)

        vector = [
            (term_id, float(weight))
            for term_id, weight
            in zip(term_ids, term_weights)
        ]
        return vector


class OkapiBM25Model(BM25ABC):
    """The original Okapi BM25 scoring function of Robertson et al. [2]_.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.corpora import Dictionary
        >>> from gensim.models import OkapiBM25Model
        >>> from gensim.test.utils import common_texts
        >>>
        >>> dictionary = Dictionary(common_texts)  # fit dictionary
        >>> model = OkapiBM25Model(dictionary=dictionary)  # fit model
        >>>
        >>> corpus = [dictionary.doc2bow(line) for line in common_texts]  # convert corpus to BoW format
        >>> vector = model[corpus[0]]  # apply model to the first corpus document

    References
    ----------
    .. [2] Robertson S. E., Walker S., Jones S., Hancock-Beaulieu M. M., Gatford M. (1995).
       `Okapi at TREC-3 <http://research.microsoft.com/pubs/67649/okapi_trec3.pdf>`_.
       *NIST Special Publication 500-226*.

    """
    def __init__(self, corpus=None, dictionary=None, k1=1.5, b=0.75, epsilon=0.25):
        r"""Pre-compute the average length of a document and inverse term document frequencies,
        which will be used to weight term frequencies for the documents.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int) or None, optional
            An input corpus, which will be used to compute the average length of a document and
            inverse term document frequencies. If None, then `dictionary` will be used to compute
            the statistics. If both `corpus` and `dictionary` are None, the statistics will be left
            unintialized. Default is None.
        dictionary : :class:`~gensim.corpora.Dictionary`
            An input dictionary, which will be used to compute the average length of a document and
            inverse term document frequencies.  If None, then `corpus` will be used to compute the
            statistics. If both `corpus` and `dictionary` are None, the statistics will be left
            unintialized. Default is None.
        k1 : float
            A positive tuning parameter that determines the impact of the term frequency on its BM25
            weight. Singhal [5]_ suggests to set `k1` between 1.0 and 2.0. Default is 1.5.
        b : float
            A tuning parameter between 0.0 and 1.0 that determines the document length
            normalization: 1.0 corresponds to full document normalization, while 0.0 corresponds to
            no length normalization. Singhal [5]_ suggests to set `b` to 0.75, which is the default.
        epsilon : float
            A positive tuning parameter that lower-bounds an inverse document frequency.
            Defaults to 0.25.

        Attributes
        ----------
        k1 : float
            A positive tuning parameter that determines the impact of the term frequency on its BM25
            weight. Singhal [3]_ suggests to set `k1` between 1.0 and 2.0. Default is 1.5.
        b : float
            A tuning parameter between 0.0 and 1.0 that determines the document length
            normalization: 1.0 corresponds to full document normalization, while 0.0 corresponds to
            no length normalization. Singhal [3]_ suggests to set `b` to 0.75, which is the default.
        epsilon : float
            A positive tuning parameter that lower-bounds an inverse document frequency.
            Defaults to 0.25.

        References
        ----------
        .. [3] Singhal, A. (2001). `Modern information retrieval: A brief overview
           <http://singhal.info/ieee2001.pdf>`_. *IEEE Data Eng. Bull.*, 24(4), 35–43.

        """
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

    def get_term_weights(self, num_tokens, term_frequencies, idfs):
        term_weights = idfs * (term_frequencies * (self.k1 + 1)
                              / (term_frequencies + self.k1 * (1 - self.b + self.b
                                                              * num_tokens / self.avgdl)))
        return term_weights


class LuceneBM25Model(BM25ABC):
    """The scoring function of Apache Lucene 8 [4]_.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.corpora import Dictionary
        >>> from gensim.models import LuceneBM25Model
        >>> from gensim.test.utils import common_texts
        >>>
        >>> dictionary = Dictionary(common_texts)  # fit dictionary
        >>> corpus = [dictionary.doc2bow(line) for line in common_texts]  # convert corpus to BoW format
        >>>
        >>> model = LuceneBM25Model(dictionary=dictionary)  # fit model
        >>> vector = model[corpus[0]]  # apply model to the first corpus document

    References
    ----------
    .. [4] Kamphuis, C., de Vries, A. P., Boytsov, L., Lin, J. (2020). Which
       BM25 Do You Mean? `A Large-Scale Reproducibility Study of Scoring Variants
       <https://doi.org/10.1007/978-3-030-45442-5_4>`_. In: Advances in Information Retrieval.
       28–34.

    """
    def __init__(self, corpus=None, dictionary=None, k1=1.5, b=0.75):
        r"""Pre-compute the average length of a document and inverse term document frequencies,
        which will be used to weight term frequencies for the documents.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int) or None, optional
            An input corpus, which will be used to compute the average length of a document and
            inverse term document frequencies. If None, then `dictionary` will be used to compute
            the statistics. If both `corpus` and `dictionary` are None, the statistics will be left
            unintialized. Default is None.
        dictionary : :class:`~gensim.corpora.Dictionary`
            An input dictionary, which will be used to compute the average length of a document and
            inverse term document frequencies.  If None, then `corpus` will be used to compute the
            statistics. If both `corpus` and `dictionary` are None, the statistics will be left
            unintialized. Default is None.
        k1 : float
            A positive tuning parameter that determines the impact of the term frequency on its BM25
            weight. Singhal [5]_ suggests to set `k1` between 1.0 and 2.0. Default is 1.5.
        b : float
            A tuning parameter between 0.0 and 1.0 that determines the document length
            normalization: 1.0 corresponds to full document normalization, while 0.0 corresponds to
            no length normalization. Singhal [5]_ suggests to set `b` to 0.75, which is the default.

        Attributes
        ----------
        k1 : float
            A positive tuning parameter that determines the impact of the term frequency on its BM25
            weight. Singhal [3]_ suggests to set `k1` between 1.0 and 2.0. Default is 1.5.
        b : float
            A tuning parameter between 0.0 and 1.0 that determines the document length
            normalization: 1.0 corresponds to full document normalization, while 0.0 corresponds to
            no length normalization. Singhal [3]_ suggests to set `b` to 0.75, which is the default.

        """
        self.k1, self.b = k1, b
        super().__init__(corpus, dictionary)

    def precompute_idfs(self, dfs, num_docs):
        idfs = dict()
        for term_id, freq in dfs.items():
            idf = math.log(num_docs + 1.0) - math.log(freq + 0.5)
            idfs[term_id] = idf
        return idfs

    def get_term_weights(self, num_tokens, term_frequencies, idfs):
        term_weights = idfs * (term_frequencies
                              / (term_frequencies + self.k1 * (1 - self.b + self.b
                                                              * num_tokens / self.avgdl)))
        return term_weights


class AtireBM25Model(BM25ABC):
    """The scoring function of Trotman et al. [5]_.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.corpora import Dictionary
        >>> from gensim.models import AtireBM25Model
        >>> from gensim.test.utils import common_texts
        >>>
        >>> dictionary = Dictionary(common_texts)  # fit dictionary
        >>> corpus = [dictionary.doc2bow(line) for line in common_texts]  # convert corpus to BoW format
        >>>
        >>> model = AtireBM25Model(dictionary=dictionary)  # fit model
        >>> vector = model[corpus[0]]  # apply model to the first corpus document

    References
    ----------
    .. [5] Trotman, A., Jia X., Crane M., `Towards an Efficient and Effective Search Engine
       <http://www.cs.otago.ac.nz/homepages/andrew/involvement/2012-SIGIR-OSIR.pdf#page=45>`_,
       In: SIGIR 2012 Workshop on Open Source Information Retrieval. 40–47.

    """
    def __init__(self, corpus=None, dictionary=None, k1=1.5, b=0.75):
        r"""Pre-compute the average length of a document and inverse term document frequencies,
        which will be used to weight term frequencies for the documents.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int) or None, optional
            An input corpus, which will be used to compute the average length of a document and
            inverse term document frequencies. If None, then `dictionary` will be used to compute
            the statistics. If both `corpus` and `dictionary` are None, the statistics will be left
            unintialized. Default is None.
        dictionary : :class:`~gensim.corpora.Dictionary`
            An input dictionary, which will be used to compute the average length of a document and
            inverse term document frequencies.  If None, then `corpus` will be used to compute the
            statistics. If both `corpus` and `dictionary` are None, the statistics will be left
            unintialized. Default is None.
        k1 : float
            A positive tuning parameter that determines the impact of the term frequency on its BM25
            weight. Singhal [5]_ suggests to set `k1` between 1.0 and 2.0. Default is 1.5.
        b : float
            A tuning parameter between 0.0 and 1.0 that determines the document length
            normalization: 1.0 corresponds to full document normalization, while 0.0 corresponds to
            no length normalization. Singhal [5]_ suggests to set `b` to 0.75, which is the default.

        Attributes
        ----------
        k1 : float
            A positive tuning parameter that determines the impact of the term frequency on its BM25
            weight. Singhal [3]_ suggests to set `k1` between 1.0 and 2.0. Default is 1.5.
        b : float
            A tuning parameter between 0.0 and 1.0 that determines the document length
            normalization: 1.0 corresponds to full document normalization, while 0.0 corresponds to
            no length normalization. Singhal [3]_ suggests to set `b` to 0.75, which is the default.

        """
        self.k1, self.b = k1, b
        super().__init__(corpus, dictionary)

    def precompute_idfs(self, dfs, num_docs):
        idfs = dict()
        for term_id, freq in dfs.items():
            idf = math.log(num_docs) - math.log(freq)
            idfs[term_id] = idf
        return idfs

    def get_term_weights(self, num_tokens, term_frequencies, idfs):
        term_weights = idfs * (term_frequencies * (self.k1 + 1)
                              / (term_frequencies + self.k1 * (1 - self.b + self.b
                                                              * num_tokens / self.avgdl)))
        return term_weights
