#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Emil Rijcken, https://github.com/ERijck
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Fuzzy topic models.

FlsaModel contains three topic modeling algorithms: FLSA, FLSA-W and FLSA-E. 

1. FLSA - https://link.springer.com/article/10.1007/s40815-017-0327-9
2. FLSA-W - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9660139
3. FLSA-E - https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9945594

In experimental work, FLSA-W outperforms other topic modeling algorithms on
various open datasets in terms of the coherence- and diversity score:
    
https://pure.tue.nl/ws/portalfiles/portal/222725628/Pure_ExperimentalStudyOfFlsa_wForTopicModeling.pdf

The algorithms go through similar steps:

FLSA-W:
    1. Create a document-term matrix
    2. Calculate global term weights. 
    3. Perform SVD and obtain the V matrix
    4. Perform fuzzy clustering (the default is fuzzy C-means) on the V matrix.
        To obtain the partition matrix. 
    5. Perform various matrix multiplications to obtain the output matrices
        P(W|T) and P(T|D).

FLSA: Works similar to FLSA-W. However, the U-matrix is used in step 3 and 4 and
        for this reason the calculations in step 5 are different. 
        
FLSA-E:
    1: Train a word embedding (currently, only Word2Vec is supported) on the corpus. 
    2: Perform fuzzy clustering on the word embedding to obtain the partition matrix. 
    3. Perform various matrix multiplications to obtain P(W|T) and P(T|D).

Since these algorithms go through similar steps, the FlsaModel object contains 
the operations shared by the algorithms. When one of the models is initialized, 
it goes through the algorithm steps and calls FlsaModel to execute a step. 

EXAMPLE: 
    Suppose we have our dataset and id2word, then we can initialize/train a model with:

flsaw = FlsaW(
    corpus=corpus, 
    id2word=idword)

TOPICS: 
    To see the produced topics:
    
    topics = flsaw.show_topics()
    
EVALUATION:
    To evaluate the produced topics use:
        coherence = flsaw.get_coherence_score()
        diversity = flsaw.get_diversity_score()
        interpretability = flsaw.get_interpretability_score()
        
WARNING work-in-progress, do not use this module!

"""

from abc import abstractmethod
from collections import Counter
import itertools

import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import dok_matrix

try:
    from pyfume import Clustering
except ImportError as e:
    raise ImportError("FlsaModel requires pyfume; please install it with `pip install gensim[flsamodel]`")

import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import Word2Vec


class FlsaModel:
    """
    Initialize and train a fuzzy topic model.

    Parameters
    ----------
            corpus : either a list of list of str (tokens), or a list of list of tuples `(int, int)`.
                The input corpus.
                FIXME: Accept only BoW = standard Gensim streaming format. No need for "list of list of str".

            num_topics: int
                The number of topics to be trained.

            algorithm: str ['flsa', 'flsa-w', 'flsa-e']
                The algorithm to train.

            id2word: :class:`~gensim.corpora.dictionary.Dictionary`
                Object that maps word IDs to words.
                Only used when the corpus is passed into the object as a BoW.
                FIXME: Simply use this. Drop all the internal `index_to_word` and `word_to_index` shenanigans.

            word_weighting: str ['normal', 'idf', 'probidf', 'entropy']
                Global term weighting mechanism.

            cluster_method: str ['fcm', 'gk', 'fst-pso']
                Fuzzy clustering method.

            svd_factors: int
                The number of singular values to use.

    """
    def __init__(
            self,
            corpus,
            num_topics,
            algorithm,
            num_words=20,
            word_weighting='normal',
            cluster_method='fcm',
            svd_factors=2,
            id2word=None,
            min_count=None,
            window=None,
            vector_size=None,
        ):
        self.corpus = self._set_corpus(corpus, id2word)
        self.num_topics = num_topics
        self.algorithm = algorithm
        self.num_topics = num_topics
        self.num_words = num_words
        self.word_weighting = word_weighting
        self.cluster_method = cluster_method
        self.svd_factors = svd_factors
        self.min_count = min_count
        self.window = window
        self.vector_size = vector_size
        self._check_variables()
        self._vocabulary = set(el for lis in corpus for el in lis)
        self._index_to_word = list(self._vocabulary)
        self._word_to_index = {w: i for i, w in enumerate(self._index_to_word)}
        self._word_frequencies = Counter(itertools.chain(*self.corpus))
        self._prob_word_i = None
        self._prob_document_j = None
        self._prob_topic_k = None
        self._prob_word_given_topic = None
        self._prob_word_given_document = None
        self.coherence_score = None
        self.diversity_score = None
        self.pwgt, self.ptgd = self._get_matrices()

    def _set_corpus(self, corpus, id2word):
        """
        Set the corpus to FuzzyTM's required input format.

        If a list of list of str is passed into the method for corpus, then
        it returns the same corpus. If a BoW (list of list of tuples) is passed
        into the class, it transforms this into a list of list of str.

        Parameters
        ----------
            corpus : either a list of list of str (tokens), or a list of list of tuples `(int, int)`.
                The input corpus.

            id2word: :class:`~gensim.corpora.dictionary.Dictionary`
                Object that maps word IDs to words.
                Only used when the corpus is passed into the object as a BoW.

        Returns
        -------
            list of list of str
                The corpus in FuzzyTM's required input format.
        """
        if self._check_bow(corpus):
            if is2word is None:
                raise ValueError("id2word must be specified when using a BoW input corpus.")
            return self._convert_bow(corpus, id2word)
        return corpus

    @staticmethod
    def _check_bow(corpus):
        """
        Is the input corpus in the bag-of-word (BoW) format?

        Parameters
        ----------
            corpus : either a list of list of str (tokens), or a list of list of tuples `(int, int)`.
                The input corpus.

        Returns
        -------
            bool
                True if `corpus` in BoW format.
        """
        if not isinstance(corpus, list):
            return False
        for doc in corpus:
            if not isinstance(doc, list):
                return False
            for tup in doc:
                if not isinstance(tup, tuple):
                    return False
                if not isinstance(tup[0], int) or not isinstance(tup[1], int):
                    return False
        return True

    @staticmethod
    def _convert_bow(corpus, id2word):
        """
        Convert the BoW format into a list of list of str.

        Parameters
        ----------
            corpus : list of list of tuples `(int, int)`.
                The input corpus in BoW format.

            id2word: :class:`~gensim.corpora.dictionary.Dictionary`
                Object that maps word IDs to words.

        Returns
        -------
            list of list of str
                The corpus in FuzzyTM's required input format.

        """
        # FIXME remove; model must work directly over streaming BoW input, not lists of strings!
        doc_list = []
        for doc in corpus:
            str_doc = []
            for wordid, count in doc:
                str_doc.extend([id2word[wordid]] * count)
            data_list.append(str_doc)
        return data_list

    @abstractmethod
    def _get_matrices(self):
        raise NotImplementedError

    def _check_variables(self):
        """
        Is the input data in the right format? Raise a ValueError if not.

        Correct format: list of list of str (tokens).

        """
        for i, doc in enumerate(self.corpus):
            if not isinstance(doc, list):
                raise TypeError(f"corpus variable at index {i} is not a list")
            if not len(doc) > 0:
                raise ValueError(f"The corpus has an empty list at index {i} and should contain at least one str value")
            for j, word in enumerate(doc):
                if not isinstance(word, str):
                    raise TypeError(f"Word {j} of document {i} is not a str")

        if not isinstance(self.num_topics, int) or self.num_topics < 1:
            raise ValueError(f"Please use a positive int for num_topics, not {self.num_topics}")
        if not isinstance(self.num_words, int) or self.num_words < 1:
            raise ValueError(f"Please use a positive int for num_words, not {self.num_words}")

        allowed_word_weighting = {"entropy", "idf", "normal", "probidf"}
        if self.algorithm in {"flsa", "flsa-w"} and self.word_weighting not in allowed_word_weighting:
            raise ValueError(
                f"Invalid word weighting method {self.word_weighting}; must be one of {allowed_word_weighting}"
            )

        allowed_cluster_method = {"fcm", "fst-pso", "gk"}
        if self.cluster_method not in allowed_cluster_method:
            raise ValueError(f"Invalid cluster_method {self.cluster_method}; must be one of {allowed_cluster_method}")

        if not isinstance(self.svd_factors, int) or self.svd_factors <= 0:
            raise ValueError(f"Invalid svd_factors {self.svd_factors}; must be a positive integer")

        allowed_algorithms = {'flsa', 'flsa-w', 'flsa-e'}
        if self.algorithm not in allowed_algorithms:
            raise ValueError(f'Invalid algorithm {self.algorithm}; must be one of {allowed_algorithms}')

    @staticmethod
    def _create_sparse_local_term_weights(
            corpus,
            vocabulary_size,
            word_to_index,
        ):
        """
        Create a sparse matrix showing the frequency of each words in documents.

        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html

        Axes:
            rows: documents (size: number of documents in corpus)
            columns: words (size: vocabulary length)

        Parameters
        ----------
            corpus : list of lists of str
                The input file used to initialize the model.
            vocabulary_size : int
                Number of unique words in the corpus.
            word_to_index: dict {str : int}
                Maps each unique vocabulary word to a unique index number.

        Returns
        -------
            scipy.sparse.dok_matrix
                sparse matrix representation of the local term weights.
        """
        sparse_local_term_weights = dok_matrix(
            (len(corpus), vocabulary_size),
            dtype=np.float32,
        )
        for document_index, document in enumerate(corpus):
            # FIXME: This seems to be reconstructing BoW from strings again. What for? BoW was the original input.
            document_counter = Counter(document)
            for word in document_counter.keys():
                sparse_local_term_weights[document_index, word_to_index[word]] = document_counter[word]
        return sparse_local_term_weights

    def _create_sparse_global_term_weights(
            self,
            corpus,
            word_weighting,
            vocabulary_size=None,
            sparse_local_term_weights=None,
            index_to_word=None,
            word_to_index=None,
            sum_words=None,
        ):
        """
        Apply a word_weighting method on the sparse_local_term_weights to create sparse_global_term_weights.
        See: https://link.springer.com/article/10.1007/s40815-017-0327-9

        Parameters
        ----------
            corpus : list of lists of str
                The input file used to initialize the model.
            word_weighting : str
                Indicates the method used for word_weighting. Choose from:
                  - entropy
                  - normal
                  - idf
                  - probidf
             vocabulary_size : int
                Number of unique words in the corpus.
             sparse_local_term_weights : scipy.sparse.dok_matrix
                 A sparse matrix showing the frequency of each words in documents.
             word_to_index : dict {str : int}
                Maps each unique vocabulary word to a unique index number.
             index_to_word : list of str
                 Maps each unique index number to a unique vocabulary word.
             sum_words : collections.Counter {str : int}
                 The count of each word in the corpus.

        Returns
        -------
            scipy.sparse.dok_matrix
                sparse matrix representation of the global term weights.
        """
        num_documents = len(corpus)
        if word_weighting == 'entropy':
            global_term_weights = self._calculate_entropy(
                num_documents,
                vocabulary_size,
                sparse_local_term_weights,
                index_to_word, sum_words,
            )
        elif word_weighting == 'idf':
            global_term_weights = self._calculate_idf(
                num_documents,
                vocabulary_size,
                corpus,
                word_to_index,
            )
        elif word_weighting == 'normal':
            global_term_weights = self._calculate_normal(sparse_local_term_weights)
        elif word_weighting == 'probidf':
            global_term_weights = self._calculate_probidf(
                num_documents,
                vocabulary_size,
                corpus,
                word_to_index,
            )
        else:
            raise ValueError('Invalid word weighting method')
        return sparse_local_term_weights.multiply(global_term_weights).tocsc()

    def _calculate_entropy(
            self,
            num_documents,
            vocabulary_size,
            sparse_local_term_weights,
            index_to_word,
            sum_words,
        ):
        """
        Use the entropy word weighting method, for the second step in the
        FLSA-W and FLSA algorithm. 

        See: https://link.springer.com/article/10.1007/s40815-017-0327-9

        Parameters
        ----------
            num_documents : int
                The number of documents in the corpus.
            vocabulary_size : int
                Number of unique words in the corpus.
            sparse_local_term_weights : scipy.sparse.dok_matrix
                A sparse matrix showing the frequency of each words in documents.
            index_to_word : dict {int : str}
                Maps each unique index number to a unique vocabulary word.
            sum_words : collections.Counter {str : int}
                The count of each word in the corpus.

        Returns
        -------
            numpy.array : float
               A document-term matrix with globally weighted values.
        """
        p_log_p_ij = self._create_p_log_p_ij(
            num_documents,
            vocabulary_size,
            sparse_local_term_weights,
            index_to_word,
            sum_words,
        )
        summed_p_log_p = p_log_p_ij.sum(0).tolist()[0]
        return np.array([1 + summed_p_log_p_i / np.log2(num_documents) for summed_p_log_p_i in summed_p_log_p])

    def _calculate_idf(
            self,
            num_documents,
            vocabulary_size,
            corpus,
            word_to_index,
        ):
        """
        Use the IDF word weighting method, for the second step in the
        FLSA-W and FLSA algorithm. 

        See: https://link.springer.com/article/10.1007/s40815-017-0327-9

        Parameters
        ----------
            num_documents : int
                The number of documents in the corpus.
            vocabulary_size : int
                Number of unique words in the corpus.
            corpus : list of lists of str
                The input file used to initialize the model.
            word_to_index: dict {str : int}
                Maps each unique vocabulary word to a unique index number.

        Returns
        -------
            numpy.array : float
               A document-term matrix with globally weighted values.
        """
        # FIXME we already have streamed Bow / IDF methods implementd in Gensim, e.g. in the TfIdf models.
        # What's the overlap – is this really needed?
        #ERijck: once we have BOW implemented in FlsaModel, 
        #       I suggest we change to implemented Gensim models.
        binary_sparse_dtm = self._create_sparse_binary_dtm(
            num_documents,
            vocabulary_size,
            corpus,
            word_to_index,
        )
        summed_words = binary_sparse_dtm.sum(0).tolist()[0]
        return np.array([np.log2(num_documents / word_count) for word_count in summed_words])

    @staticmethod
    def _calculate_normal(sparse_local_term_weights):
        """
        Use the normal word weighting method, for the second step in the
        FLSA-W and FLSA algorithm. 

        See: https://link.springer.com/article/10.1007/s40815-017-0327-9

        Parameters
        ----------
             sparse_local_term_weights : scipy.sparse.dok_matrix
                 A sparse matrix showing the frequency of each words in documents.

        Returns
        -------
            numpy.array : float
               A document-term matrix with globally weighted values.
        """
        squared_dtm = sparse_local_term_weights.multiply(sparse_local_term_weights)
        summed_words = squared_dtm.sum(0).tolist()[0]
        return np.array([1 / (np.sqrt(word_count)) for word_count in summed_words])

    def _calculate_probidf(
            self,
            num_documents,
            vocabulary_size,
            corpus,
            word_to_index,
        ):
        """
        Use the probidf word weighting method, for the second step in the
        FLSA-W and FLSA algorithm. 

        See: https://link.springer.com/article/10.1007/s40815-017-0327-9

        Parameters
        ----------
            num_documents : int
                The number of documents in the corpus.
            vocabulary_size : int
                Number of unique words in the corpus.
            corpus : list of lists of str
                The input file used to initialize the model.
            word_to_index: dict {str : int}
                Maps each unique vocabulary word to a unique index number.

        Returns
        -------
            numpy.array : float
               A document-term matrix with globally weighted values.
        """
        binary_sparse_dtm = self._create_sparse_binary_dtm(
            num_documents,
            vocabulary_size,
            corpus,
            word_to_index,
            )
        summed_binary_words_list = binary_sparse_dtm.sum(0).tolist()[0]

        return np.array([np.log2((num_documents - binary_word_count) / binary_word_count)
                         for binary_word_count in summed_binary_words_list])

    @staticmethod
    def _create_p_log_p_ij(
            num_documents,
            vocabulary_size,
            sparse_local_term_weights,
            index_to_word,
            sum_words,
        ):
        """
        Calculate probability of word i in document j, multiplied by its base-2 logarithm.

        See: https://link.springer.com/article/10.1007/s40815-017-0327-9

        Parameters
        ----------
            num_documents : int
                The number of documents in the corpus.
            vocabulary_size : int
                Number of unique words in the corpus.
            sparse_local_term_weights : scipy.sparse.dok_matrix
                A sparse matrix showing the frequency of each words in documents.
            index_to_word : list of str
                Maps each unique index number to a unique vocabulary word.
            sum_words : collections.Counter {str : int}
                The count of each word in the corpus.

        Returns
        -------
             scipy.sparse.dok_matrix
        """
        p_log_p_ij = dok_matrix(
            (num_documents, vocabulary_size), dtype=np.float32,
            )
        for j in range(num_documents):
            row_counts = sparse_local_term_weights.getrow(j).toarray()[0]
            word_index = row_counts.nonzero()[0]
            non_zero_row_counts = row_counts[row_counts != 0]
            for i, count in enumerate(non_zero_row_counts):
                word = index_to_word[word_index[i]]
                prob_ij = count / sum_words[word]
                p_log_p_ij[j, word_index[i]] = prob_ij * np.log2(prob_ij)
        return p_log_p_ij

    @staticmethod
    def _create_sparse_binary_dtm(
            num_documents,
            vocabulary_size,
            corpus,
            word_to_index,
        ):
        """
        Create a binary sparse document-term-matrix (used for idf and probidf).

        See: https://link.springer.com/article/10.1007/s40815-017-0327-9

        Parameters
        ----------
            num_documents : int
                The number of documents in the corpus.
            vocabulary_size : int
                Number of unique words in the corpus.
            corpus : list of lists of str
                The input file used to initialize the model.
            word_to_index: dict {str : int}
                Maps each unique vocabulary word to a unique index number.

       Returns
        -------
             scipy.sparse.dok_matrix
        """
        binary_sparse_dtm = dok_matrix(
            (num_documents, vocabulary_size), dtype=np.float32,
            )
        for doc_index, document in enumerate(corpus):
            binary_document_counter = dict.fromkeys(document, 1)
            for word in set(document):
                binary_sparse_dtm[doc_index,
                                  word_to_index[word]] = binary_document_counter[word]
        return binary_sparse_dtm

    @staticmethod
    def _create_projected_data(
            algorithm,
            sparse_weighted_matrix,
            svd_factors,
        ):
        """
        Perform singular decomposition for dimensionality reduction.

        (See: https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm)
        For SVD on a sparse matrix, the sparsesvd package is used
        (https://pypi.org/project/sparsesvd/)

        Parameters
        ----------
             algorithm : str
                 Indicator for which algorithm is being trained ('flsa' or 'flsa-w').
             sparse_weighted_matrix : scipy.sparse.dok_matrix
                 Sparse global term matrix.
             svd_factors : int
                 The number of singular values to include.

        Returns
        -------
            numpy.array : float
        """
        svd_u, _, svd_v = svds(
            sparse_weighted_matrix,
            svd_factors,
        )
        if algorithm in ['flsa']:
            return svd_u
        elif algorithm in ['flsa-w']:
            return svd_v.T
        else:
            raise ValueError(f'Invalid algorithm {algorithm}; must be one of "flsa" or "flsa-w".')

    @staticmethod
    def _create_partition_matrix(
            data,
            number_of_clusters,
            method='fcm',
        ):
        """
        Cluster the projected data.

        The pyFUME package is used for clustering: https://pyfume.readthedocs.io/en/latest/Clustering.html

        Parameters
        ----------
             data: numpy.array
                 The output from self._create_projected_data().
             number_of_clusters : int
                 The number of clusters (topics).
             method : str
                 The cluster method, choose from: 'fcm', 'gk', 'fst-pso'.

        Returns
        -------
            numpy.array : float
        """
        clusterer = Clustering.Clusterer(
            nr_clus=number_of_clusters,
            data=data,
            )
        _, partition_matrix, _ = clusterer.cluster(method=method)
        return partition_matrix

    @staticmethod
    def _create_prob_document_j(sparse_matrix):
        """
        Get the probability of document j.

        Parameters
        ----------
            sparse_matrix : scipy.sparse.dok_matrix
                A sparse matrix representation of the global term weights.
        Returns
        -------
            numpy.array : float
                (shape: number of documents x 1)
        """
        # Vector with the length of num_document,
        # each cell represents the sum of all weights of a document
        document_sum = np.array([doc[0] for doc in sparse_matrix.sum(1).tolist()])
        # sum of all the elements in the weighted matrix
        total_sum_d = sum(sparse_matrix.sum(0).tolist()[0])
        return document_sum / total_sum_d  # normalized probability

    @staticmethod
    def _create_prob_word_i(sparse_matrix):
        """
        Get the probability of word i.

        Parameters
        ----------
            sparse_matrix : scipy.sparse.dok_matrix
                A sparse matrix representation of the global term weights.

        Returns
        -------
            numpy.array : float
                (shape: vocabulary_size x 1)
        """
        word_sum = np.array(sparse_matrix.sum(0).tolist())
        # Sum of all the elements in the weighted matrix
        total_sum_w = sum(sparse_matrix.sum(0).tolist()[0])
        return (word_sum / total_sum_w)[0]  # normalized probability

    @staticmethod
    def _create_prob_topic_k(
            prob_topic_given_word_transpose,
            prob_word_i,
        ):
        """
        Get the probability of topic k.

        Parameters
        ----------
            prob_topic_given_word_transpose : numpy.array : float
                The output from self._create_partition_matrix().
            prob_word_i : numpy.array : float
                The output from self._create_prob_word_i().

        Returns
        -------
            numpy.array : float
                (shape: 1 x number of topics)
        """
        return np.matmul(prob_topic_given_word_transpose.T, prob_word_i)

    @staticmethod
    def _check_passed_variables(
            algorithm,
            prob_topic_given_document_transpose,
            prob_topic_given_word_transpose,
            local_term_weights,
            global_term_weights,
        ):
        """Check whether the algorithms are being fed the right attributes."""
        if algorithm in ['flsa']:
            if prob_topic_given_document_transpose is None:
                raise ValueError(
                    "The `prob_topic_given_document_transpose` parameter must be set when using `flsa` algorithm."
                )
            if global_term_weights is None:
                raise ValueError(
                    "The `global_term_weights` parameter must be set when using `flsa` algorithm."
                )
        elif algorithm in ['flsa-w']:
            if prob_topic_given_word_transpose is None:
                raise ValueError(
                    "The `prob_topic_given_word_transpose` parameter must be set when using `flsa-w` algorithm."
                )
            if global_term_weights is None:
                raise ValueError(
                    "The `global_term_weights` parameter must be set when using `flsa-w` algorithm."
                )
        elif algorithm in ['flsa-e']:
            if prob_topic_given_word_transpose is None:
                raise ValueError(
                    "The `prob_topic_given_word_transpose` parameter must be set when using `flsa-e` algorithm."
                )
            if local_term_weights is None:
                raise ValueError(
                    "The `local_term_weights` parameter must be set when using `flsa-e` algorithm."
                )
        else:
            raise ValueError(f'Unsupported algorithm {algorithm}')

    def _create_probability_matrices(
            self,
            algorithm,
            prob_topic_given_document_transpose=None,
            prob_topic_given_word_transpose=None,
            local_term_weights=None,
            global_term_weights=None,
        ):
        """
        This method corresponds to FLSA and FLSA-W's step 5 and FLSA-E's step 3.
        Given an algorithm, it performs the matrix multiplications needed to 
        obtain the P(W|T) and P(T|D) matrices.

        The 'algorithm' parameter is needed for all algoriths. Whereas the other 
        parameters depend on the selected algorithm.

        Parameters
        ----------
            algorithm : str
                 Indicator for which algorithm is being trained ('flsa' or 'flsa-w').
            global_term_weights : scipy.sparse.dok_matrix
                The output from self._create_partition_matrix().
            prob_topic_given_document_transpose : numpy.array : float
                The output from self._create_partition_matrix() (flsa)
            prob_topic_given_word_transpose : numpy.array : float
                 (flsa-w)

        Returns
        -------
            numpy.array : float
                The probability of a word given a topic.
            numpy.array : float
                The probability of a topic given a document.
        """
        # Check whether the right variable are passed into the method.
        self._check_passed_variables(
            algorithm,
            prob_topic_given_document_transpose,
            prob_topic_given_word_transpose,
            local_term_weights,
            global_term_weights,
            )

        # Calculate the initial probabilities
        if algorithm in ['flsa', 'flsa-w']:
            self._prob_word_i = self._create_prob_word_i(global_term_weights)
            self._prob_document_j = self._create_prob_document_j(global_term_weights)
            if algorithm in ['flsa-w']:
                self._prob_topic_k = self._create_prob_topic_k(
                    prob_topic_given_word_transpose,
                    self._prob_word_i,
                    )
        elif algorithm in ['flsa-e']:
            self._prob_word_i = self._create_prob_word_i(local_term_weights)
            self._prob_document_j = self._create_prob_document_j(local_term_weights)
            self._prob_topic_k = self._create_prob_topic_k(
                prob_topic_given_word_transpose, self._prob_word_i,
                )
        if algorithm in ['flsa']:
            prob_document_and_topic = (prob_topic_given_document_transpose.T * self._prob_document_j).T
            prob_document_given_topic = prob_document_and_topic / prob_document_and_topic.sum(axis=0)
            self._prob_word_given_document = np.asarray(global_term_weights / global_term_weights.sum(1))
            self._prob_word_given_topic = np.matmul(
                self._prob_word_given_document.T,
                prob_document_given_topic,
                )
            prob_topic_given_document = prob_topic_given_document_transpose.T
            return self._prob_word_given_topic, prob_topic_given_document

        elif algorithm in ['flsa-w', 'flsa-e']:
            prob_word_and_topic = (prob_topic_given_word_transpose.T * self._prob_word_i).T
            self._prob_word_given_topic = prob_word_and_topic / prob_word_and_topic.sum(axis=0)
            if algorithm in ['flsa-w']:
                self._prob_word_given_document = np.asarray(global_term_weights / global_term_weights.sum(1)).T
            elif algorithm in ['flsa-e']:
                self._prob_word_given_document = np.asarray(local_term_weights / local_term_weights.sum(1)).T
            prob_document_given_word = (
                (self._prob_word_given_document * self._prob_document_j).T / np.array(self._prob_word_i)
            )
            prob_document_given_topic = np.matmul(
                prob_document_given_word,
                self._prob_word_given_topic,
            )
            prob_topic_given_document = (
                ((prob_document_given_topic * self._prob_topic_k).T / self._prob_document_j)
            )
            return self._prob_word_given_topic, prob_topic_given_document
        raise ValueError(f'Unsupported algorithm {algorithm}')

    @staticmethod
    def _create_dictlist_topn(
            topn,
            prob_word_given_topic,
            index_to_word,
        ):
        """
        Create a list with dictionaries of word probabilities
        per topic based on the top-n words.

        Parameters
        ----------
             topn : int
                The top-n words to include
                (needs only to be used when 'method=topn').
             prob_word_given_topic : numpy.array : float
                Matrix that gives the probability of a word given a topic.
             index_to_word : list of str
                Maps each unique index number to a unique vocabulary word.

        Returns
        -------
             list of dicts {int : float}
                Keys: all the indices of words from prob_word_given_topic
                with weights amongst the top percentage.
                Values: the probability associated to a word.
        """
        if not isinstance(topn, int) or topn <= 0:
            raise ValueError(f"`topn` must be a positive integer, not {topn}.")
        top_dictionaries = []
        for topic_index in range(prob_word_given_topic.shape[1]):
            new_dict = dict()
            highest_weight_indices = prob_word_given_topic[:, topic_index].argsort()[-topn:]
            for word_index in highest_weight_indices:
                new_dict[index_to_word[word_index]] = prob_word_given_topic[word_index, topic_index]
            top_dictionaries.append(new_dict)
        return top_dictionaries

    @staticmethod
    def _create_dictlist_percentile(
            perc,
            prob_word_given_topic,
            index_to_word,
        ):
        """
        Create a list with dictionaries of word probabilities per topic based on the percentile.
         - Keys: all the indices of words from prob_word_given_topic
             who's weight's are amongst the top percentage.
         - Values: the probability associated to a word.

        Parameters
        ----------
             perc : float
                The top percentile words to include
                (needs only to be used when 'method=percentile').
             prob_word_given_topic : numpy.array : float
                Matrix that gives the probability of a word given a topic.
             index_to_word : list of str
                Maps each unique index number to a unique vocabulary word.

        Returns
        -------
             list of dicts {int : float}
                Keys: all the indices of words from prob_word_given_topic
                    with weights amongst the top percentage.
                Values: the probability associated to a word.
        """
        if not isinstance(perc, float) and 0 <= perc <= 1:
            raise ValueError("Please choose a number between 0 and 1 for 'perc'")
        top_list = []
        for top in range(prob_word_given_topic.shape[1]):
            new_dict = dict()
            count = 0
            i = 0
            weights = np.sort(prob_word_given_topic[:, top])[::-1]
            word_indices = np.argsort(prob_word_given_topic[:, top])[::-1]
            while count < perc:
                new_dict[index_to_word[word_indices[i]]] = weights[i]
                count += weights[i]
                i += 1
            top_list.append(new_dict)
        return top_list

    def show_topics(
            self,
            formatted=True,
            prob_word_given_topic=None,
            num_words=-1,
            index_to_word=None,
        ):
        """
        Get a representation for topics.

        Parameters
        ----------
            formatted : bool
                Whether the topic representations should be formatted as strings.
                If False, they are returned as 2 tuples of (word, probability).
            prob_word_given_topic : numpy.array : float
                Matrix that gives the probability of a word given a topic.
            num_words : int
                 Indicates how many words per topic should be shown.
            index_to_word : list of str
                Maps each unique index number to a unique vocabulary word.

        Returns
        -------
            list of tuples (int, str)
                The produced topics.
        """
        if prob_word_given_topic is None:
            prob_word_given_topic = self._prob_word_given_topic

        if num_words < 0:
            num_words = self.num_words
        if index_to_word is None:
            index_to_word = self._index_to_word
        if not isinstance(prob_word_given_topic, np.ndarray):
            raise TypeError(
                "Please feed a np.ndarray to 'prob_word_given_topic'"
            )
        if not isinstance(index_to_word, dict):
            raise TypeError("Please feed a dict to 'index_to_word'")
        if not isinstance(num_words, int) or num_words <= 0:
            raise TypeError("Please use a positive int for 'num_words'.")
        if prob_word_given_topic.shape[0] < prob_word_given_topic.shape[1]:
            raise ValueError("'prob_word_given_topic' has more columns then rows,",
                             " probably you need to take the transpose.")  # FIXME What? Why?
            #ERijck: A model with more topics than words makes no sense. 
        if prob_word_given_topic.shape[0] != len(index_to_word.keys()):
            raise ValueError(
                f"The shape of prob_word_given_topic={prob_word_given_topic.shape} doesn't match "
                f"the length of index_to_word={len(index_to_word)}."
            )
        if not isinstance(formatted, bool):
            raise ValueError('Please choose a boolean for "formatted"')
        topic_list = []
        if not formatted:
            for topic_index in range(prob_word_given_topic.shape[1]):
                sorted_highest_weight_indices = prob_word_given_topic[:, topic_index].argsort()[-num_words:][::-1]
                weight_words = [
                    f"{round(prob_word_given_topic[word_index, topic_index], 4)} * {index_to_word[word_index]}"
                    for word_index in sorted_highest_weight_indices
                ]
                topic_list.append((topic_index, ' + '.join(weight_words)))
            return topic_list
        else:
            for topic_index in range(prob_word_given_topic.shape[1]):
                sorted_highest_weight_indices = prob_word_given_topic[:, topic_index].argsort()[-num_words:][::-1]
                word_list = [
                    index_to_word[word_index]
                    for word_index in sorted_highest_weight_indices
                ]
                topic_list.append(word_list)
            return topic_list

    def get_topic_embedding(
            self,
            corpus,
            prob_word_given_topic=None,
            method='topn',
            topn=20,
            perc=0.05,
        ):
        """
        Create a topic embedding for each input document,
        to be used as input to predictive models.

        Parameters
        ----------
            corpus : list of lists of str
                The input file used to initialize the model.
            prob_word_given_topic : numpy.array : float
                Matrix that gives the probability of a word given a topic.
            method : str
                Method to select words to be included in the embedding.
                (choose from 'topn', 'percentile'):
                    - topn: for each topic the top n words with the highest
                        probability are included.
                    - percentile: for each topic all words with highest
                        probabilities are assigned while the cumulative
                        probability is lower than the percentile.
            topn : int
                The top-n words to include
                (needs only to be used when 'method=topn').
            perc: float
                The benchmark percentile until which words need to be added
                (between 0 and 1).

        Returns
        -------
            numpy.array : float
                Array in which each row gives the topic embedding for
                the associated document.
        """
        self._check_variables()
        if prob_word_given_topic is None:
            prob_word_given_topic = self._prob_word_given_topic
        top_dist = []
        if method not in ['topn', 'percentile']:
            raise ValueError(f"The `method` parameter must be one of {{'topn', 'percentile'}}, not {method}.")
        if method == 'topn':
            dictlist = self._create_dictlist_topn(topn, prob_word_given_topic, self._index_to_word)
        else:
            dictlist = self._create_dictlist_percentile(perc, prob_word_given_topic, self._index_to_word)
        for doc in corpus:
            topic_weights = [0] * prob_word_given_topic.shape[1]
            for word in doc:
                for i in range(prob_word_given_topic.shape[1]):
                    topic_weights[i] += dictlist[i].get(word, 0)
            top_dist.append(topic_weights)
        return np.array(top_dist)

    def get_coherence_score(
            self,
            corpus=None,
            topics=None,
            coherence='c_v',
        ):
        """
        Calculate the coherence score for a set of topics (https://dl.acm.org/doi/10.1145/2684822.2685324).
        This method can be called on the topics trained by the FlsaModel and by topics trained elsewhere.
        In case of the first, the method can be called without passing any variables.
        In case of the latter, the topics and corpus should be fed to the method. 
        
        Parameters
        ----------
             corpus : list of lists of str
                The input file used to initialize the model.
             topics : list of lists of str
                 The words per topics with the same format as produced by self.show_topics(formatted=True).
             coherence : str
                 The type of coherence to be calculated.
                 Choose from: 'u_mass', 'c_v', 'c_uci', 'c_npmi'.

        Returns
        -------
             float
                 The coherence score.
        """
        if corpus is None and topics is None:
            corpus = self.corpus
            topics = self.show_topics(formatted=True)

        id2word = corpora.Dictionary(corpus)
        corpus_bow = [id2word.doc2bow(text) for text in corpus]
        self.coherence_score = CoherenceModel(
            topics=topics,
            texts=corpus,
            corpus=corpus_bow,
            dictionary=id2word,
            coherence=coherence,
            topn=len(topics[0]),
            ).get_coherence()
        return self.coherence_score

    def get_diversity_score(self, topics=None):
        """
        Calculate the diversity score for a set of topics.
        This method can be called on the topics trained by the FlsaModel and by topics trained elsewhere.
        In case of the first, the method can be called without passing any variables.
        In case of the latter, the topics and corpus should be fed to the method. 

        Diversity = number of unique words / number of total words.
        See: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00325/96463/Topic-Modeling-in-Embedding-Spaces

        Parameters
        ----------
             topics : list of lists of str
                 The words per topics,
                 equivalent to self.show_topics(formatted=True).

        Returns
        -------
             float
                 The diversity score.
        """
        if topics is None:
            topics = self.show_topics(formatted=True)
        unique_words = set()
        total_words = 0
        for top in topics:
            unique_words.update(top)
            total_words += len(top)
        self.diversity_score = len(unique_words) / total_words
        return self.diversity_score

    def get_interpretability_score(self, corpus=None, topics=None, coherence='c_v'):
        """
        Calculate the interpretability score for a set of topics.
        This method can be called on the topics trained by the FlsaModel and by topics trained elsewhere.
        In case of the first, the method can be called without passing any variables.
        In case of the latter, the topics and corpus should be fed to the method. 

        Interpretability = coherence * diversity.
        See: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00325/96463/Topic-Modeling-in-Embedding-Spaces

        Parameters
        ----------
             corpus : list of lists of str
                The input file used to initialize the model.
             topics : list of lists of str
                 The words per topics, equivalent to
                 self.show_topics(formatted=True).
             coherence : str
                 The type of coherence to be calculated.
                 Choose from: 'u_mass', 'c_v', 'c_uci', 'c_npmi'.

        Returns
        -------
             float
                 The interpretability score.
        """
        if corpus is None and topics is None:
            corpus = self.corpus
            topics = self.show_topics(formatted=True)
        if self.coherence_score is None:
            self.coherence_score = self.get_coherence_score(
                corpus,
                topics,
                coherence,
            )
        if self.diversity_score is None:
            self.diversity_score = self.get_diversity_score(topics)
        return self.coherence_score * self.diversity_score

    def get_topics(self):
        """
        Get the term-topic matrix.

        Returns
        -------
        numpy.ndarray of shape (num_topics, vocabulary_size)
            The probability for each word in each topic.
        """
        return self.pwgt

    def get_word_to_index(self):
        """
        Obtain a dictionary that maps each vocabulary word to an index.

        Returns
        -------
        dict of {str : int}
            word to int mapping.
        """
        return self._word_to_index

    def get_corpus(self):
        """
        Return the input file.

        Returns
        -------
            list of list of str
                The input file 'corpus'.
        """
        return self.corpus

    def get_prob_word_i(self):
        """
        Return the probabilities per word.

        Returns
        -------
            np.array of float
                The probabilities per word.
        """
        return self._prob_word_i

    def get_prob_document_j(self):
        """
        Return the probabilities per document.

        Returns
        -------
            np.array of float
                The probabilities per document.
        """
        return self._prob_document_j

    def get_prob_topic_k(self):
        """
        Return the probabilities per topic.

        Returns
        -------
            np.array of float
                The probabilities per topic.
        """
        return self._prob_topic_k


class Flsa(FlsaModel):
    """
    The FLSA algorithm.
    See https://link.springer.com/article/10.1007/s40815-017-0327-9

    Parameters
        ----------
            corpus : list of lists of str
                The input file used to initialize the model.
            num_topics : int
                The number of topics that the model should train.
            num_words : int
                 Indicates how many words per topic should be shown.
            word_weighting : str
                Indicates the method used for word_weighting. Choose from:
                  - entropy
                  - normal
                  - idf
                  - probidf
            svd_factors : int
                 The number of singular values to include.
            cluster_method : str
                 The cluster algorithm to be used ('fcm', 'gk', 'fst-pso').
    """
    def __init__(
            self,
            corpus,
            num_topics,
            num_words=10,
            word_weighting='normal',
            svd_factors=2,
            cluster_method='fcm',
        ):
        super().__init__(
            algorithm='flsa',
            corpus=corpus,
            num_topics=num_topics,
            num_words=num_words,
            word_weighting=word_weighting,
            cluster_method=cluster_method,
            svd_factors=svd_factors,
        )

    def _get_matrices(self):
        """
        Method to obtain the matrices after the model has been initialized.

        Returns
        -------
            numpy.array : float
                The probability of a word given a topic.
            numpy.array : float
                The probability of a topic given a document.
        """
        sparse_document_term_matrix = self._create_sparse_local_term_weights(
            self.corpus,
            len(self.vocabulary),
            self._word_to_index,
        )
        sparse_global_term_weighting = self._create_sparse_global_term_weights(
            corpus=self.corpus,
            word_weighting=self.word_weighting,
            vocabulary_size=len(self.vocabulary),
            sparse_local_term_weights=sparse_document_term_matrix,
            index_to_word=self._index_to_word,
            word_to_index=self._word_to_index,
            sum_words=self._word_frequencies,
        )
        projected_data = self._create_projected_data(
            algorithm='flsa',
            sparse_weighted_matrix=sparse_global_term_weighting,
            svd_factors=self.svd_factors,
        )
        partition_matrix = self._create_partition_matrix(
            data=projected_data,
            number_of_clusters=self.num_topics,
            method=self.cluster_method
        )
        return self._create_probability_matrices(
            algorithm='flsa',
            prob_topic_given_document_transpose=partition_matrix,
            global_term_weights=sparse_global_term_weighting,
        )


class FlsaW(FlsaModel):
    """
    Train the FLSA-W algorithm.

    See: https://ieeexplore.ieee.org/abstract/document/9660139

    Parameters
        ----------
            corpus : list of lists of str
                The input file used to initialize the model.
            num_topics : int
                The number of topics that the model should train.
            num_words : int
                 Indicates how many words per topic should be shown.
            word_weighting : str
                Indicates the method used for word_weighting. Choose from:
                  - entropy
                  - normal
                  - idf
                  - probidf
            svd_factors : int
                 The number of singular values to include.
            cluster_method : str
                 The cluster algorithm to be used ('fcm', 'gk', 'fst-pso').
    """
    def __init__(
            self,
            corpus,
            num_topics,
            num_words=10,
            word_weighting='normal',
            svd_factors=2,
            cluster_method='fcm',
        ):

        super().__init__(
            algorithm='flsa-w',
            corpus=corpus,
            num_topics=num_topics,
            num_words=num_words,
            word_weighting=word_weighting,
            cluster_method=cluster_method,
            svd_factors=svd_factors,
        )

    def _get_matrices(self):
        """
        Obtain the matrices after the model has been initialized.

        Returns
        -------
            numpy.array : float
                The probability of a word given a topic.
            numpy.array : float
                The probability of a topic given a document.
        """
        sparse_document_term_matrix = self._create_sparse_local_term_weights(
            self.corpus,
            len(self.vocabulary),
            self._word_to_index,
        )
        sparse_global_term_weighting = self._create_sparse_global_term_weights(
            corpus=self.corpus,
            word_weighting=self.word_weighting,
            vocabulary_size=len(self.vocabulary),
            sparse_local_term_weights=sparse_document_term_matrix,
            index_to_word=self._index_to_word,
            word_to_index=self._word_to_index,
            sum_words=self._word_frequencies,
        )
        projected_data = self._create_projected_data(
            algorithm='flsa-w',
            sparse_weighted_matrix=sparse_global_term_weighting,
            svd_factors=self.svd_factors,
        )
        partition_matrix = self._create_partition_matrix(
            data=projected_data,
            number_of_clusters=self.num_topics,
            method=self.cluster_method,
        )
        return self._create_probability_matrices(
            algorithm='flsa-w',
            prob_topic_given_word_transpose=partition_matrix,
            global_term_weights=sparse_global_term_weighting,
        )


class FlsaE(FlsaModel):
    """
    Train the FLSA-E algorithm.

    See: https://research.tue.nl/nl/publications/exploring-embedding-spaces-for-more-coherent-topic-modeling-in-el

    Parameters
        ----------
            corpus : list of lists of str
                The input file used to initialize the model.
            num_topics : int
                The number of topics that the model should train.
            num_words : int
                Indicates how many words per topic should be shown.
            cluster_method : str
                The cluster algorithm to be used ('fcm', 'gk', 'fst-pso').
            min_count : int
                Ignores all words with total frequency lower than this.
            window : int
                Maximum distance between the current and predicted word within a sentence.
            vector_size : int
                Dimensionality of the word vectors.
    """

    def __init__(
            self,
            corpus,
            num_topics,
            num_words=10,
            cluster_method='fcm',
            min_count=1,
            window=5,
            vector_size=20,
        ):

        self.model = ...   # FIXME what is this?
        self.word_embedding = ...  # FIXME what is this?
        #ERijck: this way users can do post-analysis. But storing them as an attribute 
        #       is not crucial.

        super().__init__(
            algorithm='flsa-e',
            corpus=corpus,
            num_topics=num_topics,
            num_words=num_words,
            cluster_method=cluster_method,
            min_count=min_count,
            window=window,
            vector_size=vector_size,
        )

    def get_word_embedding(
            self,
            data,
            vector_size,
            window,
            min_count,
        ):
        """
            Train a word embedding on the corpus.

            Parameters
            ----------
                data : list of lists of str
                    The input file used to initialize the model.
                min_count : int
                    Ignores all words with total frequency lower than this.
                window : int
                    Maximum distance between the current and predicted word within a sentence.
                vector_size : int
                    Dimensionality of the word vectors.
        """
        self.model = Word2Vec(
            sentences=data,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
        )
        # FIXME is the whole Word2Vec model really needed? Why are we storing it as an attribute?
        #ERijck: This way users can do a post analysis. However, it is not crucial and can be removed.
        return self.model.wv.vectors

    def _get_matrices(self):
        """
        Run after the FLSA_E class has been initialized to obtain the output matrices.

        Returns
        -------
            numpy.array : float
                The probability of a word given a topic.
            numpy.array : float
                The probability of a topic given a document.
        """
        sparse_document_term_matrix = self._create_sparse_local_term_weights(
            self.corpus,
            len(self.vocabulary),
            self._word_to_index,
        )

        self.word_embedding = self.get_word_embedding(
            data=self.corpus,
            min_count=self.min_count,
            vector_size=self.vector_size,
            window=self.window,
        )

        partition_matrix = self._create_partition_matrix(
            data=self.word_embedding,
            number_of_clusters=self.num_topics,
            method=self.cluster_method,
        )

        return self._create_probability_matrices(
            algorithm='flsa-e',
            prob_topic_given_word_transpose=partition_matrix,
            local_term_weights=sparse_document_term_matrix,
        )
