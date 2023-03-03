# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:04:27 2022

@author: 20200016
"""

from collections import Counter
import itertools
import math
import pickle
import sys
import warnings

import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import dok_matrix

import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import Word2Vec


class FlsaModel():
    """
    Class to initialize and train fuzzy topic models with methods similar
    to Gensim's LdaModel'

    Parameters
    ----------
            corpus : The input corpus.
                either: list of list of str.
                or: list of list of tuples (int, int) (bow).

            num_topics: int
                The number of topics to be trained.

            algorithm: str ['flsa', 'flsa-w', 'flsa-e']
                The algorithm to train.

            id2word: gensim.corpora.dictionary.Dictionary
                Object to map id's to words
                (only used when the corpus is passed into the object as a bow).

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
            workers=None,
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
        self.workers = workers
        self._check_variables()
        self._vocabulary, self._vocabulary_size = self._create_vocabulary(self.corpus)
        self._word_to_index, self._index_to_word = self._create_index_dicts(self._vocabulary)
        self._sum_words = self._create_sum_words(self.corpus)
        self._prob_word_i = None
        self._prob_document_j = None
        self._prob_topic_k = None
        self._prob_word_given_topic = None
        self._prob_word_given_document = None
        self.coherence_score = None
        self.diversity_score = None
        self.pwgt, self.ptgd = self._get_matrices()

    def _set_corpus(
            self,
            corpus,
            id2word,
            ):
        """
        Method that sets the corpus to FuzzyTM's required input format.
        If a list of list of str is passed into the method for corpus, then
        it returns the same corpus. If a bow (list of list of tuples) is passed
        into the class, it transforms this into a list of list of str.

        Parameters
        ----------
            corpus : either: list of list of str (tokens). or: list of list of tuples (int, int).
                The input corpus.
            id2word: gensim.corpora.dictionary.Dictionary
                Object to map id's to words
                (only used when the corpus is passed into the object as a bow)

        Returns
        -------
            list of list of str
                The corpus in FuzzyTM's required input format.
        """
        if self._check_bow(corpus):
            if not isinstance(id2word, corpora.dictionary.Dictionary):
                raise ValueError("Please pass 'id2word' when using a bow for 'corpus'.")
            return self._convert_bow(corpus, id2word)
        return corpus

    @staticmethod
    def _check_bow(
            corpus,
            ):
        """
        Method to check if the input format has the bow format.

        Parameters
        ----------
            corpus : either: list of list of str (tokens). or: list of list of tuples (int, int).
                The input corpus.

        Returns
        -------
            bool
                True if bow format
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
    def _convert_bow(
            corpus,
            id2word,
            ):
        """
        Method to convert the bow format into a list of list of str.

        Parameters
        ----------
            corpus : The input corpus.
                either: list of list of str (tokens).
                or: list of list of tuples (int, int).

            id2word: gensim.corpora.dictionary.Dictionary
                Object to map id's to words

        Returns
        -------
            list of list of str
                The corpus in FuzzyTM's required input format.
        """
        id2token = {v: k for k, v in id2word.token2id.items()}
        data_list = []
        for doc in corpus:
            doc_list = []
            for tup in doc:
                for _ in itertools.repeat(None, tup[1]):
                    doc_list.append(id2token[tup[0]])
            data_list.append(doc_list)
        return data_list

    def _check_variables(self):
        """
        Check whether the input data has the right format.

        Correct format: list of list of str (tokens)
        The function raises an error if the format is incorrect.
        """
        for i, doc in enumerate(self.corpus):
            if not isinstance(doc, list):
                raise TypeError("corpus variable at index ",
                                str(i),
                                " is not a list")
            if not len(doc) > 0:
                raise ValueError(
                    "The corpus has an empty list at index ",
                    str(i),
                    " and should contain at least one str value")
            for j, word in enumerate(doc):
                if not isinstance(word, str):
                    raise TypeError(f"Word {j} of document {i} is not a str")
        if not isinstance(self.num_topics, int) or self.num_topics < 1:
            raise ValueError("Please use a positive int for num_topics")
        if not isinstance(self.num_words, int) or self.num_words < 1:
            raise ValueError("Please use a positive int for num_words")
        if self.algorithm in [
                "flsa",
                "flsa-w",
                ] and self.word_weighting not in [
                "entropy",
                "idf",
                "normal",
                "probidf",
                ]:
            warning = ["Invalid word weighting method",
                       "Please choose between:",
                       "'entropy', 'idf', 'normal' and'probidf'",
                       ]
            raise ValueError(' '.join(warning))
        if self.cluster_method not in [
                "fcm",
                "fst-pso",
                "gk",
                ]:
            raise ValueError(
                "Invalid 'cluster_method. Please choose: 'fcm', 'fst-pso' or 'gk'")
        if not isinstance(self.svd_factors, int) and self.svd_factors > 0:
            raise ValueError("Please use a positive int for svd_factors")
        if self.algorithm not in [
                'flsa',
                'flsa-w',
                'flsa-e',
                ]:
            raise ValueError('Please select a correct "algoritm"')

    @staticmethod
    def _create_vocabulary(corpus):
        """
        Create the vocabulary from 'corpus'.

        Parameters
        ----------
             corpus : list of lists of str
                 The input file used to initialize the model.

        Returns
        -------
            set of str
                All the vocabulary words.
        """
        vocabulary = set(el for lis in corpus for el in lis)
        return vocabulary, len(vocabulary)

    @staticmethod
    def _create_index_dicts(vocabulary):
        """
        Create the dictionaries with mappings between words and indices.

        Parameters
        ----------
            vocabulary : set of str
                All the words in the corpus.

        Returns
        -------
            dict of {str : int}
                Dictionary that maps a vocabulary word to and index number.
            dict of {int : str}
                Dictionary that maps an index number to each vocabulary word.
        """
        if not isinstance(vocabulary, set):
            raise ValueError("Please use a 'set' type for 'vocabulary'.")
        word_to_index = dict()
        index_to_word = dict()
        for i, word in enumerate(vocabulary):
            word_to_index[word] = i
            index_to_word[i] = word
        return word_to_index, index_to_word

    @staticmethod
    def _create_sum_words(corpus):
        """
        Creates a Counter object that stores the count of each word in the corpus (corpus).

        Parameters
        ----------
            corpus : list of lists of str
                The input file used to initialize the model.

        Returns
        -------
            collections.Counter {str : int}
                The count of each word in the corpus.
        """
        sum_words = Counter()
        for document in corpus:
            sum_words.update(Counter(document))
        return sum_words

    @staticmethod
    def _create_sparse_local_term_weights(
            corpus,
            vocabulary_size,
            word_to_index,
            ):
        """
        Creates a sparse matrix showing the frequency of each words in documents.

        (See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html)
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
            (len(corpus),
             vocabulary_size),
            dtype=np.float32,
            )
        for document_index, document in enumerate(corpus):
            document_counter = Counter(document)
            for word in document_counter.keys():
                sparse_local_term_weights[
                    document_index, word_to_index[word],
                    ] = document_counter[word]
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
        Apply a word_weighting method on the sparse_local_term_weights
        to create sparse_global_term_weights.
        (See: https://link.springer.com/article/10.1007/s40815-017-0327-9)

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
             index_to_word : dict {int : str}
                 Maps each unique index number to a unique vocabulary word.
             sum_words : collections.Counter {str : int}
                 The count of each word in the corpus.

        Returns
        -------
            scipy.sparse.dok_matrix
                sparse matrix representation of the global term weights.
        """
        num_documents = len(corpus)
        if word_weighting in ['entropy', 'normal']:
            if sparse_local_term_weights is None:
                raise ValueError("Please feed the algorithm 'sparse_local_term_weights'")
        if word_weighting in ['entropy']:
            if index_to_word is None:
                raise ValueError("Please feed the algorithm 'index_to_word'")
            if sum_words is None:
                raise ValueError("Please feed the algorithm 'sum_words'")
        if word_weighting in ['entropy', 'idf', 'probidf']:
            if vocabulary_size is None:
                raise ValueError("Please feed the algorithm 'vocabulary_size'")
        if word_weighting in ['idf', 'probidf']:
            if word_to_index is None:
                raise ValueError("Please feed the algorithm 'word_to_index'")
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
        Use the entropy word weighting method.

        (See: https://link.springer.com/article/10.1007/s40815-017-0327-9)

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
        Use the idf word weightingg method.

        (See: https://link.springer.com/article/10.1007/s40815-017-0327-9)

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
        """
        binary_sparse_dtm = self._create_sparse_binary_dtm(
            num_documents,
            vocabulary_size,
            corpus,
            word_to_index,
            )
        summed_words = binary_sparse_dtm.sum(0).tolist()[0]
        return np.array([np.log2(num_documents / word_count) for word_count in summed_words])

    @staticmethod
    def _calculate_normal(
            sparse_local_term_weights,
            ):
        """
        Use the normal word weightingg method.

        (See: https://link.springer.com/article/10.1007/s40815-017-0327-9)

        Parameters
        ----------
             sparse_local_term_weights : scipy.sparse.dok_matrix
                 A sparse matrix showing the frequency of each words in documents.

        Returns
        -------
            numpy.array : float
        """
        squared_dtm = sparse_local_term_weights.multiply(sparse_local_term_weights)
        summed_words = squared_dtm.sum(0).tolist()[0]
        return np.array([1 / (math.sqrt(word_count)) for word_count in summed_words])

    def _calculate_probidf(
            self,
            num_documents,
            vocabulary_size,
            corpus,
            word_to_index,
            ):
        """
        Use the probidf word weightingg method.

        (See: https://link.springer.com/article/10.1007/s40815-017-0327-9)

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
        Create probability of word i in document j, multiplied by its base-2 logarithm.

        (See: https://link.springer.com/article/10.1007/s40815-017-0327-9)

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

        (See: https://link.springer.com/article/10.1007/s40815-017-0327-9)

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
        if algorithm in ['flsa-w']:
            return svd_v.T
        raise ValueError('Invalid algorithm selected.',
                         'Only "flsa" ans "flsa-w" are currently supported.')

    @staticmethod
    def _create_partition_matrix(
            data,
            number_of_clusters,
            method='fcm',
            ):
        """
        Perform clustering on the projected data.

        The pyFUME package is used for clustering:
            (https://pyfume.readthedocs.io/en/latest/Clustering.html)

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

        try:
            from pyfume import Clustering
        except ImportError:
            msg = (
                "FlsaModel requires pyfume; install manually via "
                "`pip install pyfume` or otherwise"
            )
            print('-' * len(msg), file=sys.stderr)
            print(msg, file=sys.stderr)
            print('-' * len(msg), file=sys.stderr)
            raise

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
        """
        Check whether the algorithms are being fed the right attributes.
        """
        if algorithm in ['flsa']:
            if prob_topic_given_document_transpose is None:
                raise ValueError("Please feed the method",
                                 "'prob_topic_given_document_transpose' to run flsa")
            if global_term_weights is None:
                raise ValueError("Please feed the method 'global_term_weights', to run flsa")
        elif algorithm in ['flsa-w']:
            if prob_topic_given_word_transpose is None:
                raise ValueError("Please feed the method",
                                 "'prob_topic_given_word_transpose' to run flsa-w")
            if global_term_weights is None:
                raise ValueError("Please feed the method 'global_term_weights'",
                                 " to run flsa-w")
        elif algorithm in [
                           'flsa-e',
                           ]:
            if prob_topic_given_word_transpose is None:
                raise ValueError("Please feed the method",
                                 "'prob_topic_given_word_transpose' to run model")
            if local_term_weights is None:
                raise ValueError("Please feed the method 'local_term_weights', to run model")

        else:
            raise ValueError('Your algorithm is currently not supported')

    def _create_probability_matrices(
            self,
            algorithm,
            prob_topic_given_document_transpose=None,
            prob_topic_given_word_transpose=None,
            local_term_weights=None,
            global_term_weights=None,
            ):
        """
        Method that performs matrix multiplications to obtain the output matrices.

        The 'algorithm' parameter is generic and the other ones depend on the selected algorithm.
        The other parameters passed into this method depend on the used algorithm.

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
                The prbability of a word given a topic.
            numpy.array : float
                The prbability of a topic given a document.
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
        if algorithm in [
                'flsa',
                'flsa-w',
                ]:
            self._prob_word_i = self._create_prob_word_i(global_term_weights)
            self._prob_document_j = self._create_prob_document_j(global_term_weights)
            if algorithm in ['flsa-w']:
                self._prob_topic_k = self._create_prob_topic_k(
                    prob_topic_given_word_transpose,
                    self._prob_word_i,
                    )
        elif algorithm in [
                           'flsa-e',
                           ]:
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

        elif algorithm in [
                'flsa-w',
                'flsa-e'
                ]:
            prob_word_and_topic = (prob_topic_given_word_transpose.T * self._prob_word_i).T
            self._prob_word_given_topic = prob_word_and_topic / prob_word_and_topic.sum(axis=0)
            if algorithm in ['flsa-w']:
                self._prob_word_given_document = np.asarray(global_term_weights / global_term_weights.sum(1)).T
            elif algorithm in [
                               'flsa-e',
                               ]:
                self._prob_word_given_document = np.asarray(local_term_weights / local_term_weights.sum(1)).T
            prob_document_given_word = ((self._prob_word_given_document * self._prob_document_j).T
                                        / np.array(self._prob_word_i))
            prob_document_given_topic = np.matmul(
                prob_document_given_word,
                self._prob_word_given_topic,
                )
            prob_topic_given_document = ((prob_document_given_topic * self._prob_topic_k).T
                                               / self._prob_document_j)
            return self._prob_word_given_topic, prob_topic_given_document
        raise ValueError('"algorithm" is unknown.')

    @staticmethod
    def _create_dictlist_topn(
            topn,
            prob_word_given_topic,
            index_to_word,
            ):
        """
        Creates a list with dictionaries of word probabilities
        per topic based on the top-n words.

        Parameters
        ----------
             topn : int
                The top-n words to include
                (needs only to be used when 'method=topn').
             prob_word_given_topic : numpy.array : float
                Matrix that gives the probability of a word given a topic.
             index_to_word : dict {int : str}
                Maps each unique index number to a unique vocabulary word.

        Returns
        -------
             list of dicts {int : float}
                Keys: all the indices of words from prob_word_given_topic
                who's weight's are amongst the top percentage.
                Values: the probability associated to a word.
        """
        if not isinstance(topn, int) and topn > 0:
            raise ValueError("Please choose a positive integer for 'topn'")
        top_dictionaries = []
        for topic_index in range(prob_word_given_topic.shape[1]):
            new_dict = dict()
            highest_weight_indices = prob_word_given_topic[:, topic_index].argsort()[-topn:]
            for word_index in highest_weight_indices:
                new_dict[index_to_word[word_index]] = prob_word_given_topic[
                    word_index, topic_index,
                    ]
            top_dictionaries.append(new_dict)
        return top_dictionaries

    @staticmethod
    def _create_dictlist_percentile(
            perc,
            prob_word_given_topic,
            index_to_word,
            ):
        """
        Create a list with dictionaries of word probabilities
        per topic based on the percentile.
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
             index_to_word : dict {int : str}
                Maps each unique index number to a unique vocabulary word.

        Returns
        -------
             list of dicts {int : float}
                Keys: all the indices of words from prob_word_given_topic
                    who's weight's are amongst the top percentage.
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
        Get a representation for the topics.

        Parameters
        ----------
            formatted : bool
                Whether the topic representations should be formatted as strings.
                If False, they are returned as 2 tuples of (word, probability).
            prob_word_given_topic : numpy.array : float
                Matrix that gives the probability of a word given a topic.
            num_words : int
                 Indicates how many words per topic should be shown.
            index_to_word : dict {int : str}
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
            raise TypeError("Please feed the algorithm 'prob_word_given_topic' as a np.ndarray")
        if not isinstance(index_to_word, dict):
            raise TypeError("Please feed the algorithm 'index_to_word' as a dict")
        if not isinstance(num_words, int) or num_words <= 0:
            raise TypeError("Please use a positive int for 'num_words'.")
        if prob_word_given_topic.shape[0] < prob_word_given_topic.shape[1]:
            raise ValueError("'prob_word_given_topic' has more columns then rows,",
                             " probably you need to take the transpose.")
        warning = ["It seems like 'prob_word_given_topic' and 'index_to_word",
                   "are not aligned. The number of vocabulary words in",
                   "'prob_word_given_topic' deviate from the ",
                   "number of words in 'index_to_word'."]
        if prob_word_given_topic.shape[0] != len(index_to_word.keys()):
            warnings.warn(' '.join(warning))
        if not isinstance(formatted, bool):
            raise ValueError('Please choose a boolean for "formatted"')
        topic_list = []
        if not formatted:
            for topic_index in range(prob_word_given_topic.shape[1]):
                weight_words = ""
                sorted_highest_weight_indices = prob_word_given_topic[:, topic_index].argsort()[-num_words:][::-1]
                for word_index in sorted_highest_weight_indices:
                    weight_words += (str(round(prob_word_given_topic[word_index, topic_index], 4))
                                     + '*"' + index_to_word[word_index] + '" + ')
                topic_list.append((topic_index, weight_words[:-3]))
            return topic_list
        else:
            for topic_index in range(prob_word_given_topic.shape[1]):
                word_list = []
                sorted_highest_weight_indices = prob_word_given_topic[:, topic_index].argsort()[-num_words:][::-1]
                for word_index in sorted_highest_weight_indices:
                    word_list.append(index_to_word[word_index])
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
            raise ValueError(method, "is not a valid option for 'method'.",
                             " Choose either 'topn' or 'percentile'")
        if method == 'topn':
            dictlist = self._create_dictlist_topn(
                topn, prob_word_given_topic, self._index_to_word,
                )
        else:
            dictlist = self._create_dictlist_percentile(
                perc, prob_word_given_topic, self._index_to_word,
                )
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
        Calculate the coherence score for the generated topic.

        Parameters
        ----------
             corpus : list of lists of str
                The input file used to initialize the model.
             topics : list of lists of str
                 The words per topics,
                 equivalent to self.show_topics(formatted=True).
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

    def get_diversity_score(
            self,
            topics=None,
            ):
        """''
        Calculate the diversity score for the generated topic.

        Diversity = number of unique words / number of total words.
        See: https://tinyurl.com/2bs84zd8

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

    def get_interpretability_score(
            self,
            corpus=None,
            topics=None,
            coherence='c_v',
            ):
        """''
        Calculate the interpretability score for the generated topics.

        Interpretability = coherence * diversity.
        (see: https://tinyurl.com/2bs84zd8)

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

    def get_vocabulary(self):
        """
        Returns a set of all the words in the corpus

        Example:
        After initializing an instance of the flsamodel models as 'model'

        corpus = [['this', 'is', 'the', 'first', 'file'],
             ['and', 'this', 'is', 'second', 'file']]

        model.get_vocabulary()

        >>> {'this', 'is', 'the', 'first', 'file', 'and', 'second'}
        """
        return self._vocabulary

    def get_topics(self):
        """
        Get the term-topic matrix.

        Returns
        -------
        numpy.ndarray
            The probability for each word in each topic,
            shape (num_topics, vocabulary_size).
        """
        return self.pwgt

    def get_vocabulary_size(self):
        """
        Returns the number of words in the vocabulary

        Example:
            After initializing an instance of the flsamodel models as 'model'

        corpus = [['this', 'is', 'the', 'first', 'file'],
             ['and', 'this', 'is', 'second', 'file']]

        model.get_vocabulary_size()

        >>> 7
        """
        return self._vocabulary_size

    def get_word_to_index(self):
        """
        Obtain a dictionary that maps each vocabulary word to an index.

        Returns
        -------
        dict of {str : int}
            word to int mapping.
        """
        return self._word_to_index

    def get_index_to_word(self):
        """
        Obtain a dictionary that maps index numbers to vocabulary words.

        Returns
        -------
        dict of {int : str}
            int to word mapping.
        """
        return self._index_to_word

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

    def save(
            self,
            filepath,
            ):
        """''
        Saves the object to the drive, using the pickle library.

        Parameters
        ----------
             filepath : str
                The directory in which the file should be stored,
                either with or without the file name.

        Returns
        -------
             float
                 The interpretability score.
        """
        if not isinstance(filepath, str):
            raise ValueError('Make sure that "filepath" has type "str"')
        if filepath.endswith('.pickle'):
            pickle_out = open(filepath, 'wb')
        elif filepath.endswith('/'):
            pickle_out = open(filepath + 'model.pickle', 'wb')
        else:
            pickle_out = open(filepath + '.pickle', 'wb')
        pickle.dump(self, pickle_out)
        pickle_out.close()

    def load(
            self,
            filepath,
            ):
        """''
        Loads the object from the drive, using the pickle library.

        Parameters
        ----------
             filepath : str
                The directory in which the file should be stored,
                either with or without the file name.

        Returns
        -------
             float
                 The interpretability score.
        """
        if not isinstance(filepath, str):
            raise ValueError('Make sure that "filepath" has type "str"')
        if not filepath.endswith('.pickle'):
            if filepath.endswith('/'):
                filepath += 'model.pickle'
            else:
                filepath += '/model.pickle'
        infile = open(filepath, 'rb')
        self.__dict__ = pickle.load(infile).__dict__
        infile.close()


class Flsa(FlsaModel):
    """
    Class to run the FLSA algorithm (see: https://tinyurl.com/mskjaeuu).

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
                The prbability of a word given a topic.
            numpy.array : float
                The prbability of a topic given a document.
        """
        sparse_document_term_matrix = self._create_sparse_local_term_weights(
            self.corpus,
            self._vocabulary_size,
            self._word_to_index,
            )
        sparse_global_term_weighting = self._create_sparse_global_term_weights(
            corpus=self.corpus,
            word_weighting=self.word_weighting,
            vocabulary_size=self._vocabulary_size,
            sparse_local_term_weights=sparse_document_term_matrix,
            index_to_word=self._index_to_word,
            word_to_index=self._word_to_index,
            sum_words=self._sum_words,
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
    Class to train the FLSA-W algorithm.

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
        Method to obtain the matrices after the model has been initialized.

        Returns
        -------
            numpy.array : float
                The prbability of a word given a topic.
            numpy.array : float
                The prbability of a topic given a document.
        """
        sparse_document_term_matrix = self._create_sparse_local_term_weights(
            self.corpus,
            self._vocabulary_size,
            self._word_to_index,
            )
        sparse_global_term_weighting = self._create_sparse_global_term_weights(
            corpus=self.corpus,
            word_weighting=self.word_weighting,
            vocabulary_size=self._vocabulary_size,
            sparse_local_term_weights=sparse_document_term_matrix,
            index_to_word=self._index_to_word,
            word_to_index=self._word_to_index,
            sum_words=self._sum_words,
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
    Class to train the FLSA-E algorithm. See: https://tinyurl.com/5n8utppk

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
            workers : int
                Use these many worker threads to train the model
                ( = faster training with multicore machines).
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
            workers=4,
            ):

        self.model = ...
        self.word_embedding = ...

        super().__init__(
            algorithm='flsa-e',
            corpus=corpus,
            num_topics=num_topics,
            num_words=num_words,
            cluster_method=cluster_method,
            min_count=min_count,
            window=window,
            vector_size=vector_size,
            workers=workers,
            )

    def get_word_embedding(
            self,
            data,
            vector_size,
            window,
            min_count,
            workers,
            ):
        """
            Method to train a word embedding on the corpus.

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
                    workers : int
                        Use these many worker threads to train the model
                        ( = faster training with multicore machines).
        """

        self.model = Word2Vec(
            sentences=data,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            )

        return self.model.wv.vectors

    def _get_matrices(
            self,
            ):
        '''
        Method to run after the FLSA_E class has been initialized to obtain the output matrices.

        Returns:
                  - Numpy array: prob_word_given_topic
                  - Numpy array: prob_topic_given_document
        '''
        sparse_document_term_matrix = self._create_sparse_local_term_weights(
            self.corpus,
            self._vocabulary_size,
            self._word_to_index,
            )

        self.word_embedding = self.get_word_embedding(
            data=self.corpus,
            min_count=self.min_count,
            vector_size=self.vector_size,
            window=self.window,
            workers=self.workers,
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
