# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:05:39 2022

@author: 20200016
"""
import itertools
from FuzzyTM import FLSA_W, FLSA
import gensim

class FlsaModel:
    """
    Class to initialize and train fuzzy topic models with methods similar
    to Gensim's LdaModel'

    Parameters
    ----------
            corpus :  either: list of list of str. or: list of list of tuples (int, int) (bow).
                The input corpus.

            num_topics: int
                The number of topics to be trained.

            algorithm: str ['flsa', 'flsa-w']
                The algorithm to train.

            id2word: gensim.corpora.dictionary.Dictionary
                Object to map id's to words
                (only used when the corpus is passed into the object as a bow).

            word_weighting: str ['normal','idf','probidf','entropy']
                Global term weighting mechanism.

            cluster_method: str ['fcm','gk','fst-pso']
                Fuzzy clustering method.

            svd_factors: int
                The number of singular values to use.
    """
    def __init__(
            self,
            corpus,
            num_topics,
            algorithm='flsa-w',
            id2word=None,
            word_weighting='normal',
            cluster_method='fcm',
            svd_factors=2,
            ):
        if algorithm not in [
                'flsa',
                'flsa-w',
                ]:
            raise ValueError('Please select a correct "algoritm"')

        self.corpus = self._set_corpus(corpus, id2word)
        self.num_topics = num_topics

        if algorithm == 'flsa':
            self.model = FLSA(
                input_file=self.corpus,
                num_topics=num_topics,
                word_weighting=word_weighting,
                cluster_method=cluster_method,
                svd_factors=svd_factors,
                )
        else:
            self.model = FLSA_W(
                input_file=self.corpus,
                num_topics=num_topics,
                word_weighting=word_weighting,
                cluster_method=cluster_method,
                svd_factors=svd_factors,
                )
        self.pwgt, self.ptgd = self.model.get_matrices()

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
            if not isinstance(id2word, gensim.corpora.dictionary.Dictionary):
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
            corpus : either: list of list of str (tokens). or: list of list of tuples (int, int).
                The input corpus.
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

    def show_topics(
            self,
            formatted=True,
            num_words=20,
            ):
        """
        Get a representation for selected topics.

        Parameters
        ----------
            formatted : bool
                Whether the topic representations should be formatted as strings.
                If False, they are returned as 2 tuples of (word, probability).
            num_words: int
                Number of words to be presented for each topic.
                These will be the most relevant words
                (assigned the highest probability for each topic).

        Returns
        -------
            A list of topics, each represented either as a string (when formatted == True)
            or word-probability pairs.
        """
        if not isinstance(formatted, bool):
            raise ValueError('Please choose a boolean for "formatted"')
        bool_map = {True: "words", False: "both"}

        return self.model.show_topics(
            representation=bool_map[formatted],
            num_words=num_words,
            )

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
