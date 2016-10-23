#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
"""
scikit learn interface for gensim for easy use of gensim with scikit-learn
"""
import numpy as np
import gensim.models.ldamodel


class BaseClass(object):
    def __init__(self):
        """init
        base class to be always inherited
        to be used in the future
        """
    def run(self):   # to test
        return np.array([0, 0, 0])


class LdaModel(object):
    """
    Base LDA module
    """
    def __init__(self, n_topics=5, n_iter=2000, alpha=0.1, eta=0.01, random_state=None,
                 refresh=10, lda_model=None, id2word=None, passes=20, ex=None):
        """
        base LDA code . Uses mapper function
        n_topics : num_topics
        .fit  : init call // corpus not used
        //none : id2word
        n_iter : passes // assumed
        random_state : random_state
        alpha : alpha
        eta : eta
        refresh : update_every
        id2word: id2word
        """
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.alpha = alpha
        self.eta = eta
        self.random_state = random_state
        self.refresh = refresh
        self.id2word = id2word
        self.passes = passes
        # use lda_model variable as object
        self.lda_model = lda_model
        # perform appropriate checks
        if alpha <= 0:
            raise ValueError("alpha value must be larger than zero")
        if eta <= 0:
            raise ValueError("eta value must be larger than zero")

    def get_params(self, deep=True):
        if deep:
            return {"alpha": self.alpha, "n_iter": self.n_iter, "eta": self.eta, "random_state": self.random_state,
                    "lda_model": self.lda_model, "id2word": self.id2word, "passes": self.passes}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def fit(self, X, y=None):
        """
        call gensim.model.LdaModel from this
        // todo: convert fit and relevant,corpus still requires gensim preprocessing
        calling :
        >>>gensim.models.LdaModel(corpus=corpus,num_topics=n_topics,id2word=None,passes=n_iter,update_every=refresh,alpha=alpha,iterations=n_iter,eta=eta,random_state=random_state)
        """
        if X is None:
            raise AttributeError("Corpus defined as none")
        self.lda_model = gensim.models.LdaModel(
                         corpus=X, num_topics=self.n_topics, id2word=self.id2word, passes=self.passes,
                         update_every=self.refresh, alpha=self.alpha, iterations=self.n_iter,
                         eta=self.eta, random_state=self.random_state)
        return self.lda_model

    def print_topics(self, n_topics, num_words, log):
        """
        print all the topics
        using the object lda_model
        """
        return self.lda_model.show_topics(num_topics=n_topics, num_words=num_words, log=log)

    def transform(self, bow, minimum_probability, minimum_phi_value, per_word_topics):
        """
        takes as an input a new document (bow) and
        Return topic distribution for the given document bow, as a list of (topic_id, topic_probability) 2-tuples.
        """
        return self.lda_model.get_document_topics(bow, minimum_probability=minimum_probability,
                                                  minimum_phi_value=minimum_phi_value, per_word_topics=per_word_topics)

    def get_term_topics(self, wordid, minimum_probability):
        """
        returns the most likely topic associated with a particular word
        use wordid or simply pass the word itself
        """
        return self.lda_model.get_term_topics(wordid, minimum_probability=minimum_probability)

    def get_topic_terms(self, topicid, topn):
        """
        return a tuple of (wordid,probability) for given topic
        topn can be used to restrict
        """
        return self.lda_model.get_topic_terms(topicid=topicid, topn=topn)

    def partial_fit(self, X, y=None):
        """
        train model over X
        """
        self.lda_model.update(corpus=X)

