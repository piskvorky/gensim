#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
"""
scikit learn interface for gensim for easy use of gensim with scikit-learn
follows on scikit learn API conventions
"""
from gensim import models
from gensim import matutils
from scipy.sparse.csr import csr_matrix


class SklearnWrapperLdaModel(models.LdaModel,object):
    """
    Base LDA module
    """
    def __init__(self, corpus=None, num_topics=100, id2word=None,
                 distributed=False, chunksize=2000, passes=1, update_every=1,
                 alpha='symmetric', eta=None, decay=0.5, offset=1.0,
                 eval_every=10, iterations=50, gamma_threshold=0.001,
                 minimum_probability=0.01, random_state=None):
        """
        sklearn wrapper for LDA model. derived class for gensim.model.LdaModel
        """
        self.corpus = corpus
        self.num_topics = num_topics
        self.id2word = id2word
        self.distributed = distributed
        self.chunksize = chunksize
        self.passes = passes
        self.update_every = update_every
        self.alpha = alpha
        self.eta = eta
        self.decay = decay
        self.offset = offset
        self.eval_every = eval_every
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold
        self.minimum_probability = minimum_probability
        self.random_state = random_state
        """
        if no fit function is used , then corpus is given in init
        """
        if self.corpus:
            models.LdaModel.__init__(
                                     self, corpus=self.corpus, num_topics=self.num_topics, id2word=self.id2word,
                                     distributed=self.distributed, chunksize=self.chunksize, passes=self.passes,
                                     update_every=self.update_every,alpha=self.alpha, eta=self.eta, decay=self.decay,
                                     offset=self.offset,eval_every=self.eval_every, iterations=self.iterations,
                                     gamma_threshold=self.gamma_threshold,minimum_probability=self.minimum_probability,
                                     random_state=self.random_state)

    def get_params(self, deep=True):
        """
        returns all parameters as dictionary.
        Warnings: Must for sklearn API.Do not Remove.
        """
        if deep:
            return {"corpus":self.corpus,"num_topics":self.num_topics,"id2word":self.id2word,
                    "distributed":self.distributed,"chunksize":self.chunksize,"passes":self.passes,
                    "update_every":self.update_every,"alpha":self.alpha," eta":self.eta," decay":self.decay,
                    "offset":self.offset,"eval_every":self.eval_every," iterations":self.iterations,
                    "gamma_threshold":self.gamma_threshold,"minimum_probability":self.minimum_probability,
                    "random_state":self.random_state}

    def set_params(self, **parameters):
        """
        set all parameters.
        Warnings: Must for sklearn API.Do not Remove.
        """
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def fit(self, X):
        """
        For fitting corpus into the class object.
        calls gensim.model.LdaModel:
        >>>gensim.models.LdaModel(corpus=corpus,num_topics=num_topics,id2word=id2word,passes=passes,update_every=update_every,alpha=alpha,iterations=iterations,eta=eta,random_state=random_state)
        Warnings: Must for sklearn API. Do not Remove.
        """
        if isinstance(X, csr_matrix):
            self.corpus = matutils.Sparse2Corpus(X)
        else:
            self.corpus = X

        models.LdaModel.__init__(
                                 self, corpus=self.corpus, num_topics=self.num_topics, id2word=self.id2word,
                                 distributed=self.distributed, chunksize=self.chunksize, passes=self.passes,
                                 update_every=self.update_every, alpha=self.alpha, eta=self.eta, decay=self.decay,
                                 offset=self.offset, eval_every=self.eval_every, iterations=self.iterations,
                                 gamma_threshold=self.gamma_threshold, minimum_probability=self.minimum_probability,
                                 random_state=self.random_state)
        return self

    def transform(self, bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False):
        """
        takes as an input a new document (bow) and
        Return topic distribution for the given document bow, as a list of (topic_id, topic_probability) 2-tuples.
        Warnings: Must for sklearn API. Do not Remove.
        """
        return self.get_document_topics(
                                        bow, minimum_probability=minimum_probability,
                                        minimum_phi_value=minimum_phi_value, per_word_topics=per_word_topics)

    def partial_fit(self, X):
        """
        train model over X.
        """
        if isinstance(X, csr_matrix):
            X = matutils.Sparse2Corpus(X)

        self.update(corpus=X)
