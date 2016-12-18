#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
"""
scikit learn interface for gensim for easy use of gensim with scikit-learn
"""
from gensim import models


class LdaModel(models.LdaModel,object):
    """
    Base LDA module
    """
    def __init__(self, corpus=None, num_topics=100, id2word=None,
                 distributed=False, chunksize=2000, passes=1, update_every=1,
                 alpha='symmetric', eta=None, decay=0.5, offset=1.0,
                 eval_every=10, iterations=50, gamma_threshold=0.001,
                 minimum_probability=0.01, random_state=None):
        """
        base LDA code.Use corpus=corpus when fit is not involved. Otherwise leave corpus as None and use fit to give
        the parameter as corpus.
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
        if self.corpus is not None:
            models.LdaModel.__init__(
                                     self, corpus=self.corpus, num_topics=self.num_topics, id2word=self.id2word,
                                     distributed=self.distributed, chunksize=self.chunksize, passes=self.passes,
                                     update_every=self.update_every,alpha=self.alpha, eta=self.eta, decay=self.decay,
                                     offset=self.offset,eval_every=self.eval_every, iterations=self.iterations,
                                     gamma_threshold=self.gamma_threshold,minimum_probability=self.minimum_probability,
                                     random_state=self.random_state)

    def get_params(self, deep=True):
        if deep:
            return {"corpus":self.corpus,"num_topics":self.num_topics,"id2word":self.id2word,
                    "distributed":self.distributed,"chunksize":self.chunksize,"passes":self.passes,
                    "update_every":self.update_every,"alpha":self.alpha," eta":self.eta," decay":self.decay,
                    "offset":self.offset,"eval_every":self.eval_every," iterations":self.iterations,
                    "gamma_threshold":self.gamma_threshold,"minimum_probability":self.minimum_probability,
                    "random_state":self.random_state}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def fit(self, X):
        """
        call gensim.model.LdaModel from this
        calling :
        >>>gensim.models.LdaModel(corpus=corpus,num_topics=num_topics,id2word=id2word,passes=passes,update_every=update_every,alpha=alpha,iterations=iterations,eta=eta,random_state=random_state)
        """
        self.corpus=X
        models.LdaModel.__init__(
                                 self, corpus=X, num_topics=self.num_topics, id2word=self.id2word,
                                 distributed=self.distributed, chunksize=self.chunksize, passes=self.passes,
                                 update_every=self.update_every,alpha=self.alpha, eta=self.eta, decay=self.decay,
                                 offset=self.offset,eval_every=self.eval_every, iterations=self.iterations,
                                 gamma_threshold=self.gamma_threshold,minimum_probability=self.minimum_probability,
                                 random_state=self.random_state)
        return self

    def transform(self, bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False):
        """
        takes as an input a new document (bow) and
        Return topic distribution for the given document bow, as a list of (topic_id, topic_probability) 2-tuples.
        """
        return self.get_document_topics(
                                        bow, minimum_probability=minimum_probability,
                                        minimum_phi_value=minimum_phi_value, per_word_topics=per_word_topics)

    def partial_fit(self, X):
        """
        train model over X
        """
        self.update(corpus=X)
