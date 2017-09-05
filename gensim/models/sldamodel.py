#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import logging
import numbers
import os
from random import sample

import numpy as np
import six
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from six.moves import xrange

from gensim import interfaces, utils, matutils
from gensim.matutils import dirichlet_expectation
from gensim.matutils import kullback_leibler, hellinger, jaccard_distance, jensen_shannon
from gensim.models import basemodel, CoherenceModel

# log(sum(exp(x))) that tries to avoid overflow
try:
    # try importing from here if older scipy is installed
    from scipy.maxentropy import logsumexp
except ImportError:
    # maxentropy has been removed in recent releases, logsumexp now in misc
    from scipy.misc import logsumexp


logger = logging.getLogger('gensim.models.sldamodel')

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

def get_lambda(self):
        return self.eta + self.sstats
    
def get_Elogbeta(self):
        return dirichlet_expectation(self.get_lambda())

class sLdaModel(interfaces.TransformationABC, basemodel.BaseTopicModel):
    
    def __init__(self, corpus=None, n_topics=100, alpha='symmetric', beta, mu, nu, nu2, 
                 sigma2, iterations = 100, report_iter=10, seed=None):
    """
    Supervised latent Dirichlet allocation, using collapsed Gibbs
    sampling.
    Args
        n_topics : int
            Number of topics
        alpha : array-like, shape = (n_topics,)
            Dirichlet distribution parameter for each document's topic
            distribution.
        beta : array-like, shape = (n_terms,)
            Dirichlet distribution parameter for each topic's term distribution.
        mu : float
            Mean of regression coefficients (eta).
        nu2 : float
            Variance of regression coefficients (eta).
        sigma2 : float
            Variance of response (y).
        n_iter : int, default=100
            Number of iterations of Gibbs sampler
        n_report_iter : int, default=10
            Number of iterations of Gibbs sampler between progress reports.
        seed : int, optional
            Seed for random number generator
    """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nu2 = nu2
        self.sigma2 = sigma2
        self.iterations = iterations
        self.report_iter = n_report_iter
        self.seed = seed
        
    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        """
        Args:
            num_topics (int): show results for first `num_topics` topics.
                Unlike LSA, there is no natural ordering between the topics in sLDA.
                The returned `num_topics <= self.num_topics` subset of all topics is
                therefore arbitrary and may change between two sLDA training runs.
            num_words (int): include top `num_words` with highest probabilities in topic.
            log (bool): If True, log output in addition to returning it.
            formatted (bool): If True, format topics as strings, otherwise return them as
                `(word, probability) 2-tuples.
        Returns:
            list: `num_words` most significant words for `num_topics` number of topics
            (10 words for top 10 topics, by default).
        """
        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics = self.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics = min(num_topics, self.num_topics)

            # add a little random jitter, to randomize results around the same alpha
            sort_alpha = self.alpha + 0.0001 * self.random_state.rand(len(self.alpha))

            sorted_topics = list(matutils.argsort(sort_alpha))
            chosen_topics = sorted_topics[:num_topics // 2] + sorted_topics[-num_topics // 2:]

        shown = []

        topic = self.state.get_lambda()
        for i in chosen_topics:
            topic_ = topic[i]
            topic_ = topic_ / topic_.sum()  # normalize to probability distribution
            bestn = matutils.argsort(topic_, num_words, reverse=True)
            topic_ = [(self.id2word[id], topic_[id]) for id in bestn]
            if formatted:
                topic_ = ' + '.join(['%.3f*"%s"' % (v, k) for k, v in topic_])

            shown.append((i, topic_))
            if log:
                logger.info("topic #%i (%.3f): %s", i, self.alpha[i], topic_)

        return shown

    def show_topic(self, topicid, topn=10):
        """
        Args:
            topn (int): Only return 2-tuples for the topn most probable words
                (ignore the rest).

        Returns:
            list: of `(word, probability)` 2-tuples for the most probable
            words in topic `topicid`.
        """
        return [(self.id2word[id], value) for id, value in self.get_topic_terms(topicid, topn)]

    def get_topics(self):
        """
        Returns:
            np.ndarray: `num_topics` x `vocabulary_size` array of floats which represents
            the term topic matrix learned during inference.
        """
        topics = self.state.get_lambda()
        return topics / topics.sum(axis=1)[:, None]
    
    def save(self, fname, ignore=['state', 'dispatcher'], separately=None, *args, **kwargs):
        """
        Save the model to file.
        """
        if self.state is not None:
            self.state.save(utils.smart_extension(fname, '.state'), *args, **kwargs)
        if 'id2word' not in ignore:
            utils.pickle(self.id2word, utils.smart_extension(fname, '.id2word'))

        if ignore is not None and ignore:
            if isinstance(ignore, six.string_types):
                ignore = [ignore]
            ignore = [e for e in ignore if e]
            ignore = list(set(['state', 'dispatcher', 'id2word']) | set(ignore))
        else:
            ignore = ['state', 'dispatcher', 'id2word']

        separately_explicit = ['expElogbeta', 'sstats']
        if (isinstance(self.alpha, six.string_types) and self.alpha == 'auto') or (isinstance(self.alpha, np.ndarray) and len(self.alpha.shape) != 1):
            separately_explicit.append('alpha')
        if (isinstance(self.eta, six.string_types) and self.mu == 'auto') or (isinstance(self.eta, np.ndarray) and len(self.eta.shape) != 1):
            separately_explicit.append('eta')
        if separately:
            if isinstance(separately, six.string_types):
                separately = [separately]
            separately = [e for e in separately if e]  # make sure None and '' are not in the list
            separately = list(set(separately_explicit) | set(separately))
        else:
            separately = separately_explicit
        super(sLdaModel, self).save(fname, ignore=ignore, separately=separately, *args, **kwargs)
