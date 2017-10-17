#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import logging
import numpy as np
from six import string_types
from six.moves import xrange
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from scipy.misc import logsumexp

from gensim import interfaces, utils, matutils

def sampling_from_dist(prob):
    """ 
    Sample index from a list of unnormalised probability distribution
    same as np.random.multinomial(1, prob/np.sum(prob)).argmax()
    
    Parameters
    ----------
    prob: ndarray
        array of unnormalised probability distribution
    
    Returns
    -------
    new_topic: return a sampled index
    """
    
    thr = prob.sum() * np.random.rand()
    new_topic = 0
    tmp = prob[new_topic]
    while tmp < thr:
        new_topic += 1
        tmp += prob[new_topic]
    return new_topic

def get_top_words(topic_word_matrix, vocab, topic, n_words=20):
    if not isinstance(vocab, np.ndarray):
        vocab = np.array(vocab)
    top_words = vocab[topic_word_matrix[topic].argsort()[::-1][:n_words]]
    return top_words


class SLdaModel(interfaces.TransformationABC, basemodel.BaseTopicModel):

    def __init__(self, corpus=None, n_topics=100, alpha='symmetric', beta, mu, nu, nu2,
                 sigma2, iterations=100, report_iter=10, seed=None):
        """
        Supervised latent Dirichlet allocation, using collapsed Gibbs
        sampling.
        Args:
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
        super(SLdaModel, self).__init__(n_doc=n_doc, n_voca=n_voca, n_topic=n_topic, alpha=alpha, beta=beta,
                                                 **kwargs)
        self.eta = np.random.normal(scale=5, size=self.n_topic)
        self.sigma = sigma
        
    
    def random_init(self, docs):
        """
        Random initialization of topics
        """
        for di in xrange(len(docs)):
            doc = docs[di]
            topics = np.random.randint(self.n_topic, size=len(doc))
            self.topic_assignment.append(topics)

            for wi in xrange(len(doc)):
                topic = topics[wi]
                word = doc[wi]
                self.TW[topic, word] += 1
                self.sum_T[topic] += 1
                self.DT[di, topic] += 1
                
    def log_likelihood(self, docs, responses):
        """
        Calculate log-likelihood function
        """
        l1 = 0

        l1 += len(docs) * gammaln(self.alpha * self.n_topic)
        l1 -= len(docs) * self.n_topic * gammaln(self.alpha)
        l1 += self.n_topic * gammaln(self.beta * self.n_voca)
        l1 -= self.n_topic * self.n_voca * gammaln(self.beta)

        for di in xrange(self.n_doc):
            l1 += gammaln(self.DT[di, :]).sum() - gammaln(self.DT[di, :].sum())
            z_bar = self.DT[di] / np.sum(self.DT[di])
            mean = np.dot(z_bar, self.eta)
            l1 += norm.logpdf(responses[di], mean, np.sqrt(self.sigma))
        for ki in xrange(self.n_topic):
            l1 += gammaln(self.TW[ki, :]).sum() - gammaln(self.TW[ki, :].sum())

        return l1
    
    def sample_heldout_doc(self, max_iter, heldout_docs):
        """
        Calculate Topic sum on heldout docs. 
        """
        h_doc_topics = list()
        h_doc_topic_sum = np.zeros([len(heldout_docs), self.n_topic]) + self.alpha

        # random init
        for di in xrange(len(heldout_docs)):
            doc = heldout_docs[di]
            topics = np.random.randint(self.n_topic, size=len(doc))
            h_doc_topics.append(topics)

            for wi in xrange(len(doc)):
                topic = topics[wi]
                h_doc_topic_sum[di, topic] += 1

        for iter in xrange(max_iter):
            for di in xrange(len(heldout_docs)):
                doc = heldout_docs[di]
                for wi in xrange(len(doc)):
                    word = doc[wi]
                    old_topic = h_doc_topics[di][wi]

                    h_doc_topic_sum[di, old_topic] -= 1

                    # update
                    prob = (self.TW[:, word] / self.sum_T) * (self.DT[di, :])

                    new_topic = sampling_from_dist(prob)

                    h_doc_topics[di][wi] = new_topic
                    h_doc_topic_sum[di, new_topic] += 1

        return h_doc_topic_sum
    
    
    def fit(self, docs, responses, max_iter=100):
        """ 
        Fit sLDA model using Stochastic Expectation Maximisation.
        """
        self.random_init(docs)
        for iteration in xrange(max_iter):

            for di in xrange(len(docs)):
                doc = docs[di]
                for wi in xrange(len(doc)):
                    word = doc[wi]
                    old_topic = self.topic_assignment[di][wi]

                    self.TW[old_topic, word] -= 1
                    self.sum_T[old_topic] -= 1
                    self.DT[di, old_topic] -= 1
                    
                    # Calculate z-bar 
                    z_bar = np.zeros([self.n_topic, self.n_topic]) + self.DT[di, :] + np.identity(self.n_topic)
                    z_bar /= self.DT[di, :].sum() + 1

                    # update
                    prob = (self.TW[:, word]) / (self.sum_T) * (self.DT[di, :]) * np.exp(
                        np.negative((responses[di] - np.dot(z_bar, self.eta)) ** 2) / 2 / self.sigma)

                    new_topic = sampling_from_dist(prob)

                    self.topic_assignment[di][wi] = new_topic
                    self.TW[new_topic, word] += 1
                    self.sum_T[new_topic] += 1
                    self.DT[di, new_topic] += 1

            # estimate parameters
            z_bar = self.DT / self.DT.sum(1)[:, np.newaxis]  # DxK
            self.eta = solve(np.dot(z_bar.T, z_bar), np.dot(z_bar.T, responses))

            # compute mean absolute error
            mae = np.mean(np.abs(responses - np.dot(z_bar, self.eta)))
            if self.verbose:
                logger.info('[ITER] %d,\tMAE:%.2f,\tlog_likelihood:%.2f', iteration, mae,
                            self.log_likelihood(docs, responses))
