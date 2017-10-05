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
from gensim.matutils import dirichlet_expectation


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
            chosen_topics = xrange(num_topics)
        else:
            num_topics = min(num_topics, self.num_topics)

            # add a little random jitter, to randomize results around the same
            # alpha
            sort_alpha = self.alpha + 0.0001 * \
                self.random_state.rand(len(self.alpha))

            sorted_topics = list(matutils.argsort(sort_alpha))
            chosen_topics = sorted_topics[:num_topics //
                                          2] + sorted_topics[-num_topics // 2:]

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
        return [(self.id2word[id], value)
                for id, value in self.get_topic_terms(topicid, topn)]

    def do_estep(self, chunk, state=None):

        if state is None:
            state = self.state
        gamma, sstats = self.inference(chunk, collect_sstats=True)
        state.sstats += sstats
        # avoids calling len(chunk) on a generator
        state.numdocs += gamma.shape[0]
        return gamma

    def do_mstep(self, rho, other, extra_pass=False):
        diff = np.log(self.expElogbeta)
        self.state.blend(rho, other)
        diff -= self.state.get_Elogbeta()
        self.sync_state()

        self.print_topics(5)
        logger.info("topic diff=%f, rho=%f", np.mean(np.abs(diff)), rho)

        if self.optimize_eta:
            self.update_eta(self.state.get_lambda(), rho)

        if not extra_pass:
            self.num_updates += other.numdocs

    def accuracy(self, goldlabel):
        right = 0
        for d in xrange(0, self._D):
            if (self._predictions[d] == goldlabel[d]):
                right = right + 1
        accuracy = float(right) / float(self._D)
        return accuracy

    def save_parameters(self):
        np.savetxt("lambda-%d.txt" % self._iterations, self._lambda)
        np.savetxt("mu-%d.txt" % self._iterations, self._mu)

    def calculate_mu(self, phi, expmu, cts, label):
        gra_mu = n.zeros(expmu.shape)
        nphi = (phi.T * cts).T
        avephi = n.average(nphi, axis=0)
        gra_mu[label, :] = avephi
        N = float(n.sum(cts))
        sf_aux = np.dot(expmu, phi.T)
        sf_aux_power = np.power(sf_aux, cts)

        sf_aux_prod = np.prod(sf_aux_power, axis=1) + 1e-100
        kappa_1 = 1.0 / np.sum(sf_aux_prod)

        sf_pra = np.zeros((self._C, self._K))

        temp = (sf_aux_prod[:, np.newaxis] / sf_aux)
        for c in xrange(0, self._C):
            temp1 = np.outer(temp[c, :], (1.0 / N) * expmu[c, :])
            temp1 = temp1 * nphi
            sf_pra[c, :] = np.sum(temp1, axis=0)
