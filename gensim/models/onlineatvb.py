#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Author-topic model.
"""

import pdb
from pdb import set_trace as st

import logging
import numpy
import numbers

from gensim import utils
from gensim.models.ldamodel import dirichlet_expectation, get_random_state
from gensim.models.hdpmodel import log_normalize  # For efficient normalization of variational parameters.
from six.moves import xrange

# log(sum(exp(x))) that tries to avoid overflow
try:
    # try importing from here if older scipy is installed
    from scipy.maxentropy import logsumexp
except ImportError:
    # maxentropy has been removed in recent releases, logsumexp now in misc
    from scipy.misc import logsumexp

logger = logging.getLogger(__name__)


class OnlineAtVb:
    """
    Train the author-topic model using online variational Bayes.
    """
    # TODO: inherit interfaces.TransformationABC.

    def __init__(self, corpus=None, num_topics=100, id2word=None,
            author2doc=None, doc2author=None, threshold=0.001,
            iterations=10, alpha=None, eta=None, decay=0.5, offset=1.0,
            eval_every=1, random_state=None):

        # TODO: require only author2doc OR doc2author, and construct the missing one automatically.

        if alpha is None:
            alpha = 50 / num_topics
        if eta is None:
            eta = 0.01

        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0

        if self.num_terms == 0:
            raise ValueError("cannot compute LDA over an empty collection (no terms)")

        self.corpus = corpus
        self.iterations = iterations
        self.num_topics = num_topics
        self.threshold = threshold
        self.alpha = alpha
        self.eta = eta
        self.decay = decay
        self.offset = offset
        self.author2doc = author2doc
        self.doc2author = doc2author
        self.num_docs = len(corpus)
        self.num_authors = len(author2doc)
        self.eval_every = eval_every
        self.random_state = random_state

        # TODO: find a way out of this nonsense.
        self.authorid2idx = dict(zip(list(author2doc.keys()), xrange(self.num_authors)))
        self.authoridx2id = dict(zip(xrange(self.num_authors), list(author2doc.keys())))

        self.random_state = get_random_state(random_state)

        if corpus is not None and author2doc is not None and doc2author is not None:
            self.inference(corpus, author2doc, doc2author)

    def rho(self, iteration):
        return pow(self.offset + iteration, -self.decay)

    def inference(self, corpus=None, author2doc=None, doc2author=None):
        if corpus is None:
            corpus = self.corpus.copy()

        # Initial values of gamma and lambda.
        # NOTE: parameters of gamma distribution same as in `ldamodel`.
        init_gamma = self.random_state.gamma(100., 1. / 100.,
                (self.num_authors, self.num_topics))
        init_lambda = self.random_state.gamma(100., 1. / 100.,
                (self.num_topics, self.num_terms))

        converged = 0

        # TODO: consider making phi and mu sparse.
        var_phi = numpy.zeros((self.num_terms, self.num_topics))
        var_mu = numpy.zeros((self.num_terms, self.num_authors))

        var_gamma = init_gamma.copy()
        var_lambda = init_lambda.copy()
        tilde_gamma = init_gamma.copy()
        tilde_lambda = init_lambda.copy()

        # Initialize dirichlet expectations.
        Elogtheta = dirichlet_expectation(var_gamma)
        Elogbeta = dirichlet_expectation(var_lambda)
        expElogbeta = numpy.exp(Elogbeta)
        st()
        for d, doc in enumerate(corpus):
            ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
            cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
            authors_d = doc2author[d]  # List of author IDs for document d.

            # Initialize mu.
            # mu is 1/|A_d| if a is in A_d, zero otherwise.
            # NOTE: I could do random initialization instead.
            for v in ids:
                for aid in authors_d:
                    a = self.authorid2idx[aid]
                    var_mu[v, a] = 1 / len(authors_d)

            for iteration in xrange(self.iterations):
                #logger.info('iteration %i', iteration)

                lastgamma = tilde_gamma.copy()
                lastlambda = tilde_lambda.copy()

                # Update phi.
                for v in ids:
                    for k in xrange(self.num_topics):
                        # Average Elogtheta over authors a in document d.
                        avgElogtheta = 0.0
                        for ad in authors_d:
                            a = self.authorid2idx[aid]
                            avgElogtheta += var_mu[v, a] * Elogtheta[a, k]
                        expavgElogtheta = numpy.exp(avgElogtheta)

                        # Compute phi.
                        # TODO: avoid computing phi if possible.
                        var_phi[v, k] = expavgElogtheta * expElogbeta[k, v]  # FIXME: may have an alignment issue here.

                    # Normalize phi over k.
                    (log_var_phi_v, _) = log_normalize(var_phi[v, :])  # NOTE: it might be possible to do this out of the v loop.
                    var_phi[v, :] = numpy.exp(log_var_phi_v)

                # Update mu.
                for v in ids:
                    # Prior probability of observing author a in document d is one
                    # over the number of authors in document d.
                    author_prior_prob = 1.0 / len(authors_d)
                    for aid in authors_d:
                        a = self.authorid2idx[aid]
                        # Average Elogtheta over topics k.
                        avgElogtheta = 0.0
                        for k in xrange(self.num_topics):
                            avgElogtheta += var_phi[v, k] * Elogtheta[a, k]
                        expavgElogtheta = numpy.exp(avgElogtheta)

                        # Compute mu over a.
                        # TODO: avoid computing mu if possible.
                        var_mu[v, a] = author_prior_prob * expavgElogtheta

                    # Normalize mu.
                    (log_var_mu_v, _) = log_normalize(var_mu[v, :])
                    var_mu[v, :] = numpy.exp(log_var_mu_v)


                # Update gamma.
                for a in xrange(self.num_authors):
                    for k in xrange(self.num_topics):
                        tilde_gamma[a, k] = 0.0
                        for vi, v in enumerate(ids):
                            tilde_gamma[a, k] += cts[vi] * var_mu[v, a] * var_phi[v, k]
                        aid = self.authoridx2id[a]
                        tilde_gamma[a, k] *= len(author2doc[aid])
                        tilde_gamma[a, k] += self.alpha

                # Update lambda.
                #tilde_lambda = self.eta + self.num_docs * cts * var_phi[ids, :].T
                for k in xrange(self.num_topics):
                    for vi, v in enumerate(ids):
                        tilde_lambda[k, v] = self.eta + self.num_docs * cts[vi] * var_phi[v, k]

                # Check for convergence.
                # Criterion is mean change in "local" gamma and lambda.
                if iteration > 0:
                    meanchange_gamma = numpy.mean(abs(tilde_gamma - lastgamma))
                    meanchange_lambda = numpy.mean(abs(tilde_lambda - lastlambda))
                    #logger.info('Mean change in gamma: %.3e', meanchange_gamma)
                    #logger.info('Mean change in lambda: %.3e', meanchange_lambda)
                    if meanchange_gamma < self.threshold and meanchange_lambda < self.threshold:
                        converged += 1
                        break
            # End of iterations loop.

            # Update gamma and lambda.
            # Interpolation between document d's "local" gamma (tilde_gamma),
            # and "global" gamma (var_gamma). Same goes for lambda.
            rhot = self.rho(d)
            var_gamma = (1 - rhot) * var_gamma + rhot * tilde_gamma
            # Note that we only changed the elements in lambda corresponding to 
            # the words in document d, hence the [:, ids] indexing.
            var_lambda[:, ids] = (1 - rhot) * var_lambda[:, ids] + rhot * tilde_lambda[:, ids]

            # Update Elogtheta and Elogbeta, since gamma and lambda have been updated.
            Elogtheta = dirichlet_expectation(var_gamma)
            Elogbeta = dirichlet_expectation(var_lambda)
            expElogbeta = numpy.exp(Elogbeta)

            word_prob = self.eval_word_prob(Elogtheta, Elogbeta)
            logger.info('Word probabilities: %.3e', word_prob)
            logger.info('Converged documents: %d', converged)
        # End of corpus loop.

        return var_gamma, var_lambda

    def eval_word_prob(self, Elogtheta, Elogbeta, doc_ids=None):
        """
        Compute the conditional liklihood of a set of documents,

            p(D | theta, beta, A).

        theta and beta are estimated by exponentiating the expectations of
        log theta and log beta.
        """

        # TODO: allow for evaluating test corpus. This will require inferring on unseen documents.


        if doc_ids is None:
            docs = self.corpus
        else:
            docs = [self.corpus[d] for d in doc_ids]

        word_prob = 0.0
        for d, doc in enumerate(docs):
            ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
            cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
            authors_d = self.doc2author[d]
            word_prob_d = 0.0
            for vi, v in enumerate(ids):
                for k in xrange(self.num_topics):
                    for aid in authors_d:
                        a = self.authorid2idx[aid]
                        word_prob_d += cts[vi] * numpy.exp(Elogtheta[a, k] + Elogbeta[k, v])
            author_prior_prob = 1.0 / len(authors_d)
            word_prob_d *= author_prior_prob
            word_prob += word_prob_d

            # TODO: can I do this?
            # bound += author_prior_prob * numpy.sum(cnt * sum(logsumexp(Elogtheta[authors_d, :] + Elogbeta[:, id])) for id, cnt in doc)

        return word_prob







