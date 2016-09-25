#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""

"""

import logging
import numpy
import numbers

from gensim import utils
from gensim.models.ldamodel import dirichlet_expectation
from gensim.models.hdpmodel import log_normalize  # For efficient normalization of variational parameters.
from six.moves import xrange

# log(sum(exp(x))) that tries to avoid overflow
try:
    # try importing from here if older scipy is installed
    from scipy.maxentropy import logsumexp
except ImportError:
    # maxentropy has been removed in recent releases, logsumexp now in misc
    from scipy.misc import logsumexp

logger = logging.getLogger('gensim.models.atmodel')

def get_random_state(seed):
     """ Turn seed into a np.random.RandomState instance.
         Method originally from maciejkula/glove-python, and written by @joshloyal
     """
     if seed is None or seed is numpy.random:
         return numpy.random.mtrand._rand
     if isinstance(seed, (numbers.Integral, numpy.integer)):
         return numpy.random.RandomState(seed)
     if isinstance(seed, numpy.random.RandomState):
        return seed
     raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                      ' instance' % seed)


class AtVb:
    """
    Train the author-topic model using variational Bayes.
    """
    # TODO: inherit interfaces.TransformationABC.

    def __init__(self, corpus=None, num_topics=100, id2word=None,
                 author2doc=None, doc2author=None, threshold=0.001,
                 iterations=10, alpha=None, eta=None,
                 eval_every=10):

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
        self.author2doc = author2doc
        self.doc2author = doc2author
        self.num_docs = len(corpus)
        self.num_authors = len(doc2author)
        self.eval_every = eval_every

        self.random_state = get_random_state(random_state)

        if corpus is not None and author2doc is not None and doc2author is not None:
            self.inference(corpus, author2doc, doc2author)

    def inference(corpus=None, author2doc=None, doc2author=None):
        if corpus is None:
            corpus = self.corpus

        # Initial value of gamma and lambda.
        # NOTE: parameters of gamma distribution same as in `ldamodel`.
        var_gamma_init = self.random_state.gamma(100., 1. / 100.,
                                   (self.num_authors, self.num_topics))
        var_lambda_init = self.random_state.gamma(100., 1. / 100.,
                                    (self.num_topics, self.num_terms))

        var_gamma = numpy.zeros((self.num_authors, self.num_topics))
        for a in xrange(self.num_authors):
            var_gamma[a, :] = var_gamma_init

        var_lambda = numpy.zeros((self.num_authors, self.num_topics))
        for k in xrange(self.num_topics):
            var_lambda[k, :] = var_lambda_init

        # Initialize mu.
        # mu is 1/|A_d| if a is in A_d, zero otherwise.
        mu = numpy.zeros((self.num_docs, self.num_terms, self.num_authors))
        for d, doc in enumerate(corpus):
            ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
            for v in ids:
                authors_d = doc2author[d]  # List of author IDs for document d.
                for a in authors_d:
                    mu[d, v, a] = 1 / len(authors_d)

        # TODO: consider how to vectorize opterations as much as
        # possible.
        # TODO: check vector and matrix dimensions, and ensure that
        # things are multiplied along the correct dimensions.
        # TODO: rename variational parameters to "var_[parameter name]".

        Elogtheta = dirichlet_expectation(var_gamma)
        Elogbeta = dirichlet_expectation(var_lambda)
        expElogbeta = numpy.exp(Elogbeta)
        likelihood = eval_likelihood(docs=corpus, Elogtheta, Elogbeta)
        for iteration in xrange(self.iterations):
            # Update phi.
            for d, doc in enumerate(corpus):
                ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
                authors_d = doc2author[d]  # List of author IDs for document d.

                expElogbetad = expElogbeta[:, ids]

                for v in ids:
                    for k in xrange(self.num_topics):
                        # Average Elogtheta over authors a in document d.
                        avgElogtheta = 0.0
                        for a in authors_d:
                            avgElogtheta += var_mu[d, v, a] * Elogtheta[a, k]
                        expavgElogtheta = numpy.exp(avgElogtheta)

                        # Compute phi.
                        # TODO: avoid computing phi if possible.
                        var_phi[d, v, k] = expavgElogtheta * expElogbetad.T[k, v]  # FIXME: may have an alignment issue here.
                        # Normalize phi.
                        (log_var_phi, _) = log_normalize(var_phi[d, v, k])
                        var_phi[d, v, k] = numpy.exp(log_var_phi)

            # Update mu.
            for d, doc in enumerate(corpus):
                ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
                authors_d = doc2author[d]  # List of author IDs for document d.

                # Prior probability of observing author a in document d is one
                # over the number of authors in document d.
                author_prior_prob = 1.0 / len(authors_d)
                for v in ids:
                    for a in authors_d:
                        # Average Elogtheta over topics k.
                        avgElogtheta = 0.0
                        for k in xrange(self.num_topics):
                            avgElogtheta += var_phi[d, v, k] * Elogtheta[a, k]
                        expavgElogtheta = numpy.exp(avgElogtheta)

                        # Compute mu.
                        # TODO: avoid computing mu if possible.
                        var_mu[d, v, a] = author_prior_prob * avgexpElogtheta[a, k]  # FIXME: may have an alignment issue here.
                        # Normalize mu.
                        (log_var_mu, _) = log_normalize(var_mu[d, v, a])
                        var_mu[d, v, a] = numpy.exp(log_var_mu)

            # Update gamma.
            for a in xrange(self.num_authors):
                for k in xrange(self.num_topics):
                    docs_a = author2doc[a]
                    var_gamma[a, k] = 0.0
                    var_gamma[a, k] += self.alpha
                    for d in docs_a:
                        doc = corpus[d]
                        ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                        cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
                        for v in ids:
                            var_gamma[a, k] += cts[v] * var_mu[d, v, a] * var_phi[d, v, k]

            # Update Elogtheta, since gamma has been updated.
            Elogtheta = dirichlet_expectation(var_gamma)

            # Update lambda.
            for k in xrange(self.num_topics):
                for v in xrange(self.num_terms):
                    var_lambda[k, v] = 0.0
                    var_lambda[k, v] += self.eta
                    for d, doc in enumerate(corpus):
                        ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                        cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
                        for v in ids:
                            var_lambda += cts[v] * var_phi[d, v, k]

            # Update Elogbeta, since lambda has been updated.
            Elogbeta = dirichlet_expectation(var_lambda)
            expElogbeta = numpy.exp(Elogbeta)

            # Evaluate likelihood.
            if (iteration + 1) % self.eval_every == 0:
                prev_likelihood = likelihood
                likelihood = eval_likelihood(docs=corpus, Elogtheta, Elogbeta)
                if numpy.abs(likelihood - prev_likelihood) / prev_likelihood < self.threshold:
                    break
        # End of update loop (iterations).

        return var_gamma, var_lambda

    def eval_likelihood(doc_ids=None, Elogtheta, Elogbeta):
        """
        Compute the conditional liklihood of a set of documents,

            p(D | theta, beta, A).

        theta and beta are estimated by exponentiating the expectations of
        log theta and log beta.
        """

        # TODO: allow for evaluating test corpus. This will require inferring on unseen documents.


        if doc_ids is None:
            docs = corpus
        else:
            docs = [corpus[d] for d in doc_ids]

        likelihood = 0.0
        for d, doc in enumerate(docs):
            authors_d = self.doc2author[d]
            likelihood_d = 0.0
            for v in ids:
                for k in self.num_topics:
                    for a in authors_d:
                        likelihood_d += cnt[v] * numpy.exp(Elogtheta[a, k] + Elogbeta[k, v])
            author_prior_prob = 1.0 / len(authors_d)
            likelihood_d *= author_prior_prob
            likelihood += likelihood_d

            # TODO: can I do this?
            # bound += author_prior_prob * numpy.sum(cnt * sum(logsumexp(Elogtheta[authors_d, :] + Elogbeta[:, id])) for id, cnt in doc)







