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

from gensim import utils, matutils
from gensim.models.ldamodel import dirichlet_expectation, get_random_state
from gensim.models import LdaModel
from gensim.models.hdpmodel import log_normalize  # For efficient normalization of variational parameters.
from six.moves import xrange

from pprint import pprint

# log(sum(exp(x))) that tries to avoid overflow
try:
    # try importing from here if older scipy is installed
    from scipy.maxentropy import logsumexp
except ImportError:
    # maxentropy has been removed in recent releases, logsumexp now in misc
    from scipy.misc import logsumexp

logger = logging.getLogger('gensim.models.atmodel')


class AtVb(LdaModel):
    """
    Train the author-topic model using variational Bayes.
    """
    # TODO: inherit interfaces.TransformationABC. Probably not necessary if I'm inheriting LdaModel.

    def __init__(self, corpus=None, num_topics=100, id2word=None,
            author2doc=None, doc2author=None, threshold=0.001,
            iterations=10, alpha=None, eta=None,
            eval_every=1, random_state=None):

        # TODO: require only author2doc OR doc2author, and construct the missing one automatically.

        if alpha is None:
            alpha = 1.0 / num_topics
        if eta is None:
            eta = 1.0 / num_topics

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
        
        logger.info('Vocabulary consists of %d words.', self.num_terms)

        self.corpus = corpus
        self.iterations = iterations
        self.num_topics = num_topics
        self.threshold = threshold
        self.alpha = alpha
        self.eta = eta
        self.author2doc = author2doc
        self.doc2author = doc2author
        self.num_docs = len(corpus)
        self.num_authors = len(author2doc)
        self.eval_every = eval_every
        self.random_state = random_state

        logger.info('Number of authors: %d.', self.num_authors)

        # TODO: find a way out of this nonsense.
        self.authorid2idx = dict(zip(list(author2doc.keys()), xrange(self.num_authors)))
        self.authoridx2id = dict(zip(xrange(self.num_authors), list(author2doc.keys())))

        self.random_state = get_random_state(random_state)

        if corpus is not None:
            self.inference(corpus, author2doc, doc2author)

    def inference(self, corpus=None, author2doc=None, doc2author=None):
        if corpus is None:
            corpus = self.corpus

        logger.info('Starting inference. Training on %d documents.', len(corpus))

        # Initial value of gamma and lambda.
        # NOTE: parameters of gamma distribution same as in `ldamodel`.
        var_gamma = self.random_state.gamma(100., 1. / 100.,
                (self.num_authors, self.num_topics))
        var_lambda = self.random_state.gamma(100., 1. / 100.,
                (self.num_topics, self.num_terms))

        # Initialize mu.
        # mu is 1/|A_d| if a is in A_d, zero otherwise.
        # var_mu is essentially a (self.num_docs, self.num_terms, self.num_authors) sparse matrix,
        # which we represent using a dictionary.
        # TODO: consider initializing mu randomly.
        var_mu = dict()
        for d, doc in enumerate(corpus):
            ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
            for v in ids:
                authors_d = doc2author[d]  # List of author IDs for document d.
                for aid in authors_d:
                    a = self.authorid2idx[aid]
                    # Draw mu from gamma distribution.
                    # var_mu[(d, v, a)] = self.random_state.gamma(100., 1. / 100., (1,))[0]
                    var_mu[(d, v, a)] = 1 / len(authors_d)
                # Normalize mu.
                # mu_sum = 0.0
                # for aid_prime in authors_d:
                #     a_prime = self.authorid2idx[aid]
                #     mu_sum += var_mu[(d, v, a)]

                # for aid_prime in authors_d:
                #     a_prime = self.authorid2idx[aid]
                #     var_mu[(d, v, a)] *= 1 / mu_sum

        var_phi = numpy.zeros((self.num_docs, self.num_terms, self.num_topics))

        # TODO: consider how to vectorize opterations as much as
        # possible.
        # TODO: check vector and matrix dimensions, and ensure that
        # things are multiplied along the correct dimensions.

        Elogtheta = dirichlet_expectation(var_gamma)
        Elogbeta = dirichlet_expectation(var_lambda)
        expElogbeta = numpy.exp(Elogbeta)
        likelihood = self.eval_likelihood(Elogtheta, Elogbeta)
        logger.info('Likelihood: %.3e', likelihood)
        for iteration in xrange(self.iterations):
            # Update phi.
            for d, doc in enumerate(corpus):
                ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                authors_d = doc2author[d]  # List of author IDs for document d.

                # Update phi.
                for v in ids:
                    for k in xrange(self.num_topics):
                        # Average Elogtheta over authors a in document d.
                        avgElogtheta = 0.0
                        for aid in authors_d:
                            a = self.authorid2idx[aid]
                            avgElogtheta += var_mu[(d, v, a)] * Elogtheta[a, k]
                        expavgElogtheta = numpy.exp(avgElogtheta)

                        # Compute phi.
                        # TODO: avoid computing phi if possible.
                        var_phi[d, v, k] = expavgElogtheta * expElogbeta[k, v]
                    # Normalize phi.
                    #(log_var_phi_dv, _) = log_normalize(var_phi[d, v, :])
                    (log_var_phi_dv, _) = log_normalize(numpy.log(var_phi[d, v, :]))
                    var_phi[d, v, :] = numpy.exp(log_var_phi_dv)

            # Update mu.
            for d, doc in enumerate(corpus):
                ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                authors_d = doc2author[d]  # List of author IDs for document d.
                # author_prior_prob = 1. / len(authors_d)
                for v in ids:
                    mu_sum = 0.0
                    for aid in authors_d:
                        a = self.authorid2idx[aid]
                        # Average Elogtheta over topics k.
                        avgElogtheta = 0.0
                        for k in xrange(self.num_topics):
                            avgElogtheta += var_phi[d, v, k] * Elogtheta[a, k]
                        expavgElogtheta = numpy.exp(avgElogtheta)

                        # Compute mu.
                        # TODO: avoid computing mu if possible.
                        # var_mu[(d, v, a)] = author_prior_prob * expavgElogtheta
                        var_mu[(d, v, a)] = expavgElogtheta
                        mu_sum += var_mu[(d, v, a)]

                    mu_norm_const = 1.0 / mu_sum
                    for aid in authors_d:
                        a = self.authorid2idx[aid]
                        var_mu[(d, v, a)] *= mu_norm_const

            # Update gamma.
            for a in xrange(self.num_authors):
                for k in xrange(self.num_topics):
                    aid = self.authoridx2id[a]
                    docs_a = self.author2doc[aid]
                    var_gamma[a, k] = 0.0
                    var_gamma[a, k] += self.alpha
                    for d in docs_a:
                        # TODO: if this document doesn't exist, we will have problems here. Could to an "if corpus.get(d)" type of thing.
                        doc = corpus[d]
                        ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                        cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
                        for vi, v in enumerate(ids):
                            var_gamma[a, k] += cts[vi] * var_mu[(d, v, a)] * var_phi[d, v, k]

            # Update Elogtheta, since gamma has been updated.
            Elogtheta = dirichlet_expectation(var_gamma)

            # Update lambda.
            for k in xrange(self.num_topics):
                for v in xrange(self.num_terms):
                    # TODO: highly unnecessary:
                    var_lambda[k, v] = 0.0
                    var_lambda[k, v] += self.eta
                    for d, doc in enumerate(corpus):
                        # Get the count of v in doc. If v is not in doc, return 0.
                        cnt = dict(doc).get(v, 0)
                        var_lambda[k, v] += cnt * var_phi[d, v, k]
                        #ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                        #cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
                        #for vi, v in enumerate(ids):
                        #    # FIXME: I'm 90% sure this is wrong.
                        #    var_lambda[k, v] += cts[vi] * var_phi[d, v, k]

            # Update Elogbeta, since lambda has been updated.
            Elogbeta = dirichlet_expectation(var_lambda)
            expElogbeta = numpy.exp(Elogbeta)


            logger.info('All variables updated.')

            # Print topics:
            self.var_lambda = var_lambda
            #pprint(self.show_topics())

            # Evaluate likelihood.
            if (iteration + 1) % self.eval_every == 0:
                prev_likelihood = likelihood
                likelihood = self.eval_likelihood(Elogtheta, Elogbeta)
                logger.info('Likelihood: %.3e', likelihood)
                if numpy.abs(likelihood - prev_likelihood) / abs(prev_likelihood) < self.threshold:
                    break
        # End of update loop (iterations).

        return var_gamma, var_lambda

    def eval_likelihood(self, Elogtheta, Elogbeta, doc_ids=None):
        """
        Note that this is not strictly speaking a likelihood.

        Compute the expectation of the log conditional likelihood of the data,

            E_q[log p(w_d | theta, beta, A_d)],

        where p(w_d | theta, beta, A_d) is the log conditional likelihood of the data.
        """
        
        # TODO: call this something other than "likelihood".

        # TODO: allow for evaluating test corpus. This will require inferring on unseen documents.

        if doc_ids is None:
            docs = self.corpus
        else:
            docs = [self.corpus[d] for d in doc_ids]

        likelihood = 0.0
        for d, doc in enumerate(docs):
            ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
            cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
            authors_d = self.doc2author[d]
            likelihood_d = 0.0
            for vi, v in enumerate(ids):
                for k in xrange(self.num_topics):
                    for aid in authors_d:
                        a = self.authorid2idx[aid]
                        likelihood_d += numpy.log(cts[vi]) + Elogtheta[a, k] + Elogbeta[k, v]
            author_prior_prob = 1.0 / len(authors_d)
            likelihood_d += numpy.log(author_prior_prob)
            likelihood += likelihood_d

        # For per-word likelihood, do:
        # likelihood *= 1 /sum(len(doc) for doc in docs)

        # TODO: can I do something along the lines of:
        # likelihood += author_prior_prob * numpy.sum(cnt * sum(logsumexp(Elogtheta[authors_d, :] + Elogbeta[:, id])) for id, cnt in doc)

        return likelihood

    # Overriding LdaModel.get_topic_terms.
    def get_topic_terms(self, topicid, topn=10):
        """
        Return a list of `(word_id, probability)` 2-tuples for the most
        probable words in topic `topicid`.
        Only return 2-tuples for the topn most probable words (ignore the rest).
        """
        topic = self.var_lambda[topicid, :]
        topic = topic / topic.sum()  # normalize to probability distribution
        bestn = matutils.argsort(topic, topn, reverse=True)
        return [(id, topic[id]) for id in bestn]







