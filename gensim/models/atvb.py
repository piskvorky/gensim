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
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from scipy.optimize import line_search

from six.moves import xrange

from pprint import pprint
from random import sample
from copy import deepcopy

# log(sum(exp(x))) that tries to avoid overflow
try:
    # try importing from here if older scipy is installed
    from scipy.maxentropy import logsumexp
except ImportError:
    # maxentropy has been removed in recent releases, logsumexp now in misc
    from scipy.misc import logsumexp

logger = logging.getLogger('gensim.models.atmodel')

def update_dir_prior(prior, N, logphat, rho):
    """
    Updates a given prior using Newton's method, described in
    **Huang: Maximum Likelihood Estimation of Dirichlet Distribution Parameters.**
    http://jonathan-huang.org/research/dirichlet/dirichlet.pdf
    """
    dprior = numpy.copy(prior)
    gradf = N * (psi(numpy.sum(prior)) - psi(prior) + logphat)

    c = N * polygamma(1, numpy.sum(prior))
    q = -N * polygamma(1, prior)

    b = numpy.sum(gradf / q) / (1 / c + numpy.sum(1 / q))

    dprior = -(gradf - b) / q

    # NOTE: in the LDA code, the criterion below is:
    # if all(rho * dprior + prior > 0)
    # But this causes an error for me, but the below criterion works.
    if (rho * dprior + prior > 0).all():
        prior += rho * dprior
    else:
        logger.warning("updated prior not positive")

    return prior

def dir_mle_search_direction(prior, N, logphat):
    """
    Updates a given prior using Newton's method, described in
    **Huang: Maximum Likelihood Estimation of Dirichlet Distribution Parameters.**
    http://jonathan-huang.org/research/dirichlet/dirichlet.pdf
    """
    dprior = numpy.copy(prior)
    gradf = N * (psi(numpy.sum(prior)) - psi(prior) + logphat)

    c = N * polygamma(1, numpy.sum(prior))
    q = -N * polygamma(1, prior)

    b = numpy.sum(gradf / q) / (1 / c + numpy.sum(1 / q))

    dprior = -(gradf - b) / q

    return dprior

class AtVb(LdaModel):
    """
    Train the author-topic model using variational Bayes.
    """

    def __init__(self, corpus=None, num_topics=100, id2word=None, id2author=None,
            author2doc=None, doc2author=None, threshold=0.001,
            iterations=10, alpha='symmetric', eta='symmetric', minimum_probability=0.01,
            eval_every=1, random_state=None, var_lambda=None):

        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')

        # NOTE: this stuff is confusing to me (from LDA code). Why would id2word not be none, but have length 0?
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

        if doc2author is None and author2doc is None:
            raise ValueError('at least one of author2doc/doc2author must be specified, to establish input space dimensionality')

        # TODO: consider whether there is a more elegant way of doing this (more importantly, a more efficient way).
        # If either doc2author or author2doc is missing, construct them from the other.
        if doc2author is None:
            # Make a mapping from document IDs to author IDs.
            doc2author = {}
            for d, _ in enumerate(corpus):
                author_ids = []
                for a, a_doc_ids in author2doc.items():
                    if d in a_doc_ids:
                        author_ids.append(a)
                doc2author[d] = author_ids
        elif author2doc is None:
            # Make a mapping from author IDs to document IDs.

            # First get a set of all authors.
            authors_ids = set()
            for d, a_doc_ids in doc2author.items():
                for a in a_doc_ids:
                    authors_ids.add(a)

            # Now construct the dictionary.
            author2doc = {}
            for a in range(len(authors_ids)):
                author2doc[a] = []
                for d, a_ids in doc2author.items():
                    if a in a_ids:
                        author2doc[a].append(d)

        self.author2doc = author2doc
        self.doc2author = doc2author

        self.num_authors = len(self.author2doc)
        logger.info('Number of authors: %d.', self.num_authors)

        self.id2author = id2author
        if self.id2author is None:
            logger.warning("no author id mapping provided; initializing from corpus, assuming identity")
            author_integer_ids = [str(i) for i in range(len(author2doc))]
            self.id2author = dict(zip(range(len(author2doc)), author_integer_ids))

        self.corpus = corpus
        self.iterations = iterations
        self.num_topics = num_topics
        self.threshold = threshold
        self.minimum_probability = minimum_probability 
        self.num_docs = len(corpus)
        self.num_authors = len(author2doc)
        self.eval_every = eval_every
        self.random_state = random_state

        self.random_state = get_random_state(random_state)

        # NOTE: I don't think this necessarily is a good way to initialize the topics.
        self.alpha = numpy.asarray([1.0 / self.num_topics for i in xrange(self.num_topics)])
        self.eta = numpy.asarray([1.0 / self.num_terms for i in xrange(self.num_terms)])

        if alpha == 'auto':
            self.optimize_alpha = True
        else:
            self.optimize_alpha = False

        if  eta == 'auto':
            self.optimize_eta = True
        else:
            self.optimize_eta = False

        if corpus is not None:
            self.inference(corpus, author2doc, doc2author, var_lambda)

    def update_alpha(self, var_gamma):
        """
        Update parameters for the Dirichlet prior on the per-document
        topic weights `alpha` given the last `var_gamma`.
        """
        N = float(var_gamma.shape[0])

        # NOTE: there might be possibility for overflow if number
        # of authors is very high.
        logphat = 0.0
        for a in xrange(self.num_authors):
            logphat += dirichlet_expectation(var_gamma[a, :])
        logphat *= 1 / N

        self.alpha = update_dir_prior(self.alpha, N, logphat, 1)

    def update_alpha_ls(self, var_gamma):
        """
        Work in progress.
        MLE of alpha with line search.
        """
        N = float(var_gamma.shape[0])

        # NOTE: there might be possibility for overflow if number
        # of authors is very high.
        logphat = 0.0
        for a in xrange(self.num_authors):
            logphat += dirichlet_expectation(var_gamma[a, :])
        logphat *= 1 / N

        def f(alpha):
            '''Compute the Dirichlet likelihood.'''
            return -N * (gammaln(numpy.sum(alpha)) - numpy.sum(gammaln(alpha)) + numpy.sum((alpha - 1) * logphat))

        def g(alpha):
            '''Compute the first derivative of the Dirichlet likelihood.'''
            return -N * (psi(numpy.sum(alpha)) - psi(alpha) + logphat)


        # TODO: consider what stopping criterion to use here, and
        # how many maximum iterations to use.
        # TODO: consider using line search.
        f1 = f(self.alpha)
        #print(f1)
        #print(0)
        for i in xrange(10):
            # Obtain search direction for Newton step.
            pk = dir_mle_search_direction(self.alpha, N, logphat)
            # Obtain stepsize using Wolfe condition.
            stepsize = line_search(f, g, self.alpha, pk)[0]
            # Update alpha.
            # NOTE: need to check that update is positive.
            self.alpha += stepsize * pk
            f2 = f(self.alpha)
            if (f2 - f1) / f1 < 0.01:
                break
            else:
                f1 = f2
            #print(f2)

        # logger.info("optimized eta %s", list(self.alpha))

        return self.alpha

    def update_eta(self, var_lambda):
        """
        Update parameters for the Dirichlet prior on the per-document
        topic weights `eta` given the last `var_lambda`.
        """
        N = float(len(var_lambda))

        logphat = 0.0
        for k in xrange(self.num_topics):
            logphat += dirichlet_expectation(var_lambda[k, :])
        logphat *= 1 / N

        self.eta = update_dir_prior(self.eta, N, logphat, rho)
        # logger.info("optimized eta %s", list(self.eta))

        return self.eta

    def inference(self, corpus=None, author2doc=None, doc2author=None, var_lambda=None):
        if corpus is None:
            corpus = self.corpus

        logger.info('Starting inference. Training on %d documents.', len(corpus))

        if var_lambda is None:
            optimize_lambda = True
        else:
            # We have topics from LDA, thus we do not train the topics.
            optimize_lambda = False

        # Initial value of gamma and lambda.
        # NOTE: parameters of gamma distribution same as in `ldamodel`.
        var_gamma = self.random_state.gamma(100., 1. / 100.,
                (self.num_authors, self.num_topics))

        if var_lambda is None:
            var_lambda = self.random_state.gamma(100., 1. / 100.,
                    (self.num_topics, self.num_terms))

        self.var_lambda = var_lambda
        self.var_gamma = var_gamma

        # Initialize mu.
        # mu is 1/|A_d| if a is in A_d, zero otherwise.
        # var_mu is essentially a (self.num_docs, self.num_terms, self.num_authors) sparse matrix,
        # which we represent using a dictionary.
        # TODO: consider initializing mu randomly. i.e.:
        # var_mu[(d, v, a)] = self.random_state.gamma(100., 1. / 100., (1,))[0]
        var_mu = dict()
        for d, doc in enumerate(corpus):
            ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
            for v in ids:
                authors_d = doc2author[d]  # List of author IDs for document d.
                for a in authors_d:
                    # Draw mu from gamma distribution.
                    var_mu[(d, v, a)] = 1 / len(authors_d)

        var_phi = numpy.zeros((self.num_docs, self.num_terms, self.num_topics))

        # TODO: consider how to vectorize opterations as much as
        # possible.
        # TODO: check vector and matrix dimensions, and ensure that
        # things are multiplied along the correct dimensions.

        Elogtheta = dirichlet_expectation(var_gamma)
        # NOTE: computing the Dirichlet expectation of lambda may
        # cause overflow when the vocabulary is very large, as it
        # requires a sum over vocab words.
        Elogbeta = dirichlet_expectation(var_lambda)
        expElogbeta = numpy.exp(Elogbeta)

        word_bound = self.word_bound(Elogtheta, Elogbeta)
        theta_bound = self.theta_bound(Elogtheta, var_gamma)
        beta_bound = self.beta_bound(Elogbeta, var_lambda)
        bound = word_bound + theta_bound + beta_bound
        #likelihood = self.log_word_prob(var_gamma, var_lambda)
        logger.info('Total bound: %.3e. Word bound: %.3e. theta bound: %.3e. beta bound: %.3e.', bound, word_bound, theta_bound, beta_bound)
        for iteration in xrange(self.iterations):
            #logger.info('Starting iteration %d.', iteration)
            # Update phi.
            for d, doc in enumerate(corpus):
                #logger.info('Updating phi, document %d.', d)
                ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                authors_d = doc2author[d]  # List of author IDs for document d.

                # Update phi.
                for v in ids:
                    for k in xrange(self.num_topics):
                        # Average Elogtheta over authors a in document d.
                        # NOTE: avgElogtheta may become numerically unsable. If
                        # it is a large positive number, exponentiating it may
                        # cause overflow, which probably results in the value 
                        # "inf". If it is a large negative number, exponentiating
                        # it may result in 0.0.
                        avgElogtheta = 0.0
                        for a in authors_d:
                            avgElogtheta += var_mu[(d, v, a)] * Elogtheta[a, k]
                        expavgElogtheta = numpy.exp(avgElogtheta)

                        # Compute phi.
                        # TODO: avoid computing phi if possible.
                        var_phi[d, v, k] = expavgElogtheta * expElogbeta[k, v]
                    # Normalize phi.
                    var_phi[d, v, :] = var_phi[d, v, :] / (var_phi[d, v, :].sum() + 1e-100)

            # Update mu.
            for d, doc in enumerate(corpus):
                #logger.info('Updating mu, document %d.', d)
                ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                authors_d = doc2author[d]  # List of author IDs for document d.
                for v in ids:
                    mu_sum = 0.0
                    for a in authors_d:
                        # Average Elogtheta over topics k.
                        # NOTE: we may have same problems as with phi update, above.
                        avgElogtheta = 0.0
                        for k in xrange(self.num_topics):
                            avgElogtheta += var_phi[d, v, k] * Elogtheta[a, k]
                        expavgElogtheta = numpy.exp(avgElogtheta)

                        # Compute mu.
                        # TODO: avoid computing mu if possible.
                        var_mu[(d, v, a)] = expavgElogtheta
                        mu_sum += var_mu[(d, v, a)]

                    mu_norm_const = 1.0 / (mu_sum + 1e-100)
                    for a in authors_d:
                        var_mu[(d, v, a)] *= mu_norm_const

            # Update gamma.
            #logger.info('Updating gamma.')
            for a in xrange(self.num_authors):
                for k in xrange(self.num_topics):
                    docs_a = self.author2doc[a]
                    var_gamma[a, k] = 0.0
                    var_gamma[a, k] += self.alpha[k]
                    for d in docs_a:
                        # TODO: if this document doesn't exist, we will have problems here. Could to an "if corpus.get(d)" type of thing.
                        doc = corpus[d]
                        ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                        cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
                        for vi, v in enumerate(ids):
                            var_gamma[a, k] += cts[vi] * var_mu[(d, v, a)] * var_phi[d, v, k]

            if self.optimize_alpha:
                # NOTE: taking a full Newton step seems to yield good results.
                # In the LDA code, they use rho() as step size. This seems 
                # very arbitrary; if a carefully chosen stepsize is needed,
                # linesearch would probably be better.
                stepsize = 1  
                self.update_alpha(var_gamma)

            # Update Elogtheta, since gamma has been updated.
            Elogtheta = dirichlet_expectation(var_gamma)

            # Update lambda.
            if optimize_lambda:
                #logger.info('Updating lambda.')
                for k in xrange(self.num_topics):
                    #logger.info('k = %d.', k)
                    for v in xrange(self.num_terms):
                        #logger.info('v = %d.', v)
                        var_lambda[k, v] = self.eta[v]

                        # The following commented-out code is used for "sampling" documents when
                        # updating lambda:
                        # sample_ratio = 1.0  # When sample_ratio is 1.0, the whole dataset is used.
                        # nsamples = int(numpy.ceil(self.num_docs * sample_ratio))
                        # doc_idxs = sample(xrange(self.num_docs), nsamples)

                        # TODO: this would be more efficient if there was a mapping from words
                        # to the documents that contain that word, although that mapping would be
                        # very large.
                        # NOTE: the below might cause overflow if number of documents is very large,
                        # although it seems somewhat unlikely.
                        for d, doc in enumerate(corpus):
                            # Get the count of v in doc. If v is not in doc, return 0.
                            cnt = dict(doc).get(v, 0)
                            var_lambda[k, v] += cnt * var_phi[d, v, k]

                if self.optimize_eta:
                    stepsize = 1
                    self.update_eta(var_lambda)

                # Update Elogbeta, since lambda has been updated.
                Elogbeta = dirichlet_expectation(var_lambda)
                expElogbeta = numpy.exp(Elogbeta)

                self.var_lambda = var_lambda

            self.var_gamma = var_gamma

            # Print topics:
            #pprint(self.show_topics())


            # Evaluate bound.
            if (iteration + 1) % self.eval_every == 0:
                #logger.info('Computing bound.')
                prev_bound = bound
                word_bound = self.word_bound(Elogtheta, Elogbeta)
                theta_bound = self.theta_bound(Elogtheta, var_gamma)
                beta_bound = self.beta_bound(Elogbeta, var_lambda)
                bound = word_bound + theta_bound + beta_bound
                #likelihood = self.log_word_prob(var_gamma, var_lambda)
                logger.info('Total bound: %.3e. Word bound: %.3e. theta bound: %.3e. beta bound: %.3e.', bound, word_bound, theta_bound, beta_bound)
                if numpy.abs(bound - prev_bound) / abs(prev_bound) < self.threshold:
                    break
        # End of update loop (iterations).

        return var_gamma, var_lambda

    def word_bound(self, Elogtheta, Elogbeta, doc_ids=None):
        """
        Note that this is not strictly speaking a likelihood.

        Compute the expectation of the log conditional likelihood of the data,

            E_q[log p(w_d | theta, beta, A_d)],

        where p(w_d | theta, beta, A_d) is the log conditional likelihood of the data.
        """

        # TODO: allow for evaluating test corpus. This will require inferring on unseen documents.

        if doc_ids is None:
            docs = self.corpus
        else:
            docs = [self.corpus[d] for d in doc_ids]

        # NOTE: computing the bound this way is very numerically unstable, which is why
        # "logsumexp" is used in the LDA code.
        bound= 0.0
        for d, doc in enumerate(docs):
            authors_d = self.doc2author[d]
            ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
            cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
            bound_d = 0.0
            for vi, v in enumerate(ids):
                bound_v = 0.0
                for k in xrange(self.num_topics):
                    for a in authors_d:
                        bound_v += numpy.exp(Elogtheta[a, k] + Elogbeta[k, v])
                bound_d += cts[vi] * numpy.log(bound_v)
            bound += numpy.log(1.0 / len(authors_d)) + bound_d

        # For per-word likelihood, do:
        # likelihood *= 1 /sum(len(doc) for doc in docs)

        # TODO: can I do something along the lines of (as in ldamodel):
        # likelihood += numpy.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, id]) for id, cnt in doc)
        # If I computed the LDA bound the way I compute the author-topic bound above:
        # bound = 0.0
        # for d, doc in enumerate(docs):
        #     ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
        #     cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
        #     bound_d = 0.0
        #     for vi, v in enumerate(ids):
        #         bound_v = 0.0
        #         for k in xrange(self.num_topics):
        #             bound_v += numpy.exp(Elogtheta[d, k] + Elogbeta[k, v])
        #         bound_d += cts[vi] * numpy.log(bound_v)
        #     bound += bound_d

        return bound

    def theta_bound(self, Elogtheta, var_gamma, doc_ids=None):
        """
        """

        if doc_ids is None:
            docs = self.corpus
        else:
            docs = [self.corpus[d] for d in doc_ids]

        bound = 0.0
        for a in xrange(self.num_authors):
            var_gamma_a = var_gamma[a, :]
            Elogtheta_a = Elogtheta[a, :]
            # E[log p(theta | alpha) - log q(theta | gamma)]; assumes alpha is a vector
            bound += numpy.sum((self.alpha - var_gamma_a) * Elogtheta_a)
            bound += numpy.sum(gammaln(var_gamma_a) - gammaln(self.alpha))
            bound += gammaln(numpy.sum(self.alpha)) - gammaln(numpy.sum(var_gamma_a))

        return bound

    def beta_bound(self, Elogbeta, var_lambda, doc_ids=None):
        bound = 0.0
        bound += numpy.sum((self.eta - var_lambda) * Elogbeta)
        bound += numpy.sum(gammaln(var_lambda) - gammaln(self.eta))
        bound += numpy.sum(gammaln(numpy.sum(self.eta)) - gammaln(numpy.sum(var_lambda, 1)))

        return bound

    def log_word_prob(self, var_gamma, var_lambda, doc_ids=None):
        """
        Compute the liklihood of the corpus under the model, by first 
        computing the conditional probabilities of the words in a
        document d,

            p(w_d | theta, beta, A_d),

        summing over all documents, and dividing by the number of documents.
        """

        norm_gamma = var_gamma.copy()
        norm_lambda = var_lambda.copy()
        for a in xrange(self.num_authors):
            norm_gamma[a, :] = var_gamma[a, :] / var_gamma.sum(axis=1)[a]
        for k in xrange(self.num_topics):
            norm_lambda[k, :] = var_lambda[k, :] / var_lambda.sum(axis=1)[k]

        if doc_ids is None:
            docs = self.corpus
        else:
            docs = [self.corpus[d] for d in doc_ids]

        log_word_prob = 0.0
        for d, doc in enumerate(docs):
            ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
            cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
            authors_d = self.doc2author[d]
            log_word_prob_d = 0.0
            for vi, v in enumerate(ids):
                log_word_prob_v = 0.0
                for k in xrange(self.num_topics):
                    for a in authors_d:
                        log_word_prob_v += norm_gamma[a, k] * norm_lambda[k, v]
                log_word_prob_d += cts[vi] * numpy.log(log_word_prob_v)
            log_word_prob += numpy.log(1.0 / len(authors_d)) + log_word_prob_d
            #authors_idxs = [self.authorid2idx[aid] for aid in authors_d]
            #likelihood += author_prior_prob * numpy.sum(cnt * numpy.log(numpy.sum(numpy.exp(logsumexp(Elogtheta[a, :] + Elogbeta[:, id])) for a in authors_idxs)) for id, cnt in doc)

        return log_word_prob

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


    def get_author_topics(self, author_id, minimum_probability=None):
        """
        Return topic distribution the given author, as a list of
        (topic_id, topic_probability) 2-tuples.
        Ignore topics with very low probability (below `minimum_probability`).
        """
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)  # never allow zero values in sparse output

        topic_dist = self.var_gamma[author_id, :] / sum(self.var_gamma[author_id, :])

        author_topics = [(topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)
                if topicvalue >= minimum_probability]

        # author_name = self.id2author[author_id]

        return author_topics






