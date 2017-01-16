#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2016 Olavur Mortensen <olavurmortensen@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Author-topic model.
"""

# NOTE: from what I understand, my name as well as Radim's should be attributed copyright above?

import pdb
from pdb import set_trace as st

import logging
import numpy
import numbers

from gensim import utils, matutils
from gensim.models.ldamodel import dirichlet_expectation, get_random_state
from gensim.models import LdaModel
from six.moves import xrange
from scipy.special import gammaln

from pprint import pprint

# log(sum(exp(x))) that tries to avoid overflow
try:
    # try importing from here if older scipy is installed
    from scipy.maxentropy import logsumexp
except ImportError:
    # maxentropy has been removed in recent releases, logsumexp now in misc
    from scipy.misc import logsumexp

logger = logging.getLogger(__name__)


class MinibatchAtVb(LdaModel):
    """
    Train the author-topic model using online variational Bayes.
    """
    # TODO: inherit interfaces.TransformationABC.

    def __init__(self, corpus=None, num_topics=100, id2word=None, id2author=None,
            author2doc=None, doc2author=None, threshold=0.001, minimum_probability=0.01,
            iterations=10, passes=1, alpha=None, eta=None, decay=0.5, offset=1.0,
            eval_every=1, random_state=None, var_lambda=None, chunksize=2000):

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

        # Make the reverse mapping, from author names to author IDs.
        self.author2id = dict(zip(self.id2author.values(), self.id2author.keys()))

        self.corpus = corpus
        self.iterations = iterations
        self.passes = passes
        self.num_topics = num_topics
        self.threshold = threshold
        self.minimum_probability = minimum_probability 
        self.decay = decay
        self.offset = offset
        self.num_docs = len(corpus)
        self.num_authors = len(author2doc)
        self.eval_every = eval_every
        self.chunksize = chunksize
        self.random_state = random_state

        # NOTE: I don't think this necessarily is a good way to initialize the topics.
        self.alpha = numpy.asarray([1.0 / self.num_topics for i in xrange(self.num_topics)])
        self.eta = numpy.asarray([1.0 / self.num_terms for i in xrange(self.num_terms)])

        self.random_state = get_random_state(random_state)

        if corpus is not None:
            self.inference(corpus, var_lambda=var_lambda)

    def rho(self, t):
        return pow(self.offset + t, -self.decay)

    def inference(self, corpus=None, var_lambda=None):
        if corpus is None:
            # TODO: I can't remember why I used "copy()" here.
            corpus = self.corpus.copy()

        self.num_docs = len(corpus)  # TODO: this needs to be different if the algorithm is truly online.

        logger.info('Starting inference. Training on %d documents.', len(corpus))

        # Whether or not to evaluate bound and log probability, respectively.
        bound_eval = True
        logprob_eval = False

        if var_lambda is None:
            self.optimize_lambda = True
        else:
            # We have topics from LDA, thus we do not train the topics.
            self.optimize_lambda = False

        # Initial values of gamma and lambda.
        # Parameters of gamma distribution same as in `ldamodel`.
        var_gamma = self.random_state.gamma(100., 1. / 100.,
                (self.num_authors, self.num_topics))
        tilde_gamma = var_gamma.copy()
        self.var_gamma = var_gamma

        if var_lambda is None:
            var_lambda = self.random_state.gamma(100., 1. / 100.,
                    (self.num_topics, self.num_terms))
            tilde_lambda = var_lambda.copy()
        else:
            self.norm_lambda = var_lambda.copy()
            for k in xrange(self.num_topics):
                self.norm_lambda[k, :] = var_lambda[k, :] / var_lambda.sum(axis=1)[k]
        
        self.var_lambda = var_lambda

        # TODO: consider making phi sparse. Each document does not contain all terms.
        var_phi = numpy.zeros((self.num_terms, self.num_topics))

        # Initialize dirichlet expectations.
        Elogtheta = dirichlet_expectation(var_gamma)
        Elogbeta = dirichlet_expectation(var_lambda)
        expElogbeta = numpy.exp(Elogbeta)

        t = 0
        if self.eval_every > 0:
            if bound_eval:
                word_bound = self.word_bound(Elogtheta, Elogbeta)
                theta_bound = self.theta_bound(Elogtheta)
                beta_bound = self.beta_bound(Elogbeta)
                bound = word_bound + theta_bound + beta_bound
                logger.info('Total bound: %.3e. Word bound: %.3e. theta bound: %.3e. beta bound: %.3e.', bound, word_bound, theta_bound, beta_bound)
            if logprob_eval:
                logprob = self.eval_logprob()
                logger.info('Log prob: %.3e.', logprob)
        for _pass in xrange(self.passes):
            for chunk_no, chunk in enumerate(utils.grouper(corpus, self.chunksize, as_numpy=False)):
                converged = 0  # Number of documents converged for current pass over corpus.
                rhot = self.rho(chunk_no + _pass)
                for d, doc in enumerate(chunk):
                    ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                    cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
                    authors_d = self.doc2author[d]  # List of author IDs for document d.

                    # Initialize mu.
                    # mu is 1/|A_d| if a is in A_d, zero otherwise.
                    # TODO: consider doing random initialization instead.
                    var_mu = dict()
                    for v in ids:
                        for a in authors_d:
                            var_mu[(v, a)] = 1 / len(authors_d)

                    for iteration in xrange(self.iterations):
                        #logger.info('iteration %i', iteration)

                        lastgamma = tilde_gamma.copy()

                        # Update phi.
                        for v in ids:
                            for k in xrange(self.num_topics):
                                # Average Elogtheta over authors a in document d.
                                avgElogtheta = 0.0
                                for a in authors_d:
                                    avgElogtheta += var_mu[(v, a)] * Elogtheta[a, k]
                                expavgElogtheta = numpy.exp(avgElogtheta)

                                # Compute phi.
                                var_phi[v, k] = expavgElogtheta * expElogbeta[k, v]

                            # Normalize phi over k.
                            var_phi[v, :] = var_phi[v, :] / (var_phi[v, :].sum() + 1e-100)

                        # Update mu.
                        for v in ids:
                            # Prior probability of observing author a in document d is one
                            # over the number of authors in document d.
                            mu_sum = 0.0
                            for a in authors_d:
                                # Average Elogtheta over topics k.
                                avgElogtheta = 0.0
                                for k in xrange(self.num_topics):
                                    avgElogtheta += var_phi[v, k] * Elogtheta[a, k]
                                expavgElogtheta = numpy.exp(avgElogtheta)

                                # Compute mu over a.
                                var_mu[(v, a)] = expavgElogtheta
                                mu_sum += var_mu[(v, a)]

                            # Normalize mu.
                            mu_norm_const = 1.0 / (mu_sum + 1e-100)
                            for a in authors_d:
                                var_mu[(v, a)] *= mu_norm_const

                        # Update gamma.
                        for a in authors_d:
                            for k in xrange(self.num_topics):
                                tilde_gamma[a, k] = 0.0
                                for vi, v in enumerate(ids):
                                    tilde_gamma[a, k] += cts[vi] * var_mu[(v, a)] * var_phi[v, k]
                                tilde_gamma[a, k] *= len(self.author2doc[a])
                                tilde_gamma[a, k] += self.alpha[k]

                        # Update gamma and lambda.
                        # Interpolation between document d's "local" gamma (tilde_gamma),
                        # and "global" gamma (var_gamma). Same goes for lambda.
                        # TODO: I may need to be smarter about computing rho. In ldamodel,
                        # it's: pow(offset + pass_ + (self.num_updates / chunksize), -decay).
                        # FIXME: if tilde_gamma is computed like this in every iteration, then I can't compare
                        # lastgamma to it for convergence test. FIXME.
                        var_gamma_temp = (1 - rhot) * var_gamma + rhot * tilde_gamma

                        # Update Elogtheta and Elogbeta, since gamma and lambda have been updated.
                        # FIXME: I don't need to update the entire gamma, as I only updated a few rows of it,
                        # corresponding to the authors in the document. The same goes for Elogtheta.
                        Elogtheta = dirichlet_expectation(var_gamma_temp)
                        
                        # Check for convergence.
                        # Criterion is mean change in "local" gamma and lambda.
                        if iteration > 0:
                            meanchange_gamma = numpy.mean(abs(var_gamma_temp - lastgamma))
                            gamma_condition = meanchange_gamma < self.threshold
                            # logger.info('Mean change in gamma: %.3e', meanchange_gamma)
                            if gamma_condition:
                                # logger.info('Converged after %d iterations.', iteration)
                                converged += 1
                                break
                    # End of iterations loop.

                    # FIXME: there are too many different gamma variables!
                    var_gamma = var_gamma_temp.copy()

                if self.optimize_lambda:
                     # Update lambda.
                     # only one update per document).
                     for k in xrange(self.num_topics):
                         for vi, v in enumerate(ids):
                             # cnt = dict(doc).get(v, 0)
                             cnt = cts[vi]
                             tilde_lambda[k, v] = self.eta[v] + self.num_docs * cnt * var_phi[v, k]

                     # This is a little bit faster:
                     # tilde_lambda[:, ids] = self.eta[ids] + self.num_docs * cts * var_phi[ids, :].T

                     # Note that we only changed the elements in lambda corresponding to 
                     # the words in document d, hence the [:, ids] indexing.
                     var_lambda[:, ids] = (1 - rhot) * var_lambda[:, ids] + rhot * tilde_lambda[:, ids]
                     Elogbeta = dirichlet_expectation(var_lambda)
                     expElogbeta = numpy.exp(Elogbeta)
                     var_lambda = var_lambda.copy()

                # Print topics:
                # pprint(self.show_topics())

            # End of corpus loop.

            if _pass % self.eval_every == 0:
                self.var_gamma = var_gamma
                self.var_lambda = var_lambda
                if self.eval_every > 0:
                    if bound_eval:
                        prev_bound = bound
                        word_bound = self.word_bound(Elogtheta, Elogbeta)
                        theta_bound = self.theta_bound(Elogtheta)
                        beta_bound = self.beta_bound(Elogbeta)
                        bound = word_bound + theta_bound + beta_bound
                        logger.info('Total bound: %.3e. Word bound: %.3e. theta bound: %.3e. beta bound: %.3e.', bound, word_bound, theta_bound, beta_bound)
                    if logprob_eval:
                        logprob = self.eval_logprob()
                        logger.info('Log prob: %.3e.', logprob)

            #logger.info('Converged documents: %d/%d', converged, self.num_docs)

            # TODO: consider whether to include somthing like this:
            #if numpy.abs(bound - prev_bound) / abs(prev_bound) < self.bound_threshold:
            #    break
        # End of pass over corpus loop.

        # Ensure that the bound (or log probabilities) is computed at the very last pass.
        if self.eval_every != 0 and not _pass % self.eval_every == 0:
            # If the bound should be computed, and it wasn't computed at the last pass,
            # then compute the bound.
            self.var_gamma = var_gamma
            self.var_lambda = var_lambda
            if self.eval_every > 0:
                if bound_eval:
                    prev_bound = bound
                    word_bound = self.word_bound(Elogtheta, Elogbeta)
                    theta_bound = self.theta_bound(Elogtheta)
                    beta_bound = self.beta_bound(Elogbeta)
                    bound = word_bound + theta_bound + beta_bound
                    logger.info('Total bound: %.3e. Word bound: %.3e. theta bound: %.3e. beta bound: %.3e.', bound, word_bound, theta_bound, beta_bound)
                if logprob_eval:
                    logprob = self.eval_logprob()
                    logger.info('Log prob: %.3e.', logprob)


        self.var_lambda = var_lambda
        self.var_gamma = var_gamma

        return var_gamma, var_lambda

    def word_bound(self, Elogtheta, Elogbeta, doc_ids=None):
        """
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
        # NOTE: computing bound is very very computationally intensive. I could, for example,
        # only use a portion of the data to do that (even a held-out set).
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

    def theta_bound(self, Elogtheta):
        bound = 0.0
        for a in xrange(self.num_authors):
            var_gamma_a = self.var_gamma[a, :]
            Elogtheta_a = Elogtheta[a, :]
            # E[log p(theta | alpha) - log q(theta | gamma)]; assumes alpha is a vector
            bound += numpy.sum((self.alpha - var_gamma_a) * Elogtheta_a)
            bound += numpy.sum(gammaln(var_gamma_a) - gammaln(self.alpha))
            bound += gammaln(numpy.sum(self.alpha)) - gammaln(numpy.sum(var_gamma_a))

        return bound

    def beta_bound(self, Elogbeta):
        bound = 0.0
        bound += numpy.sum((self.eta - self.var_lambda) * Elogbeta)
        bound += numpy.sum(gammaln(self.var_lambda) - gammaln(self.eta))
        bound += numpy.sum(gammaln(numpy.sum(self.eta)) - gammaln(numpy.sum(self.var_lambda, 1)))

        return bound

    def eval_logprob(self, doc_ids=None):
        """
        Compute the liklihood of the corpus under the model, by first 
        computing the conditional probabilities of the words in a
        document d,

            p(w_d | theta, beta, A_d),

        summing over all documents, and dividing by the number of documents.
        """

        # TODO: if var_lambda is supplied from LDA, normalizing it every time
        # is unnecessary.
        norm_gamma = self.var_gamma.copy()
        for a in xrange(self.num_authors):
            norm_gamma[a, :] = self.var_gamma[a, :] / self.var_gamma.sum(axis=1)[a]

        if self.optimize_lambda:
            norm_lambda = self.var_lambda.copy()
            for k in xrange(self.num_topics):
                norm_lambda[k, :] = self.var_lambda[k, :] / self.var_lambda.sum(axis=1)[k]
        else:
            norm_lambda = self.norm_lambda

        if doc_ids is None:
            docs = self.corpus
        else:
            docs = [self.corpus[d] for d in doc_ids]

        logprob = 0.0
        for d, doc in enumerate(docs):
            ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
            cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
            authors_d = self.doc2author[d]
            logprob_d = 0.0
            for vi, v in enumerate(ids):
                logprob_v = 0.0
                for k in xrange(self.num_topics):
                    for a in authors_d:
                        logprob_v += norm_gamma[a, k] * norm_lambda[k, v]
                logprob_d += cts[vi] * numpy.log(logprob_v)
            logprob += numpy.log(1.0 / len(authors_d)) + logprob_d

        return logprob

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



