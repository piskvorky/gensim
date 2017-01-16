#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author-topic model.
"""

import pdb
from pdb import set_trace as st

import logging
import numpy
import numbers
import numpy as np

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

logger = logging.getLogger('gensim.models.atmodel')


class AtBlocking(LdaModel):
    """
    Train the author-topic model using variational Bayes.
    """

    def __init__(self, corpus=None, num_topics=100, id2word=None, id2author=None,
            author2doc=None, doc2author=None, threshold=0.001,
            iterations=10, alpha='symmetric', eta='symmetric', minimum_probability=0.01,
            eval_every=1, random_state=None):

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

        if doc2author is None and author2doc is None:
            raise ValueError('at least one of author2doc/doc2author must be specified, to establish input space dimensionality')

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

        self.author2id = dict(zip(author2doc.keys(), range(len(author2doc.keys()))))
        self.id2author = dict(zip(range(len(author2doc.keys())), author2doc.keys()))

        self.alpha = numpy.asarray([1.0 / self.num_topics for i in xrange(self.num_topics)])
        self.eta = numpy.asarray([1.0 / self.num_terms for i in xrange(self.num_terms)])

        if corpus is not None:
            self.inference(corpus)

    def inference(self, corpus=None):
        if corpus is None:
            corpus = self.corpus

        logger.info('Starting inference. Training on %d documents.', len(corpus))

        # Initial value of gamma and lambda.
        var_gamma = self.random_state.gamma(100., 1. / 100.,
                (self.num_authors, self.num_topics))
        var_lambda = self.random_state.gamma(100., 1. / 100.,
                (self.num_topics, self.num_terms))

        self.var_lambda = var_lambda
        self.var_gamma = var_gamma

        # Initialize phi.
        var_phi = numpy.zeros((self.num_docs, self.num_terms, self.num_authors, self.num_topics))

        Elogtheta = dirichlet_expectation(var_gamma)
        Elogbeta = dirichlet_expectation(var_lambda)

        if self.eval_every > 0:
            self.Elogtheta = Elogtheta
            self.Elogbeta = Elogbeta
            corpus_words = sum(cnt for document in corpus for _, cnt in document)
            perwordbound = self.bound(corpus) / corpus_words
            logger.info('perwordbound: %.3e.', perwordbound)
        for iteration in xrange(self.iterations):
            lastgamma = var_gamma.copy()
            lastlambda = var_lambda.copy()

            # Update phi.
            for d, doc in enumerate(corpus):
                ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                authors_d = self.doc2author[d]  # List of author IDs for document d.
                authors_d = [self.author2id[a] for a in authors_d]

                # Update phi.
                for v in ids:
                    for a in authors_d:
                        for k in xrange(self.num_topics):
                            # Compute phi.
                            var_phi[d, v, a, k] = np.exp(Elogtheta[a, k]) * np.exp(Elogbeta[k, v])
                    # Normalize phi.
                    var_phi[d, v, :, :] = var_phi[d, v, :, :] / (var_phi[d, v, :, :].sum() + 1e-100)

            # Update gamma.
            for a in xrange(self.num_authors):
                for k in xrange(self.num_topics):
                    author_name = self.id2author[a]
                    docs_a = self.author2doc[author_name]
                    var_gamma[a, k] = 0.0
                    var_gamma[a, k] += self.alpha[k]
                    for d in docs_a:
                        doc = corpus[d]
                        ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
                        cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
                        for vi, v in enumerate(ids):
                            var_gamma[a, k] += cts[vi] * var_phi[d, v, a, k]

            # Update Elogtheta, since gamma has been updated.
            Elogtheta = dirichlet_expectation(var_gamma)

            # Update lambda.
            for k in xrange(self.num_topics):
                for v in xrange(self.num_terms):
                    var_lambda[k, v] = self.eta[v]

                    for d, doc in enumerate(corpus):
                        # Get the count of v in doc. If v is not in doc, return 0.
                        cnt = dict(doc).get(v, 0)
                        phi_sum = 0.0
                        for author_name in self.doc2author[d]:
                            a = self.author2id[author_name]
                            phi_sum += var_phi[d, v, a, k]
                        var_lambda[k, v] += cnt * phi_sum

            # Update Elogbeta, since lambda has been updated.
            Elogbeta = dirichlet_expectation(var_lambda)

            self.var_lambda = var_lambda

            self.var_gamma = var_gamma

            # Evaluate bound.
            if (iteration + 1) % self.eval_every == 0:
                self.Elogtheta = Elogtheta
                self.Elogbeta = Elogbeta
                corpus_words = sum(cnt for document in corpus for _, cnt in document)
                perwordbound = self.bound(corpus) / corpus_words
                logger.info('perwordbound: %.3e.', perwordbound)
        # End of update loop (iterations).

        # Ensure that the bound (or log probabilities) is computed after the last iteration.
        if self.eval_every != 0 and not (iteration + 1) % self.eval_every == 0:
            self.Elogtheta = Elogtheta
            self.Elogbeta = Elogbeta
            corpus_words = sum(cnt for document in corpus for _, cnt in document)
            perwordbound = self.bound(corpus) / corpus_words
            logger.info('perwordbound: %.3e.', perwordbound)

        return var_gamma, var_lambda

    def bound(self, corpus):

        # FIXME: check that this is correct.
        word_score = 0.0
        for d, doc in enumerate(corpus):
            authors_d = self.doc2author[d]
            authors_d = [self.author2id[a] for a in authors_d]
            ids = numpy.array([id for id, _ in doc])  # Word IDs in doc.
            cts = numpy.array([cnt for _, cnt in doc])  # Word counts.
            word_score_d = 0.0
            for vi, v in enumerate(ids):
                word_score_v = 0.0
                for k in xrange(self.num_topics):
                    for a in authors_d:
                        word_score_v += numpy.exp(self.Elogtheta[a, k] + self.Elogbeta[k, v])
                word_score_d += cts[vi] * numpy.log(word_score_v)
            word_score += numpy.log(1.0 / len(authors_d)) + word_score_d

        # E[log p(theta | alpha) - log q(theta | gamma)]
        theta_score = 0.0
        for a in self.author2id.values():
            theta_score += np.sum((self.alpha - self.var_gamma[a, :]) * self.Elogtheta[a, :])
            theta_score += np.sum(gammaln(self.var_gamma[a, :]) - gammaln(self.alpha))
            theta_score += gammaln(np.sum(self.alpha)) - gammaln(np.sum(self.var_gamma[a, :]))

        # E[log p(beta | eta) - log q (beta | lambda)]
        beta_score = 0.0
        beta_score += np.sum((self.eta - self.var_lambda) * self.Elogbeta)
        beta_score += np.sum(gammaln(self.var_lambda) - gammaln(self.eta))
        sum_eta = np.sum(self.eta)
        beta_score += np.sum(gammaln(sum_eta) - gammaln(np.sum(self.var_lambda, 1)))

        total_score = word_score + theta_score + beta_score

        return total_score

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






