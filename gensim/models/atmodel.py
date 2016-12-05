#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2016 Olavur Mortensen <olavurmortensen@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Author-topic model in Python.

"""

# TODO: write proper docstrings.

import pdb
from pdb import set_trace as st
from pprint import pprint

import logging
import numpy as np # for arrays, array broadcasting etc.
import numbers

from gensim import utils
from gensim.models import LdaModel
from gensim.models.ldamodel import dirichlet_expectation, get_random_state, LdaState
from itertools import chain
from scipy.special import gammaln  # gamma function utils
from six.moves import xrange
import six

logger = logging.getLogger('gensim.models.atmodel')

class AuthorTopicState(LdaState):
    """
    Encapsulate information for distributed computation of AuthorTopicModel objects.

    Objects of this class are sent over the network, so try to keep them lean to
    reduce traffic.

    """
    def __init__(self, eta, lambda_shape, gamma_shape):
        self.eta = eta
        self.sstats = np.zeros(lambda_shape)
        self.gamma = np.zeros(gamma_shape)
        self.numdocs = 0

def construct_doc2author(corpus, author2doc):
    """Make a mapping from document IDs to author IDs."""
    doc2author = {}
    for d, _ in enumerate(corpus):
        author_ids = []
        for a, a_doc_ids in author2doc.items():
            if d in a_doc_ids:
                author_ids.append(a)
        doc2author[d] = author_ids
    return doc2author

def construct_author2doc(corpus, doc2author):
    """Make a mapping from author IDs to document IDs."""

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
    return author2doc

class AuthorTopicModel(LdaModel):
    """
    """
    def __init__(self, corpus=None, num_topics=100, id2word=None,
                author2doc=None, doc2author=None, id2author=None, var_lambda=None,
                 chunksize=2000, passes=1, update_every=1,
                 alpha='symmetric', eta='symmetric', decay=0.5, offset=1.0,
                 eval_every=10, iterations=50, gamma_threshold=0.001,
                 minimum_probability=0.01, random_state=None, ns_conf={},
                 minimum_phi_value=0.01, per_word_topics=False):
        """
        """

        distributed = False  # TODO: implement distributed version.

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

        # If either doc2author or author2doc is missing, construct them from the other.
        # FIXME: make the code below into methods, so the user can construct either doc2author or author2doc *once* and then not worry about it.
        # TODO: consider whether there is a more elegant way of doing this (more importantly, a more efficient way).
        if doc2author is None:
            doc2author = construct_doc2author(corpus, author2doc)
        elif author2doc is None:
            author2doc = construct_author2doc(corpus, doc2author)

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

        self.distributed = distributed
        self.num_topics = num_topics
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability 
        self.num_updates = 0

        self.passes = passes
        self.update_every = update_every
        self.eval_every = eval_every
        self.minimum_phi_value = minimum_phi_value
        self.per_word_topics = per_word_topics

        self.corpus = corpus
        self.num_authors = len(author2doc)

        self.alpha, self.optimize_alpha = self.init_dir_prior(alpha, 'alpha')

        assert self.alpha.shape == (self.num_topics,), "Invalid alpha shape. Got shape %s, but expected (%d, )" % (str(self.alpha.shape), self.num_topics)

        if isinstance(eta, six.string_types):
            if eta == 'asymmetric':
                raise ValueError("The 'asymmetric' option cannot be used for eta")

        self.eta, self.optimize_eta = self.init_dir_prior(eta, 'eta')

        self.random_state = get_random_state(random_state)

        assert (self.eta.shape == (self.num_terms,) or self.eta.shape == (self.num_topics, self.num_terms)), (
                "Invalid eta shape. Got shape %s, but expected (%d, 1) or (%d, %d)" %
                (str(self.eta.shape), self.num_terms, self.num_topics, self.num_terms))

        if not distributed:
            self.dispatcher = None
            self.numworkers = 1
        else:
            # TODO: implement distributed version.
            pass

        # VB constants
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold

        # Initialize the variational distributions q(beta|lambda) and q(theta|gamma)
        self.state = AuthorTopicState(self.eta, (self.num_topics, self.num_terms), (self.num_authors, self.num_topics))
        self.state.sstats = self.random_state.gamma(100., 1. / 100., (self.num_topics, self.num_terms))
        self.state.gamma = self.random_state.gamma(100., 1. / 100., (self.num_authors, self.num_topics))
        self.expElogbeta = np.exp(dirichlet_expectation(self.state.sstats))
        self.expElogtheta = np.exp(dirichlet_expectation(self.state.gamma))

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            use_numpy = self.dispatcher is not None
            self.update(corpus, chunks_as_numpy=use_numpy)

    def __str__(self):
        return "AuthorTopicModel(num_terms=%s, num_topics=%s, num_authors=%s, decay=%s, chunksize=%s)" % \
            (self.num_terms, self.num_topics, self.num_authors, self.decay, self.chunksize)

    def compute_phinorm(self, ids, authors_d, expElogthetad, expElogbetad):
        """Efficiently computes the normalizing factor in phi."""
        phinorm = np.zeros(len(ids))
        expElogtheta_sum = np.zeros(self.num_topics)
        for a in xrange(len(authors_d)):
            expElogtheta_sum += expElogthetad[a, :]
        phinorm = expElogtheta_sum.dot(expElogbetad)

        return phinorm

    def inference(self, chunk, collect_sstats=False, chunk_no=None):
        """
        Given a chunk of sparse document vectors, estimate gamma (parameters
        controlling the topic weights) for each document in the chunk.

        This function does not modify the model (=is read-only aka const). The
        whole input chunk of document is assumed to fit in RAM; chunking of a
        large corpus must be done earlier in the pipeline.

        If `collect_sstats` is True, also collect sufficient statistics needed
        to update the model's topic-word distributions, and return a 2-tuple
        `(gamma, sstats)`. Otherwise, return `(gamma, None)`. `gamma` is of shape
        `len(chunk_authors) x self.num_topics`, where `chunk_authors` is the number
        of authors in the documents in the current chunk.

        Avoids computing the `phi` variational parameter directly using the
        optimization presented in **Lee, Seung: Algorithms for non-negative matrix factorization, NIPS 2001**.

        """
        try:
            _ = len(chunk)
        except:
            # convert iterators/generators to plain list, so we have len() etc.
            chunk = list(chunk)
        if len(chunk) > 1:
            logger.debug("performing inference on a chunk of %i documents", len(chunk))

        # Initialize the variational distribution q(theta|gamma) for the chunk
        if collect_sstats:
            sstats = np.zeros_like(self.expElogbeta)
        else:
            sstats = None
        converged = 0

        chunk_authors = set()

        # Now, for each document d update that document's gamma and phi
        for d, doc in enumerate(chunk):
            doc_no = chunk_no + d  # TODO: can it safely be assumed that this is the case?
            if doc and not isinstance(doc[0][0], six.integer_types):
                # make sure the term IDs are ints, otherwise np will get upset
                ids = [int(id) for id, _ in doc]
            else:
                ids = [id for id, _ in doc]
            cts = np.array([cnt for _, cnt in doc])
            authors_d = self.doc2author[doc_no]  # List of author IDs for the current document.

            gammad = self.state.gamma[authors_d, :]
            tilde_gamma = gammad.copy()

            Elogthetad = dirichlet_expectation(tilde_gamma)
            expElogthetad = np.exp(Elogthetad)
            expElogbetad = self.expElogbeta[:, ids]

            phinorm = self.compute_phinorm(ids, authors_d, expElogthetad, expElogbetad)

            # Iterate between gamma and phi until convergence
            for iteration in xrange(self.iterations):
                #logger.info('iteration %i', iteration)

                lastgamma = tilde_gamma.copy()

                # Update gamma.
                for ai, a in enumerate(authors_d):
                    tilde_gamma[ai, :] = self.alpha + len(self.author2doc[a]) * expElogthetad[ai, :] * np.dot(cts / phinorm, expElogbetad.T)

                # Update gamma and lambda.
                # Interpolation between document d's "local" gamma (tilde_gamma),
                # and "global" gamma (var_gamma).
                tilde_gamma = (1 - self.rho) * gammad + self.rho * tilde_gamma

                # Update Elogtheta and Elogbeta, since gamma and lambda have been updated.
                Elogthetad = dirichlet_expectation(tilde_gamma)
                expElogthetad = np.exp(Elogthetad)

                phinorm = self.compute_phinorm(ids, authors_d, expElogthetad, expElogbetad)

                # Check for convergence.
                # Criterion is mean change in "local" gamma and lambda.
                meanchange_gamma = np.mean(abs(tilde_gamma - lastgamma))
                gamma_condition = meanchange_gamma < self.gamma_threshold
                # logger.info('Mean change in gamma: %.3e', meanchange_gamma)
                if gamma_condition:
                    # logger.info('Converged after %d iterations.', iteration)
                    converged += 1
                    break
            # End of iterations loop.

            self.state.gamma[authors_d, :] = tilde_gamma

            # NOTE: this may be slow. Especially when there are many authors per document. It is
            # imporant to find a faster way to handle this.
            chunk_authors = chunk_authors.union(set(authors_d))

            if collect_sstats:
                # Contribution of document d to the expected sufficient
                # statistics for the M step.
                expElogtheta_sum_a = expElogthetad.sum(axis=0)
                sstats[:, ids] += np.outer(expElogtheta_sum_a.T, cts/phinorm)

        if len(chunk) > 1:
            logger.debug("%i/%i documents converged within %i iterations",
                         converged, len(chunk), self.iterations)

        if collect_sstats:
            # This step finishes computing the sufficient statistics for the
            # M step, so that
            # sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
            # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
            sstats *= self.expElogbeta
        gamma_chunk = self.state.gamma[list(chunk_authors), :]
        return gamma_chunk, sstats

    def do_estep(self, chunk, state=None, chunk_no=None):
        """
        Perform inference on a chunk of documents, and accumulate the collected
        sufficient statistics in `state` (or `self.state` if None).

        """
        if state is None:
            state = self.state
        gamma, sstats = self.inference(chunk, collect_sstats=True, chunk_no=chunk_no)
        state.sstats += sstats
        state.numdocs += len(chunk)
        return gamma

    def bound(self, corpus, chunk_no=None, gamma=None, subsample_ratio=1.0, doc2author=None, ):
        """
        Estimate the variational bound of documents from `corpus`:
        E_q[log p(corpus)] - E_q[log q(corpus)]

        `gamma` are the variational parameters on topic weights for each `corpus`
        document (=2d matrix=what comes out of `inference()`).
        If not supplied, will be inferred from the model.

        Computing the bound of unseen data is not recommended, unless one knows what one is doing.
        In this case, gamma must be inferred in advance, and doc2author for this new data must be
        provided.

        """

        _lambda = self.state.get_lambda()
        Elogbeta = dirichlet_expectation(_lambda)
        expElogbeta = np.exp(dirichlet_expectation(_lambda))

        if gamma is not None:
            logger.warning('bound() assumes gamma to be None and uses the gamma provided is self.state.')
            # NOTE: alternatively:
            #assert gamma is None, 'bound() assumes gamma to be None and uses the gamma provided is self.state.'
        else:
            gamma = self.state.gamma

        if chunk_no is None:
            logger.warning('No chunk_no provided to bound().')
            # NOTE: alternatively:
            #assert chunk_no is not None, 'chunk_no must be provided to bound().'
            chunk_no = 0

        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(dirichlet_expectation(gamma))

        word_score = 0.0
        authors_set = set()  # Used in computing theta bound.
        theta_score = 0.0
        for d, doc in enumerate(corpus):  # stream the input doc-by-doc, in case it's too large to fit in RAM
            doc_no = chunk_no + d
            authors_d = self.doc2author[doc_no]
            ids = np.array([id for id, _ in doc])  # Word IDs in doc.
            cts = np.array([cnt for _, cnt in doc])  # Word counts.

            if d % self.chunksize == 0:
                logger.debug("bound: at document #%i", d)

            # Computing the bound requires summing over expElogtheta[a, k] * expElogbeta[k, v], which
            # is the same computation as in normalizing phi.
            phinorm = self.compute_phinorm(ids, authors_d, expElogtheta[authors_d, :], expElogbeta[:, ids])
            word_score += np.log(1.0 / len(authors_d)) + cts.dot(np.log(phinorm))

            # E[log p(theta | alpha) - log q(theta | gamma)]
            # The code blow ensure we compute the score of each author only once.
            for a in authors_d:
                if a not in authors_set:
                    theta_score += np.sum((self.alpha - gamma[a, :]) * Elogtheta[a, :])
                    theta_score += np.sum(gammaln(gamma[a, :]) - gammaln(self.alpha))
                    theta_score += gammaln(np.sum(self.alpha)) - gammaln(np.sum(gamma[a, :]))
                    authors_set.add(a)

        # Compensate likelihood for when `corpus` above is only a sample of the whole corpus. This ensures
        # that the likelihood is always rougly on the same scale.
        word_score *= subsample_ratio

        # theta_score is rescaled in a similar fashion.
        theta_score *= self.num_authors / len(authors_set)

        # E[log p(beta | eta) - log q (beta | lambda)]
        beta_score = 0.0
        beta_score += np.sum((self.eta - _lambda) * Elogbeta)
        beta_score += np.sum(gammaln(_lambda) - gammaln(self.eta))
        sum_eta = np.sum(self.eta)
        beta_score += np.sum(gammaln(sum_eta) - gammaln(np.sum(_lambda, 1)))

        total_score = word_score + theta_score + beta_score

        #print("%.3e\t%.3e\t%.3e\t%.3e" %(total_score, word_score, theta_score, beta_score))

        return total_score

    def get_author_topics(self, author_id, minimum_probability=None):
        """
        Return topic distribution the given author, as a list of
        (topic_id, topic_probability) 2-tuples.
        Ignore topics with very low probability (below `minimum_probability`).
        """
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)  # never allow zero values in sparse output

        topic_dist = self.state.gamma[author_id, :] / sum(self.state.gamma[author_id, :])

        author_topics = [(topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)
                if topicvalue >= minimum_probability]

        return author_topics

    # NOTE: method `top_topics` is used directly. There is no topic coherence measure for 
    # the author-topic model. c_v topic coherence is a valid measure of topic quality in 
    # the author-topic model, although it does not take authorship information into account.

    def __getitem__(self, bow, eps=None):
        """
        """
        # TODO: this.
        # E.g. assume bow is a list of documents for this particular author, and that the author 
        # is not in the corpus beforehand. Then add an author to doc2author and author2doc,
        # and call self.update to infer the new author's topic distribution.
        pass

    def save(self, fname, ignore=['state', 'dispatcher'], *args, **kwargs):
        """
        Save the model to file.

        Large internal arrays may be stored into separate files, with `fname` as prefix.

        `separately` can be used to define which arrays should be stored in separate files.

        `ignore` parameter can be used to define which variables should be ignored, i.e. left
        out from the pickled author-topic model. By default the internal `state` is ignored as it uses
        its own serialisation not the one provided by `AuthorTopicModel`. The `state` and `dispatcher`
        will be added to any ignore parameter defined.


        Note: do not save as a compressed file if you intend to load the file back with `mmap`.

        Note: If you intend to use models across Python 2/3 versions there are a few things to
        keep in mind:

          1. The pickled Python dictionaries will not work across Python versions
          2. The `save` method does not automatically save all NumPy arrays using NumPy, only
             those ones that exceed `sep_limit` set in `gensim.utils.SaveLoad.save`. The main
             concern here is the `alpha` array if for instance using `alpha='auto'`.

        Please refer to the wiki recipes section (https://github.com/piskvorky/gensim/wiki/Recipes-&-FAQ#q9-how-do-i-load-a-model-in-python-3-that-was-trained-and-saved-using-python-2)
        for an example on how to work around these issues.
        """
        if self.state is not None:
            self.state.save(utils.smart_extension(fname, '.state'), *args, **kwargs)

        # make sure 'state' and 'dispatcher' are ignored from the pickled object, even if
        # someone sets the ignore list themselves
        if ignore is not None and ignore:
            if isinstance(ignore, six.string_types):
                ignore = [ignore]
            ignore = [e for e in ignore if e] # make sure None and '' are not in the list
            ignore = list(set(['state', 'dispatcher']) | set(ignore))
        else:
            ignore = ['state', 'dispatcher']
        # TODO: the only difference between this save method and LdaModel's is the use of
        # "AuthorTopicModel" below. This should be an easy refactor.
        # Same goes for load method below.
        super(AuthorTopicModel, self).save(fname, *args, ignore=ignore, **kwargs)

    @classmethod
    def load(cls, fname, *args, **kwargs):
        """
        Load a previously saved object from file (also see `save`).

        Large arrays can be memmap'ed back as read-only (shared memory) by setting `mmap='r'`:

            >>> AuthorTopicModel.load(fname, mmap='r')

        """
        kwargs['mmap'] = kwargs.get('mmap', None)
        result = super(AuthorTopicModel, cls).load(fname, *args, **kwargs)
        state_fname = utils.smart_extension(fname, '.state')
        try:
            result.state = super(LdaModel, cls).load(state_fname, *args, **kwargs)
        except Exception as e:
            logging.warning("failed to load state from %s: %s", state_fname, e)
        return result
# endclass LdaModel
