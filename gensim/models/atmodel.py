#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2016 Olavur Mortensen <olavurmortensen@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Author-topic model in Python.

This module trains the author-topic model on documents and corresponding author-document
dictionaries. The training is online and is constant in memory w.r.t. the number of 
documents. The model is *not* constant in memory w.r.t. the number of authors.

The model can be updated with additional documents after taining has been completed. It is
also possible to continue training on the existing data.

The model is closely related to Latent Dirichlet Allocation. Usage of the AuthorTopicModel
class is likewise similar to the usage of the LdaModel class.

"""

# FIXME: at the moment the input corpus is treated as a list. It must be possible to treat 
# it as an MmCorpus. The reason for this is that the corpus must be indexable, so that it
# is possible to find out what authors correspond to a particular document (variables
# author2doc and doc2author). If the input corpus is a list, just keep treating it as a list.
# If the input document is an MmCorpus, just keep treating it as an MmCorpus. If the input 
# document is something else, for example some sort of iterable, it should be saved as an 
# MmCorpus (and it should be checked that it is actually indexable, i.e. corpus[d] is possible).

import pdb
from pdb import set_trace as st
from pprint import pprint

import logging
import numpy as np # for arrays, array broadcasting etc.
import numbers
from copy import deepcopy

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
    NOTE: distributed mode not available yet in the author-topic model. This AuthorTopicState
    object is kept so that when the time comes to imlement it, it will be easier.

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
    for a in authors_ids:
        author2doc[a] = []
        for d, a_ids in doc2author.items():
            if a in a_ids:
                author2doc[a].append(d)
    return author2doc

class AuthorTopicModel(LdaModel):
    """
    The constructor estimates the author-topic model parameters based
    on a training corpus:

    >>> model = AuthorTopicModel(corpus, num_topics=10, author2doc=author2doc)

    The model can be updated (trained) with new documents via

    >>> model.update(other_corpus, other_author2doc)

    Model persistency is achieved through its `load`/`save` methods.
    """

    def __init__(self, corpus=None, num_topics=100, id2word=None,
                author2doc=None, doc2author=None, id2author=None, var_lambda=None,
                 chunksize=2000, passes=1, update_every=1,
                 alpha='symmetric', eta='symmetric', decay=0.5, offset=1.0,
                 eval_every=10, iterations=50, gamma_threshold=0.001,
                 minimum_probability=0.01, random_state=None, ns_conf={},
                 minimum_phi_value=0.01, per_word_topics=False):
        """
        If the iterable corpus and one of author2doc/doc2author dictionaries are given,
        start training straight away. If not given, the model is left untrained 
        (presumably because you want to call `update()` manually).

        `num_topics` is the number of requested latent topics to be extracted from
        the training corpus.

        `id2word` is a mapping from word ids (integers) to words (strings). It is
        used to determine the vocabulary size, as well as for debugging and topic
        printing.

        `author2doc` is a dictionary where the keys are the names of authors, and the
        values are lists of documents that the author contributes to.

        `doc2author` is a dictionary where the keys are document IDs (indexes to corpus)
        and the values are lists of author names. I.e. this is the reverse mapping of
        `author2doc`. Only one of the two, `author2doc` and `doc2author` have to be
        supplied.

        `alpha` and `eta` are hyperparameters that affect sparsity of the author-topic
        (theta) and topic-word (lambda) distributions. Both default to a symmetric
        1.0/num_topics prior.

        `alpha` can be set to an explicit array = prior of your choice. It also
        support special values of 'asymmetric' and 'auto': the former uses a fixed
        normalized asymmetric 1.0/topicno prior, the latter learns an asymmetric
        prior directly from your data.

        `eta` can be a scalar for a symmetric prior over topic/word
        distributions, or a vector of shape num_words, which can be used to 
        impose (user defined) asymmetric priors over the word distribution. 
        It also supports the special value 'auto', which learns an asymmetric
        prior over words directly from your data. `eta` can also be a matrix
        of shape num_topics x num_words, which can be used to impose 
        asymmetric priors over the word distribution on a per-topic basis
        (can not be learned from data).

        Calculate and log perplexity estimate from the latest mini-batch every
        `eval_every` model updates. Set to None to disable perplexity estimation.

        `decay` and `offset` parameters are the same as Kappa and Tau_0 in
        Hoffman et al, respectively.

        `minimum_probability` controls filtering the topics returned for a document (bow).

        `random_state` can be a np.random.RandomState object or the seed for one

        Example:

        >>> model = AuthorTopicModel(corpus, num_topics=100, author2doc=author2doc)  # train model
        >>> model.update(corpus2) # update the author-topic model with additional documents

        >>> model = AuthorTopicModel(corpus, num_topics=50, author2doc=author2doc, alpha='auto', eval_every=5)  # train asymmetric alpha from data

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
            raise ValueError("cannot compute the author-topic model over an empty collection (no terms)")

        logger.info('Vocabulary consists of %d words.', self.num_terms)

        self.id2author = id2author
        self.author2doc = {}
        self.doc2author = {}
        self.corpus = []  # FIXME: should be either a list or an MmCorpus instance.

        self.distributed = distributed
        self.num_topics = num_topics
        self.num_authors = 0
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability 
        self.num_updates = 0
        self.total_docs = 0

        self.passes = passes
        self.update_every = update_every
        self.eval_every = eval_every
        self.minimum_phi_value = minimum_phi_value
        self.per_word_topics = per_word_topics

        self.author2id = {}
        self.id2author = {}

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
            # NOTE: distributed processing is not implemented for the author-topic model.
            pass

        # VB constants
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold

        # Initialize the variational distributions q(beta|lambda) and q(theta|gamma)
        self.state = AuthorTopicState(self.eta, (self.num_topics, self.num_terms), (self.num_authors, self.num_topics))
        self.state.sstats = self.random_state.gamma(100., 1. / 100., (self.num_topics, self.num_terms))
        self.expElogbeta = np.exp(dirichlet_expectation(self.state.sstats))

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None and (author2doc is not None or doc2author is not None):
            use_numpy = self.dispatcher is not None
            self.update(corpus, author2doc, doc2author, chunks_as_numpy=use_numpy)

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

    def inference(self, chunk, author2doc, doc2author, rhot, collect_sstats=False, chunk_doc_idx=None):
        """
        Given a chunk of sparse document vectors, update gamma (parameters
        controlling the topic weights) for each author corresponding to the
        documents in the chunk.

        The whole input chunk of document is assumed to fit in RAM; chunking of
        a large corpus must be done earlier in the pipeline.

        If `collect_sstats` is True, also collect sufficient statistics needed
        to update the model's topic-word distributions, and return a 2-tuple
        `(gamma_chunk, sstats)`. Otherwise, return `(gamma_chunk, None)`. 
        `gamma_cunk` is of shape `len(chunk_authors) x self.num_topics`, where 
        `chunk_authors` is the number of authors in the documents in the
        current chunk.

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

        # Stack all the computed gammas into this output array.
        gamma_chunk = np.zeros((0, self.num_topics))

        # Now, for each document d update gamma and phi w.r.t. all authors in those documents.
        for d, doc in enumerate(chunk):
            if chunk_doc_idx is not None:
                doc_no = chunk_doc_idx[d]
            else:
                doc_no = d
            # Get the IDs and counts of all the words in the current document.
            if doc and not isinstance(doc[0][0], six.integer_types):
                # make sure the term IDs are ints, otherwise np will get upset
                ids = [int(id) for id, _ in doc]
            else:
                ids = [id for id, _ in doc]
            cts = np.array([cnt for _, cnt in doc])

            # Get all the authors in the current document.
            authors_d = self.doc2author[doc_no]  # List of author names.
            authors_d = [self.author2id[a] for a in authors_d]  # Convert names to integer IDs.

            gammad = self.state.gamma[authors_d, :]  # gamma of document d before update.
            tilde_gamma = gammad.copy()  # gamma that will be updated.

            # Compute the expectation of the log of the Dirichlet parameters theta and beta.
            Elogthetad = dirichlet_expectation(tilde_gamma)
            expElogthetad = np.exp(Elogthetad)
            expElogbetad = self.expElogbeta[:, ids]

            # Compute the normalizing constant of phi for the current document.
            phinorm = self.compute_phinorm(ids, authors_d, expElogthetad, expElogbetad)

            # Iterate between gamma and phi until convergence
            for iteration in xrange(self.iterations):

                lastgamma = tilde_gamma.copy()

                # Update gamma.
                # phi is computed implicitly below,
                for ai, a in enumerate(authors_d):
                    tilde_gamma[ai, :] = self.alpha + len(self.author2doc[self.id2author[a]]) * expElogthetad[ai, :] * np.dot(cts / phinorm, expElogbetad.T)

                # Update gamma.
                # Interpolation between document d's "local" gamma (tilde_gamma),
                # and "global" gamma (gammad).
                tilde_gamma = (1 - rhot) * gammad + rhot * tilde_gamma

                # Update Elogtheta and Elogbeta, since gamma and lambda have been updated.
                Elogthetad = dirichlet_expectation(tilde_gamma)
                expElogthetad = np.exp(Elogthetad)

                # Update the normalizing constant in phi.
                phinorm = self.compute_phinorm(ids, authors_d, expElogthetad, expElogbetad)

                # Check for convergence.
                # Criterion is mean change in "local" gamma.
                meanchange_gamma = np.mean(abs(tilde_gamma - lastgamma))
                gamma_condition = meanchange_gamma < self.gamma_threshold
                if gamma_condition:
                    converged += 1
                    break
            # End of iterations loop.

            # Store the updated gammas in the model state.
            self.state.gamma[authors_d, :] = tilde_gamma

            # Stack the new gammas into the output array.
            gamma_chunk = np.vstack([gamma_chunk, tilde_gamma])

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
            # sstats[k, w] = \sum_d n_{dw} * \sum_a phi_{dwak}
            # = \sum_d n_{dw} * exp{Elogtheta_{ak} + Elogbeta_{kw}} / phinorm_{dw}.
            sstats *= self.expElogbeta
        return gamma_chunk, sstats

    def do_estep(self, chunk, author2doc, doc2author, rhot, state=None, chunk_doc_idx=None):
        """
        Perform inference on a chunk of documents, and accumulate the collected
        sufficient statistics in `state` (or `self.state` if None).

        """
        if state is None:
            state = self.state
        gamma, sstats = self.inference(chunk, author2doc, doc2author, rhot, collect_sstats=True, chunk_doc_idx=chunk_doc_idx)
        state.sstats += sstats
        state.numdocs += len(chunk)
        return gamma

    def log_perplexity(self, chunk, chunk_doc_idx=None, total_docs=None):
        """
        Calculate and return per-word likelihood bound, using the `chunk` of
        documents as evaluation corpus. Also output the calculated statistics. incl.
        perplexity=2^(-bound), to log at INFO level.

        """
        if total_docs is None:
            total_docs = len(chunk)
        corpus_words = sum(cnt for document in chunk for _, cnt in document)
        subsample_ratio = 1.0 * total_docs / len(chunk)
        perwordbound = self.bound(chunk, chunk_doc_idx, subsample_ratio=subsample_ratio) / (subsample_ratio * corpus_words)
        print(perwordbound)
        logger.info("%.3f per-word bound, %.1f perplexity estimate based on a held-out corpus of %i documents with %i words" %
                    (perwordbound, np.exp2(-perwordbound), len(chunk), corpus_words))
        return perwordbound

    def update(self, corpus=None, author2doc=None, doc2author=None, chunksize=None, decay=None, offset=None,
               passes=None, update_every=None, eval_every=None, iterations=None,
               gamma_threshold=None, chunks_as_numpy=False):
        """
        Train the model with new documents, by EM-iterating over `corpus` until
        the topics converge (or until the maximum number of allowed iterations
        is reached). `corpus` must be an iterable (repeatable stream of documents),

        This update also supports updating an already trained model (`self`)
        with new documents from `corpus`; the two models are then merged in
        proportion to the number of old vs. new documents. This feature is still
        experimental for non-stationary input streams.

        For stationary input (no topic drift in new documents), on the other hand,
        this equals the online update of Hoffman et al. and is guaranteed to
        converge for any `decay` in (0.5, 1.0>. Additionally, for smaller
        `corpus` sizes, an increasing `offset` may be beneficial (see
        Table 1 in Hoffman et al.)

        If update is called with authors that already exist in the model, it will
        resume training on not only new documents for that author, but also the 
        previously seen documents. This is necessary for those authors' topic
        distributions to converge.

        Every time `update(corpus, author2doc)` is called, the new documents are
        to appended to all the previously seen documents, and author2doc is
        combined with the previously seen authors.

        To resume training on all the data seen by the model, simply call 
        `update()`.

        It is not possible to add new authors to existing documents, as all
        documents in `corpus` are assumed to be new documents.

        Args:
            corpus (gensim corpus): The corpus with which the author-topic model should be updated.

            author2doc (dictionary): author to document mapping corresponding to indexes in input 
            corpus.

            doc2author (dictionary): document to author mapping corresponding to indexes in input 
            corpus.

            chunks_as_numpy (bool): Whether each chunk passed to `.inference` should be a np
                array of not. np can in some settings turn the term IDs
                into floats, these will be converted back into integers in
                inference, which incurs a performance hit. For distributed
                computing it may be desirable to keep the chunks as np
                arrays.

        For other parameter settings, see :class:`AuthorTopicModel` constructor.

        """

        # use parameters given in constructor, unless user explicitly overrode them
        if decay is None:
            decay = self.decay
        if offset is None:
            offset = self.offset
        if passes is None:
            passes = self.passes
        if update_every is None:
            update_every = self.update_every
        if eval_every is None:
            eval_every = self.eval_every
        if iterations is None:
            iterations = self.iterations
        if gamma_threshold is None:
            gamma_threshold = self.gamma_threshold

        # NOTE: it is not possible to add new authors to an existing document (all input documents are treated
        # as completely new documents). Perhaps this functionality could be implemented.
        # If it's absolutely necessary, the user can delete the documents that have new authors, and call update
        # on them with the new and old authors.

        if corpus is None:
            # Just keep training on the already available data.
            # Assumes self.update() has been called before with input documents and corresponding authors.
            assert self.total_docs > 0, 'update() was called with no documents to train on.'
            train_corpus_idx = [d for d in xrange(self.total_docs)]
            num_input_authors = len(self.author2doc)
        else:
            if doc2author is None and author2doc is None:
                raise ValueError('at least one of author2doc/doc2author must be specified, to establish input space dimensionality')

            # Avoid overwriting the user's dictionaries.
            author2doc = deepcopy(author2doc)
            doc2author = deepcopy(doc2author)

            # If either doc2author or author2doc is missing, construct them from the other.
            if doc2author is None:
                doc2author = construct_doc2author(corpus, author2doc)
            elif author2doc is None:
                author2doc = construct_author2doc(corpus, doc2author)

            # Number of authors that need to be updated.
            num_input_authors = len(author2doc)

            try:
                len_input_corpus = len(corpus)
            except:
                logger.warning("input corpus stream has no len(); counting documents")
                len_input_corpus = sum(1 for _ in corpus)
            if len_input_corpus == 0:
                logger.warning("AuthorTopicModel.update() called with an empty corpus")
                return

            self.total_docs += len_input_corpus

            # FIXME: don't treat the corpus as a list. It's either a list or an MmCorpus instance.
            # Perhaps if it is some sort of other iterable, it can be stored as an MmCorpus anyway.
            self.corpus.extend(corpus)

            # Obtain a list of new authors.
            new_authors = []
            for a in author2doc.keys():
                if not self.author2doc.get(a):
                    new_authors.append(a)

            num_new_authors = len(new_authors)

            # Add new authors do author2id/id2author dictionaries.
            for a_id, a_name in enumerate(new_authors):
                self.author2id[a_name] = a_id + self.num_authors
                self.id2author[a_id] = a_name

            # Increment the number of total authors seen.
            self.num_authors += num_new_authors

            # Initialize the variational distributions q(theta|gamma)
            gamma_new = self.random_state.gamma(100., 1. / 100., (num_new_authors, self.num_topics))
            self.state.gamma = np.vstack([self.state.gamma, gamma_new])

            # Combine author2doc with self.author2doc.
            # First, increment the document IDs by the number of previously seen documents.
            for a, doc_ids in author2doc.items():
                doc_ids = [d + self.total_docs - len_input_corpus for d in doc_ids]

            # For all authors in the input corpus, add the new documents.
            for a, doc_ids in author2doc.items():
                if self.author2doc.get(a):
                    # This is not a new author, append new documents.
                    self.author2doc[a].extend(doc_ids)
                else:
                    # This is a new author, create index.
                    self.author2doc[a] = doc_ids

            # Add all new documents to self.doc2author.
            for d, a_list in doc2author.items():
                self.doc2author[d] = a_list

            # Train on all documents of authors in input_corpus.
            train_corpus_idx = []
            for a in author2doc.keys():  # For all authors in input corpus.
                for doc_ids in self.author2doc.values():  # For all documents in total corpus.
                    train_corpus_idx.extend(doc_ids)

            # Make the list of training documents unique.
            train_corpus_idx = list(set(train_corpus_idx))


        # train_corpus_idx is only a list of indexes, so "len" is valid.
        lencorpus = len(train_corpus_idx)

        if chunksize is None:
            chunksize = min(lencorpus, self.chunksize)

        self.state.numdocs += lencorpus

        if update_every:
            updatetype = "online"
            updateafter = min(lencorpus, update_every * self.numworkers * chunksize)
        else:
            updatetype = "batch"
            updateafter = lencorpus
        evalafter = min(lencorpus, (eval_every or 0) * self.numworkers * chunksize)

        updates_per_pass = max(1, lencorpus / updateafter)
        logger.info("running %s author-topic training, %s topics, %s authors, %i passes over "
                    "the supplied corpus of %i documents, updating model once "
                    "every %i documents, evaluating perplexity every %i documents, "
                    "iterating %ix with a convergence threshold of %f",
                    updatetype, self.num_topics, num_input_authors, passes, lencorpus,
                        updateafter, evalafter, iterations,
                        gamma_threshold)

        if updates_per_pass * passes < 10:
            logger.warning("too few updates, training might not converge; consider "
                           "increasing the number of passes or iterations to improve accuracy")

        # rho is the "speed" of updating; TODO try other fncs
        # pass_ + num_updates handles increasing the starting t for each pass,
        # while allowing it to "reset" on the first pass of each update
        def rho():
            return pow(offset + pass_ + (self.num_updates / chunksize), -decay)

        for pass_ in xrange(passes):
            if self.dispatcher:
                logger.info('initializing %s workers' % self.numworkers)
                self.dispatcher.reset(self.state)
            else:
                # gamma is not needed in "other", thus its shape is (0, 0).
                other = AuthorTopicState(self.eta, self.state.sstats.shape, (0, 0))
            dirty = False

            reallen = 0
            for chunk_no, chunk_doc_idx in enumerate(utils.grouper(train_corpus_idx, chunksize, as_numpy=chunks_as_numpy)):
                chunk = [self.corpus[d] for d in chunk_doc_idx]
                reallen += len(chunk)  # keep track of how many documents we've processed so far

                if eval_every and ((reallen == lencorpus) or ((chunk_no + 1) % (eval_every * self.numworkers) == 0)):
                    # log_perplexity requires the indexes of the documents being evaluated, to know what authors 
                    # correspond to the documents.
                    self.log_perplexity(chunk, chunk_doc_idx, total_docs=lencorpus)

                if self.dispatcher:
                    # add the chunk to dispatcher's job queue, so workers can munch on it
                    logger.info('PROGRESS: pass %i, dispatching documents up to #%i/%i',
                                pass_, chunk_no * chunksize + len(chunk), lencorpus)
                    # this will eventually block until some jobs finish, because the queue has a small finite length
                    self.dispatcher.putjob(chunk)
                else:
                    logger.info('PROGRESS: pass %i, at document #%i/%i',
                                pass_, chunk_no * chunksize + len(chunk), lencorpus)
                    # do_estep requires the indexes of the documents being trained on, to know what authors 
                    # correspond to the documents.
                    gammat = self.do_estep(chunk, self.author2doc, self.doc2author, rho(), other, chunk_doc_idx)

                    if self.optimize_alpha:
                        self.update_alpha(gammat, rho())

                dirty = True
                del chunk

                # perform an M step. determine when based on update_every, don't do this after every chunk
                if update_every and (chunk_no + 1) % (update_every * self.numworkers) == 0:
                    if self.dispatcher:
                        # distributed mode: wait for all workers to finish
                        logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                        other = self.dispatcher.getstate()
                    self.do_mstep(rho(), other, pass_ > 0)
                    del other  # frees up memory

                    if self.dispatcher:
                        logger.info('initializing workers')
                        self.dispatcher.reset(self.state)
                    else:
                        other = AuthorTopicState(self.eta, self.state.sstats.shape, (0, 0))
                    dirty = False
            # endfor single corpus iteration
            if reallen != lencorpus:
                raise RuntimeError("input corpus size changed during training (don't use generators as input)")

            if dirty:
                # finish any remaining updates
                if self.dispatcher:
                    # distributed mode: wait for all workers to finish
                    logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                    other = self.dispatcher.getstate()
                self.do_mstep(rho(), other, pass_ > 0)
                del other
                dirty = False
        # endfor entire corpus update

    def do_mstep(self, rho, other, extra_pass=False):
        """
        M step: use linear interpolation between the existing topics and
        collected sufficient statistics in `other` to update the topics.

        """
        logger.debug("updating topics")
        # update self with the new blend; also keep track of how much did
        # the topics change through this update, to assess convergence
        diff = np.log(self.expElogbeta)
        self.state.blend(rho, other)
        diff -= self.state.get_Elogbeta()
        self.sync_state()

        # print out some debug info at the end of each EM iteration
        self.print_topics(5)
        logger.info("topic diff=%f, rho=%f", np.mean(np.abs(diff)), rho)

        if self.optimize_eta:
            self.update_eta(self.state.get_lambda(), rho)

        if not extra_pass:
            # only update if this isn't an additional pass
            self.num_updates += other.numdocs

    def bound(self, chunk, chunk_doc_idx=None, subsample_ratio=1.0, author2doc=None, doc2author=None):
        """
        Estimate the variational bound of documents from `corpus`:
        E_q[log p(corpus)] - E_q[log q(corpus)]

        `gamma` are the variational parameters on topic weights for each author
        document (=2d matrix=what comes out of `inference()`). 

        There are basically two use cases of this method:
        1. `chunk` is a subset of the training corpus, and `chunk_doc_idx` is provided,
        indicating the indexes of the documents in the training corpus.
        2. `chunk` is a test set (held-out data), `chunk_doc_idx` is not needed, but 
        author2doc and doc2author corresponding to this test set are provided. It is
        not recommended to call this method with data that has authors the model has 
        not seen; if this is the case, those documents will simply be discarded.

        To obtain the per-word bound, compute:
        >>> corpus_words = sum(cnt for document in corpus for _, cnt in document)
        >>> model.bound(corpus, author2doc=author2doc, doc2author=doc2author) / corpus_words

        """

        # NOTE: it may be possible to enable evaluation of documents with new authors. To
        # do this, self.inference() has to be altered so that it uses gamma = self.state.gamma[a, :]
        # if author a is already trained on, but initializes gamma randomly if author a is
        # not already in the model.

        _lambda = self.state.get_lambda()
        Elogbeta = dirichlet_expectation(_lambda)
        expElogbeta = np.exp(dirichlet_expectation(_lambda))

        if author2doc is None and doc2author is None:
            gamma = self.state.gamma
            chunk_idx = [d for d in xrange(len(chunk))]
            author2doc = self.author2doc
            doc2author = self.doc2author
        else:
            # Infer gamma based on input corpus.

            # Will be needed in self.inference().
            def rho():
                return pow(self.offset + self.passes + (self.num_updates / self.chunksize), -self.decay)

            # sstats are not collected, thus lambda is not updated
            gamma, _ = self.inference(chunk, author2doc, doc2author, rho())

            # Bound of held-out (test) data can only be computed with authors
            # that are already existing in the data.
            # Documents that contain new authors are discarded.
            num_docs_new_authors = 0
            chunk_idx = []
            for d in xrange(len(chunk)):
                authors_d = doc2author[d]
                doc_new_authors = False
                for a in authors_d:
                    if not self.author2doc.get(a):
                        doc_new_authors = True
                if not doc_new_authors:
                    chunk_idx.append(d)
                else:
                    num_docs_new_authors += 1
            if num_docs_new_authors > 0:
                logger.warning('bound() called with held-out data with new authors; discarding %d documents.' % (num_docs_new_authors))


        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(dirichlet_expectation(gamma))

        word_score = 0.0
        authors_set = set()  # Used in computing theta bound.
        theta_score = 0.0
        for d in chunk_idx:
            if author2doc is None:
                doc_no = chunk_doc_idx[d]
            else:
                doc_no = d
            doc = chunk[d]
            authors_d = doc2author[doc_no]
            authors_d = [self.author2id[a] for a in authors_d]
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

        # Compensate likelihood for when `chunk` above is only a sample of the whole corpus. This ensures
        # that the likelihood is always rougly on the same scale.
        word_score *= subsample_ratio

        # theta_score is rescaled in a similar fashion.
        # TODO: treat this in a more general way, similar to how it is done with word_score.
        theta_score *= self.num_authors / len(authors_set)

        # E[log p(beta | eta) - log q (beta | lambda)]
        beta_score = 0.0
        beta_score += np.sum((self.eta - _lambda) * Elogbeta)
        beta_score += np.sum(gammaln(_lambda) - gammaln(self.eta))
        sum_eta = np.sum(self.eta)
        beta_score += np.sum(gammaln(sum_eta) - gammaln(np.sum(_lambda, 1)))

        total_score = word_score + theta_score + beta_score

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

    def __getitem__(self, data):
        """
        `data` must be a list consisting of two elements: `bow` and `author_name`, described below.

        `bow` is a list of documents in BOW representation.

        `author_name` is the name of the author of the documents in `bow`.
        
        If `author_name`
        already exists in model (e.g. self.author2doc), the model will be updated w.r.t. all
        the documents that the author is responsible.

        """

        assert False, '__getitem__ (model[data]) is not ready for use.'

        # FIXME: it is not clear at all what a __getitem__ method should accomplish in the author-topic
        # model. In the attempt below, it assumed that multiple documents corresponding to a single 
        # author is passed to this method, and then update is called on that data. Then, get_author_topics
        # is called on the author.

        bow = data[0]
        author_name = data[1]

        # TODO: perhaps this method should assume author_name if it is not provided. This is problematic
        # if the author names are strings, though.

        assert author_name not in self.author2doc, '__getitem__ (model[data]) called on an existing author.'

        author2doc = {author_name: list(xrange(len(bow)))}

        self.update(bow, author2doc)

        return self.get_author_topics(self.author2id[author_name])

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
