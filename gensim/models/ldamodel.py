#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#


"""
**For a faster implementation of LDA (parallelized for multicore machines), see** :mod:`gensim.models.ldamulticore`.

Latent Dirichlet Allocation (LDA) in Python.

This module allows both LDA model estimation from a training corpus and inference of topic
distribution on new, unseen documents. The model can also be updated with new documents
for online training.

The core estimation code is based on the `onlineldavb.py` script by M. Hoffman [1]_, see
**Hoffman, Blei, Bach: Online Learning for Latent Dirichlet Allocation, NIPS 2010.**

The algorithm:

* is **streamed**: training documents may come in sequentially, no random access required,
* runs in **constant memory** w.r.t. the number of documents: size of the
  training corpus does not affect memory footprint, can process corpora larger than RAM, and
* is **distributed**: makes use of a cluster of machines, if available, to
  speed up model estimation.

.. [1] http://www.cs.princeton.edu/~mdhoffma

"""


import logging
import numpy as np  # for arrays, array broadcasting etc.
import numbers
import os

from gensim import interfaces, utils, matutils
from gensim.matutils import dirichlet_expectation
from gensim.models import basemodel

from itertools import chain
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from six.moves import xrange
import six

# log(sum(exp(x))) that tries to avoid overflow
try:
    # try importing from here if older scipy is installed
    from scipy.maxentropy import logsumexp
except ImportError:
    # maxentropy has been removed in recent releases, logsumexp now in misc
    from scipy.misc import logsumexp


logger = logging.getLogger('gensim.models.ldamodel')


def update_dir_prior(prior, N, logphat, rho):
    """
    Updates a given prior using Newton's method, described in
    **Huang: Maximum Likelihood Estimation of Dirichlet Distribution Parameters.**
    http://jonathan-huang.org/research/dirichlet/dirichlet.pdf
    """
    dprior = np.copy(prior)
    gradf = N * (psi(np.sum(prior)) - psi(prior) + logphat)

    c = N * polygamma(1, np.sum(prior))
    q = -N * polygamma(1, prior)

    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))

    dprior = -(gradf - b) / q

    if all(rho * dprior + prior > 0):
        prior += rho * dprior
    else:
        logger.warning("updated prior not positive")

    return prior


class LdaState(utils.SaveLoad):
    """
    Encapsulate information for distributed computation of LdaModel objects.

    Objects of this class are sent over the network, so try to keep them lean to
    reduce traffic.

    """
    def __init__(self, eta, shape):
        self.eta = eta
        self.sstats = np.zeros(shape)
        self.numdocs = 0

    def reset(self):
        """
        Prepare the state for a new EM iteration (reset sufficient stats).

        """
        self.sstats[:] = 0.0
        self.numdocs = 0

    def merge(self, other):
        """
        Merge the result of an E step from one node with that of another node
        (summing up sufficient statistics).

        The merging is trivial and after merging all cluster nodes, we have the
        exact same result as if the computation was run on a single node (no
        approximation).

        """
        assert other is not None
        self.sstats += other.sstats
        self.numdocs += other.numdocs

    def blend(self, rhot, other, targetsize=None):
        """
        Given LdaState `other`, merge it with the current state. Stretch both to
        `targetsize` documents before merging, so that they are of comparable
        magnitude.

        Merging is done by average weighting: in the extremes, `rhot=0.0` means
        `other` is completely ignored; `rhot=1.0` means `self` is completely ignored.

        This procedure corresponds to the stochastic gradient update from Hoffman
        et al., algorithm 2 (eq. 14).

        """
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        # stretch the current model's expected n*phi counts to target size
        if self.numdocs == 0 or targetsize == self.numdocs:
            scale = 1.0
        else:
            scale = 1.0 * targetsize / self.numdocs
        self.sstats *= (1.0 - rhot) * scale

        # stretch the incoming n*phi counts to target size
        if other.numdocs == 0 or targetsize == other.numdocs:
            scale = 1.0
        else:
            logger.info("merging changes from %i documents into a model of %i documents",
                        other.numdocs, targetsize)
            scale = 1.0 * targetsize / other.numdocs
        self.sstats += rhot * scale * other.sstats

        self.numdocs = targetsize

    def blend2(self, rhot, other, targetsize=None):
        """
        Alternative, more simple blend.
        """
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        # merge the two matrices by summing
        self.sstats += other.sstats
        self.numdocs = targetsize

    def get_lambda(self):
        return self.eta + self.sstats

    def get_Elogbeta(self):
        return dirichlet_expectation(self.get_lambda())
# endclass LdaState


class LdaModel(interfaces.TransformationABC, basemodel.BaseTopicModel):
    """
    The constructor estimates Latent Dirichlet Allocation model parameters based
    on a training corpus:

    >>> lda = LdaModel(corpus, num_topics=10)

    You can then infer topic distributions on new, unseen documents, with

    >>> doc_lda = lda[doc_bow]

    The model can be updated (trained) with new documents via

    >>> lda.update(other_corpus)

    Model persistency is achieved through its `load`/`save` methods.
    """
    def __init__(self, corpus=None, num_topics=100, id2word=None,
                 distributed=False, chunksize=2000, passes=1, update_every=1,
                 alpha='symmetric', eta=None, decay=0.5, offset=1.0,
                 eval_every=10, iterations=50, gamma_threshold=0.001,
                 minimum_probability=0.01, random_state=None, ns_conf={},
                 minimum_phi_value=0.01, per_word_topics=False):
        """
        If given, start training from the iterable `corpus` straight away. If not given,
        the model is left untrained (presumably because you want to call `update()` manually).

        `num_topics` is the number of requested latent topics to be extracted from
        the training corpus.

        `id2word` is a mapping from word ids (integers) to words (strings). It is
        used to determine the vocabulary size, as well as for debugging and topic
        printing.

        `alpha` and `eta` are hyperparameters that affect sparsity of the document-topic
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

        Turn on `distributed` to force distributed computing (see the `web tutorial <http://radimrehurek.com/gensim/distributed.html>`_
        on how to set up a cluster of machines for gensim).

        Calculate and log perplexity estimate from the latest mini-batch every
        `eval_every` model updates (setting this to 1 slows down training ~2x;
        default is 10 for better performance). Set to None to disable perplexity estimation.

        `decay` and `offset` parameters are the same as Kappa and Tau_0 in
        Hoffman et al, respectively.

        `minimum_probability` controls filtering the topics returned for a document (bow).

        `random_state` can be a np.random.RandomState object or the seed for one

        Example:

        >>> lda = LdaModel(corpus, num_topics=100)  # train model
        >>> print(lda[doc_bow]) # get topic probability distribution for a document
        >>> lda.update(corpus2) # update the LDA model with additional documents
        >>> print(lda[doc_bow])

        >>> lda = LdaModel(corpus, num_topics=50, alpha='auto', eval_every=5)  # train asymmetric alpha from data

        """

        # store user-supplied parameters
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

        self.distributed = bool(distributed)
        self.num_topics = int(num_topics)
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

        self.alpha, self.optimize_alpha = self.init_dir_prior(alpha, 'alpha')

        assert self.alpha.shape == (self.num_topics,), "Invalid alpha shape. Got shape %s, but expected (%d, )" % (str(self.alpha.shape), self.num_topics)

        if isinstance(eta, six.string_types):
            if eta == 'asymmetric':
                raise ValueError("The 'asymmetric' option cannot be used for eta")

        self.eta, self.optimize_eta = self.init_dir_prior(eta, 'eta')

        self.random_state = utils.get_random_state(random_state)

        assert (self.eta.shape == (self.num_terms,) or self.eta.shape == (self.num_topics, self.num_terms)), (
                "Invalid eta shape. Got shape %s, but expected (%d, 1) or (%d, %d)" %
                (str(self.eta.shape), self.num_terms, self.num_topics, self.num_terms))

        # VB constants
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold

        # set up distributed environment if necessary
        if not distributed:
            logger.info("using serial LDA version on this node")
            self.dispatcher = None
            self.numworkers = 1
        else:
            if self.optimize_alpha:
                raise NotImplementedError("auto-optimizing alpha not implemented in distributed LDA")
            # set up distributed version
            try:
                import Pyro4
                with utils.getNS(**ns_conf) as ns:
                    from gensim.models.lda_dispatcher import LDA_DISPATCHER_PREFIX
                    self.dispatcher = Pyro4.Proxy(ns.list(prefix=LDA_DISPATCHER_PREFIX)[LDA_DISPATCHER_PREFIX])
                    logger.debug("looking for dispatcher at %s" % str(self.dispatcher._pyroUri))
                    self.dispatcher.initialize(id2word=self.id2word, num_topics=self.num_topics,
                                               chunksize=chunksize, alpha=alpha, eta=eta, distributed=False)
                    self.numworkers = len(self.dispatcher.getworkers())
                    logger.info("using distributed version with %i workers" % self.numworkers)
            except Exception as err:
                logger.error("failed to initialize distributed LDA (%s)", err)
                raise RuntimeError("failed to initialize distributed LDA (%s)" % err)

        # Initialize the variational distribution q(beta|lambda)
        self.state = LdaState(self.eta, (self.num_topics, self.num_terms))
        self.state.sstats = self.random_state.gamma(100., 1. / 100., (self.num_topics, self.num_terms))
        self.expElogbeta = np.exp(dirichlet_expectation(self.state.sstats))

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            use_numpy = self.dispatcher is not None
            self.update(corpus, chunks_as_numpy=use_numpy)

    def init_dir_prior(self, prior, name):
        if prior is None:
            prior = 'symmetric'

        if name == 'alpha':
            prior_shape = self.num_topics
        elif name == 'eta':
            prior_shape = self.num_terms
        else:
            raise ValueError("'name' must be 'alpha' or 'eta'")

        is_auto = False

        if isinstance(prior, six.string_types):
            if prior == 'symmetric':
                logger.info("using symmetric %s at %s", name, 1.0 / prior_shape)
                init_prior = np.asarray([1.0 / self.num_topics for i in xrange(prior_shape)])
            elif prior == 'asymmetric':
                init_prior = np.asarray([1.0 / (i + np.sqrt(prior_shape)) for i in xrange(prior_shape)])
                init_prior /= init_prior.sum()
                logger.info("using asymmetric %s %s", name, list(init_prior))
            elif prior == 'auto':
                is_auto = True
                init_prior = np.asarray([1.0 / self.num_topics for i in xrange(prior_shape)])
                if name == 'alpha':
                    logger.info("using autotuned %s, starting with %s", name, list(init_prior))
            else:
                raise ValueError("Unable to determine proper %s value given '%s'" % (name, prior))
        elif isinstance(prior, list):
            init_prior = np.asarray(prior)
        elif isinstance(prior, np.ndarray):
            init_prior = prior
        elif isinstance(prior, np.number) or isinstance(prior, numbers.Real):
            init_prior = np.asarray([prior] * prior_shape)
        else:
            raise ValueError("%s must be either a np array of scalars, list of scalars, or scalar" % name)

        return init_prior, is_auto

    def __str__(self):
        return "LdaModel(num_terms=%s, num_topics=%s, decay=%s, chunksize=%s)" % \
            (self.num_terms, self.num_topics, self.decay, self.chunksize)

    def sync_state(self):
        self.expElogbeta = np.exp(self.state.get_Elogbeta())

    def clear(self):
        """Clear model state (free up some memory). Used in the distributed algo."""
        self.state = None
        self.Elogbeta = None

    def inference(self, chunk, collect_sstats=False):
        """
        Given a chunk of sparse document vectors, estimate gamma (parameters
        controlling the topic weights) for each document in the chunk.

        This function does not modify the model (=is read-only aka const). The
        whole input chunk of document is assumed to fit in RAM; chunking of a
        large corpus must be done earlier in the pipeline.

        If `collect_sstats` is True, also collect sufficient statistics needed
        to update the model's topic-word distributions, and return a 2-tuple
        `(gamma, sstats)`. Otherwise, return `(gamma, None)`. `gamma` is of shape
        `len(chunk) x self.num_topics`.

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
        gamma = self.random_state.gamma(100., 1. / 100., (len(chunk), self.num_topics))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        if collect_sstats:
            sstats = np.zeros_like(self.expElogbeta)
        else:
            sstats = None
        converged = 0

        # Now, for each document d update that document's gamma and phi
        # Inference code copied from Hoffman's `onlineldavb.py` (esp. the
        # Lee&Seung trick which speeds things up by an order of magnitude, compared
        # to Blei's original LDA-C code, cool!).
        for d, doc in enumerate(chunk):
            if doc and not isinstance(doc[0][0], six.integer_types):
                # make sure the term IDs are ints, otherwise np will get upset
                ids = [int(id) for id, _ in doc]
            else:
                ids = [id for id, _ in doc]
            cts = np.array([cnt for _, cnt in doc])
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self.expElogbeta[:, ids]

            # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_w.
            # phinorm is the normalizer.
            # TODO treat zeros explicitly, instead of adding 1e-100?
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

            # Iterate between gamma and phi until convergence
            for _ in xrange(self.iterations):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self.alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = np.mean(abs(gammad - lastgamma))
                if (meanchange < self.gamma_threshold):
                    converged += 1
                    break
            gamma[d, :] = gammad
            if collect_sstats:
                # Contribution of document d to the expected sufficient
                # statistics for the M step.
                sstats[:, ids] += np.outer(expElogthetad.T, cts / phinorm)

        if len(chunk) > 1:
            logger.debug("%i/%i documents converged within %i iterations",
                         converged, len(chunk), self.iterations)

        if collect_sstats:
            # This step finishes computing the sufficient statistics for the
            # M step, so that
            # sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
            # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
            sstats *= self.expElogbeta
        return gamma, sstats

    def do_estep(self, chunk, state=None):
        """
        Perform inference on a chunk of documents, and accumulate the collected
        sufficient statistics in `state` (or `self.state` if None).

        """
        if state is None:
            state = self.state
        gamma, sstats = self.inference(chunk, collect_sstats=True)
        state.sstats += sstats
        state.numdocs += gamma.shape[0]  # avoids calling len(chunk) on a generator
        return gamma

    def update_alpha(self, gammat, rho):
        """
        Update parameters for the Dirichlet prior on the per-document
        topic weights `alpha` given the last `gammat`.
        """
        N = float(len(gammat))
        logphat = sum(dirichlet_expectation(gamma) for gamma in gammat) / N

        self.alpha = update_dir_prior(self.alpha, N, logphat, rho)
        logger.info("optimized alpha %s", list(self.alpha))

        return self.alpha

    def update_eta(self, lambdat, rho):
        """
        Update parameters for the Dirichlet prior on the per-topic
        word weights `eta` given the last `lambdat`.
        """
        N = float(lambdat.shape[0])
        logphat = (sum(dirichlet_expectation(lambda_) for lambda_ in lambdat) / N).reshape((self.num_terms,))

        self.eta = update_dir_prior(self.eta, N, logphat, rho)

        return self.eta

    def log_perplexity(self, chunk, total_docs=None):
        """
        Calculate and return per-word likelihood bound, using the `chunk` of
        documents as evaluation corpus. Also output the calculated statistics. incl.
        perplexity=2^(-bound), to log at INFO level.

        """
        if total_docs is None:
            total_docs = len(chunk)
        corpus_words = sum(cnt for document in chunk for _, cnt in document)
        subsample_ratio = 1.0 * total_docs / len(chunk)
        perwordbound = self.bound(chunk, subsample_ratio=subsample_ratio) / (subsample_ratio * corpus_words)
        logger.info("%.3f per-word bound, %.1f perplexity estimate based on a held-out corpus of %i documents with %i words" %
                    (perwordbound, np.exp2(-perwordbound), len(chunk), corpus_words))
        return perwordbound

    def update(self, corpus, chunksize=None, decay=None, offset=None,
               passes=None, update_every=None, eval_every=None, iterations=None,
               gamma_threshold=None, chunks_as_numpy=False):
        """
        Train the model with new documents, by EM-iterating over `corpus` until
        the topics converge (or until the maximum number of allowed iterations
        is reached). `corpus` must be an iterable (repeatable stream of documents),

        In distributed mode, the E step is distributed over a cluster of machines.

        This update also supports updating an already trained model (`self`)
        with new documents from `corpus`; the two models are then merged in
        proportion to the number of old vs. new documents. This feature is still
        experimental for non-stationary input streams.

        For stationary input (no topic drift in new documents), on the other hand,
        this equals the online update of Hoffman et al. and is guaranteed to
        converge for any `decay` in (0.5, 1.0>. Additionally, for smaller
        `corpus` sizes, an increasing `offset` may be beneficial (see
        Table 1 in Hoffman et al.)

        Args:
            corpus (gensim corpus): The corpus with which the LDA model should be updated.

            chunks_as_numpy (bool): Whether each chunk passed to `.inference` should be a np
                array of not. np can in some settings turn the term IDs
                into floats, these will be converted back into integers in
                inference, which incurs a performance hit. For distributed
                computing it may be desirable to keep the chunks as np
                arrays.

        For other parameter settings, see :class:`LdaModel` constructor.

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

        try:
            lencorpus = len(corpus)
        except:
            logger.warning("input corpus stream has no len(); counting documents")
            lencorpus = sum(1 for _ in corpus)
        if lencorpus == 0:
            logger.warning("LdaModel.update() called with an empty corpus")
            return

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
        logger.info("running %s LDA training, %s topics, %i passes over "
                    "the supplied corpus of %i documents, updating model once "
                    "every %i documents, evaluating perplexity every %i documents, "
                    "iterating %ix with a convergence threshold of %f",
                    updatetype, self.num_topics, passes, lencorpus,
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
                other = LdaState(self.eta, self.state.sstats.shape)
            dirty = False

            reallen = 0
            for chunk_no, chunk in enumerate(utils.grouper(corpus, chunksize, as_numpy=chunks_as_numpy)):
                reallen += len(chunk)  # keep track of how many documents we've processed so far

                if eval_every and ((reallen == lencorpus) or ((chunk_no + 1) % (eval_every * self.numworkers) == 0)):
                    self.log_perplexity(chunk, total_docs=lencorpus)

                if self.dispatcher:
                    # add the chunk to dispatcher's job queue, so workers can munch on it
                    logger.info('PROGRESS: pass %i, dispatching documents up to #%i/%i',
                                pass_, chunk_no * chunksize + len(chunk), lencorpus)
                    # this will eventually block until some jobs finish, because the queue has a small finite length
                    self.dispatcher.putjob(chunk)
                else:
                    logger.info('PROGRESS: pass %i, at document #%i/%i',
                                pass_, chunk_no * chunksize + len(chunk), lencorpus)
                    gammat = self.do_estep(chunk, other)

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
                        other = LdaState(self.eta, self.state.sstats.shape)
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

    def bound(self, corpus, gamma=None, subsample_ratio=1.0):
        """
        Estimate the variational bound of documents from `corpus`:
        E_q[log p(corpus)] - E_q[log q(corpus)]

        `gamma` are the variational parameters on topic weights for each `corpus`
        document (=2d matrix=what comes out of `inference()`).
        If not supplied, will be inferred from the model.

        """
        score = 0.0
        _lambda = self.state.get_lambda()
        Elogbeta = dirichlet_expectation(_lambda)

        for d, doc in enumerate(corpus):  # stream the input doc-by-doc, in case it's too large to fit in RAM
            if d % self.chunksize == 0:
                logger.debug("bound: at document #%i", d)
            if gamma is None:
                gammad, _ = self.inference([doc])
            else:
                gammad = gamma[d]
            Elogthetad = dirichlet_expectation(gammad)

            # E[log p(doc | theta, beta)]
            score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, id]) for id, cnt in doc)

            # E[log p(theta | alpha) - log q(theta | gamma)]; assumes alpha is a vector
            score += np.sum((self.alpha - gammad) * Elogthetad)
            score += np.sum(gammaln(gammad) - gammaln(self.alpha))
            score += gammaln(np.sum(self.alpha)) - gammaln(np.sum(gammad))

        # Compensate likelihood for when `corpus` above is only a sample of the whole corpus. This ensures
        # that the likelihood is always rougly on the same scale.
        score *= subsample_ratio

        # E[log p(beta | eta) - log q (beta | lambda)]; assumes eta is a scalar
        score += np.sum((self.eta - _lambda) * Elogbeta)
        score += np.sum(gammaln(_lambda) - gammaln(self.eta))

        if np.ndim(self.eta) == 0:
            sum_eta = self.eta * self.num_terms
        else:
            sum_eta = np.sum(self.eta)

        score += np.sum(gammaln(sum_eta) - gammaln(np.sum(_lambda, 1)))

        return score

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        """
        For `num_topics` number of topics, return `num_words` most significant words
        (10 words per topic, by default).

        The topics are returned as a list -- a list of strings if `formatted` is
        True, or a list of `(word, probability)` 2-tuples if False.

        If `log` is True, also output this result to log.

        Unlike LSA, there is no natural ordering between the topics in LDA.
        The returned `num_topics <= self.num_topics` subset of all topics is therefore
        arbitrary and may change between two LDA training runs.

        """
        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics = self.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics = min(num_topics, self.num_topics)

            # add a little random jitter, to randomize results around the same alpha
            sort_alpha = self.alpha + 0.0001 * self.random_state.rand(len(self.alpha))

            sorted_topics = list(matutils.argsort(sort_alpha))
            chosen_topics = sorted_topics[:num_topics // 2] + sorted_topics[-num_topics // 2:]

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
        Return a list of `(word, probability)` 2-tuples for the most probable
        words in topic `topicid`.

        Only return 2-tuples for the topn most probable words (ignore the rest).

        """
        return [(self.id2word[id], value) for id, value in self.get_topic_terms(topicid, topn)]

    def get_topic_terms(self, topicid, topn=10):
        """
        Return a list of `(word_id, probability)` 2-tuples for the most
        probable words in topic `topicid`.

        Only return 2-tuples for the topn most probable words (ignore the rest).

        """
        topic = self.state.get_lambda()[topicid]
        topic = topic / topic.sum()  # normalize to probability distribution
        bestn = matutils.argsort(topic, topn, reverse=True)
        return [(id, topic[id]) for id in bestn]

    def top_topics(self, corpus, num_words=20):
        """
        Calculate the Umass topic coherence for each topic. Algorithm from
        **Mimno, Wallach, Talley, Leenders, McCallum: Optimizing Semantic Coherence in Topic Models, CEMNLP 2011.**
        """
        is_corpus, corpus = utils.is_corpus(corpus)
        if not is_corpus:
            logger.warning("LdaModel.top_topics() called with an empty corpus")
            return

        topics = []
        str_topics = []
        for topic in self.state.get_lambda():
            topic = topic / topic.sum()  # normalize to probability distribution
            bestn = matutils.argsort(topic, topn=num_words, reverse=True)
            topics.append(bestn)
            beststr = [(topic[id], self.id2word[id]) for id in bestn]
            str_topics.append(beststr)

        # top_ids are limited to every topics top words. should not exceed the
        # vocabulary size.
        top_ids = set(chain.from_iterable(topics))

        # create a document occurence sparse matrix for each word
        doc_word_list = {}
        for id in top_ids:
            id_list = set()
            for n, document in enumerate(corpus):
                if id in frozenset(x[0] for x in document):
                    id_list.add(n)

            doc_word_list[id] = id_list

        coherence_scores = []
        for t, top_words in enumerate(topics):
            # Calculate each coherence score C(t, top_words)
            coherence = 0.0
            # Sum of top words m=2..M
            for m in top_words[1:]:
                # m_docs is v_m^(t)
                m_docs = doc_word_list[m]
                m_index = np.where(top_words == m)[0][0]

                # Sum of top words l=1..m
                # i.e., all words ranked higher than the current word m
                for l in top_words[:m_index]:
                    # l_docs is v_l^(t)
                    l_docs = doc_word_list[l]

                    # make sure this word appears in some documents.
                    if len(l_docs) > 0:
                        # co_doc_frequency is D(v_m^(t), v_l^(t))
                        co_doc_frequency = len(m_docs.intersection(l_docs))

                        # add to the coherence sum for these two words m, l
                        coherence += np.log((co_doc_frequency + 1.0) / len(l_docs))

            coherence_scores.append((str_topics[t], coherence))

        top_topics = sorted(coherence_scores, key=lambda t: t[1], reverse=True)
        return top_topics

    def get_document_topics(self, bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False):
        """
        Return topic distribution for the given document `bow`, as a list of
        (topic_id, topic_probability) 2-tuples.

        Ignore topics with very low probability (below `minimum_probability`).

        If per_word_topics is True, it also returns a list of topics, sorted in descending order of most likely topics for that word.
        It also returns a list of word_ids and each words corresponding topics' phi_values, multiplied by feature length (i.e, word count)

        """
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)  # never allow zero values in sparse output

        if minimum_phi_value is None:
            minimum_phi_value = self.minimum_probability
        minimum_phi_value = max(minimum_phi_value, 1e-8)  # never allow zero values in sparse output

        # if the input vector is a corpus, return a transformed corpus
        is_corpus, corpus = utils.is_corpus(bow)
        if is_corpus:
            kwargs = dict(
                per_word_topics = per_word_topics,
                minimum_probability = minimum_probability,
                minimum_phi_value = minimum_phi_value
            )
            return self._apply(corpus, **kwargs)

        gamma, phis = self.inference([bow], collect_sstats=True)
        topic_dist = gamma[0] / sum(gamma[0])  # normalize distribution

        document_topics = [(topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)
                    if topicvalue >= minimum_probability]

        if not per_word_topics:
            return document_topics
        else:
            word_topic = [] # contains word and corresponding topic
            word_phi = [] # contains word and phi values
            for word_type, weight in bow:
                phi_values = [] # contains (phi_value, topic) pairing to later be sorted
                phi_topic = [] # contains topic and corresponding phi value to be returned 'raw' to user
                for topic_id in range(0, self.num_topics):
                    if phis[topic_id][word_type] >= minimum_phi_value:
                        # appends phi values for each topic for that word
                        # these phi values are scaled by feature length
                        phi_values.append((phis[topic_id][word_type], topic_id))
                        phi_topic.append((topic_id, phis[topic_id][word_type]))

                # list with ({word_id => [(topic_0, phi_value), (topic_1, phi_value) ...]).
                word_phi.append((word_type, phi_topic))
                # sorts the topics based on most likely topic
                # returns a list like ({word_id => [topic_id_most_probable, topic_id_second_most_probable, ...]).
                sorted_phi_values = sorted(phi_values, reverse=True)
                topics_sorted = [x[1] for x in sorted_phi_values]
                word_topic.append((word_type, topics_sorted))
            return (document_topics, word_topic, word_phi) # returns 2-tuple

    def get_term_topics(self, word_id, minimum_probability=None):
        """
        Returns most likely topics for a particular word in vocab.

        """
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)  # never allow zero values in sparse output

        # if user enters word instead of id in vocab, change to get id
        if isinstance(word_id, str):
            word_id = self.id2word.doc2bow([word_id])[0][0]

        values = []
        for topic_id in range(0, self.num_topics):
            if self.expElogbeta[topic_id][word_id] >= minimum_probability:
                values.append((topic_id, self.expElogbeta[topic_id][word_id]))

        return values


    def __getitem__(self, bow, eps=None):
        """
        Return topic distribution for the given document `bow`, as a list of
        (topic_id, topic_probability) 2-tuples.

        Ignore topics with very low probability (below `eps`).

        """
        return self.get_document_topics(bow, eps, self.minimum_phi_value, self.per_word_topics)

    def save(self, fname, ignore=['state', 'dispatcher'], separately=None, *args, **kwargs):
        """
        Save the model to file.

        Large internal arrays may be stored into separate files, with `fname` as prefix.

        `separately` can be used to define which arrays should be stored in separate files.

        `ignore` parameter can be used to define which variables should be ignored, i.e. left
        out from the pickled lda model. By default the internal `state` is ignored as it uses
        its own serialisation not the one provided by `LdaModel`. The `state` and `dispatcher`
        will be added to any ignore parameter defined.


        Note: do not save as a compressed file if you intend to load the file back with `mmap`.

        Note: If you intend to use models across Python 2/3 versions there are a few things to
        keep in mind:

          1. The pickled Python dictionaries will not work across Python versions
          2. The `save` method does not automatically save all np arrays using np, only
             those ones that exceed `sep_limit` set in `gensim.utils.SaveLoad.save`. The main
             concern here is the `alpha` array if for instance using `alpha='auto'`.

        Please refer to the wiki recipes section (https://github.com/piskvorky/gensim/wiki/Recipes-&-FAQ#q9-how-do-i-load-a-model-in-python-3-that-was-trained-and-saved-using-python-2)
        for an example on how to work around these issues.
        """
        if self.state is not None:
            self.state.save(utils.smart_extension(fname, '.state'), *args, **kwargs)
        # Save the dictionary separately if not in 'ignore'.
        if 'id2word' not in ignore:
            utils.pickle(self.id2word, utils.smart_extension(fname, '.id2word'))

        # make sure 'state', 'id2word' and 'dispatcher' are ignored from the pickled object, even if 
        # someone sets the ignore list themselves
        if ignore is not None and ignore:
            if isinstance(ignore, six.string_types):
                ignore = [ignore]
            ignore = [e for e in ignore if e] # make sure None and '' are not in the list
            ignore = list(set(['state', 'dispatcher', 'id2word']) | set(ignore))
        else:
            ignore = ['state', 'dispatcher', 'id2word']
        
        # make sure 'expElogbeta' and 'sstats' are ignored from the pickled object, even if
        # someone sets the separately list themselves.
        separately_explicit = ['expElogbeta', 'sstats']
        # Also add 'alpha' and 'eta' to separately list if they are set 'auto' or some
        # array manually.
        if (isinstance(self.alpha, six.string_types) and self.alpha == 'auto') or len(self.alpha.shape) != 1:
            separately_explicit.append('alpha')
        if (isinstance(self.eta, six.string_types) and self.eta == 'auto') or len(self.eta.shape) != 1:
            separately_explicit.append('eta')
        # Merge separately_explicit with separately.
        if separately:
            if isinstance(separately, six.string_types):
                separately = [separately]
            separately = [e for e in separately if e] # make sure None and '' are not in the list
            separately = list(set(separately_explicit) | set(separately))
        else:
            separately = separately_explicit
        super(LdaModel, self).save(fname, ignore=ignore, separately = separately, *args, **kwargs)
       
    @classmethod
    def load(cls, fname, *args, **kwargs):
        """
        Load a previously saved object from file (also see `save`).

        Large arrays can be memmap'ed back as read-only (shared memory) by setting `mmap='r'`:

            >>> LdaModel.load(fname, mmap='r')

        """
        kwargs['mmap'] = kwargs.get('mmap', None)
        result = super(LdaModel, cls).load(fname, *args, **kwargs)

        random_state_fname = utils.smart_extension(fname, '.random_state')
        try:
            result.random_state = super(LdaModel, cls).load(random_state_fname, *args, **kwargs)
        except Exception as e:
            logging.warning("failed to load random_state from %s: %s", random_state_fname, e)
        state_fname = utils.smart_extension(fname, '.state')
        try:
            result.state = super(LdaModel, cls).load(state_fname, *args, **kwargs)
        except Exception as e:
            logging.warning("failed to load state from %s: %s", state_fname, e)
        id2word_fname = utils.smart_extension(fname, '.id2word')
        if (os.path.isfile(id2word_fname)):
            try:
                result.id2word = utils.unpickle(id2word_fname)
            except Exception as e:
                logging.warning("failed to load id2word dictionary from %s: %s", id2word_fname, e)
        else:
            result.id2word = None
        return result
# endclass LdaModel
