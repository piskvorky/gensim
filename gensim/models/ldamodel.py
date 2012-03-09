#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# Parts of the LDA inference code come from Dr. Hoffman's `onlineldavb.py` script,
# (C) 2010  Matthew D. Hoffman, GNU GPL 3.0


"""
This module encapsulates functionality for the Latent Dirichlet Allocation algorithm.

It allows both model estimation from a training corpus and inference of topic
distribution on new, unseen documents.

The core estimation code is directly adapted from the `onlineldavb.py` script
by M. Hoffman [1]_, see
**Hoffman, Blei, Bach: Online Learning for Latent Dirichlet Allocation, NIPS 2010.**

The algorithm:

  * is **streamed**: training documents come in sequentially, no random access,
  * runs in **constant memory** w.r.t. the number of documents: size of the
    training corpus does not affect memory footprint, and
  * is **distributed**: makes use of a cluster of machines, if available, to
    speed up model estimation.

.. [1] http://www.cs.princeton.edu/~mdhoffma

"""


import logging
import itertools

logger = logging.getLogger('gensim.models.ldamodel')


import numpy # for arrays, array broadcasting etc.
#numpy.seterr(divide='ignore') # ignore 0*log(0) errors

from scipy.special import gammaln, digamma, psi # gamma function utils
try:
    from scipy.maxentropy import logsumexp # log(sum(exp(x))) that tries to avoid overflow
except ImportError: # maxentropy has been removed for next release
    from scipy.misc import logsumexp
from gensim import interfaces, utils




def dirichlet_expectation(alpha):
    """
    For a vector `theta~Dir(alpha)`, compute `E[log(theta)]`.
    """
    if (len(alpha.shape) == 1):
        result = psi(alpha) - psi(numpy.sum(alpha))
    else:
        result = psi(alpha) - psi(numpy.sum(alpha, 1))[:, numpy.newaxis]
    return result.astype(alpha.dtype) # keep the same precision as input



class LdaState(utils.SaveLoad):
    """
    Encapsulate information for distributed computation of LdaModel objects.

    Objects of this class are sent over the network, so try to keep them lean to
    reduce traffic.
    """
    def __init__(self, eta, shape):
        self.eta = eta
        self.sstats = numpy.zeros(shape)
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
            logger.info("merging changes from %i documents into a model of %i documents" %
                        (other.numdocs, targetsize))
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
#endclass LdaState



class LdaModel(interfaces.TransformationABC):
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
    def __init__(self, corpus=None, num_topics=100, id2word=None, distributed=False,
                 chunksize=2000, passes=1, update_every=1, alpha=None, eta=None, decay=0.5):
        """
        `num_topics` is the number of requested latent topics to be extracted from
        the training corpus.

        `id2word` is a mapping from word ids (integers) to words (strings). It is
        used to determine the vocabulary size, as well as for debugging and topic
        printing.

        `alpha` and `eta` are hyperparameters that affect sparsity of the document-topic
        (theta) and topic-word (lambda) distributions. Both default to a symmetric
        1.0/num_topics (but can be set to a vector, for asymmetric priors).

        Turn on `distributed` to force distributed computing (see the web tutorial
        on how to set up a cluster of machines for gensim).

        Example:

        >>> lda = LdaModel(corpus, num_topics=100)
        >>> print lda[doc_bow] # get topic probability distribution for a document
        >>> lda.update(corpus2) # update the LDA model with additional documents
        >>> print lda[doc_bow]

        """
        # store user-supplied parameters
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')

        if self.id2word is None:
            logger.info("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        else:
            self.num_terms = 1 + max([-1] + self.id2word.keys())

        if self.num_terms == 0:
            raise ValueError("cannot compute LDA over an empty collection (no terms)")

        self.distributed = bool(distributed)
        self.num_topics = int(num_topics)
        self.chunksize = chunksize
        self.decay = decay
        self.num_updates = 0

        self.passes = passes
        self.update_every = update_every

        if alpha is None:
            self.alpha = 1.0 / num_topics
        else:
            self.alpha = alpha
        if eta is None:
            self.eta = 1.0 / num_topics
        else:
            self.eta = eta

        # VB constants
        self.VAR_MAXITER = 50
        self.VAR_THRESH = 0.001

        # set up distributed environment if necessary
        if not distributed:
            logger.info("using serial LDA version on this node")
            self.dispatcher = None
            self.numworkers = 1
        else:
            # set up distributed version
            try:
                import Pyro4
                dispatcher = Pyro4.Proxy('PYRONAME:gensim.lda_dispatcher')
                dispatcher._pyroOneway.add("exit")
                logger.debug("looking for dispatcher at %s" % str(dispatcher._pyroUri))
                dispatcher.initialize(id2word=self.id2word, num_topics=num_topics,
                                      chunksize=chunksize, alpha=alpha, eta=eta, distributed=False)
                self.dispatcher = dispatcher
                self.numworkers = len(dispatcher.getworkers())
                logger.info("using distributed version with %i workers" % self.numworkers)
            except Exception, err:
                logger.error("failed to initialize distributed LDA (%s)" % err)
                raise RuntimeError("failed to initialize distributed LDA (%s)" % err)

        # Initialize the variational distribution q(beta|lambda)
        self.state = LdaState(self.eta, (self.num_topics, self.num_terms))
        self.state.sstats = numpy.random.gamma(100., 1./100., (self.num_topics, self.num_terms))
        self.sync_state()

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            self.update(corpus)


    def sync_state(self):
        self.expElogbeta = numpy.exp(self.state.get_Elogbeta())


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
        `len(chunk) x topics`.
        """
        try:
            _ = len(chunk)
        except:
            chunk = list(chunk) # convert iterators/generators to plain list, so we have len() etc.
        logger.debug("performing inference on a chunk of %i documents" % len(chunk))

        # Initialize the variational distribution q(theta|gamma) for the chunk
        gamma = numpy.random.gamma(100., 1./100., (len(chunk), self.num_topics))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = numpy.exp(Elogtheta)
        if collect_sstats:
            sstats = numpy.zeros_like(self.expElogbeta)
        else:
            sstats = None
        converged = 0

        # Now, for each document d update that document's gamma and phi
        # Inference code copied from Hoffman's `onlineldavb.py` (esp. the
        # Lee&Seung trick which speeds things up by an order of magnitude, compared
        # to Blei's original LDA-C code, cool!).
        for d, doc in enumerate(chunk):
            ids = [id for id, _ in doc]
            cts = numpy.array([cnt for _, cnt in doc])
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self.expElogbeta[:, ids]

            # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_w.
            # phinorm is the normalizer.
            phinorm = numpy.dot(expElogthetad, expElogbetad) + 1e-100 # TODO treat zeros explicitly, instead of adding eps?

            # Iterate between gamma and phi until convergence
            for _ in xrange(self.VAR_MAXITER):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self.alpha + expElogthetad * numpy.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = numpy.exp(Elogthetad)
                phinorm = numpy.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = numpy.mean(abs(gammad - lastgamma))
                if (meanchange < self.VAR_THRESH):
                    converged += 1
                    break
            gamma[d, :] = gammad
            if collect_sstats:
                # Contribution of document d to the expected sufficient
                # statistics for the M step.
                sstats[:, ids] += numpy.outer(expElogthetad.T, cts / phinorm)

        if len(chunk) > 1:
            logger.info("%i/%i documents converged within %i iterations" %
                         (converged, len(chunk), self.VAR_MAXITER))

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
        state.numdocs += gamma.shape[0] # avoid calling len(chunk), might be a generator
        return gamma


    def update(self, corpus, chunksize=None, decay=None, passes=None, update_every=None):
        """
        Train the model with new documents, by EM-iterating over `corpus` until
        the topics converge (or until the maximum number of allowed iterations
        is reached).

        In distributed mode, the E step is distributed over a cluster of machines.

        This update also supports updating an already trained model (`self`)
        with new documents from `corpus`; the two models are then merged in
        proportion to the number of old vs. new documents. This feature is still
        experimental for non-stationary input streams.

        For stationary input (no topic drift in new documents), on the other hand,
        this equals the online update of Hoffman et al. and is guaranteed to
        converge for any `decay` in (0.5, 1.0>.
        """
        # use parameters given in constructor, unless user explicitly overrode them
        if chunksize is None:
            chunksize = self.chunksize
        if decay is None:
            decay = self.decay
        if passes is None:
            passes = self.passes
        if update_every is None:
            update_every = self.update_every

        # rho is the "speed" of updating; TODO try other fncs
        rho = lambda: pow(1.0 + self.num_updates, -decay)

        try:
            lencorpus = len(corpus)
        except:
            logger.warning("input corpus stream has no len(); counting documents")
            lencorpus = sum(1 for _ in corpus)
        if lencorpus == 0:
            logger.warning("LdaModel.update() called with an empty corpus")
            return
        self.state.numdocs += lencorpus

        if update_every > 0:
            updatetype = "online"
            updateafter = min(lencorpus, update_every * self.numworkers * chunksize)
        else:
            updatetype = "batch"
            updateafter = lencorpus

        updates_per_pass = max(1, lencorpus / updateafter)
        logger.info("running %s LDA training, %s topics, %i passes over "
                    "the supplied corpus of %i documents, updating model once "
                    "every %i documents" %
                    (updatetype, self.num_topics, passes, lencorpus, updateafter))
        if updates_per_pass * passes < 10:
            logger.warning("too few updates, training might not converge; consider "
                           "increasing the number of passes to improve accuracy")

        for iteration in xrange(passes):
            if self.dispatcher:
                logger.info('initializing %s workers' % self.numworkers)
                self.dispatcher.reset(self.state)
            else:
                other = LdaState(self.eta, self.state.sstats.shape)
            dirty = False

            for chunk_no, chunk in enumerate(utils.grouper(corpus, chunksize, as_numpy=True)):
                if self.dispatcher:
                    # add the chunk to dispatcher's job queue, so workers can munch on it
                    logger.info('PROGRESS: iteration %i, dispatching documents up to #%i/%i' %
                                (iteration, chunk_no * chunksize + len(chunk), lencorpus))
                    # this will eventually block until some jobs finish, because the queue has a small finite length
                    self.dispatcher.putjob(chunk)
                else:
                    logger.info('PROGRESS: iteration %i, at document #%i/%i' %
                                (iteration, chunk_no * chunksize + len(chunk), lencorpus))
                    self.do_estep(chunk, other)
                dirty = True
                del chunk

                # perform an M step. determine when based on update_every, don't do this after every chunk
                if update_every and (chunk_no + 1) % (update_every * self.numworkers) == 0:
                    if self.dispatcher:
                        # distributed mode: wait for all workers to finish
                        logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                        other = self.dispatcher.getstate()
                    self.do_mstep(rho(), other)
                    del other # free up some mem

                    if self.dispatcher:
                        logger.info('initializing workers')
                        self.dispatcher.reset(self.state)
                    else:
                        other = LdaState(self.eta, self.state.sstats.shape)
                    dirty = False
            #endfor single corpus iteration

            if dirty:
                # finish any remaining updates
                if self.dispatcher:
                    # distributed mode: wait for all workers to finish
                    logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                    other = self.dispatcher.getstate()
                self.do_mstep(rho(), other)
                del other
                dirty = False
        #endfor entire corpus update


    def do_mstep(self, rho, other):
        """
        M step: use linear interpolation between the existing topics and
        collected sufficient statistics in `other` to update the topics.
        """
        logger.debug("updating topics")
        # update self with the new blend; also keep track of how much did
        # the topics change through this update, to assess convergence
        diff = numpy.log(self.expElogbeta)
        self.state.blend(rho, other)
        del other
        diff -= self.state.get_Elogbeta()
        self.sync_state()
        self.print_topics(15) # print out some debug info at the end of each EM iteration
        logger.info("topic diff=%f, rho=%f" % (numpy.mean(numpy.abs(diff)), rho))
        self.num_updates += 1


    def bound(self, corpus, gamma=None):
        """
        Estimate the variational bound of documents from `corpus`.

        `gamma` are the variational parameters on topic weights (one for each
        document in `corpus`). If not supplied, will be automatically inferred
        from the model.
        """
        score = 0.0
        _lambda = self.state.get_lambda()
        Elogbeta = dirichlet_expectation(_lambda)

        for d, doc in enumerate(corpus):
            if d % self.chunksize == 0:
                logger.info("PROGRESS: at document #%i" % d)
            if gamma is None:
                gammad, _ = self.inference([doc])
            else:
                gammad = gamma[d]
            Elogthetad = dirichlet_expectation(gammad)
            ids = [id for id, _ in doc]
            cts = numpy.array([cnt for _, cnt in doc])
            phinorm = numpy.zeros(len(ids))
            for i in xrange(len(ids)):
                phinorm[i] = logsumexp(Elogthetad + Elogbeta[:, ids[i]])

            # E[log p(docs | theta, beta)]
            score += numpy.sum(cts * phinorm)

            # E[log p(theta | alpha) - log q(theta | gamma)]
            score += numpy.sum((self.alpha - gammad) * Elogthetad)
            score += numpy.sum(gammaln(gammad) - gammaln(self.alpha))
            score += gammaln(self.alpha * self.num_topics) - gammaln(numpy.sum(gammad))

        # E[log p(beta | eta) - log q (beta | lambda)]
        score += numpy.sum((self.eta - _lambda) * Elogbeta)
        score += numpy.sum(gammaln(_lambda) - gammaln(self.eta))
        score += numpy.sum(gammaln(self.eta * self.num_terms) - gammaln(numpy.sum(_lambda, 1)))
        return score


    def print_topics(self, topics=10, topn=10):
        self.show_topics(topics, topn, True)

    def show_topics(self, topics=10, topn=10, log=False, formatted=True):
        """
        Print the `topN` most probable words for (randomly selected) `topics`
        number of topics. Set `topics=-1` to print all topics.

        Unlike LSA, there is no ordering between the topics in LDA.
        The printed `topics <= self.num_topics` subset of all topics is therefore
        arbitrary and may change between two runs.
        """
        if topics < 0:
            # print all topics if `topics` is negative
            topics = self.num_topics
        topics = min(topics, self.num_topics)
        shown  = []
        for i in xrange(topics):
            if formatted:
                topic = self.print_topic(i, topn=topn)
            else:
                topic = self.show_topic(i, topn=topn)
            shown.append(topic)
            if log:
                logger.info("topic #%i: %s" % (i, topic))
        return shown

    def show_topic(self, topicid, topn=10):
        topic = self.state.get_lambda()[topicid]
        topic = topic / topic.sum() # normalize to probability dist
        bestn = numpy.argsort(topic)[::-1][:topn]
        beststr = [(topic[id], self.id2word[id]) for id in bestn]
        return beststr

    def print_topic(self, topicid, topn=10):
        return ' + '.join(['%.3f*%s' % v for v in self.show_topic(topicid, topn)])


    def __getitem__(self, bow, eps=0.01):
        """
        Return topic distribution for the given document `bow`, as a list of
        (topic_id, topic_probability) 2-tuples.

        Ignore topics with very low probability (below `eps`).
        """
        # if the input vector is in fact a corpus, return a transformed corpus as result
        is_corpus, corpus = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(corpus)

        gamma, _ = self.inference([bow])
        topic_dist = gamma[0] / sum(gamma[0]) # normalize to proper distribution
        return [(topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)
                if topicvalue >= eps] # ignore document's topics that have prob < eps
#endclass LdaModel
