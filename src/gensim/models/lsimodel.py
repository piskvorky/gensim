#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Module for Latent Semantic Indexing.

This module actually contains several algorithms for decomposition of large corpora, a
combination of which effectively and transparently allows building LSI models for:

* corpora much larger than RAM: only constant memory is needed, independent of
  the corpus size (though still dependent on the feature set size)
* corpora that are streamed: documents are only accessed sequentially, no
  random-access
* corpora that cannot be even temporarily stored: each document can only be
  seen once and must be processed immediately (one-pass algorithm)
* distributed computing for very large corpora, making use of a cluster of
  machines

Wall-clock performance on the English Wikipedia (2G corpus positions, 3.2M
documents, 100K features, 0.5G non-zero entries in the final TF-IDF matrix),
requesting the top 400 LSI factors:


====================================================== ============ ==================
 algorithm                                             serial       distributed
====================================================== ============ ==================
 one-pass merge algorithm                              5h14m        1h41m
 multi-pass stochastic algo (with 2 power iterations)  5h39m        N/A [1]_
====================================================== ============ ==================


*serial* = Core 2 Duo MacBook Pro 2.53Ghz, 4GB RAM, libVec

*distributed* = cluster of four logical nodes on three physical machines, each
with dual core Xeon 2.0GHz, 4GB RAM, ATLAS

.. [1] The stochastic algo could be distributed too, but most time is already spent
   reading/decompressing the input from disk in its 4 passes. The extra network
   traffic due to data distribution across cluster nodes would likely make it
   *slower*.

"""


import logging
import itertools
import sys

import numpy
import scipy.sparse
from scipy.sparse import sparsetools

from gensim import interfaces, matutils, utils


logger = logging.getLogger('lsimodel')
logger.setLevel(logging.INFO)


# accuracy defaults for the multi-pass stochastic algo
P2_EXTRA_DIMS = 100 # set to `None` for dynamic P2_EXTRA_DIMS=k
P2_EXTRA_ITERS = 2


def clipSpectrum(s, k, discard=0.001):
    """
    Given eigenvalues `s`, return how many factors should be kept to avoid
    storing spurious (tiny, numerically instable) values.

    This will ignore the tail of the spectrum with relative combined mass < min(`discard`, 1/k).

    The returned value is clipped against `k` (= never return more than `k`).
    """
    # compute relative contribution of eigenvalues towards the energy spectrum
    rel_spectrum = numpy.abs(1.0 - numpy.cumsum(s / numpy.sum(s)))
    # ignore the last `discard` mass (or 1/k, whichever is smaller) of the spectrum
    small = 1 + len(numpy.where(rel_spectrum > min(discard, 1.0 / k))[0])
    k = min(k, small) # clip against k
    logger.info("keeping %i factors (discarding %.3f%% of energy spectrum)" %
                (k, 100 * rel_spectrum[k - 1]))
    return k


class Projection(utils.SaveLoad):
    def __init__(self, m, k, docs=None, use_svdlibc=False):
        """
        Construct the (U, S) projection from a corpus `docs`. The projection can
        be later updated by merging it with another Projection via `self.merge()`.

        This is the class taking care of the 'core math'; interfacing with corpora,
        splitting large corpora into chunks and merging them etc. is done through
        the LsiModel class.
        """
        self.m, self.k = m, k
        if docs is not None:
            # base case decomposition: given a job `docs`, compute its decomposition,
            # *in-core*.
            if not use_svdlibc:
                u, s = stochasticSvd(docs, k, chunks=sys.maxint, num_terms=m,
                    power_iters=P2_EXTRA_ITERS, extra_dims=P2_EXTRA_DIMS)
            else:
                try:
                    import sparsesvd
                except ImportError:
                    raise ImportError("for LSA, the `sparsesvd` module is needed but not found; run `easy_install sparsesvd`")
                logger.info("computing sparse SVD of %s matrix" % str(docs.shape))
                if not scipy.sparse.issparse(docs):
                    docs = matutils.corpus2csc(docs)
                ut, s, vt = sparsesvd.sparsesvd(docs, k + 30) # ask for extra factors, because for some reason SVDLIBC sometimes returns fewer factors than requested
                u = ut.T
                del ut, vt
                k = clipSpectrum(s ** 2, self.k)
            self.u, self.s = u[:, :k], s[:k]
        else:
            self.u, self.s = None, None


    def empty_like(self):
        return Projection(self.m, self.k)


    def merge(self, other, decay=1.0):
        """
        Merge this Projection with another.

        The content of `other` is destroyed in the process, so pass this function a
        copy of `other` if you need it further.
        """
        if other.u is None:
            # the other projection is empty => do nothing
            return
        if self.u is None:
            # we are empty => result of merge is the other projection, whatever it is
            self.u = other.u.copy('F')
            self.s = other.s.copy()
            return
        if self.m != other.m:
            raise ValueError("vector space mismatch: update is using %s features, expected %s" %
                             (other.m, self.m))
        logger.info("merging projections: %s + %s" % (str(self.u.shape), str(other.u.shape)))
        m, n1, n2 = self.u.shape[0], self.u.shape[1], other.u.shape[1]
        # TODO Maybe keep the bases as elementary reflectors, without
        # forming explicit matrices with ORGQR.
        # The only operation we ever need is basis^T*basis ond basis*component.
        # But how to do that in scipy? And is it fast(er)?

        # find component of u2 orthogonal to u1
        # IMPORTANT: keep matrices in memory suitable order for matrix products; failing to do so gives 8x lower performance :(
        self.u = numpy.asfortranarray(self.u) # does nothing if input already fortran-order array
        other.u = numpy.asfortranarray(other.u)
        gemm = matutils.blas('gemm', self.u)
        logger.debug("constructing orthogonal component")
        c = gemm(1.0, self.u, other.u, trans_a = True)
        gemm(-1.0, self.u, c, beta = 1.0, c = other.u, overwrite_c = True)

        other.u = [other.u] # do some reference magic and call qr_destroy, to save RAM
        q, r = matutils.qr_destroy(other.u) # q, r = QR(component)
        assert not other.u

        # find the rotation that diagonalizes r
        k = numpy.bmat([[numpy.diag(decay * self.s), c * other.s], [matutils.pad(numpy.matrix([]).reshape(0, 0), min(m, n2), n1), r * other.s]])
        logger.debug("computing SVD of %s dense matrix" % str(k.shape))
        try:
            # in numpy < 1.1.0, running SVD sometimes results in "LinAlgError: SVD did not converge'.
            # for these early versions of numpy, catch the error and try to compute
            # SVD again, but over k*k^T.
            # see http://www.mail-archive.com/numpy-discussion@scipy.org/msg07224.html and
            # bug ticket http://projects.scipy.org/numpy/ticket/706
            u_k, s_k, _ = numpy.linalg.svd(k, full_matrices = False) # TODO *ugly overkill*!! only need first self.k SVD factors... but there is no LAPACK wrapper for partial svd/eigendecomp in numpy :(
        except numpy.linalg.LinAlgError:
            logging.error("SVD(A) failed; trying SVD(A * A^T)")
            u_k, s_k, _ = numpy.linalg.svd(numpy.dot(k, k.T), full_matrices = False) # if this fails too, give up with an exception
            s_k = numpy.sqrt(s_k) # go back from eigen values to singular values

        k = clipSpectrum(s_k ** 2, self.k)
        u1_k, u2_k, s_k = u_k[:n1, :k].copy('F'), u_k[n1:, :k].copy('F'), s_k[:k]

        # update & rotate current basis U = [U, U']*[U1_k, U2_k]
        logger.debug("updating orthonormal basis U")
        self.u = gemm(1.0, self.u, u1_k) # TODO temporarily creates an extra (m,k) dense array in memory. find a way to avoid this!
        gemm(1.0, q, u2_k, beta = 1.0, c = self.u, overwrite_c = True)
        self.s = s_k
#        diff = numpy.dot(self.u.T, self.u) - numpy.eye(self.u.shape[1])
#        logger.info('orth error after=%f' % numpy.sum(diff * diff))


    def __setstate__(self, state):
        """
        This is a hack to work around a bug in numpy, where a FORTRAN-order array
        unpickled from disk segfaults on using it.
        """
        self.__dict__ = state
        if self.u is not None:
            self.u = self.u.copy('F') # simply making a fresh copy fixes the broken array
#endclass Projection



class LsiModel(interfaces.TransformationABC):
    """
    Objects of this class allow building and maintaining a model for Latent
    Semantic Indexing (also known as Latent Semantic Analysis).

    The main methods are:

    1. constructor, which initializes the projection into latent topics space,
    2. the ``[]`` method, which returns representation of any input document in the
       latent space,
    3. the `addDocuments()` method, which allows for incrementally updating the model with new documents.

    Model persistency is achieved via its load/save methods.

    """
    def __init__(self, corpus=None, numTopics=200, id2word=None, chunks=10000,
                 decay=1.0, distributed=False, onepass=True,
                 power_iters=P2_EXTRA_ITERS, extra_samples=P2_EXTRA_DIMS):
        """
        `numTopics` is the number of requested factors (latent dimensions).

        After the model has been trained, you can estimate topics for an
        arbitrary, unseen document, using the ``topics = self[document]`` dictionary
        notation. You can also add new training documents, with ``self.addDocuments``,
        so that training can be stopped and resumed at any time, and the
        LSI transformation is available at any point.

        If you specify a `corpus`, it will be used to train the model. See the
        method `addDocuments` for a description of the `chunks` and `decay` parameters.

        Turn `onepass` off to force a multi-pass stochastic algorithm.

        `power_iters` and `extra_samples` affect the accuracy of the stochastic
        multi-pass algorithm, which is used either internally (`onepass=True`) or
        as the front-end algorithm (`onepass=False`). Increasing the number of
        power iterations improves accuracy, but lowers performance. See [2]_ for
        some hard numbers.

        Turn on `distributed` to enable distributed computing.

        Example:

        >>> lsi = LsiModel(corpus, numTopics=10)
        >>> print lsi[doc_tfidf] # project some document into LSI space
        >>> lsi.addDocuments(corpus2) # update LSI on additional documents
        >>> print lsi[doc_tfidf]

        .. [2] http://nlp.fi.muni.cz/~xrehurek/nips/rehurek_nips.pdf

        """
        self.id2word = id2word
        self.numTopics = int(numTopics)
        self.chunks = int(chunks)
        self.decay = float(decay)
        if distributed:
            if not onepass:
                logger.warning("forcing the one-pass algorithm for distributed LSA")
                onepass = True
        self.onepass = onepass
        self.extra_samples, self.power_iters = extra_samples, power_iters

        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')

        if self.id2word is None:
            logger.info("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dictFromCorpus(corpus)
            self.numTerms = len(self.id2word)
        else:
            self.numTerms = 1 + max([-1] + self.id2word.keys())

        self.docs_processed = 0
        self.projection = Projection(self.numTerms, self.numTopics)

        self.numworkers = 1
        if not distributed:
            logger.info("using serial LSI version on this node")
            self.dispatcher = None
        else:
            if not onepass:
                raise NotImplementedError("distributed stochastic LSA not implemented yet; "
                                          "run either distributed one-pass, or serial randomized.")
            try:
                import Pyro
                ns = Pyro.naming.locateNS()
                dispatcher = Pyro.core.Proxy('PYRONAME:gensim.lsi_dispatcher@%s' % ns._pyroUri.location)
                dispatcher._pyroOneway.add("exit")
                logger.debug("looking for dispatcher at %s" % str(dispatcher._pyroUri))
                dispatcher.initialize(id2word = self.id2word, numTopics = numTopics,
                                      chunks = chunks, decay = decay,
                                      distributed = False, onepass = onepass)
                self.dispatcher = dispatcher
                self.numworkers = len(dispatcher.getworkers())
                logger.info("using distributed version with %i workers" % self.numworkers)
            except Exception, err:
                # distributed version was specifically requested, so this is an error state
                logger.error("failed to initialize distributed LSI (%s)" % err)
                raise RuntimeError("failed to initialize distributed LSI (%s)" % err)

        if corpus is not None:
            self.addDocuments(corpus)


    def addDocuments(self, corpus, chunks=None, decay=None):
        """
        Update singular value decomposition to take into account a new
        corpus of documents.

        Training proceeds in chunks of `chunks` documents at a time. The size of
        `chunks` is a tradeoff between increased speed (bigger `chunks`)
        vs. lower memory footprint (smaller `chunks`). If the distributed mode
        is on, each chunk is sent to a different worker/computer.

        Setting `decay` < 1.0 causes re-orientation towards new data trends in the
        input document stream, by giving less emphasis to old observations. This allows
        LSA to gradually "forget" old observations (documents) and give more
        preference to new ones.
        """
        logger.info("updating SVD with new documents")

        # get computation parameters; if not specified, use the ones from constructor
        if chunks is None:
            chunks = self.chunks
        if decay is None:
            decay = self.decay

        if not scipy.sparse.issparse(corpus):
            if not self.onepass:
                # we are allowed multiple passes over the input => use a faster, randomized two-pass algo
                update = Projection(self.numTerms, self.numTopics, None)
                update.u, update.s = stochasticSvd(corpus, self.numTopics,
                    num_terms=self.numTerms, chunks=chunks,
                    extra_dims=self.extra_samples, power_iters=self.power_iters)
                self.projection.merge(update, decay=decay)
            else:
                # the one-pass algo

                doc_no = 0
                # the corpus will be processed in chunks of `chunks` of documents.
                # keep preparing new chunks in a separate thread, so that we don't
                # waste time waiting for chunks to be read from disk. instead, fill
                # a (relatively short) chunk queue asynchronously in utils.chunkize,
                # and pop already-ready chunks from it as needed.
                for chunk_no, chunk in enumerate(utils.chunkize(corpus, chunks, self.numworkers)):
                    # construct the job as a sparse matrix, to minimize memory overhead
                    # definitely avoid materializing it as a dense matrix!
                    job = matutils.corpus2csc(chunk, num_terms=self.numTerms)
                    del chunk
                    doc_no += job.shape[1]
                    if self.dispatcher:
                        # distributed version: add this job to the job queue, so workers can work on it
                        logger.debug("creating job #%i" % chunk_no)
                        self.dispatcher.putjob(job) # put job into queue; this will eventually block, because the queue has a small finite size
                        del job
                        logger.info("dispatched documents up to #%s" % doc_no)
                    else:
                        # serial version, there is only one "worker" (myself) => process the job directly
                        update = Projection(self.numTerms, self.numTopics, job)
                        del job
                        self.projection.merge(update, decay = decay)
                        del update
                        logger.info("processed documents up to #%s" % doc_no)
                        self.printTopics(5) # TODO see if printDebug works and remove one of these..

                # wait for all workers to finish (distributed version only)
                if self.dispatcher:
                    logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                    self.projection = self.dispatcher.getstate()
#            logging.info("top topics after adding %i documents" % doc_no)
#            self.printDebug(10)
        else:
            assert not self.dispatcher, "must be in serial mode to receive jobs"
            assert self.onepass, "distributed two-pass algo not supported yet"
            update = Projection(self.numTerms, self.numTopics, corpus.tocsc())
            self.projection.merge(update, decay=decay)
            logger.info("processed sparse job of %i documents" % (corpus.shape[1]))


    def __str__(self):
        return "LsiModel(numTerms=%s, numTopics=%s, decay=%s, chunks=%s)" % \
                (self.numTerms, self.numTopics, self.decay, self.chunks)


    def __getitem__(self, bow, scaled=False):
        """
        Return latent representation, as a list of (topic_id, topic_value) 2-tuples.

        This is done by folding input document into the latent topic space.

        Note that this function returns the latent space representation **scaled by the
        singular values**. To return non-scaled embedding, set `scaled` to False.
        """
        # if the input vector is in fact a corpus, return a transformed corpus as a result
        is_corpus, bow = utils.isCorpus(bow)
        if is_corpus:
            return self._apply(bow)

        assert self.projection.u is not None, "decomposition not initialized yet"
        vec = matutils.sparse2full(bow, self.numTerms).astype(self.projection.u.dtype)
        vec.shape = (self.numTerms, 1)
        assert self.projection.u.flags.f_contiguous
        dgemv = matutils.blas('gemv', self.projection.u)
        topicDist = dgemv(1.0, self.projection.u, vec, trans=True) # u^T * x
        if scaled:
            topicDist = (1.0 / self.projection.s) * topicDist # s^-1 * u^T * x

        nnz = topicDist.nonzero()[0]
        return zip(nnz, topicDist[nnz])


    def printTopic(self, topicNo, topN=10):
        """
        Return a specified topic (=left singular vector), 0 <= `topicNo` < `self.numTopics`,
        as string.

        Return only the `topN` words which contribute the most to the direction
        of the topic (both negative and positive).

        >>> lsimodel.printTopic(10, topN=5)
        '-0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + -0.174 * "functor" + -0.168 * "operator"'

        """
        # size of the projection matrix can actually be smaller than `self.numTopics`,
        # if there were not enough factors (real rank of input matrix smaller than
        # `self.numTopics`). in that case, return empty string
        if topicNo >= len(self.projection.u.T):
            return ''
        c = numpy.asarray(self.projection.u.T[topicNo, :]).flatten()
        norm = numpy.sqrt(numpy.sum(c * c))
        most = numpy.abs(c).argsort()[::-1][:topN]
        return ' + '.join(['%.3f*"%s"' % (1.0 * c[val] / norm, self.id2word[val]) for val in most])


    def printTopics(self, numTopics = 5, numWords = 10):
        for i in xrange(min(numTopics, self.numTopics)):
            if i < len(self.projection.s):
                logger.info("topic #%i(%.3f): %s" %
                            (i, self.projection.s[i],
                             self.printTopic(i, topN = numWords)))


    def printDebug(self, numTopics=5, numWords=10):
        """
        Print (to log) the most salient words of the first `numTopics` topics.

        Unlike `printTopics()`, this looks for words that are significant for a
        particular topic *and* not for others. This *should* result in a more
        human-interpretable description of topics.
        """
        # only wrap the module-level fnc
        printDebug(self.id2word,
                   self.projection.u, self.projection.s,
                   range(min(numTopics, len(self.projection.u.T))),
                   numWords = numWords)
#endclass LsiModel


def printDebug(id2token, u, s, topics, numWords=10, numNeg=None):
    if numNeg is None:
        # by default, print half as many salient negative words as positive
        numNeg = numWords / 2

    logger.info('computing word-topic salience for %i topics' % len(topics))
    topics, result = set(topics), {}
    # TODO speed up by block computation
    for uvecno, uvec in enumerate(u):
        uvec = numpy.abs(numpy.asarray(uvec).flatten())
        udiff = uvec / numpy.sqrt(numpy.sum(uvec * uvec))
        for topic in topics:
            result.setdefault(topic, []).append((udiff[topic], uvecno))

    logger.debug("printing %i+%i salient words" % (numWords, numNeg))
    for topic in sorted(result.iterkeys()):
        weights = sorted(result[topic], key=lambda x: -abs(x[0]))
        _, most = weights[0]
        if u[most, topic] < 0.0: # the most significant word has a negative sign => flip sign of u[most]
            normalize = -1.0
        else:
            normalize = 1.0

        # order features according to salience; ignore near-zero entries in u
        pos, neg = [], []
        for weight, uvecno in weights:
            if normalize * u[uvecno, topic] > 0.0001:
                pos.append('%s(%.3f)' % (id2token[uvecno], u[uvecno, topic]))
                if len(pos) >= numWords:
                    break

        for weight, uvecno in weights:
            if normalize * u[uvecno, topic] < -0.0001:
                neg.append('%s(%.3f)' % (id2token[uvecno], u[uvecno, topic]))
                if len(neg) >= numNeg:
                    break

        logger.info('topic #%s(%.3f): %s, ..., %s' % (topic, s[topic], ', '.join(pos), ', '.join(neg)))


def svdUpdate(U, S, V, a, b):
    """
    Update SVD of an (m x n) matrix `X = U * S * V^T` so that
    `[X + a * b^T] = U' * S' * V'^T`
    and return `U'`, `S'`, `V'`.

    The original matrix X is not needed at all, so this function implements one-pass
    streaming rank-1 updates to an existing decomposition.

    `a` and `b` are (m, 1) and (n, 1) matrices.

    You can set V to None if you're not interested in the right singular
    vectors. In that case, the returned V' will also be None (saves memory).

    The blocked merge algorithm in LsiModel.addDocuments() is much faster; I keep this fnc here
    purely for backup reasons.

    This is the rank-1 update as described in
    **Brand, 2006: Fast low-rank modifications of the thin singular value decomposition**,
    but without separating the basis from rotations.
    """
    # convert input to matrices (no copies of data made if already numpy.ndarray or numpy.matrix)
    S = numpy.asmatrix(S)
    U = numpy.asmatrix(U)
    if V is not None:
        V = numpy.asmatrix(V)
    a = numpy.asmatrix(a).reshape(a.size, 1)
    b = numpy.asmatrix(b).reshape(b.size, 1)

    rank = S.shape[0]

    # eq (6)
    m = U.T * a
    p = a - U * m
    Ra = numpy.sqrt(p.T * p)
    if float(Ra) < 1e-10:
        logger.debug("input already contained in a subspace of U; skipping update")
        return U, S, V
    P = (1.0 / float(Ra)) * p

    if V is not None:
        # eq (7)
        n = V.T * b
        q = b - V * n
        Rb = numpy.sqrt(q.T * q)
        if float(Rb) < 1e-10:
            logger.debug("input already contained in a subspace of V; skipping update")
            return U, S, V
        Q = (1.0 / float(Rb)) * q
    else:
        n = numpy.matrix(numpy.zeros((rank, 1)))
        Rb = numpy.matrix([[1.0]])

    if float(Ra) > 1.0 or float(Rb) > 1.0:
        logger.debug("insufficient target rank (Ra=%.3f, Rb=%.3f); this update will result in major loss of information"
                      % (float(Ra), float(Rb)))

    # eq (8)
    K = numpy.matrix(numpy.diag(list(numpy.diag(S)) + [0.0])) + numpy.bmat('m ; Ra') * numpy.bmat('n ; Rb').T

    # eq (5)
    u, s, vt = numpy.linalg.svd(K, full_matrices = False)
    tUp = numpy.matrix(u[:, :rank])
    tVp = numpy.matrix(vt.T[:, :rank])
    tSp = numpy.matrix(numpy.diag(s[: rank]))
    Up = numpy.bmat('U P') * tUp
    if V is not None:
        Vp = numpy.bmat('V Q') * tVp
    else:
        Vp = None
    Sp = tSp

    return Up, Sp, Vp


def iterSvd(corpus, numTerms, numFactors, numIter = 200, initRate = None, convergence = 1e-4):
    """
    Perform iterative Singular Value Decomposition on a streaming corpus, returning
    `numFactors` greatest factors U,S,V^T (ie., not necessarily the full spectrum).

    The parameters `numIter` (maximum number of iterations) and `initRate` (gradient
    descent step size) guide convergency of the algorithm.

    The algorithm performs at most `numFactors*numIters` passes over the corpus.

    Use of this function is deprecated; although it works, it is several orders of
    magnitude slower than our own, direct (non-stochastic) version (which
    operates in a single pass, too, and can be distributed). I keep this function
    here purely for backup reasons.

    See **Genevieve Gorrell: Generalized Hebbian Algorithm for Incremental Singular
    Value Decomposition in Natural Language Processing. EACL 2006.**
    """
    logger.info("performing incremental SVD for %i factors" % numFactors)

    # define the document/term singular vectors, fill them with a little random noise
    sdoc = 0.01 * numpy.random.randn(len(corpus), numFactors)
    sterm = 0.01 * numpy.random.randn(numTerms, numFactors)
    if initRate is None:
        initRate = 1.0 / numpy.sqrt(numTerms)
        logger.info("using initial learn rate of %f" % initRate)

    rmse = rmseOld = numpy.inf
    for factor in xrange(numFactors):
        learnRate = initRate
        for iterNo in xrange(numIter):
            errors = 0.0
            rate = learnRate / (1.0 + 9.0 * iterNo / numIter) # gradually decrease the learning rate to 1/10 of the initial value
            logger.debug("setting learning rate to %f" % rate)
            for docNo, doc in enumerate(corpus):
                vec = dict(doc)
                if docNo % 10 == 0:
                    logger.debug('PROGRESS: at document %i/%i' % (docNo, len(corpus)))
                vdoc = sdoc[docNo, factor]
                vterm = sterm[:, factor] # create a view (not copy!) of a matrix row

                # reconstruct one document, using all previous factors <0..factor-1>
                # we do not pre-cache the dot products because that takes insane amounts of memory (=doesn't scale with corpus size)
                recon = numpy.dot(sdoc[docNo, :factor], sterm[:, :factor].T) # numpy.dot is very fast anyway, this is not the bottleneck

                for termId in xrange(numTerms): # this loop is.
                    # error of one matrix element = real value - reconstructed value
                    error = vec.get(termId, 0.0) - (recon[termId] + vdoc * vterm[termId])
                    errors += error * error

                    # update the singular vectors
                    tmp = vdoc
                    vdoc += rate * error * vterm[termId]
                    vterm[termId] += rate * error * tmp
                sdoc[docNo, factor] = vdoc

            # compute rmse = root mean square error of the reconstructed matrix
            rmse = numpy.exp(0.5 * (numpy.log(errors) - numpy.log(len(corpus)) - numpy.log(numTerms)))
            if not numpy.isfinite(rmse) or rmse > rmseOld:
                learnRate /= 2.0 # if we are not converging (oscillating), halve the learning rate
                logger.info("iteration %i diverged; halving the learning rate to %f" %
                             (iterNo, learnRate))

            # check convergence, looking for an early exit (but no sooner than 10%
            # of numIter have passed)
            converged = numpy.divide(numpy.abs(rmseOld - rmse), rmseOld)
            logger.info("factor %i, finished iteration %i, rmse=%f, rate=%f, converged=%f" %
                          (factor, iterNo, rmse, rate, converged))
            if iterNo > numIter / 10 and numpy.isfinite(converged) and converged <= convergence:
                logger.debug("factor %i converged in %i iterations" % (factor, iterNo + 1))
                break
            rmseOld = rmse

        logger.info("PROGRESS: finished SVD factor %i/%i, RMSE<=%f" %
                     (factor + 1, numFactors, rmse))

    # normalize the vectors to unit length; also keep the norms
    sdocLens = numpy.sqrt(numpy.sum(sdoc * sdoc, axis = 0))
    stermLens = numpy.sqrt(numpy.sum(sterm * sterm, axis = 0))
    sdoc /= sdocLens
    sterm /= stermLens

    # singular values are the norms
    svals = sdocLens * stermLens
    return sterm, svals, sdoc.T


def stochasticSvd(corpus, rank, num_terms, chunks=20000, extra_dims=None,
                  power_iters=0, dtype=numpy.float64, eps=1e-6):
    """
    Return (U, S): the left singular vectors and the singular values of the streamed
    input corpus `corpus` [3]_.

    This may actually return less than the requested number of top `rank` factors,
    in case the input is of lower rank. The `extra_dims` (oversampling) and especially
    `power_iters` (power iterations) parameters affect accuracy of the decomposition.

    This algorithm uses `2+power_iters` passes over the data. In case you can only
    afford a single pass over the input corpus, set `onepass=True` in :class:`LsiModel`
    and avoid using this algorithm directly.

    The decomposition algorithm is based on
    **Halko, Martinsson, Tropp. Finding structure with randomness, 2009.**

    .. [3] If `corpus` is a scipy.sparse matrix instead, it is assumed the whole
       corpus fits into core memory and a different (more efficient) code path is chosen.
    """
    rank = int(rank)
    if extra_dims is None:
        samples = max(10, 2 * rank) # use more samples than requested factors, to improve accuracy
    else:
        samples = rank + int(extra_dims)
    logger.info("using %i extra samples and %i power iterations" % (samples - rank, power_iters))

    num_terms = int(num_terms)

    eps = max(float(eps), 1e-9) # must ignore near-zero eigenvalues (probably numerical error); the associated eigenvectors are typically unstable/garbage

    # first phase: construct the orthonormal action matrix Q = orth(Y) = orth((A * A.T)^q * A * O)
    # build Y in blocks of `chunks` documents (much faster than going one-by-one
    # and more memory friendly than processing all documents at once)
    y = numpy.zeros(dtype=dtype, shape=(num_terms, samples))
    logger.info("1st phase: constructing %s action matrix" % str(y.shape))

    if scipy.sparse.issparse(corpus):
        m, n = corpus.shape
        assert num_terms == m, "mismatch in number of features: %i in sparse matrix vs. %i parameter" % (m, num_terms)
        o = numpy.random.normal(0.0, 1.0, (n, samples)).astype(y.dtype) # draw a random gaussian matrix
        sparsetools.csc_matvecs(m, n, samples, corpus.indptr, corpus.indices,
                                corpus.data, o.ravel(), y.ravel()) # y = corpus * o
        del o
        if y.dtype != dtype:
            y = y.astype(dtype)
        logger.debug("running %i power iterations" % power_iters)
        for power_iter in xrange(power_iters):
            y = corpus.T * y
            y = corpus * y
    else:
        chunker = itertools.groupby(enumerate(corpus), key = lambda (docno, doc): docno / chunks)
        num_docs = 0
        for chunk_no, (key, group) in enumerate(chunker):
            logger.info('PROGRESS: at document #%i' % (chunk_no * chunks))
            # construct the chunk as a sparse matrix, to minimize memory overhead
            # definitely avoid materializing it as a dense (num_terms x chunks) matrix!
            chunk = matutils.corpus2csc((doc for _, doc in group), num_terms=num_terms, dtype=dtype) # documents = columns of sparse CSC
            m, n = chunk.shape
            assert m == num_terms
            assert n <= chunks # the very last chunk of A is allowed to be smaller in size
            num_docs += n
            logger.debug("multiplying chunk * gauss")
            o = numpy.random.normal(0.0, 1.0, (n, samples)).astype(dtype) # draw a random gaussian matrix
            sparsetools.csc_matvecs(num_terms, n, samples, chunk.indptr, # y = y + chunk * o
                                    chunk.indices, chunk.data, o.ravel(), y.ravel())
            del chunk, o

        for power_iter in xrange(power_iters):
            logger.info("running power iteration #%i" % (power_iter + 1))
            yold = y.copy()
            y[:] = 0.0
            chunker = itertools.groupby(enumerate(corpus), key = lambda (docno, doc): docno / chunks)
            for chunk_no, (key, group) in enumerate(chunker):
                logger.info('PROGRESS: at document #%i/%i' % (chunk_no * chunks, num_docs))
                chunk = matutils.corpus2csc((doc for _, doc in group), num_terms=num_terms, dtype=dtype) # documents = columns of sparse CSC
                tmp = chunk.T * yold
                tmp = chunk * tmp
                del chunk
                y += tmp
            del yold

    logger.info("orthonormalizing %s action matrix" % str(y.shape))
    y = [y]
    q, r = matutils.qr_destroy(y) # orthonormalize the range
    del y
    samples = clipSpectrum(numpy.diag(r), samples, discard = eps)
    qt = numpy.asfortranarray(q[:, :samples].T) # discard bogus columns, in case Y was rank-deficient
    del q

    if scipy.sparse.issparse(corpus):
        b = qt * corpus
        logger.info("2nd phase: running dense svd on %s matrix" % str(b.shape))
        u, s, vt = numpy.linalg.svd(b, full_matrices=False)
        del b, vt
    else:
        # second phase: construct the covariance matrix X = B * B.T, where B = Q.T * A
        # again, construct X incrementally, in chunks of `chunks` documents from the streaming
        # input corpus A, to avoid using O(number of documents) memory
        x = numpy.zeros(shape = (samples, samples), dtype = dtype)
        logger.info("2nd phase: constructing %s covariance matrix" % str(x.shape))
        chunker = itertools.groupby(enumerate(corpus), key = lambda (docno, doc): docno / chunks)
        for chunk_no, (key, group) in enumerate(chunker):
            logger.info('PROGRESS: at document #%i/%i' % (chunk_no * chunks, num_docs))
            chunk = matutils.corpus2csc((doc for _, doc in group), num_terms=num_terms, dtype=dtype)
            b = qt * chunk # dense * sparse matrix multiply
            x += numpy.dot(b, b.T) # TODO should call the BLAS routine SYRK, but there is no SYRK wrapper in scipy :(
            del chunk, b

        # now we're ready to compute decomposition of the small matrix X
        logger.info("running dense decomposition on %s covariance matrix" % str(x.shape))
        u, s, vt = numpy.linalg.svd(x) # could use linalg.eigh, but who cares... and svd returns the factors already sorted :)
        s = numpy.sqrt(s) # sqrt to go back from singular values of X to singular values of B = singular values of the corpus

    logger.info("computing the final decomposition")
    keep = clipSpectrum(s**2, rank, discard=eps)
    u = numpy.asfortranarray(u[:, :keep])
    s = s[:keep]
    gemm = matutils.blas('gemm', u)
    u = gemm(1.0, qt, u, trans_a=True)
    return u, s


