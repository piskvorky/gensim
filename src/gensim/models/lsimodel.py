#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Module for Latent Semantic Indexing.
"""


import logging
import itertools

import numpy
import scipy.sparse

# scipy is still an experimental package and locations change, so try to work
# around differences (currently only triu in scipy 0.7 vs. 0.8)
from scipy.linalg.blas import get_blas_funcs
from scipy.linalg.lapack import get_lapack_funcs, find_best_lapack_type
try:
    from scipy.linalg.basic import triu
except ImportError:
    from scipy.linalg.special_matrices import triu

try:
    import sparsesvd
except ImportError:
    raise ImportError("sparsesvd module not found; run `easy_install sparsesvd`")


from gensim import interfaces, matutils, utils


logger = logging.getLogger('lsimodel')
logger.setLevel(logging.INFO)



def clipSpectrum(s, k):
    """
    Given singular values `s`, return how many factors should be kept to avoid
    storing spurious values. The returned value is at most `k`.
    """
    # compute relative contribution of each singular value towards the energy spectrum
    rel_spectrum = numpy.abs(1.0 - numpy.cumsum(s**2 / numpy.sum(s ** 2)))
    # ignore the last 0.1% (or 1/k percent, whichever is smaller) of the spectrum
    small = 1 + len(numpy.where(rel_spectrum > min(0.001, 1.0 / k))[0]) 
    k = min(k, small) # clip against k
    logger.info("keeping %i factors (discarding %.2f%% of the energy spectrum)" %
                (k, 100 * rel_spectrum[k - 1]))
    return k


class Projection(utils.SaveLoad):
    def __init__(self, m, k, docs = None):
        """
        Store (U, S) projection itself. This is the class taking care of 'core math';
        interfacing with corpora, training etc is done through class LsiModel.
        
        `docs` is either a spare matrix or a corpus which, when converted to a 
        sparse matrix, must fit comfortably into main memory.
        """
        self.m, self.k = m, k
        if docs is not None:
            # base case decomposition: given a job `docs`, compute its decomposition 
            # in core, algorithm 1
            if utils.isCorpus(docs):
                docs = matutils.corpus2csc(m, docs)
            if m * k < 10000:
                docs = docs.todense()
                logger.info("computing dense SVD of %s matrix" % str(docs.shape))
                # SVDLIBC gives spurious results for small matrices.. run full
                # LAPACK on them instead!
                u, s, vt = numpy.linalg.svd(docs, full_matrices = False)
            else:
                logger.info("computing sparse SVD of %s matrix" % str(docs.shape))
                ut, s, vt = sparsesvd.sparsesvd(docs, k + 30) # ask for extra factors, because for some reason SVDLIBC sometimes returns fewer factors than requested
                u = ut.T
                del ut
            del vt
            k = clipSpectrum(s, self.k)
            self.u, self.s = u[:, :k], s[:k]
        else:
            self.u, self.s = None, None
    
    
    def empty_like(self):
        return Projection(self.m, self.k)
    

    def merge(self, other, decay = 1.0):
        """
        Merge this Projection with another. 
        
        Content of `other` is destroyed in the process, so pass this function a 
        copy if you need it further.
        
        This is the optimized merge described in algorithm 5.
        """
        if other.u is None:
            # the other projection is empty => do nothing
            return
        if self.u is None:
            # we are empty => result of merge is the other projection, whatever it is
            self.u = other.u.copy()
            self.s = other.s.copy()
            return
        if self.m != other.m:
            raise ValueError("vector space mismatch: update has %s features, expected %s" %
                             (other.m, self.m))
        logger.info("merging projections: %s + %s" % (str(self.u.shape), str(other.u.shape)))
#        diff = numpy.dot(self.u.T, self.u) - numpy.eye(self.u.shape[1])
#        logger.info('orth error after=%f' % numpy.sum(diff * diff))
        m, n1, n2 = self.u.shape[0], self.u.shape[1], other.u.shape[1]
        # TODO Maybe keep the bases as elementary reflectors, without 
        # forming explicit matrices with gorgqr.
        # The only operation we ever need is basis^T*basis ond basis*component.
        # But how to do that in numpy? And is it fast(er)?
        
        # find component of u2 orthogonal to u1
        # IMPORTANT: keep matrices in suitable order for matrix products; failing to do so gives 8x lower performance :(
        self.u = numpy.asfortranarray(self.u) # does nothing if input already fortran-order array
        other.u = numpy.asfortranarray(other.u)
        gemm, = get_blas_funcs(('gemm',), (self.u,))
        logger.debug("constructing orthogonal component")
        c = gemm(1.0, self.u, other.u, trans_a = True)
        gemm(-1.0, self.u, c, beta = 1.0, c = other.u, overwrite_c = True)
        
        # perform q, r = QR(component); code hacked out of scipy.linalg.qr
        logger.debug("computing QR of %s dense matrix" % str(other.u.shape))
        geqrf, = get_lapack_funcs(('geqrf',), (other.u,))
        qr, tau, work, info = geqrf(other.u, lwork = -1, overwrite_a = True) # segfaults for overwrite_a=True!!
        logger.debug("GEQRF work size: %s" % work[0])
        qr, tau, work, info = geqrf(other.u, lwork = work[0], overwrite_a = True)
        del other.u
        assert info >= 0
        r = triu(qr[:n2, :n2])
        if m < n2: # rare case...
            qr = qr[:,:m] # retains fortran order
        gorgqr, = get_lapack_funcs(('orgqr',), (qr,))
        q, work, info = gorgqr(qr, tau, lwork = -1, overwrite_a = True)
        q, work, info = gorgqr(qr, tau, lwork = work[0], overwrite_a = True)
        assert info >= 0, "qr failed"
        assert q.flags.f_contiguous
        
        # find rotation that diagonalizes r
        k = numpy.bmat([[numpy.diag(decay * self.s), c * other.s], [matutils.pad(numpy.matrix([]).reshape(0, 0), n2, n1), r * other.s]])
        logger.debug("computing SVD of %s dense matrix" % str(k.shape))
        u_k, s_k, _ = numpy.linalg.svd(k, full_matrices = False) # TODO *ugly overkill*!! only need first self.k SVD factors... but there is no LAPACK wrapper for partial svd/eigendecomp in numpy :(
        
        k = clipSpectrum(s_k, self.k)
        u_k, s_k = u_k[:, :k], s_k[:k]
        
        # update & rotate current basis U
        logger.debug("updating orthonormal basis U")
        self.u = gemm(1.0, self.u, u_k[:n1]) # TODO temporarily creates an extra (m,k) dense array in memory. find a way to avoid this!
        gemm(1.0, q, u_k[n1:], beta = 1.0, c = self.u, overwrite_c = True) # u = [u,u']*u_k
        self.s = s_k
#        diff = numpy.dot(self.u.T, self.u) - numpy.eye(self.u.shape[1])
#        logger.info('orth error after=%f' % numpy.sum(diff * diff))
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
    def __init__(self, corpus = None, id2word = None, numTopics = 200, 
                 chunks = 10000, decay = 1.0, serial_only = None):
        """
        `numTopics` is the number of requested factors (latent dimensions). 
        
        After the model has been trained, you can estimate topics for an
        arbitrary, unseen document, using the ``topics = self[document]`` dictionary 
        notation. You can also add new training documents, with ``self.addDocuments``,
        so that training can be stopped and resumed at any time, and the
        LSI transformation is available at any point.

        If you specify a `corpus`, it will be used to train the model. See the 
        method `addDocuments` for a description of the `chunks` and `decay` parameters.
        
        The algorithm will automatically try to find active nodes on other computers
        and run in a distributed manner; if this fails, it falls back to serial mode
        (single core). To suppress distributed computing, set the `serial_only`
        constructor parameter to True.
        
        Example:
        
        >>> lsi = LsiModel(corpus, numTopics = 10)
        >>> print lsi[doc_tfidf]
        >>> lsi.addDocuments(corpus2) # update LSI on additional documents
        >>> print lsi[doc_tfidf]
        
        """
        self.id2word = id2word
        self.numTopics = numTopics # number of latent topics
        self.chunks = chunks
        self.decay = decay
        
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

        if serial_only:
            logger.info("using slave LSI version on this node")
            self.dispatcher = None
        else:
            try:
                import Pyro
                ns = Pyro.naming.locateNS()
                dispatcher = Pyro.core.Proxy('PYRONAME:gensim.dispatcher@%s' % ns._pyroUri.location)
                logger.debug("looking for dispatcher at %s" % str(dispatcher._pyroUri))
                dispatcher.initialize(id2word = self.id2word, numTopics = numTopics, 
                                      chunks = chunks, decay = decay, 
                                      serial_only = True)
                self.dispatcher = dispatcher
                logger.info("using distributed version with %i workers" % len(dispatcher.getworkers()))
            except Exception, err:
                if serial_only is not None: 
                    # distributed version was specifically requested, so this is an error state
                    logger.error("failed to initialize distributed LSI (%s)" % err)
                    raise RuntimeError("failed to initialize distributed LSI (%s)" % err)
                else:
                    # user didn't request distributed specifically; just let him know we're running in serial
                    logger.info("distributed LSI not available, running LSI in serial mode (%s)" % err)
                self.dispatcher = None

        if corpus is not None:
            self.addDocuments(corpus, chunks = chunks)
    
    
    def addDocuments(self, corpus, chunks = None, decay = None):
        """
        Update singular value decomposition factors to take into account a new 
        corpus of documents.
        
        Training proceeds in chunks of `chunks` documents at a time. If the 
        distributed mode is on, each chunk is sent to a different worker/computer.
        Size of `chunks` is a tradeoff between increased speed (bigger `chunks`) vs. 
        lower memory footprint (smaller `chunks`). Default is processing 10,000 documents
        at a time.

        Setting `decay` < 1.0 causes re-orientation towards new data trends in the 
        input document stream, by giving less emphasis to old observations. This allows
        SVD to gradually "forget" old observations and give more preference to 
        new ones. The decay is applied once after every `chunks` documents.
        """
        logger.info("updating SVD with new documents")
        
        # get computation parameters; if not specified, use the ones from constructor
        if chunks is None:
            chunks = self.chunks
        if decay is None:
            decay = self.decay
        
        if utils.isCorpus(corpus):
            # do the actual work -- perform iterative singular value decomposition.
            chunker = itertools.groupby(enumerate(corpus), key = lambda val: val[0] / chunks)
            doc_no = 0
            for chunk_no, (key, group) in enumerate(chunker):
                # construct the job as a sparse matrix, to minimize memory overhead
                # definitely avoid materializing it as a dense matrix!
                job = matutils.corpus2csc(self.numTerms, (doc for _, doc in group))
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
                    self.printDebug(5)
            
            if self.dispatcher:
                logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                import time
                while self.dispatcher.jobsdone() <= chunk_no:
                    time.sleep(0.5) # check every half a second
                logger.info("all jobs finished, downloading final projection")
                del self.projection
                self.projection = self.dispatcher.getstate()
                logger.info("decomposition complete")
        else:
            assert not self.dispatcher, "must be in serial mode to receive jobs"
            assert isinstance(corpus, scipy.sparse.csc_matrix)
            update = Projection(self.numTerms, self.numTopics, corpus)
            self.projection.merge(update, decay = decay)
            logger.info("processed sparse job of %i documents" % (corpus.shape[1]))
            self.printDebug(5)

    
    def __str__(self):
        return "LsiModel(numTerms=%s, numTopics=%s, decay=%s, chunks=%s)" % \
                (self.numTerms, self.numTopics, self.decay, self.chunks)


    def __getitem__(self, bow, scaled = False):
        """
        Return latent representation, as a list of (topic_id, topic_value) 2-tuples.
        
        This is done by folding input document into the latent topic space. 
        
        Note that this function returns the latent space representation **scaled by the
        singular values**. To return non-scaled embedding, set `scaled` to False.
        """
        # if the input vector is in fact a corpus, return a transformed corpus as result
        if utils.isCorpus(bow):
            return self._apply(bow)
        
        assert self.projection.u is not None, "decomposition not initialized yet"
        vec = numpy.asfortranarray(matutils.sparse2full(bow, self.numTerms), dtype = self.projection.u.dtype)
        vec.shape = (self.numTerms, 1)
        topicDist = scipy.linalg.fblas.dgemv(1.0, self.projection.u, vec, trans = True) # u^T * x
        if scaled:
            topicDist = (1.0 / self.projection.s) * topicDist # s^-1 * u^T * x
        return [(topicId, float(topicValue)) for topicId, topicValue in enumerate(topicDist)
                if numpy.isfinite(topicValue) and not numpy.allclose(topicValue, 0.0)]
    

    def printTopic(self, topicNo, topN = 10):
        """
        Return a specified topic (0 <= `topicNo` < `self.numTopics`) as string in human readable format.
        
        >>> lsimodel.printTopic(10, topN = 5)
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
        return ' + '.join(['%.3f * "%s"' % (1.0 * c[val] / norm, self.id2word[val]) for val in most])


    def printDebug(self, numTopics = 5, numWords = 10):
        for i in xrange(min(numTopics, self.numTopics)):
            if i < len(self.projection.s):
                logger.info("lsi topic %i (%s): %s" % (i, self.projection.s[i], self.printTopic(i, topN = numWords)))
#endclass LsiModel


def svdUpdate(U, S, V, a, b):
    """
    Update SVD of an (m x n) matrix `X = U * S * V^T` so that
    `[X + a * b^T] = U' * S' * V'^T`
    and return `U'`, `S'`, `V'`.
    
    The original matrix X is not needed at all, so this function implements flexible 
    *online* updates to an existing decomposition. 
    
    `a` and `b` are (m, 1) and (n, 1) matrices.
    
    You can set V to None if you're not interested in the right singular
    vectors. In that case, the returned V' will also be None (saves memory).
    
    This is the rank-1 update as described in
    *Brand, 2006: Fast low-rank modifications of the thin singular value decomposition*
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
    
    See **Genevieve Gorrell: Generalized Hebbian Algorithm for Incremental Singular 
    Value Decomposition in Natural Language Processing. EACL 2006.**
    
    Use of this function is deprecated; although it works, it is several orders of 
    magnitude slower than our own, direct (non-stochastic) version (which
    operates in a single pass, too, and can be distributed). 
    
    I keep this function here purely for backup reasons.
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


#def stochasticSvd(a, numTerms, numFactors, p = None, q = 0):
#    """
#    SVD decomposition based on stochastic approximation.
#    
#    See **Halko, Martinsson, Tropp. Finding structure with randomness, 2009.**
#    
#    This is the randomizing version with oversampling, but without power iteration. 
#    """
#    k = numFactors
#    if p is None:
#        l = 2 * k # default oversampling
#    else:
#        l = k + p
#    
#    # stage A: construct the "action" basis matrix Q
#    y = numpy.empty(dtype = numpy.float64, shape = (numTerms, l)) # in double precision, because we will be computing orthonormal basis on this possibly ill-conditioned projection
#    for i, row in enumerate(a):
#        y[i] = column_stack(matutils.sparse2full(doc, numTerms) * numpy.random.normal(0.0, 1.0, numTerms)
#                           for doc in corpus)
#    q = numpy.linalg.qr()
#    
    
