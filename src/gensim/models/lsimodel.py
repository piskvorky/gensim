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


from gensim import interfaces, matutils, utils


#logging.basicConfig(format = '%(asctime)s : %(module)s(%(funcName)s) : %(levelname)s : %(message)s')
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger('lsimodel')
logger.setLevel(logging.DEBUG)




class Projection(object):
    def __init__(self, m, k, docs = None):
        """
        Compute (U, S) projection from a corpus.
        
        `docs` is either a spare matrix or a corpus which, when converted to a 
        sparse matrix, must fit comfortably into main memory.
        
        This is the "base step" for computing incremental SVDs of extremely
        large matrices -- SVDs are computed from sparse submatrices that fit
        in memory, and then merged via the `merge()` method.
        """
        self.m, self.k = m, k
        if docs is not None:
            if utils.isCorpus(docs):
                docs = matutils.corpus2csc(m, docs)
            logger.info("computing sparse SVD of %s matrix" % str(docs.shape))
            import sparseSVD
            self.u, self.s, vt = sparseSVD.doSVD(docs, k)
            del vt
            self.u, self.s = self.u[:k].T, self.s[:k]
        else:
            self.u, self.s = None, None
    

    def merge(self, other, decay = 1.0, method = 'qr_inplace'):
        """
        Merge this projection with another.
        
        `other` is destroyed in the process.
        """
        if not isinstance(other, Projection):
            if not other:
                return
            if utils.isCorpus(other):
                logger.debug("constructing dense document matrix")
                other = numpy.asmatrix(numpy.column_stack(matutils.sparse2full(doc, self.m) for doc in other))
            else:
                other = numpy.asarray(other)
            logger.info("updating SVD with %i new documents" % (other.shape[1]))
            if self.u is None:
                u, s, _ = numpy.linalg.svd(other, full_matrices = False)
                u, s = u[:, :self.k], numpy.diag(s[:self.k])
                self.u, self.s = numpy.asmatrix(numpy.asfortranarray(u)), s
                return
            logger.debug("constructing orthogonal component")
            self.u = numpy.asmatrix(numpy.asfortranarray(self.u))
            other = numpy.asmatrix(numpy.asfortranarray(other))
            n1, n2 = self.u.shape[1], other.shape[1]
            m = self.u.T * other
            other -= self.u * m
            logger.debug("computing QR of %s dense matrix" % str(other.shape))
            p, r = numpy.linalg.qr(other)
            del other
            k = numpy.bmat([[decay * self.s, m], [matutils.pad(numpy.matrix([]).reshape(0, 0), n2, n1), r]])
            logger.debug("computing SVD of %s dense matrix" % str(k.shape))
            u_k, s_k, _ = numpy.linalg.svd(k, full_matrices = False)
            u_k, s_k = u_k[:, :self.k], numpy.diag(s_k[:self.k])
            self.u = self.u * u_k[:n1] # rotate current basis
            p = p * u_k[n1:] # rotate the update
            self.u += p # then add them together (avoid explicitly creating [U,P] matrix in memory)
            self.s = s_k
            return
        if other.u is None:
            # the other projection is empty => do nothing
            return
        if self.u is None:
            # we are empty => result of merge is the other projection, whatever it is
            self.u = other.u.copy()
            self.s = other.s.copy()
            return
        assert self.m == other.m
        logger.info("merging projections: %s + %s" % (str(self.u.shape), str(other.u.shape)))
        if method == 'svd':
            # merge two projections t1,t2 directly by computing svd[t1,t2]
            logger.debug("constructing update matrix")
            smat = numpy.bmat([decay * self.u, other.u])
            del self.u, other.u
            logger.debug("computing SVD of %s dense matrix" % str(smat.shape))
            # we need in-core dense SVD => might as well use LAPACK
            self.u, s, _ = numpy.linalg.svd(smat, full_matrices = False)
            self.s = numpy.diag(s[:self.k])
            self.u = self.u[:, :self.k] * self.s
        elif method == 'qr':
            logger.debug("constructing update matrix")
            smat = numpy.bmat([decay * self.u, other.u])
            del self.u, other.u
            logger.debug("computing QR of %s dense matrix" % str(smat.shape))
            self.u, r = numpy.linalg.qr(smat)
            u_r, s_r, _ = numpy.linalg.svd(r, full_matrices = False)
            u_r, s_r = u_r[:, :self.k], numpy.diag(s_r[:self.k])
            self.u = self.u * (u_r * s_r)
            self.s = s_r
        elif method == 'qr_inplace':
            m, n1, n2 = self.u.shape[0], self.u.shape[1], other.u.shape[1]
            # TODO Maybe keep the bases in their natural form (rotations), without 
            # forming explicit matrices that occupy a lot of mem.
            # The only operation we ever need is basis^T*basis ond basis*component.
            # But how to do that in numpy?
            
            # find component of u2 orthogonal to u1
            # IMPORTANT: keep matrices in suitable order for matrix products; failing to do so gives 8x lower performance :(
            self.u = numpy.asfortranarray(self.u) # does nothing if input already fortran-order array
            other.u = numpy.asfortranarray(other.u)
            logger.debug("constructing orthogonal component")
            c = scipy.linalg.fblas.dgemm(1.0, self.u, other.u, trans_a = True)
            scipy.linalg.fblas.dgemm(-1.0, self.u, c, beta = 1.0, c = other.u, overwrite_c = True)
            # perform q, r = qr(component); code hacked out of scipy.linalg.qr
            logger.debug("computing QR of %s dense matrix" % str(other.u.shape))
            qr, tau, work, info = scipy.linalg.flapack.dgeqrf(other.u, lwork = -1, overwrite_a = True)
            lwork = work[0]
            qr, tau, work, info = scipy.linalg.flapack.dgeqrf(other.u, lwork = lwork, overwrite_a = True)
            del other.u
            assert info >= 0
            r = scipy.linalg.basic.triu(qr[:n2, :n2])
            if m < n2:
                q, work, info = scipy.linalg.flapack.dorgqr(qr[:,:m], tau, lwork = -1, overwrite_a = 1)
                lwork = work[0]
                q, work, info = scipy.linalg.flapack.dorgqr(qr[:,:m], tau, lwork = lwork, overwrite_a = 1)
            else:
                q, work, info = scipy.linalg.flapack.dorgqr(qr, tau, lwork = -1, overwrite_a = 1)
                lwork = work[0]
                q, work, info = scipy.linalg.flapack.dorgqr(qr, tau, lwork = lwork, overwrite_a = 1)            
            assert q.flags.f_contiguous
            assert info >= 0
            # find rotation that diagonalizes r
            k = numpy.bmat([[numpy.diag(decay * self.s), c * other.s], [matutils.pad(numpy.matrix([]).reshape(0, 0), n2, n1), r * other.s]])
            logger.debug("computing SVD of %s dense matrix" % str(k.shape))
            u_k, s_k, _ = numpy.linalg.svd(k, full_matrices = False) # TODO NOTE only need first self.k SVD factors... but there is no LAPACK routine for partial SVD :(
            u_k, s_k = u_k[:, :self.k], s_k[:self.k]
            # update & rotate current basis U
            self.u = scipy.linalg.fblas.dgemm(1.0, self.u, u_k[:n1]) # TODO temporarily creates an extra (m,k) dense array! find a way to avoid this!
            scipy.linalg.fblas.dgemm(1.0, q, u_k[n1:], beta = 1.0, c = self.u, overwrite_c = True)
            self.s = s_k
        else:
            raise ValueError("unknown merge method: %s" % MERGE_METHOD)
    

    def as_matrix(self):
        """
        Return projection as a single matrix, in single precision and row-major 
        order.
        
        Currently, this returns U^{-1}; for some applications, (U*S)^{-1} may be
        more appropriate.
        """
        # make sure we return a row-contiguous array (C-style), for fast mat*vec multiplications
        return numpy.asmatrix(numpy.ascontiguousarray(self.u, dtype = numpy.float32).T)

    
    @staticmethod
    def from_matrix(mat):
        mat = mat.T.astype(numpy.float64)
        result = Projection(m = mat.shape[0], k = mat.shape[1], docs = None)
        lens = numpy.sqrt(numpy.sum(numpy.multiply(mat, mat), axis = 0))
        result.u = numpy.asmatrix(numpy.divide(mat, lens))
        for i, row in enumerate(result.u):
            result.u[i] = numpy.where(numpy.isfinite(row), row, 0.0) # substitute all NaNs/infs with 0.0, but proceed row-by-row to save memory
        result.s = numpy.diag(lens)
        return result
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
                 chunks = 100, decay = 1.0, serial_only = False):
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
        (single core). To suppress distributed computing, enable the `serial_only`
        constructor parameter.
        
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
            logger.info("using serial LSI version on this machine")
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
                logger.info("failed to initialize distributed LSI (%s)" % err)
                self.dispatcher = None

        if corpus is not None:
            self.add_documents(corpus, chunks = chunks)
    
    
    def add_documents(self, corpus, chunks = None, decay = None, update_projection = False): # FIXME test=>update True!!!
        """
        Update singular value decomposition factors to take into account a new 
        corpus of documents.
        
        Training proceeds in chunks of `chunks` documents at a time. If the 
        distributed mode is on, each chunk is sent to a different worker/computer.
        Size of `chunks` is a tradeoff between increased speed (bigger `chunks`) vs. 
        lower memory footprint (smaller `chunks`). Default is processing 100 documents
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
        
        self.projection_on(False) # from now on, work with the self.projection.u matrix
        
        if utils.isCorpus(corpus):
            # do the actual work -- perform iterative singular value decomposition.
            chunker = itertools.groupby(enumerate(corpus), key = lambda val: val[0] / chunks)
            doc_no = 0
            for chunk_no, (key, group) in enumerate(chunker):
#                job = [doc for _, doc in group]
#                doc_no += len(job)
#                self.projection.merge(job)
#                logger.info("processed documents up to #%s" % doc_no)
#                continue
                # construct the job as a sparse matrix, to minimize memory overhead
                # definitely avoid materializing it as a dense matrix
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
    #                self.projection.merge(other = job, decay = decay)
                    update = Projection(self.numTerms, self.numTopics, job)
                    del job
                    self.projection.merge(update, decay = decay)
                    del update
                    logger.info("processed documents up to #%s" % doc_no)
            
            if self.dispatcher:
                logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                import time
                while self.dispatcher.jobsdone() <= chunk_no:
                    time.sleep(0.5) # check every half a second
                logger.info("all jobs finished, downloading final projection")
                del self.projection
                self.projection = self.dispatcher.getstate()
        else:
            # assume we received a job which is already a chunk in CSC format
            assert not self.dispatcher, "must be in serial mode to receive jobs"
            update = Projection(self.numTerms, self.numTopics, corpus)
            self.projection.merge(update, decay = decay)
            logger.info("processed sparse job of %i documents" % (corpus.shape[1]))
        
        if update_projection:
            self.projection_on(True)
    
    
    def projection_on(self, on):
        """
        Calculate projection needed to get the topic-document matrix from 
        a term-document matrix.
       
        The way to represent a vector `x` in latent space is lsi[x] = v = self.s^-1 * self.u^-1 * x,
        so the projection is self.s^-1 * self.u^-1.
       
        The way to compare two documents `x1`, `x2` is to compute v1 * self.s^2 * v2.T, so
        we pre-multiply v * s (ie., scale axes by singular values), and return
        that directly as the representation of `x` in LSI space. This conveniently 
        simplifies to lsi[x] = self.u.T * x, so the projection is just self.u.T
        
        Note that neither `v` (the right singular vectors) nor `s` (the singular 
        values) are used at all in this scaled transformation.
        """
        # this whole Projection/matrix business is meant to save memory,
        # so that we don't need to keep both matrices in memory at the same time, 
        # or at least not for long.
        #
        # The projection matrix is smaller, well-aligned and faster to use, while
        # Projection objects offers more numerical precision. so use Projection 
        # during computation and switch to matrix once finished.
        if on: 
            # we want matrix
            if isinstance(self.projection, Projection):
                logger.info("computing transformation projection")
                self.projection = self.projection.as_matrix() # conversion to single => loses precision!
        else:
            # we want Projection opbject
            if not isinstance(self.projection, Projection):
                logger.info("initializing Projection object from matrix")
                self.projection = Projection.from_matrix(self.projection)

    
    def __str__(self):
        return "LsiModel(numTerms=%s, numTopics=%s, chunks=%s)" % \
                (self.numTerms, self.numTopics, self.chunks)


    def __getitem__(self, bow, scaled = True):
        """
        Return latent distribution, as a list of (topic_id, topic_value) 2-tuples.
        
        This is done by folding input document into the latent topic space. 
        
        Note that this function returns the latent space representation **scaled by the
        singular values**. To return non-scaled embedding, set `scaled` to False.
        """
        self.projection_on(True)
        # if the input vector is in fact a corpus, return a transformed corpus as result
        if utils.isCorpus(bow):
            return self._apply(bow)
        
        vec = matutils.sparse2full(bow, self.numTerms)
        vec.shape = (self.numTerms, 1)
        assert vec.dtype == numpy.float32 and self.projection.dtype == numpy.float32
        topicDist = self.projection * vec
        if not scaled:
            topicDist = numpy.diag(numpy.diag(1.0 / self.s)) * topicDist
        return [(topicId, float(topicValue)) for topicId, topicValue in enumerate(topicDist)
                if numpy.isfinite(topicValue) and not numpy.allclose(topicValue, 0.0)]
    

    def printTopic(self, topicNo, topN = 10):
        """
        Return a specified topic (0 <= `topicNo` < `self.numTopics`) as string in human readable format.
        
        >>> lsimodel.printTopic(10, topN = 5)
        '-0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + -0.174 * "functor" + -0.168 * "operator"'
        
        """
        self.projection_on(True)
#        c = numpy.asarray(self.u[:, topicNo]).flatten()
        c = numpy.asarray(self.projection[topicNo, :]).flatten()
        norm = numpy.sqrt(numpy.sum(c * c))
        most = numpy.abs(c).argsort()[::-1][:topN]
        return ' + '.join(['%.3f * "%s"' % (1.0 * c[val] / norm, self.id2word[val]) for val in most])
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
    
