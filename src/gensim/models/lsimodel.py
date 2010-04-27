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

from gensim import interfaces, matutils, utils


class LsiModel(interfaces.TransformationABC):
    """
    Objects of this class allow building and maintaining a model for Latent 
    Semantic Indexing (also known as Latent Semantic Analysis).
    
    The main methods are:
    
    1. constructor, which calculates the latent topics space, effectively 
       initializing the model,
    2. the [] method, which returns representation of any input document in the 
       computed latent space.
    
    Model persistency is achieved via its load/save methods.
    """
    def __init__(self, corpus, id2word = None, numTopics = 200):
        """
        Find latent space based on the corpus provided.

        `numTopics` is the number of requested factors (latent dimensions).
        
        After the model has been initialized, you can estimate topics for an
        arbitrary, unseen document, using the ``topics = self[document]`` dictionary 
        notation.
        
        Example:
        
        >>> lsi = LsiModel(corpus, numTopics = 10)
        >>> print lsi[doc_tfidf]
        
        """
        self.id2word = id2word
        self.numTopics = numTopics # number of latent topics
        if corpus is not None:
            self.initialize(corpus)

    
    def __str__(self):
        return "LsiModel(numTerms=%s, numTopics=%s)" % \
                (self.numTerms, self.numTopics)


    def initialize(self, corpus, chunks = 100, keepDecomposition = False, dtype = numpy.float64):
        """
        Run SVD decomposition on the corpus. This will define the latent space into 
        which terms and documents will be mapped.
        
        The SVD is created incrementally, in blocks of `chunks` documents. In the
        end, a `self.projection` matrix is constructed that can be used to transform 
        documents into the latent space. The `U, S, V` decomposition itself is 
        discarded, unless `keepDecomposition` is True, in which case it is stored 
        in `self.u`, `self.s` and `self.v`.
        
        `dtype` dictates precision used for intermediate computations; the final 
        projection will however always be of type numpy.float32.
        
        The algorithm is adapted from:
        **M. Brand. 2006. Fast low-rank modifications of the thin singular value decomposition**
        """
        if self.id2word is None:
            logging.info("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dictFromCorpus(corpus)
            self.numTerms = len(self.id2word)
        else:
            self.numTerms = 1 + max([-1] + self.id2word.keys())
        
        # initialize decomposition (zero documents so far)
        self.u = numpy.matrix(numpy.zeros((self.numTerms, self.numTopics)), dtype = dtype)
        self.s = numpy.matrix(numpy.zeros((self.numTopics, self.numTopics)), dtype = dtype)
        #self.v = numpy.matrix(numpy.zeros((0, self.numTopics)), dtype = dtype)
        self.v = None
        
        # do the actual work -- perform iterative singular value decomposition.
        # this is done by sequentially updating SVD with `chunks` new documents
        chunker = itertools.groupby(enumerate(corpus), key = lambda val: val[0] / chunks)
        for chunkNo, (key, group) in enumerate(chunker):
            # convert the chunk of sparse documents to full vectors
            docs = [matutils.sparse2full(doc, self.numTerms) for docNo, doc in group]
#            self.svdAddCols(docs, reorth = chunkNo % 100 == 99) # reorthogonalize once in every "100*chunks" documents
            self.svdAddCols(docs, reorth = False)
            logging.info("processed documents up to #%s" % docNo)
        

        # calculate projection needed to get document-topic matrix from term-document matrix.
        #
        # the way to represent a vector `x` in latent space is lsi[x] = v = self.s^-1 * self.u^-1 * x,
        # so the projection is self.s^-1 * self.u^-1.
        #
        # the way to compare two documents `x1`, `x2` is to compute v1 * self.s^2 * v2.T, so
        # we pre-multiply v * s (ie., scale axes by singular values), and return
        # that directly as the representation of `x` in LSI space.
        #
        # this conveniently simplifies to lsi[x] = self.u.T * x, so the projection is 
        # just self.u.T
        # 
        # note that neither `v` (the right singular vectors) nor `s` (the singular 
        # values) are used at all in the transformation
        self.projection = self.u.T.astype(numpy.float32).copy('C')
        
        if not keepDecomposition:
            # once we have the projection stored in self, discard u*s*v decomposition to free up memory
            del self.u, self.v


    def __getitem__(self, bow, scaled = True):
        """
        Return latent distribution, as a list of (topic_id, topic_value) 2-tuples.
        
        This is done by folding input document into the latent topic space. 
        
        Note that this function returns the latent space representation **scaled by the
        singular values**. To return non-scaled embedding, set `scaled` to False.
        """
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
    
    
    def svdAddCols(self, docs, decay = 1.0, reorth = False):
        """
        Update singular value decomposition factors to take into account new 
        documents `docs`.
        
        This function corresponds to the general update of Brand (section 2), 
        specialized for `A = docs.T` and `B` trivial (no update to matrix rows).

        The documents are assumed to be a list of full vectors (ie. not sparse 2-tuples).
        
        Compute new decomposition `u'`, `s'`, `v'` so that if the current matrix `X` decomposes to
        `u * s * v^T ~= X`, then
        `u' * s' * v'^T ~= [X docs^T]`
        
        `u`, `s`, `v` and their new values `u'`, `s'`, `v'` are stored within `self` (ie. as 
        `self.u`, `self.v` etc.).
        
        `self.v` can be set to `None`, in which case it is completely ignored. This saves a
        bit of speed and a lot of memory, especially for huge corpora (size of `v` is
        linear in the number of added documents).
        """
        logging.debug("updating SVD with %i new documents" % len(docs))
        keepV = self.v is not None
        if not keepV and reorth:
            raise TypeError("cannot reorthogonalize without the right singular vectors (v must not be None)")
        a = numpy.matrix(numpy.asarray(docs)).T
        m, k = self.u.shape
        if keepV:
            n, k2 = self.v.shape
            assert k == k2, "left/right singular vectors shape mismatch!"
        m2, c = a.shape
        assert m == m2, "new documents must be in the same term-space as the original documents (old %s, new %s)" % (self.u.shape, a.shape)
        
        # construct orthogonal basis for (I - U * U^T) * A
        logging.debug("constructing orthogonal component")
        m = self.u.T * a # (k, m) * (m, c) = (k, c)
        logging.debug("computing orthogonal basis")
        P, Ra = numpy.linalg.qr(a - self.u * m) # equation (2)

        # allow re-orientation towards new data trends in the document stream, by giving less emphasis on old values
        self.s *= decay
        
        # now we're ready to construct K; K will be mostly diagonal and sparse, with
        # lots of structure, and of shape only (k + c, k + c), so its direct SVD 
        # ought to be fast for reasonably small additions of new documents (ie. tens 
        # or hundreds of new documents at a time).
        empty = matutils.pad(numpy.matrix([]).reshape(0, 0), c, k)
        K = numpy.bmat([[self.s, m], [empty, Ra]]) # (k + c, k + c), equation (4)
        logging.debug("computing %s SVD" % str(K.shape))
        uK, sK, vK = numpy.linalg.svd(K, full_matrices = False) # there is no python wrapper for partial svd => request all k + c factors :(
        lost = 1.0 - numpy.sum(sK[: k]) / numpy.sum(sK)
        logging.debug("discarding %.1f%% of data variation" % (100 * lost))
        
        # clip full decomposition to the requested rank
        uK = numpy.matrix(uK[:, :k])
        sK = numpy.matrix(numpy.diag(sK[: k]))
        vK = numpy.matrix(vK.T[:, :k]) # .T because numpy transposes the right vectors V, so we need to transpose it back: V.T.T = V
        
        # and finally update the left/right singular vectors
        logging.debug('rotating subspaces')
        self.s = sK
        
        # update U piece by piece, to avoid creating (huge) temporary arrays in a complex expression
        P = P * uK[k:]
        self.u = self.u * uK[:k]
        self.u += P # (m, k) * (k, k) + (m, c) * (c, k) = (m, k), equation (5)
        del P # free up memory
        
        if keepV:
            self.v = self.v * vK[:k, :] # (n + c, k) * (k, k) = (n + c, k)
            rot = vK[k:, :]
            self.v = numpy.bmat([[self.v], [rot]])
            
            if reorth:
                # The original article contains section 4.2 on keeping the rotations separate
                # from the subspaces (decomping V into Vsubspace * Vrotate), which further reduces 
                # complexity and improves numerical properties for rank-1 updates.
                #
                # I did not implement this step yet; instead, force the (expensive)
                # reorthogonalization explicitly from time to time, by setting reorth = True
                logging.debug("re-orthogonalizing singular vectors")
                uQ, uR = numpy.linalg.qr(self.u)
                vQ, vR = numpy.linalg.qr(self.v)
                uK, sK, vK = numpy.linalg.svd(uR * self.s * vR.T, full_matrices = False)
                uK = numpy.matrix(uK[:, :k])
                sK = numpy.matrix(numpy.diag(sK[: k]))
                vK = numpy.matrix(vK.T[:, :k])
                
                logging.debug("adjusting singular values by %f%%" % 
                              (100.0 * numpy.sum(numpy.abs(self.s - sK)) / numpy.sum(numpy.abs(self.s))))
                self.u = uQ * uK
                self.s = sK
                self.v = vQ * vK
        logging.debug("added %i documents" % len(docs))


    def printTopic(self, topicNo, topN = 10):
        """
        Return a specified topic (0 <= `topicNo` < `self.numTopics`) as string in human readable format.
        
        >>> lsimodel.printTopic(10, topN = 5)
        '-0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + -0.174 * "functor" + -0.168 * "operator"'
        
        """
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
    
    `a` and `b` are (m, 1) and (n, 1) rank-1 matrices, so that svdUpdate can simulate 
    incremental addition of one new document and/or term to an already existing 
    decomposition.
    """
    rank = U.shape[1]
    m = U.T * a
    p = a - U * m
    Ra = numpy.sqrt(p.T * p)
    assert float(Ra) > 1e-10
    P = (1.0 / float(Ra)) * p
    n = V.T * b
    q = b - V * n
    Rb = numpy.sqrt(q.T * q)
    assert float(Rb) > 1e-10
    Q = (1.0 / float(Rb)) * q
    
    K = numpy.matrix(numpy.diag(list(numpy.diag(S)) + [0.0])) + numpy.bmat('m ; Ra') * numpy.bmat(' n; Rb').T
    u, s, vt = numpy.linalg.svd(K, full_matrices = False)
    tUp = numpy.matrix(u[:, :rank])
    tVp = numpy.matrix(vt.T[:, :rank])
    tSp = numpy.matrix(numpy.diag(s[: rank]))
    Up = numpy.bmat('U P') * tUp
    Vp = numpy.bmat('V Q') * tVp
    Sp = tSp
    return Up, Sp, Vp


def iterSvd(corpus, numTerms, numFactors, numIter = 200, initRate = None, convergence = 1e-4):
    """
    Perform iterative Singular Value Decomposition on a streaming matrix (corpus),
    returning `numFactors` greatest factors (ie., not necessarily full spectrum).
    
    The parameters `numIter` (maximum number of iterations) and `initRate` (gradient 
    descent step size) guide convergency of the algorithm.
    
    See **Genevieve Gorrell: Generalized Hebbian Algorithm for Incremental Singular 
    Value Decomposition in Natural Language Processing. EACL 2006.**
    
    Use of this function is deprecated; although it works, it is several orders of 
    magnitude slower than the direct (non-stochastic) version based on Brand. Use 
    svdAddCols/svdUpdate to compute SVD iteratively. I keep this function here 
    purely for backup reasons.
    """
    logging.info("performing incremental SVD for %i factors" % numFactors)

    # define the document/term singular vectors, fill them with a little random noise
    sdoc = 0.01 * numpy.random.randn(len(corpus), numFactors)
    sterm = 0.01 * numpy.random.randn(numTerms, numFactors)
    if initRate is None:
        initRate = 1.0 / numpy.sqrt(numTerms)
        logging.info("using initial learn rate of %f" % initRate)

    rmse = rmseOld = numpy.inf
    for factor in xrange(numFactors):
        learnRate = initRate
        for iterNo in xrange(numIter):
            errors = 0.0
            rate = learnRate / (1.0 + 9.0 * iterNo / numIter) # gradually decrease the learning rate to 1/10 of the initial value
            logging.debug("setting learning rate to %f" % rate)
            for docNo, doc in enumerate(corpus):
                vec = dict(doc)
                if docNo % 10 == 0:
                    logging.debug('PROGRESS: at document %i/%i' % (docNo, len(corpus)))
                vdoc = sdoc[docNo, factor]
                vterm = sterm[:, factor] # create a view (not copy!) of a matrix row
                
                # reconstruct one document, using all previous factors <0..factor-1>
                recon = numpy.dot(sdoc[docNo, :factor], sterm[:, :factor].T)
                
                for termId in xrange(numTerms):
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
                logging.info("iteration %i diverged; halving the learning rate to %f" %
                             (iterNo, learnRate))
            
            # check convergence, looking for an early exit (but no sooner than 10% 
            # of numIter have passed)
            converged = numpy.divide(numpy.abs(rmseOld - rmse), rmseOld)
            logging.info("factor %i, finished iteration %i, rmse=%f, rate=%f, converged=%f" %
                          (factor, iterNo, rmse, rate, converged))
            if iterNo > numIter / 10 and numpy.isfinite(converged) and converged <= convergence:
                logging.debug("factor %i converged in %i iterations" % (factor, iterNo + 1))
                break
            rmseOld = rmse
        
        logging.info("PROGRESS: finished SVD factor %i/%i, RMSE<=%f" % 
                     (factor + 1, numFactors, rmse))
    
    # normalize the vectors to unit length; also keep the scale
    sdocLens = numpy.sqrt(numpy.sum(sdoc * sdoc, axis = 0))
    stermLens = numpy.sqrt(numpy.sum(sterm * sterm, axis = 0))
    sdoc /= sdocLens
    sterm /= stermLens
    
    # singular value 
    svals = sdocLens * stermLens
    return sterm, svals, sdoc.T


