#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# author Radim Rehurek, radimrehurek@seznam.cz


import logging

import numpy
from scipy.maxentropy import logsumexp # sum of logarithms

import utils


class LsiModel(utils.SaveLoad):
    """
    Objects of this class allow building and maintaining a model of Latent 
    Semantic Indexing.
    
    The main methods are:
    1) LsiModel.fromCorpus(), which calculates the latent topics, initializing 
    the model, and
    
    2) iteration over LsiModel objects, which returns document representations 
    in the new, latent space. Together with the len() function, these two properties
    make LsiModel objects comply with the corpus interface.
    
    Model persistency is achieved via its load/save methods.
    """
    def __init__(self, corpus, numTerms, numTopics = 200):
        """
        Find latent space based on the corpus provided.

        numTopics is the number of requested factors (latent dimensions).
        
        After the model has been initialized, you can estimate topics for an
        arbitrary, unseen document, using the topics = self[bow] dictionary notation.
        """
        self.numTerms = numTerms
        self.numTopics = numTopics # number of latent topics
        if corpus is not None:
            self.initialize(corpus)

    
    def __str__(self):
        return "LsiModel(numTerms=%s, numTopics=%s)" % \
                (self.numTerms, self.numTopics)


    def initialize(self, corpus):
        """
        Run SVD decomposition on the corpus, which defines the latent space into 
        which terms and documents will be mapped.
        
        id2word is a mapping between word ids (integers) and words themselves 
        (utf8 strings).
        """
        # do the actual work -- perform iterative singular value decomposition
        u, s, vt = iterSvd(corpus, self.numTerms, numFactors = self.numTopics)
        
        # calculate projection needed to get document-topic matrix from term-document matrix
        # note that vt (topics of the training corpus) are discarded and not used at all
        self.projection = numpy.dot(numpy.diag(1.0 / s), u.T) # S^(-1) * U^(-1)

    
    def __getitem__(self, bow):
        """
        Return topic distribution, as a list of (topic_id, topic_value) 2-tuples.
        
        This is done by folding-in the input document into the latent topic space.
        """
        topicDist = numpy.sum(self.projection[:, termId] * val for termId, val in bow)
        return [(topicId, topicValue) for topicId, topicValue in enumerate(topicDist)
                if not numpy.allclose(topicValue, 0.0)]
#endclass LsiModel



def iterSvd(corpus, numTerms, numFactors, numIter = 200, initRate = None, convergence = 1e-4):
    """
    Performs iterative Singular Value Decomposition on a streaming matrix (corpus),
    returning numFactors greatest factors (not necessarily full spectrum).
    
    The parameters numIter (maximum number of iterations) and initRate (gradient 
    descent step size) guide convergency of the algorithm.
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
                    if not numpy.isfinite(errors):
                        acavdsv # FIXME  remove
                    
                    # update the singular vectors
                    tmp = vdoc
                    vdoc += rate * error * vterm[termId]
                    vterm[termId] += rate * error * tmp
                sdoc[docNo, factor] = vdoc
            
            # compute rmse = root mean square error of the reconstructed matrix
            rmse = numpy.exp(0.5 * (numpy.log(errors) - numpy.log(len(corpus)) - numpy.log(numTerms)))
            if rmse > rmseOld:
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


def svdUpdate(U, S, V, a, b):
    rank = U.shape[1]
    m = U.T * a
    p = a - U * m
    Ra = numpy.sqrt(p.T * p)
    P = (1.0 / float(Ra)) * p
    assert float(Ra) > 1e-10
    n = V.T * b
    q = b - V * n
    Rb = numpy.sqrt(q.T * q)
    Q = (1.0 / float(Rb)) * q
    assert float(Rb) > 1e-10
    
    K = numpy.matrix(numpy.diag(list(numpy.diag(S)) + [0.0])) + numpy.bmat('m ; Ra') * numpy.bmat(' n; Rb').T
    u, s, vt = numpy.linalg.svd(K, full_matrices = False)
    tUp = numpy.matrix(u[:, :rank])
    tVp = numpy.matrix(vt.T[:, :rank])
    tSp = numpy.matrix(numpy.diag(s[: rank]))
    Up = numpy.bmat('U P') * tUp
    Vp = numpy.bmat('V Q') * tVp
    Sp = tSp
    return Up, Sp, Vp

def svdAddDocs(sdoc, sval, sterm, docs):
    numDocs, rank = sdoc.shape
    numTerms, rank2 = sval.shape
    assert rank == rank2
    empty = numpy.matrix(numpy.zeros(len(docs), rank))
    sdocNew = numpy.bmat('sdoc; empty') # TODO factor this out into matutils.addRow/Column or something 
    a = numpy.matrix(numpy.zeros(numDocs + 1)).T
    a[-1, -1] = 1.0 # no change to terms, only add one document
    b = numpy.matrix(doc).T
    return svdUpdate(sdocNew, sval, sterm, a, b)

