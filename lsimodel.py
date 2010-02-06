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
    def __init__(self, corpus, id2word, numTopics = 200):
        """
        Find latent space based on the corpus provided.

        numTopics is the number of requested factors (latent dimensions).
        
        After the model has been initialized, you can estimate topics for an
        arbitrary, unseen document, using the topics = self[bow] dictionary notation.
        """
        self.numTerms = len(id2word)
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
        u, s, vt = iterSvd(corpus, k = self.numTopics)
        
        # calculate projection needed to get document-topic matrix from term-document matrix
        # note that vt (topics of the training corpus) are discarded and not used at all
        self.projection = numpy.dot(numpy.diag(1.0 / s), u.T) # S^(-1) * U^(-1)

    
    def __getitem__(self, bow):
        """
        Return topic distribution, as a list of (topic_id, topic_value) 2-tuples.
        """
        topicDist = numpy.sum(self.projection[termId] * val for termId, val in bow)
        return [(topicId, topicValue) for topicId, topicValue in enumerate(topicDist)
                if not numpy.allclose(topicValue, 0.0)]
#endclass LsiModel



def iterSvd(corpus, numTerms, numFactors, numIter = 200, learnRate = 0.1, minIter = 20, convergence = 1e-5):
    """
    Performs iterative Singular Value Decomposition on a streaming matrix (corpus),
    returning numFactors greatest factors (not necessarily full spectrum).
    
    The parameters nIter (maximum number of iterations) and learnRate (gradient 
    descent step size) guide convergency of the algorithm.
    """
    logging.info("performing incremental SVD for %i factors" % numFactors)

    # define the document/term singular vectors, fill them with a little random noise
    sdoc = 0.01 * numpy.random.randn(len(corpus), numFactors)
    sterm = 0.01 * numpy.random.randn(numTerms, numFactors)
    rmse = rmseOld = numpy.inf
    
    for factor in xrange(numFactors):
        for iterNo in xrange(numIter):
            errors = 0.0
            rate = learnRate / (1.0 + iterNo / 100.0) # halve the learning rate after every 100 iterations
            logging.debug("setting learning rate to %f" % rate)
            for docNo, doc in enumerate(corpus):
                recon = numpy.dot(sdoc[docNo, :factor], sterm[:, :factor].T) # reconstruct one row of the matrix, using all previous factors 0..factor-1
                for termId, value in doc:
                    error = value - (recon[termId] + sdoc[docNo, factor] * sterm[termId, factor])
                    errors += error * error
                    tmp = sdoc[docNo, factor]
                    sdoc[docNo, factor] += rate * error * sterm[termId, factor]
                    sterm[termId, factor] += rate * error * tmp
            
            # compute rmse = root mean square error of the reconstructed matrix
            rmse = numpy.sqrt(errors / (len(corpus) * numTerms))
            
            # check convergence, looking for an early exit (before numIter iterations)
            converged = numpy.divide(numpy.abs(rmseOld - rmse), rmseOld)
            logging.debug("factor %i, finished iteration %i, rmse=%f, converged=%f" %
                          (factor, iterNo, rmse, converged))
            if iterNo >= minIter and numpy.isfinite(converged) and converged <= convergence:
                logging.debug("factor %i converged in %i iterations" % (factor, iterNo + 1))
                break
            rmseOld = rmse
        logging.info("PROGRESS: finished SVD factor %i/%i" % (factor + 1, numFactors))
    
    # normalize the vectors to unit length; also keep the scale
    sdocLens = numpy.sqrt(numpy.sum(sdoc * sdoc, axis = 0))
    stermLens = numpy.sqrt(numpy.sum(sterm * sterm, axis = 0))
    sdoc /= sdocLens
    sterm /= stermLens
    
    # singular value 
    svals = sdocLens * stermLens
    return sterm, svals, sdoc.T

