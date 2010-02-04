#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# author Radim Rehurek, radimrehurek@seznam.cz


import logging

import numpy

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


def iterSvd(corpus, numTerms, numFactors, nIter = 100, learnRate = 0.001, cache = True, dtype = numpy.float64):
    """
    Performs iterative Singular Value Decomposition on a streaming matrix (corpus).
    
    Return numFactors greatest factors (only performs partial SVD).
    
    nIter (maximum number of iterations) and learnRate (gradient descent step size) 
    guide convergency of the algorithm.
    
    If cache is True, cache intermediate results between the computation of 
    successive factors; this requires *placing the entire matrix in memory* but 
    is faster than False (default).
    
    dtype determines the basic numeric type for operations as well as resulting
    vectors; default is double (numpy.float64).
    """
    logging.info("performing incremental SVD for %i factors" % numFactors)
    
    u = numpy.zeros((len(corpus), numFactors,), dtype = dtype) + 0.01 # FIXME add random rather than 0.01?
    v = numpy.zeros((numTerms, numFactors,), dtype = dtype) + 0.01
    
    if cache:
        cached = scipy.sparse.dok_matrix((len(corpus), numTerms), dtype = dtype)
    
    for factor in xrange(numFactors):
        # update the vectors for nIter iterations (or until convergence)
        for iterNo in xrange(nIter):
            for cur_row, vector in corpus:
                for cur_col, value in vector:
                    error = value - (cached[cur_row, cur_col] + u[cur_row, factor] * v[cur_col, factor])
                    u_value = u[cur_row, factor]
                    u[cur_row, axis] += learnRate * error * v[cur_col, factor]
                    v[cur_col, axis] += learnRate * error * u_value
    
        # after each factor, update the cache
        if cache:
            for (cur_row, cur_col), value in cached.iteritems():
                cached[cur_row, cur_col] += u[cur_row, factor] * v[cur_col, factor]
    
    # Factor out the svals from u and v
    u_sigma = numpy.sqrt(numpy.sum(u * u))
    v_sigma = numpy.sqrt(numpy.sum(v * v))
    
    u_tensor = DenseTensor(np.divide(u, u_sigma))
    v_tensor = DenseTensor(np.divide(v, v_sigma))
    sigma = DenseTensor(np.multiply(u_sigma, v_sigma))
    
    svdFreeSMat(predicted)
    
    if self.transposed:
        return v_tensor, u_tensor, sigma
    else:
        return u_tensor, v_tensor, sigma

