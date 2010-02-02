#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# author Radim Rehurek, radimrehurek@seznam.cz


import logging

import numpy # for arrays, array broadcasting etc.

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
    def __init__(self, id2word, numTopics = 200):
        """
        numTopics is the number of requested factors (latent dimensions).
        
        The actual latent vectors are inferred via the initialize() method. 
        """
        self.numTerms = len(id2word)
        self.numTopics = numTopics # number of latent topics
        self.corpus = None

    
    def __str__(self):
        return "LsiModel(numTerms=%s, numTopics=%s)" % \
                (self.numTerms, self.numTopics)


    def __len__(self):
        assert self.corpus is not None, "must call setCorpus() before inferring topics of a corpus"
        return len(self.corpus)

    
    def initialize(corpus):
        """
        Run SVD decomposition on the corpus, which defines the latent space into 
        which terms and documents will be mapped.
        
        id2word is a mapping between word ids (integers) and words themselves 
        (utf8 strings).
        
        After the model has been initialized, you can assign an arbitrary corpus
        with setCorpus() and get topic distribution for its documents via iter(self).
        """
        # do the actual work -- perform iterative singular value decomposition
        u, s, vt = self.doSvd(corpus)
        
        # calculate projection needed to get document-topic matrix from term-document matrix
        # note that vt (topics for the training corpus) are discarded and not used at all
        # if you care about topics of the training corpus, call setCorpus(trainingCorpus)
        # and iterate over its topics with itetr(self).
        self.projection = numpy.dot(numpy.diag(1.0 / s), u.T) # S^(-1) * U^(-1)

    
    def setCorpus(self, corpus):
        self.corpus = corpus
    
    
    def __iter__(self):
        """
        Iterate over corpus, estimating topic distribution for each document.
        
        Return this topic distribution as another.
        
        This method effectively wraps an underlying word-count corpus into another
        corpus, of the same length but with terms replaced by topics.
        
        Internally, this method performs LDA inference on each document, using 
        all the previously estimated model parameters.
        """
        logging.info("performing inference on corpus of %i documents" % len(self.corpus))
        for docNo, bow in enumerate(self.corpus):
            if docNo % 1000 == 0:
                logging.info("PROGRESS: inferring LSI topics of doc #%i/%i" %
                             (docNo, len(self.corpus)))
            
            topicDist = numpy.sum(self.projection[termId] * val for termId, val in bow)
            topicDist = numpy.where(topicDist > 0, topicDist, 0.0)
            sumDist = numpy.sum(topicDist)
            if numpy.allclose(sumDist, 0.0): # if there were no topics found, return nothing (ie for empty documents)
                yield []
            topicDist = topicDist / sumDist
            yield list(enumerate(topicDist))
#endclass LsiModel
