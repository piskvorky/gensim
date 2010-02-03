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

    
    def __str__(self):
        return "LsiModel(numTerms=%s, numTopics=%s)" % \
                (self.numTerms, self.numTopics)


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
        # note that vt (topics of the training corpus) are discarded and not used at all
        self.projection = numpy.dot(numpy.diag(1.0 / s), u.T) # S^(-1) * U^(-1)

    
    def __getitem__(self, bow):
        """
        Return topic distribution, as a list of (topic_id, topic_value) 2-tuples.
        """
        topicDist = numpy.sum(self.projection[termId] * val for termId, val in bow)
        print 'topicDist.shape', topicDist.shape
        return [(topicId, topicValue) for topicId, topicValue in enumerate(topicDist)
                if not numpy.allclose(topicValue, 0.0)]
#endclass LsiModel
