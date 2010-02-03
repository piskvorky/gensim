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
        u, s, vt = self.doSvd(corpus)
        
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
    
    def doSvd(self, corpus):
        pass # FIXME TODO
#endclass LsiModel
