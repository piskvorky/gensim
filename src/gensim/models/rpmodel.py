#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import logging
import itertools

import numpy

from gensim import interfaces, matutils, utils



class RpModel(interfaces.TransformationABC):
    """
    Objects of this class allow building and maintaining a model for Random Projections
    (also known as Random Indexing). For theoretical background on RP, see:
    
      Kanerva et al.: "Random indexing of text samples for Latent Semantic Analysis."
    
    The main methods are:
    
    1. constructor, which creates the random projection matrix
    2. the [] method, which transforms a simple count representation into the TfIdf 
       space.
    
    >>> rp = RpModel(corpus)
    >>> print rp[some_doc]
    >>> rp.save('/tmp/foo.rp_model')
    
    Model persistency is achieved via its load/save methods.
    """
    def __init__(self, corpus, id2word = None, numTopics = 300):
        """
        `id2word` is a mapping from word ids (integers) to words (strings). It is
        used to determine the vocabulary size, as well as for debugging and topic 
        printing. If not set, it will be determined from the corpus.
        """
        self.id2word = id2word
        self.numTopics = numTopics
        if corpus is not None:
            self.initialize(corpus)

    
    def __str__(self):
        return "RpModel(numTerms=%s, numTopics=%s)" % (self.numTerms, self.numTopics)


    def initialize(self, corpus):
        """
        Initialize the random projection matrix.
        """
        if self.id2word is None:
            logging.info("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dictFromCorpus(corpus)
            self.numTerms = len(self.id2word)
        else:
            self.numTerms = 1 + max([-1] + self.id2word.keys())
            
        # Now construct the projection matrix itself.
        #
        # Here i use a particular form, derived in "Achlioptas: Database-friendly random projection",
        # and his (1) scenario of Theorem 1.1 in particular (all entries are +1/-1).
        tmp = numpy.random.binomial(1, 0.5, (self.numTopics, self.numTerms)) # FIXME temporary array unnecessarily big (int32 -> int8)
        self.projection = numpy.asmatrix(1 - 2 * tmp.astype(numpy.int8)) # convert from 0/1 to +1/-1
    

    def __getitem__(self, bow):
        """
        Return RP representation of the input vector and/or corpus.
        """
        # if the input vector is in fact a corpus, return a transformed corpus as result
        if utils.isCorpus(bow):
            return self._apply(bow)
        
        vec = matutils.sparse2full(bow, self.numTerms).reshape(self.numTerms, 1)
        topicDist = (self.projection * vec) / numpy.sqrt(self.numTopics) # (1, d) * (d, k) = (1, k)
        return [(topicId, float(topicValue)) for topicId, topicValue in enumerate(topicDist.flat)
                if numpy.isfinite(topicValue) and not numpy.allclose(topicValue, 0.0)]
#endclass RpModel

