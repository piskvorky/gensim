#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import logging
import itertools

import math

from gensim import interfaces, matutils, utils



class TfidfModel(interfaces.TransformationABC):
    """
    Objects of this class realize the transformation between word-document co-occurence
    matrix (integers) into a locally/globally weighted matrix (positive floats).
    
    This is done by combining the term frequency counts (the TF part) with inverse
    document frequency counts (the IDF part), optionally normalizing the resulting
    documents to unit length.
    
    The main methods are:
    
    1. constructor, which calculates IDF weights for all terms in the training corpus.
    2. the [] method, which transforms a simple count representation into the TfIdf 
       space.
    
    >>> tfidf = TfidfModel(corpus)
    >>> print = tfidf[some_doc]
    >>> tfidf.save('/tmp/foo.tfidf_model')
    
    Model persistency is achieved via its load/save methods.
    """
    def __init__(self, corpus, id2word=None, normalize=True):
        """
        `normalize` dictates whether the resulting vectors will be set to unit length.
        """
        self.normalize = normalize
        self.numDocs = 0
        self.numNnz = 0
        if corpus is not None:
            self.initialize(corpus)

    
    def __str__(self):
        return "TfidfModel(numDocs=%s, numNnz=%s)" % (self.numDocs, self.numNnz)


    def initialize(self, corpus):
        """
        Compute inverse document weights, which will be used to modify term 
        frequencies for documents.
        """
        logging.info("calculating counts")
        idfs = {}
        numNnz = 0
        for docNo, bow in enumerate(corpus):
            if docNo % 10000 == 0:
                logging.info("PROGRESS: processing document #%i" % docNo)
            numNnz += len(bow)
            for termId, termCount in bow:
                idfs[termId] = idfs.get(termId, 0) + 1

        # keep some stats about the training corpus
        self.numDocs = docNo + 1 # HACK using leftover from enumerate(corpus) above
        self.numNnz = numNnz
        
        # and finally compute the idf weights
        logging.info("calculating IDF weights for %i documents and %i features (%i matrix non-zeros)" %
                     (self.numDocs, 1 + max([-1] + idfs.keys()), self.numNnz))
        self.idfs = dict((termId, math.log(1.0 * self.numDocs / docFreq, 2)) # the IDF weight formula
                         for termId, docFreq in idfs.iteritems())


    def __getitem__(self, bow):
        """
        Return tf-idf representation of the input vector and/or corpus.
        """
        # if the input vector is in fact a corpus, return a transformed corpus as a result
        is_corpus, bow = utils.isCorpus(bow)
        if is_corpus:
            return self._apply(bow)
        
        # unknown (new) terms will be given zero weight (NOT infinity/huge weight,
        # as strict application of the IDF formula would dictate
        vector = [(termId, tf * self.idfs.get(termId, 0.0))
                  for termId, tf in bow if self.idfs.get(termId, 0.0) != 0.0]
        if self.normalize:
            vector = matutils.unitVec(vector)
        return vector
#endclass TfidfModel

