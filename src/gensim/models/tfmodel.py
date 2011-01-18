#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import logging
import itertools


from gensim import interfaces, matutils, utils



class TfModel(interfaces.TransformationABC):
    """
    TODO: add docu

    >>> tf = TfModel(corpus)
    >>> print = tf[some_doc]
    >>> tf.save('/tmp/foo.tfidf_model')
    
    Model persistency is achieved via its load/save methods.
    """

    def __init__(self, corpus, id2word=None, normalize=True):
        """
        `normalize` dictates whether the resulting vectors will be set to unit length.
        """
        self.normalize = normalize
        self.numDocs = 0
        self.numNnz = 0

    def __str__(self):
        return "TfModel(numDocs=%s, numNnz=%s)" % (self.numDocs, self.numNnz)

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
        vector = bow
        if self.normalize:
            vector = matutils.unitVec(vector)
        return vector
