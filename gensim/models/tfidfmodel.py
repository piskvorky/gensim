#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import logging
import itertools

import math

from gensim import interfaces, matutils, utils


logger = logging.getLogger('gensim.models.tfidfmodel')


def dfs2idfs(dfs, totaldocs):
    """
    Given a mapping of `term->document frequency`, construct a mapping of
    `term->inverse document frequency`.
    """
    return dict((termid,
                math.log(1.0 * totaldocs / docfreq, 2) if docfreq > 0 else 0)
                for termid, docfreq in dfs.iteritems())


def idfs2dfs(idfs, totaldocs):
    """
    Inverse mapping for `dfs2idfs`.
    """
    return dict((termid, int(round(totaldocs / 2**weight)))
                for termid, weight in idfs.iteritems())


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
    def __init__(self, corpus=None, id2word=None, dictionary=None, normalize=True):
        """
        `normalize` dictates whether the transformed vectors will be set to unit
        length.

        If `dictionary` is specified, it must be a `corpora.Dictionary` object
        and it will be used to directly construct the inverse document frequency
        mapping (then `corpus`, if specified, is ignored).
        """
        self.normalize = normalize
        self.id2word = id2word
        self.num_docs, self.num_nnz, self.idfs = None, None, None
        if dictionary is not None:
            if corpus is not None:
                logger.warning("constructor received both corpus and explicit "
                               "inverse document frequencies; ignoring the corpus")
            self.num_docs, self.num_nnz = dictionary.num_docs, dictionary.num_nnz
            self.idfs = dfs2idfs(dictionary.dfs, dictionary.num_docs)
        elif corpus is not None:
            self.initialize(corpus)
        else:
            # NOTE: everything is left uninitialized; presumably the model will
            # be initialized in some other way
            pass


    def __str__(self):
        return "TfidfModel(num_docs=%s, num_nnz=%s)" % (self.num_docs, self.num_nnz)


    def initialize(self, corpus):
        """
        Compute inverse document weights, which will be used to modify term
        frequencies for documents.
        """
        logger.info("collecting document frequencies")
        dfs = {}
        numnnz, docno = 0, -1
        for docno, bow in enumerate(corpus):
            if docno % 10000 == 0:
                logger.info("PROGRESS: processing document #%i" % docno)
            numnnz += len(bow)
            for termid, termcount in bow:
                dfs[termid] = dfs.get(termid, 0) + 1

        # keep some stats about the training corpus
        self.num_docs = docno + 1
        self.num_nnz = numnnz

        # and finally compute the idf weights
        logger.info("calculating IDF weights for %i documents and %i features (%i matrix non-zeros)" %
                     (self.num_docs, 1 + max([-1] + dfs.keys()), self.num_nnz))
        self.idfs = dfs2idfs(dfs, self.num_docs)


    def __getitem__(self, bow):
        """
        Return tf-idf representation of the input vector and/or corpus.
        """
        # if the input vector is in fact a corpus, return a transformed corpus as a result
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)

        # unknown (new) terms will be given zero weight (NOT infinity/huge weight,
        # as strict application of the IDF formula would dictate)
        vector = [(termid, tf * self.idfs.get(termid, 0.0))
                  for termid, tf in bow if self.idfs.get(termid, 0.0) != 0.0]
        if self.normalize:
            vector = matutils.unitvec(vector)
        return vector
#endclass TfidfModel
