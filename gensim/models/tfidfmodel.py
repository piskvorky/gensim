#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import logging
import math

from gensim import interfaces, matutils, utils
from six import iteritems

import numpy as np

logger = logging.getLogger(__name__)


def resolve_weights(smartirs):
    if not isinstance(smartirs, str) or len(smartirs) != 3:
        raise ValueError('Expected a string of length 3 except got ' + smartirs)

    w_tf, w_df, w_n = smartirs

    if w_tf not in 'nlabL':
        raise ValueError('Expected term frequency weight to be one of nlabL, except got ' + w_tf)

    if w_df not in 'ntp':
        raise ValueError('Expected inverse document frequency weight to be one of ntp, except got ' + w_df)

    if w_n not in 'ncb':
        raise ValueError('Expected normalization weight to be one of ncb, except got ' + w_n)

    return w_tf, w_df, w_n


class TfidfModel(interfaces.TransformationABC):
    """
    Objects of this class realize the transformation between word-document co-occurrence
    matrix (integers) into a locally/globally weighted TF_IDF matrix (positive floats).

    The main methods are:

    1. constructor, which calculates inverse document counts for all terms in the training corpus.
    2. the [] method, which transforms a simple count representation into the TfIdf
       space.

    >>> tfidf = TfidfModel(corpus)
    >>> print(tfidf[some_doc])
    >>> tfidf.save('/tmp/foo.tfidf_model')

    Model persistency is achieved via its load/save methods.
    """

    def __init__(self, corpus=None, id2word=None, dictionary=None, smartirs="ntc",
                 wlocal=None, wglobal=None, normalize=None):
        """
        Compute tf-idf by multiplying a local component (term frequency) with a
        global component (inverse document frequency), and normalizing
        the resulting documents to unit length. Formula for unnormalized weight
        of term `i` in document `j` in a corpus of D documents::

          weight_{i,j} = frequency_{i,j} * log_2(D / document_freq_{i})

        or, more generally::

          weight_{i,j} = wlocal(frequency_{i,j}) * wglobal(document_freq_{i}, D)

        so you can plug in your own custom `wlocal` and `wglobal` functions.

        Default for `wlocal` is identity (other options: math.sqrt, math.log1p, ...)
        and default for `wglobal` is `log_2(total_docs / doc_freq)`, giving the
        formula above.

        `normalize` dictates how the final transformed vectors will be normalized.
        `normalize=True` means set to unit length (default); `False` means don't
        normalize. You can also set `normalize` to your own function that accepts
        and returns a sparse vector.

        If `dictionary` is specified, it must be a `corpora.Dictionary` object
        and it will be used to directly construct the inverse document frequency
        mapping (then `corpus`, if specified, is ignored).
        """
        self.id2word = id2word
        self.wlocal, self.wglobal, self.normalize = wlocal, wglobal, normalize
        self.num_docs, self.num_nnz, self.idfs = None, None, None
        n_tf, n_df, n_n = smartirs
        self.smartirs = smartirs

        if wlocal is None:
            def wlocal(tf, mean=None, _max=None):
                if n_tf == "n":
                    return tf
                elif n_tf == "l":
                    return 1 + math.log(tf)
                elif n_tf == "a":
                    return 0.5 + (0.5 * tf / _max)
                elif n_tf == "b":
                    return 1 if tf > 0 else 0
                elif n_tf == "L":
                    return (1 + math.log(tf)) / (1 + math.log(mean))
            self.wlocal = wlocal

        if wglobal is None:
            def wglobal(docfreq, totaldocs):
                if n_df == "n":
                    return utils.identity(docfreq)
                elif n_df == "t":
                    return math.log(1.0 * totaldocs / docfreq, 10)
                elif n_tf == "p":
                    return math.log((float(totaldocs) - docfreq) / docfreq)
            self.wglobal = wglobal

        if self.normalize is None or isinstance(self.normalize, bool):
            def normalize(x):
                if n_n == "n" or self.normalize is False:
                    return x
                elif n_n == "c" or self.normalize is True:
                    return matutils.unitvec(x)
                # TODO write byte-size normalisation
                # elif n_n == "b":
                #    pass
            self.normalize = normalize

        if dictionary is not None:
            # user supplied a Dictionary object, which already contains all the
            # statistics we need to construct the IDF mapping. we can skip the
            # step that goes through the corpus (= an optimization).
            if corpus is not None:
                logger.warning(
                    "constructor received both corpus and explicit inverse document frequencies; ignoring the corpus"
                )
            self.num_docs, self.num_nnz = dictionary.num_docs, dictionary.num_nnz
            self.dfs = dictionary.dfs.copy()

            if id2word is None:
                self.id2word = dictionary
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
                logger.info("PROGRESS: processing document #%i", docno)
            numnnz += len(bow)
            for termid, _ in bow:
                dfs[termid] = dfs.get(termid, 0) + 1

        # keep some stats about the training corpus
        self.num_docs = docno + 1
        self.num_nnz = numnnz
        self.dfs = dfs

        # and finally compute the idf weights
        n_features = max(dfs) if dfs else 0

    def __getitem__(self, bow, eps=1e-12):
        """
        Return tf-idf representation of the input vector and/or corpus.
        """
        # if the input vector is in fact a corpus, return a transformed corpus as a result
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)

        # unknown (new) terms will be given zero weight (NOT infinity/huge weight,
        # as strict application of the IDF formula would dictate)

        vector = [
            (termid, self.wlocal(tf, mean=np.mean(np.array(bow), axis=1), _max=np.max(bow, axis=1)) * self.wglobal(self.dfs[termid], self.num_docs))
            for termid, tf in bow if self.wglobal(self.dfs[termid], self.num_docs) != 0.0
        ]

        # and finally, normalize the vector either to unit length, or use a
        # user-defined normalization function

        vector = self.normalize(vector)

        # make sure there are no explicit zeroes in the vector (must be sparse)
        vector = [(termid, weight) for termid, weight in vector if abs(weight) > eps]
        return vector
