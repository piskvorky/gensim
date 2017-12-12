#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import logging
import math

from gensim import interfaces, matutils, utils
from six import iteritems


logger = logging.getLogger(__name__)


def df2idf(docfreq, totaldocs, log_base=2.0, add=0.0):
    """
    Compute default inverse-document-frequency for a term with document frequency `doc_freq`::

      idf = add + log(totaldocs / doc_freq)
    """
    return add + math.log(1.0 * totaldocs / docfreq, log_base)


def precompute_idfs(wglobal, dfs, total_docs):
    """Precompute the inverse document frequency mapping for all terms."""
    # not strictly necessary and could be computed on the fly in TfidfModel__getitem__.
    # this method is here just to speed things up a little.
    return {termid: wglobal(df, total_docs) for termid, df in iteritems(dfs)}


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

    def __init__(self, corpus=None, id2word=None, dictionary=None,
                 wlocal=utils.identity, wglobal=df2idf, normalize=True,
                 pivot_norm=False, slope=0.65, pivot=None):
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
        self.normalize = normalize
        self.id2word = id2word
        self.wlocal, self.wglobal = wlocal, wglobal
        self.num_docs, self.num_nnz, self.idfs = None, None, None
        self.pivot_norm = pivot_norm
        self.slope = slope
        self.pivot = pivot
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
            self.idfs = precompute_idfs(self.wglobal, self.dfs, self.num_docs)
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
        logger.info(
            "calculating IDF weights for %i documents and %i features (%i matrix non-zeros)",
            self.num_docs, n_features, self.num_nnz
        )
        self.idfs = precompute_idfs(self.wglobal, self.dfs, self.num_docs)

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
            (termid, self.wlocal(tf) * self.idfs.get(termid))
            for termid, tf in bow if self.idfs.get(termid, 0.0) != 0.0
        ]

        # and finally, normalize the vector either to unit length, or use a
        # user-defined normalization function
        if self.normalize is True:
            norm_vector = matutils.unitvec(vector)
        elif self.normalize:
            norm_vector = self.normalize(vector)

        # make sure there are no explicit zeroes in the vector (must be sparse)
        sparse_norm_vector = [
            (termid, weight) for termid, weight in norm_vector if                       abs(weight) > eps
        ]

        from scipy import sparse as sp
        import numpy as np

        n_samples = len(self.idfs)
        piv_lis = [0]*len(self.idfs)
        lis = piv_lis
        if self.pivot_norm is True:
            for termid, norm_weight in vector:
                pivoted_vector = (1 - self.slope)*self.pivot + self.slope*norm_weight
                piv_lis[termid] = pivoted_vector

            for termid, weight in vector:
                lis[termid] = weight

            piv_lis = np.array(piv_lis)
            piv_lis[piv_lis==0]=1

            diag_mat = sp.spdiags(1./piv_lis, diags=0, m=n_samples, n=n_samples, format='csr').toarray()
            print diag_mat.dot(np.array(lis))
            print diag_mat
            print lis
            print type(vector)
            
            
        return vector
