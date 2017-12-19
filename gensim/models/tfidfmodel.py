#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import logging

from gensim import interfaces, matutils, utils
from six import iteritems

import numpy as np

logger = logging.getLogger(__name__)


def resolve_weights(smartirs):
    """
    Checks for validity of smartirs parameter.
    """
    if not isinstance(smartirs, str) or len(smartirs) != 3:
        raise ValueError("Expected a string of length 3 except got " + smartirs)

    w_tf, w_df, w_n = smartirs

    if w_tf not in 'nlabL':
        raise ValueError("Expected term frequency weight to be one of 'nlabL', except got " + w_tf)

    if w_df not in 'ntp':
        raise ValueError("Expected inverse document frequency weight to be one of 'ntp', except got " + w_df)

    if w_n not in 'ncb':
        raise ValueError("Expected normalization weight to be one of 'ncb', except got " + w_n)

    return w_tf, w_df, w_n


def df2idf(docfreq, totaldocs, log_base=2.0, add=0.0):
    """
    Compute default inverse-document-frequency for a term with document frequency `doc_freq`::
    idf = add + log(totaldocs / doc_freq)
    """
    return add + np.log(float(totaldocs) / docfreq) / np.log(log_base)


def precompute_idfs(wglobal, dfs, total_docs):
    """
    Precompute the inverse document frequency mapping for all terms.
    """
    # not strictly necessary and could be computed on the fly in TfidfModel__getitem__.
    # this method is here just to speed things up a little.
    return {termid: wglobal(df, total_docs) for termid, df in iteritems(dfs)}


class TfidfModel(interfaces.TransformationABC):
    """
    Objects of this class realize the transformation between word-document co-occurrence
    matrix (integers) into a locally/globally weighted TF_IDF matrix (positive floats).

    Methods
    -------
    __init__(corpus=None, id2word=None, dictionary=None, wlocal=utils.identity,
                wglobal=df2idf, normalize=True, smartirs=None):
            Calculates inverse document counts for all terms in the training corpus.
    __getitem__(bow, eps=1e-12)
            which transforms a simple count representation into the TfIdf space.

    >>> tfidf = TfidfModel(corpus)
    >>> print(tfidf[some_doc])
    >>> tfidf.save('/tmp/foo.tfidf_model')

    Model persistency is achieved via its load/save methods.
    """

    def __init__(self, corpus=None, id2word=None, dictionary=None, wlocal=utils.identity,
                 wglobal=df2idf, normalize=True, smartirs=None):
        """
        Compute tf-idf by multiplying a local component (term frequency) with a
        global component (inverse document frequency), and normalizing
        the resulting documents to unit length. Formula for unnormalized weight
        of term `i` in document `j` in a corpus of D documents::

          weight_{i,j} = frequency_{i,j} * log_2(D / document_freq_{i})

        or, more generally::

          weight_{i,j} = wlocal(frequency_{i,j}) * wglobal(document_freq_{i}, D)

        so you can plug in your own custom `wlocal` and `wglobal` functions.


        Parameters
        ----------
        corpus :    dictionary.doc2bow
                    Corpus is a list of sets where each set has two elements. First being the termid and
                    second being the term frequency of each term in the document.
        id2word :   dict
                    id2word is an optional dictionary that maps the word_id to a token.
                    In case id2word isn’t specified the mapping id2word[word_id] = str(word_id) will be used.
        dictionary :corpora.Dictionary
                    If `dictionary` is specified, it must be a `corpora.Dictionary` object
                    and it will be used to directly construct the inverse document frequency
                    mapping (then `corpus`, if specified, is ignored).
        wlocals :   user specified function
                    Default for `wlocal` is identity (other options: math.sqrt, math.log1p, ...)
        wglobal :   user specified function
                    Default for `wglobal` is `log_2(total_docs / doc_freq)`, giving the
                    formula above.
        normalize : user specified function
                    It dictates how the final transformed vectors will be normalized.
                    `normalize=True` means set to unit length (default); `False` means don't
                    normalize. You can also set `normalize` to your own function that accepts
                    and returns a sparse vector.
        smartirs : {'None' ,'str'}
                    `smartirs` or SMART (System for the Mechanical Analysis and Retrieval of Text)
                    Information Retrieval System, a mnemonic scheme for denoting tf-idf weighting
                    variants in the vector space model. The mnemonic for representing a combination
                    of weights takes the form ddd, where the letters represents the term weighting
                    of the document vector.

                    Term frequency weighing:
                      natural - `n`, logarithm - `l` , augmented - `a`,  boolean `b`, log average - `L`.
                    Document frequency weighting:
                      none - `n`, idf - `t`, prob idf - `p`.
                    Document normalization:
                      none - `n`, cosine - `c`, byte size - `b`.

                    for more information visit https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System

        Returns
        -------
        x : gensim.models.tfidfmodel.TfidfModel

        """

        self.id2word = id2word
        self.wlocal, self.wglobal, self.normalize = wlocal, wglobal, normalize
        self.num_docs, self.num_nnz, self.idfs = None, None, None
        self.smartirs = smartirs

        if self.normalize is True:
            self.normalize = matutils.unitvec
        elif self.normalize is False:
            self.normalize = utils.identity

        # If smartirs is not None, override wlocal, wglobal and normalize
        if smartirs is not None:
            n_tf, n_df, n_n = resolve_weights(smartirs)

            def wlocal(tf):
                if n_tf == "n":
                    return tf
                elif n_tf == "l":
                    return 1 + np.log(tf) / np.log(2)
                elif n_tf == "a":
                    return 0.5 + (0.5 * tf / tf.max(axis=0))
                elif n_tf == "b":
                    return tf.astype('bool').astype('int')
                elif n_tf == "L":
                    return (1 + np.log(tf) / np.log(2)) / (1 + np.log(tf.mean(axis=0) / np.log(2)))
            self.wlocal = wlocal

            def wglobal(docfreq, totaldocs):
                if n_df == "n":
                    return utils.identity(docfreq)
                elif n_df == "t":
                    return np.log(1.0 * totaldocs / docfreq) / np.log(2)
                elif n_df == "p":
                    return np.log((1.0 * totaldocs - docfreq) / docfreq) / np.log(2)
            self.wglobal = wglobal

            def normalize(x):
                if n_n == "n":
                    return x
                elif n_n == "c":
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

        termid_array, tf_array = [], []
        for termid, tf in bow:
            termid_array.append(termid)
            tf_array.append(tf)

        tf_array = self.wlocal(np.array(tf_array))

        vector = [
            (termid, tf * self.idfs.get(termid))
            for termid, tf in zip(termid_array, tf_array) if self.idfs.get(termid, 0.0) != 0.0
        ]

        # and finally, normalize the vector either to unit length, or use a
        # user-defined normalization function

        vector = self.normalize(vector)

        # make sure there are no explicit zeroes in the vector (must be sparse)
        vector = [(termid, weight) for termid, weight in vector if abs(weight) > eps]
        return vector
