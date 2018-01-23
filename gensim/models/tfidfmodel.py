#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import logging
from functools import partial

from gensim import interfaces, matutils, utils
from six import iteritems

import numpy as np

logger = logging.getLogger(__name__)


def resolve_weights(smartirs):
    """Checks for validity of `smartirs` parameter.

    Parameters
    ----------
    smartirs : str
        `smartirs` or SMART (System for the Mechanical Analysis and Retrieval of Text)
        Information Retrieval System, a mnemonic scheme for denoting tf-idf weighting
        variants in the vector space model. The mnemonic for representing a combination
        of weights takes the form ddd, where the letters represents the term weighting of the document vector.
        for more information visit [1].

    Returns
    -------
    w_tf : str
        Term frequency weighing:
            * `n` - natural,
            * `l` - logarithm,
            * `a` - augmented,
            * `b` - boolean,
            * `L` - log average.
    w_df : str
        Document frequency weighting:
            * `n` - none,
            * `t` - idf,
            * `p` - prob idf.
    w_n : str
        Document normalization:
            * `n` - none,
            * `c` - cosine.

    Raises
    ------
    ValueError
        If `smartirs` is not a string of length 3 or one of the decomposed value
        doesn't fit the list of permissible values

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System

    """
    if not isinstance(smartirs, str) or len(smartirs) != 3:
        raise ValueError("Expected a string of length 3 except got " + smartirs)

    w_tf, w_df, w_n = smartirs

    if w_tf not in 'nlabL':
        raise ValueError("Expected term frequency weight to be one of 'nlabL', except got {}".format(w_tf))

    if w_df not in 'ntp':
        raise ValueError("Expected inverse document frequency weight to be one of 'ntp', except got {}".format(w_df))

    if w_n not in 'ncb':
        raise ValueError("Expected normalization weight to be one of 'ncb', except got {}".format(w_n))

    return w_tf, w_df, w_n


def df2idf(docfreq, totaldocs, log_base=2.0, add=0.0):
    """Compute default inverse-document-frequency for a term with document frequency:
    :math:`idf = add + log_{log\_base} \\frac{totaldocs}{doc\_freq}`

    Parameters
    ----------
    docfreq : float
        Document frequency.
    totaldocs : int
        Total number of documents.
    log_base : float, optional
        Base of logarithm.
    add : float, optional
        Offset.

    Returns
    -------
    float
        Inverse document frequency.

    """
    return add + np.log(float(totaldocs) / docfreq) / np.log(log_base)


def precompute_idfs(wglobal, dfs, total_docs):
    """Pre-compute the inverse document frequency mapping for all terms.

    Parameters
    ----------
    wglobal : function
        Custom function for calculation idf, look at "universal" :func:`~gensim.models.tfidfmodel.updated_wglobal`.
    dfs : dict
        Dictionary with term_id and how many documents this token appeared.
    total_docs : int
        Total number of document.

    Returns
    -------
    dict
        Precomputed idfs in format {term_id_1: idfs_1, term_id_2: idfs_2, ...}

    """
    # not strictly necessary and could be computed on the fly in TfidfModel__getitem__.
    # this method is here just to speed things up a little.
    return {termid: wglobal(df, total_docs) for termid, df in iteritems(dfs)}


def updated_wlocal(tf, n_tf):
    """A scheme to transform `tf` or term frequency based on the value of `n_tf`.

    Parameters
    ----------
    tf : int
        Term frequency.
    n_tf : {'n', 'l', 'a', 'b', 'L'}
        Parameter to decide the current transformation scheme.

    Returns
    -------
    float
        Calculated wlocal.

    """
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


def updated_wglobal(docfreq, totaldocs, n_df):
    """A scheme to transform `docfreq` or document frequency based on the value of `n_df`.

    Parameters
    ----------
    docfreq : int
        Document frequency.
    totaldocs : int
        Total number of documents.
    n_df : {'n', 't', 'p'}
        Parameter to decide the current transformation scheme.

    Returns
    -------
    float
        Calculated wglobal.

    """
    if n_df == "n":
        return utils.identity(docfreq)
    elif n_df == "t":
        return np.log(1.0 * totaldocs / docfreq) / np.log(2)
    elif n_df == "p":
        return np.log((1.0 * totaldocs - docfreq) / docfreq) / np.log(2)


def updated_normalize(x, n_n):
    """Normalizes the final tf-idf value according to the value of `n_n`.

    Parameters
    ----------
    x : numpy.ndarray
        Input array
    n_n : {'n', 'c'}
        Parameter that decides the normalizing function to be used.

    Returns
    -------
    numpy.ndarray
        Normalized array.

    """
    if n_n == "n":
        return x
    elif n_n == "c":
        return matutils.unitvec(x)


class TfidfModel(interfaces.TransformationABC):
    """Objects of this class realize the transformation between word-document co-occurrence matrix (int)
    into a locally/globally weighted TF_IDF matrix (positive floats).

    Examples
    --------
    >>> import gensim.downloader as api
    >>> from gensim.models import TfidfModel
    >>> from gensim.corpora import Dictionary
    >>>
    >>> dataset = api.load("text8")
    >>> dct = Dictionary(dataset)  # fit dictionary
    >>> corpus = [dct.doc2bow(line) for line in dataset]  # convert dataset to BoW format
    >>>
    >>> model = TfidfModel(corpus)  # fit model
    >>> vector = model[corpus[0]]  # apply model

    """

    def __init__(self, corpus=None, id2word=None, dictionary=None, wlocal=utils.identity,
                 wglobal=df2idf, normalize=True, smartirs=None):
        """Compute tf-idf by multiplying a local component (term frequency) with a global component
        (inverse document frequency), and normalizing the resulting documents to unit length.
        Formula for non-normalized weight of term :math:`i` in document :math:`j` in a corpus of :math:`D` documents

        .. math:: weight_{i,j} = frequency_{i,j} * log_2 \\frac{D}{document\_freq_{i}}

        or, more generally

        .. math:: weight_{i,j} = wlocal(frequency_{i,j}) * wglobal(document\_freq_{i}, D)

        so you can plug in your own custom :math:`wlocal` and :math:`wglobal` functions.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int), optional
            Input corpus
        id2word : {dict, :class:`~gensim.corpora.Dictionary`}, optional
            Mapping token - id, that was used for converting input data to bag of words format.
        dictionary : :class:`~gensim.corpora.Dictionary`
            If `dictionary` is specified, it must be a `corpora.Dictionary` object and it will be used.
            to directly construct the inverse document frequency mapping (then `corpus`, if specified, is ignored).
        wlocals : function, optional
            Function for local weighting, default for `wlocal` is :func:`~gensim.utils.identity`
            (other options: :func:`math.sqrt`, :func:`math.log1p`, etc).
        wglobal : function, optional
            Function for global weighting, default is :func:`~gensim.models.tfidfmodel.df2idf`.
        normalize : bool, optional
            It dictates how the final transformed vectors will be normalized. `normalize=True` means set to unit length
            (default); `False` means don't normalize. You can also set `normalize` to your own function that accepts
            and returns a sparse vector.
        smartirs : str, optional
            SMART (System for the Mechanical Analysis and Retrieval of Text) Information Retrieval System,
            a mnemonic scheme for denoting tf-idf weighting variants in the vector space model.
            The mnemonic for representing a combination of weights takes the form XYZ,
            for example 'ntc', 'bpn' and so on, where the letters represents the term weighting of the document vector.

            Term frequency weighing:
                * `n` - natural,
                * `l` - logarithm,
                * `a` - augmented,
                * `b` - boolean,
                * `L` - log average.

            Document frequency weighting:
                * `n` - none,
                * `t` - idf,
                * `p` - prob idf.

            Document normalization:
                * `n` - none,
                * `c` - cosine.

            For more information visit [1].

        """

        self.id2word = id2word
        self.wlocal, self.wglobal, self.normalize = wlocal, wglobal, normalize
        self.num_docs, self.num_nnz, self.idfs = None, None, None
        self.smartirs = smartirs

        # If smartirs is not None, override wlocal, wglobal and normalize
        if smartirs is not None:
            n_tf, n_df, n_n = resolve_weights(smartirs)

            self.wlocal = partial(updated_wlocal, n_tf=n_tf)
            self.wglobal = partial(updated_wglobal, n_df=n_df)
            self.normalize = partial(updated_normalize, n_n=n_n)

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
        """Compute inverse document weights, which will be used to modify term frequencies for documents.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Input corpus.

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
        """Get tf-idf representation of the input vector and/or corpus.

        bow : {list of (int, int), iterable of iterable of (int, int)}
            Input document or copus in BoW format.
        eps : float
            Threshold value, will remove all position that have tfidf-value less than `eps`.

        Returns
        -------
        vector : list of (int, float)
            TfIdf vector, if `bow` is document **OR**
        :class:`~gensim.interfaces.TransformedCorpus`
            TfIdf corpus, if `bow` is corpus.

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
            for termid, tf in zip(termid_array, tf_array) if abs(self.idfs.get(termid, 0.0)) > eps
        ]

        if self.normalize is True:
            self.normalize = matutils.unitvec
        elif self.normalize is False:
            self.normalize = utils.identity

        # and finally, normalize the vector either to unit length, or use a
        # user-defined normalization function
        vector = self.normalize(vector)

        # make sure there are no explicit zeroes in the vector (must be sparse)
        vector = [(termid, weight) for termid, weight in vector if abs(weight) > eps]
        return vector
