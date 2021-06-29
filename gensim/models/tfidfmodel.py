#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module implements functionality related to the `Term Frequency - Inverse Document Frequency
<https://en.wikipedia.org/wiki/Tf%E2%80%93idf>` vector space bag-of-words models.

For a more in-depth exposition of TF-IDF and its various SMART variants (normalization, weighting schemes),
see the blog post at https://rare-technologies.com/pivoted-document-length-normalisation/

"""

import logging
from functools import partial
import re

import numpy as np

from gensim import interfaces, matutils, utils
from gensim.utils import deprecated


logger = logging.getLogger(__name__)


def resolve_weights(smartirs):
    """Check the validity of `smartirs` parameters.

    Parameters
    ----------
    smartirs : str
        `smartirs` or SMART (System for the Mechanical Analysis and Retrieval of Text)
        Information Retrieval System, a mnemonic scheme for denoting tf-idf weighting
        variants in the vector space model. The mnemonic for representing a combination
        of weights takes the form ddd, where the letters represents the term weighting of the document vector.
        for more information visit `SMART Information Retrieval System
        <https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System>`_.

    Returns
    -------
    str of (local_letter, global_letter, normalization_letter)

    local_letter : str
        Term frequency weighing, one of:
            * `b` - binary,
            * `t` or `n` - raw,
            * `a` - augmented,
            * `l` - logarithm,
            * `d` - double logarithm,
            * `L` - log average.
    global_letter : str
        Document frequency weighting, one of:
            * `x` or `n` - none,
            * `f` - idf,
            * `t` - zero-corrected idf,
            * `p` - probabilistic idf.
    normalization_letter : str
        Document normalization, one of:
            * `x` or `n` - none,
            * `c` - cosine,
            * `u` - pivoted unique,
            * `b` - pivoted character length.

    Raises
    ------
    ValueError
        If `smartirs` is not a string of length 3 or one of the decomposed value
        doesn't fit the list of permissible values.
    """
    if isinstance(smartirs, str) and re.match(r"...\....", smartirs):
        match = re.match(r"(?P<ddd>...)\.(?P<qqq>...)", smartirs)
        raise ValueError(
            "The notation {ddd}.{qqq} specifies two term-weighting schemes, "
            "one for collection documents ({ddd}) and one for queries ({qqq}). "
            "You must train two separate tf-idf models.".format(
                ddd=match.group("ddd"),
                qqq=match.group("qqq"),
            )
        )
    if not isinstance(smartirs, str) or len(smartirs) != 3:
        raise ValueError("Expected a string of length 3 got " + smartirs)

    w_tf, w_df, w_n = smartirs

    if w_tf not in 'btnaldL':
        raise ValueError("Expected term frequency weight to be one of 'btnaldL', got {}".format(w_tf))

    if w_df not in 'xnftp':
        raise ValueError("Expected inverse document frequency weight to be one of 'xnftp', got {}".format(w_df))

    if w_n not in 'xncub':
        raise ValueError("Expected normalization weight to be one of 'xncub', got {}".format(w_n))

    # resolve aliases
    if w_tf == "t":
        w_tf = "n"
    if w_df == "x":
        w_df = "n"
    if w_n == "x":
        w_n = "n"

    return w_tf + w_df + w_n


def df2idf(docfreq, totaldocs, log_base=2.0, add=0.0):
    r"""Compute inverse-document-frequency for a term with the given document frequency `docfreq`:
    :math:`idf = add + log_{log\_base} \frac{totaldocs}{docfreq}`

    Parameters
    ----------
    docfreq : {int, float}
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
        Custom function for calculating the "global" weighting function.
        See for example the SMART alternatives under :func:`~gensim.models.tfidfmodel.smartirs_wglobal`.
    dfs : dict
        Dictionary mapping `term_id` into how many documents did that term appear in.
    total_docs : int
        Total number of documents.

    Returns
    -------
    dict of (int, float)
        Inverse document frequencies in the format `{term_id_1: idfs_1, term_id_2: idfs_2, ...}`.

    """
    # not strictly necessary and could be computed on the fly in TfidfModel__getitem__.
    # this method is here just to speed things up a little.
    return {termid: wglobal(df, total_docs) for termid, df in dfs.items()}


def smartirs_wlocal(tf, local_scheme):
    """Calculate local term weight for a term using the weighting scheme specified in `local_scheme`.

    Parameters
    ----------
    tf : int
        Term frequency.
    local : {'b', 'n', 'a', 'l', 'd', 'L'}
        Local transformation scheme.

    Returns
    -------
    float
        Calculated local weight.

    """
    if local_scheme == "n":
        return tf
    elif local_scheme == "l":
        return 1 + np.log2(tf)
    elif local_scheme == "d":
        return 1 + np.log2(1 + np.log2(tf))
    elif local_scheme == "a":
        return 0.5 + (0.5 * tf / tf.max(axis=0))
    elif local_scheme == "b":
        return tf.astype('bool').astype('int')
    elif local_scheme == "L":
        return (1 + np.log2(tf)) / (1 + np.log2(tf.mean(axis=0)))


def smartirs_wglobal(docfreq, totaldocs, global_scheme):
    """Calculate global document weight based on the weighting scheme specified in `global_scheme`.

    Parameters
    ----------
    docfreq : int
        Document frequency.
    totaldocs : int
        Total number of documents.
    global_scheme : {'n', 'f', 't', 'p'}
        Global transformation scheme.

    Returns
    -------
    float
        Calculated global weight.

    """
    if global_scheme == "n":
        return 1.0
    elif global_scheme == "f":
        return np.log2(1.0 * totaldocs / docfreq)
    elif global_scheme == "t":
        return np.log2((totaldocs + 1.0) / docfreq)
    elif global_scheme == "p":
        return max(0, np.log2((1.0 * totaldocs - docfreq) / docfreq))


@deprecated("Function will be removed in 4.0.0")
def smartirs_normalize(x, norm_scheme, return_norm=False):
    """Normalize a vector using the normalization scheme specified in `norm_scheme`.

    Parameters
    ----------
    x : numpy.ndarray
        The tf-idf vector.
    norm_scheme : {'n', 'c'}
        Document length normalization scheme.
    return_norm : bool, optional
        Return the length of `x` as well?

    Returns
    -------
    numpy.ndarray
        Normalized array.
    float (only if return_norm is set)
        Norm of `x`.
    """
    if norm_scheme == "n":
        if return_norm:
            _, length = matutils.unitvec(x, return_norm=return_norm)
            return x, length
        else:
            return x
    elif norm_scheme == "c":
        return matutils.unitvec(x, return_norm=return_norm)


class TfidfModel(interfaces.TransformationABC):
    """Objects of this class realize the transformation between word-document co-occurrence matrix (int)
    into a locally/globally weighted TF-IDF matrix (positive floats).

    Examples
    --------
    .. sourcecode:: pycon

        >>> import gensim.downloader as api
        >>> from gensim.models import TfidfModel
        >>> from gensim.corpora import Dictionary
        >>>
        >>> dataset = api.load("text8")
        >>> dct = Dictionary(dataset)  # fit dictionary
        >>> corpus = [dct.doc2bow(line) for line in dataset]  # convert corpus to BoW format
        >>>
        >>> model = TfidfModel(corpus)  # fit model
        >>> vector = model[corpus[0]]  # apply model to the first corpus document

    """
    def __init__(self, corpus=None, id2word=None, dictionary=None, wlocal=utils.identity,
                 wglobal=df2idf, normalize=True, smartirs=None, pivot=None, slope=0.25):
        r"""Compute TF-IDF by multiplying a local component (term frequency) with a global component
        (inverse document frequency), and normalizing the resulting documents to unit length.
        Formula for non-normalized weight of term :math:`i` in document :math:`j` in a corpus of :math:`D` documents

        .. math:: weight_{i,j} = frequency_{i,j} * log_2 \frac{D}{document\_freq_{i}}

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
        wlocals : callable, optional
            Function for local weighting, default for `wlocal` is :func:`~gensim.utils.identity`
            (other options: :func:`numpy.sqrt`, `lambda tf: 0.5 + (0.5 * tf / tf.max())`, etc.).
        wglobal : callable, optional
            Function for global weighting, default is :func:`~gensim.models.tfidfmodel.df2idf`.
        normalize : {bool, callable}, optional
            Normalize document vectors to unit euclidean length? You can also inject your own function into `normalize`.
        smartirs : str, optional
            SMART (System for the Mechanical Analysis and Retrieval of Text) Information Retrieval System,
            a mnemonic scheme for denoting tf-idf weighting variants in the vector space model.
            The mnemonic for representing a combination of weights takes the form XYZ,
            for example 'ntc', 'bpn' and so on, where the letters represents the term weighting of the document vector.

            Term frequency weighing:
                * `b` - binary,
                * `t` or `n` - raw,
                * `a` - augmented,
                * `l` - logarithm,
                * `d` - double logarithm,
                * `L` - log average.

            Document frequency weighting:
                * `x` or `n` - none,
                * `f` - idf,
                * `t` - zero-corrected idf,
                * `p` - probabilistic idf.

            Document normalization:
                * `x` or `n` - none,
                * `c` - cosine,
                * `u` - pivoted unique,
                * `b` - pivoted character length.

            Default is 'nfc'.
            For more information visit `SMART Information Retrieval System
            <https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System>`_.
        pivot : float or None, optional
            In information retrieval, TF-IDF is biased against long documents [1]_. Pivoted document length
            normalization solves this problem by changing the norm of a document to `slope * old_norm + (1.0 -
            slope) * pivot`.

            You can either set the `pivot` by hand, or you can let Gensim figure it out automatically with the following
            two steps:

                * Set either the `u` or `b` document normalization in the `smartirs` parameter.
                * Set either the `corpus` or `dictionary` parameter. The `pivot` will be automatically determined from
                  the properties of the `corpus` or `dictionary`.

            If `pivot` is None and you don't follow steps 1 and 2, then pivoted document length normalization will be
            disabled. Default is None.

            See also the blog post at https://rare-technologies.com/pivoted-document-length-normalisation/.
        slope : float, optional
            In information retrieval, TF-IDF is biased against long documents [1]_. Pivoted document length
            normalization solves this problem by changing the norm of a document to `slope * old_norm + (1.0 -
            slope) * pivot`.

            Setting the `slope` to 0.0 uses only the `pivot` as the norm, and setting the `slope` to 1.0 effectively
            disables pivoted document length normalization. Singhal [2]_ suggests setting the `slope` between 0.2 and
            0.3 for best results. Default is 0.25.

            See also the blog post at https://rare-technologies.com/pivoted-document-length-normalisation/.

        See Also
        --------
        ~gensim.sklearn_api.tfidf.TfIdfTransformer : Class that also uses the SMART scheme.
        resolve_weights : Function that also uses the SMART scheme.

        References
        ----------
        .. [1] Singhal, A., Buckley, C., & Mitra, M. (1996). `Pivoted Document Length
           Normalization <http://singhal.info/pivoted-dln.pdf>`_. *SIGIR Forum*, 51, 176–184.
        .. [2] Singhal, A. (2001). `Modern information retrieval: A brief overview <http://singhal.info/ieee2001.pdf>`_.
           *IEEE Data Eng. Bull.*, 24(4), 35–43.

        """
        self.id2word = id2word
        self.wlocal, self.wglobal, self.normalize = wlocal, wglobal, normalize
        self.num_docs, self.num_nnz, self.idfs = None, None, None
        self.smartirs = resolve_weights(smartirs) if smartirs is not None else None
        self.slope = slope
        self.pivot = pivot
        self.eps = 1e-12

        if smartirs:
            n_tf, n_df, n_n = self.smartirs
            self.wlocal = partial(smartirs_wlocal, local_scheme=n_tf)
            self.wglobal = partial(smartirs_wglobal, global_scheme=n_df)

        if dictionary:
            # user supplied a Dictionary object, which already contains all the
            # statistics we need to construct the IDF mapping. we can skip the
            # step that goes through the corpus (= an optimization).
            if corpus:
                logger.warning(
                    "constructor received both corpus and explicit inverse document frequencies; ignoring the corpus"
                )
            self.num_docs, self.num_nnz = dictionary.num_docs, dictionary.num_nnz
            self.cfs = dictionary.cfs.copy()
            self.dfs = dictionary.dfs.copy()
            self.term_lens = {termid: len(term) for termid, term in dictionary.items()}
            self.idfs = precompute_idfs(self.wglobal, self.dfs, self.num_docs)
            if not id2word:
                self.id2word = dictionary
        elif corpus:
            self.initialize(corpus)
        else:
            # NOTE: everything is left uninitialized; presumably the model will
            # be initialized in some other way
            pass

        # If smartirs is not None, override pivot and normalize
        if not smartirs:
            return
        if self.pivot is not None:
            if n_n in 'ub':
                logger.warning("constructor received pivot; ignoring smartirs[2]")
            return
        if n_n in 'ub' and callable(self.normalize):
            logger.warning("constructor received smartirs; ignoring normalize")
        if n_n in 'ub' and not dictionary and not corpus:
            logger.warning("constructor received no corpus or dictionary; ignoring smartirs[2]")
        elif n_n == "u":
            self.pivot = 1.0 * self.num_nnz / self.num_docs
        elif n_n == "b":
            self.pivot = 1.0 * sum(
                self.cfs[termid] * (self.term_lens[termid] + 1.0) for termid in dictionary.keys()
            ) / self.num_docs

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a previously saved TfidfModel class. Handles backwards compatibility from
        older TfidfModel versions which did not use pivoted document normalization.

        """
        model = super(TfidfModel, cls).load(*args, **kwargs)
        if not hasattr(model, 'pivot'):
            model.pivot = None
            logger.info('older version of %s loaded without pivot arg', cls.__name__)
            logger.info('Setting pivot to %s.', model.pivot)
        if not hasattr(model, 'slope'):
            model.slope = 0.65
            logger.info('older version of %s loaded without slope arg', cls.__name__)
            logger.info('Setting slope to %s.', model.slope)
        if not hasattr(model, 'smartirs'):
            model.smartirs = None
            logger.info('older version of %s loaded without smartirs arg', cls.__name__)
            logger.info('Setting smartirs to %s.', model.smartirs)
        return model

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
        self.cfs = None
        self.dfs = dfs
        self.term_lengths = None
        # and finally compute the idf weights
        self.idfs = precompute_idfs(self.wglobal, self.dfs, self.num_docs)
        self.add_lifecycle_event(
            "initialize",
            msg=(
                f"calculated IDF weights for {self.num_docs} documents and {max(dfs.keys()) + 1 if dfs else 0}"
                f" features ({self.num_nnz} matrix non-zeros)"
            ),
        )

    def __getitem__(self, bow, eps=1e-12):
        """Get the tf-idf representation of an input vector and/or corpus.

        bow : {list of (int, int), iterable of iterable of (int, int)}
            Input document in the `sparse Gensim bag-of-words format
            <https://radimrehurek.com/gensim/intro.html#core-concepts>`_,
            or a streamed corpus of such documents.
        eps : float
            Threshold value, will remove all position that have tfidf-value less than `eps`.

        Returns
        -------
        vector : list of (int, float)
            TfIdf vector, if `bow` is a single document
        :class:`~gensim.interfaces.TransformedCorpus`
            TfIdf corpus, if `bow` is a corpus.

        """
        self.eps = eps
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
            for termid, tf in zip(termid_array, tf_array) if abs(self.idfs.get(termid, 0.0)) > self.eps
        ]

        # and finally, normalize the vector either to unit length, or use a
        # user-defined normalization function
        if self.smartirs:
            n_n = self.smartirs[2]
            if n_n == "n" or (n_n in 'ub' and self.pivot is None):
                if self.pivot is not None:
                    _, old_norm = matutils.unitvec(vector, return_norm=True)
                norm_vector = vector
            elif n_n == "c":
                if self.pivot is not None:
                    _, old_norm = matutils.unitvec(vector, return_norm=True)
                else:
                    norm_vector = matutils.unitvec(vector)
            elif n_n == "u":
                _, old_norm = matutils.unitvec(vector, return_norm=True, norm='unique')
            elif n_n == "b":
                old_norm = sum(freq * (self.term_lens[termid] + 1.0) for termid, freq in bow)
        else:
            if self.normalize is True:
                self.normalize = matutils.unitvec
            elif self.normalize is False:
                self.normalize = utils.identity

            if self.pivot is not None:
                _, old_norm = self.normalize(vector, return_norm=True)
            else:
                norm_vector = self.normalize(vector)

        if self.pivot is None:
            norm_vector = [(termid, weight) for termid, weight in norm_vector if abs(weight) > self.eps]
        else:
            pivoted_norm = (1 - self.slope) * self.pivot + self.slope * old_norm
            norm_vector = [
                (termid, weight / float(pivoted_norm))
                for termid, weight in vector
                if abs(weight / float(pivoted_norm)) > self.eps
            ]
        return norm_vector
