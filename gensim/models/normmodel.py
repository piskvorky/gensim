#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import logging

from gensim import interfaces, matutils

logger = logging.getLogger(__name__)


class NormModel(interfaces.TransformationABC):
    """
    Objects of this class realize the explicit normalization of vectors.

    Model persistency is achieved via its load/save methods.

    Parameters
    ----------
    corpus : iterable
        Iterable of documents.
    norm : {'l1', 'l2'}
        Norm used to normalize. l1 and l2 norms are supported (l2 is default)

    Methods
    -------
    __init__(corpus=None, norm='l2')
        Normalizes the terms in the given corpus document-wise.
    calc_norm(corpus):
        Calculates the norm by calling matutils.unitvec with the norm parameter.

    Examples:
    ---------
    >>> from gensim.models import NormModel
    >>> from gensim.corpora import Dictionary
    >>> from gensim.test.utils import common_texts
    >>> dictionary = Dictionary(common_texts)
    >>> corpus = [dictionary.doc2bow(text) for text in common_texts]
    >>> norm_l2 = NormModel(corpus)
    >>> some_doc = dictionary.doc2bow(common_texts[0])
    >>> print(norm_l2[some_doc])
    >>> norm_ld.save('/tmp/foo.norm_model')
    """

    def __init__(self, corpus=None, norm='l2'):
        """
        Compute the l1 or l2 normalization by normalizing separately for
        each document in a corpus.

        If v_{i,j} is the 'i'th component of the vector representing document
        'j', the l1 normalization is:

        .. math:: norml1_{i, j} = \frac{v_{i,j}}{\sum_k |v_{k,j}|}

        The l2 normalization is:

        .. math:: norml2_{i, j} = \frac{v_{i,j}}{\sqrt{\sum_k v_{k,j}^2}}

        Parameters
        ----------
        corpus : iterable
            Iterable of documents.
        norm : {'l1', 'l2'}
            Norm used to normalize. l1 and l2 norms are supported (l2 is default)

        """
        self.norm = norm
        if corpus is not None:
            self.calc_norm(corpus)
        else:
            pass

    def __str__(self):
        return "NormModel(num_docs=%s, num_nnz=%s, norm=%s)" % (self.num_docs, self.num_nnz, self.norm)

    def calc_norm(self, corpus):
        """
        Calculates the norm by calling matutils.unitvec with the norm parameter.

        Parameters
        ----------
        corpus : iterable
            Iterable of documents.
        """
        logger.info("Performing %s normalization...", self.norm)
        norms = []
        numnnz = 0
        docno = 0
        for bow in corpus:
            docno += 1
            numnnz += len(bow)
            norms.append(matutils.unitvec(bow, self.norm))
        self.num_docs = docno
        self.num_nnz = numnnz
        self.norms = norms

    def normalize(self, bow):
        """
        Normalizes a simple count representation.

        Parameters
        ----------
        bow : :class:`~interfaces.CorpusABC` (iterable of documents) or list
        of (int, int).
        """
        vector = matutils.unitvec(bow, self.norm)
        return vector

    def __getitem__(self, bow):
        """
        Calls the self.normalize() method.

        Parameters
        ----------
        bow : :class:`~interfaces.CorpusABC` (iterable of documents) or list
        of (int, int).

        Examples:
        ---------
        >>> norm_l2 = NormModel(corpus)
        >>> print(norm_l2[some_doc])
        """
        return self.normalize(bow)
