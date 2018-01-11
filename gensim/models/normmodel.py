#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import logging

from gensim import interfaces, matutils

logger = logging.getLogger(__name__)


class NormModel(interfaces.TransformationABC):
    """Objects of this class realize the explicit normalization of vectors (l1 and l2)."""

    def __init__(self, corpus=None, norm='l2'):
        """Compute the l1 or l2 normalization by normalizing separately for each document in a corpus.

        If :math:`v_{i,j}` is the 'i'th component of the vector representing document 'j', the l1 normalization is

        .. math:: l1_{i, j} = \\frac{v_{i,j}}{\sum_k |v_{k,j}|}

        the l2 normalization is

        .. math:: l2_{i, j} = \\frac{v_{i,j}}{\sqrt{\sum_k v_{k,j}^2}}


        Parameters
        ----------
        corpus : iterable of iterable of (int, number), optional
            Input corpus.
        norm : {'l1', 'l2'}, optional
            Norm used to normalize.

        """
        self.norm = norm
        if corpus is not None:
            self.calc_norm(corpus)
        else:
            pass

    def __str__(self):
        return "NormModel(num_docs=%s, num_nnz=%s, norm=%s)" % (self.num_docs, self.num_nnz, self.norm)

    def calc_norm(self, corpus):
        """Calculates the norm by calling :func:`~gensim.matutils.unitvec` with the norm parameter.

        Parameters
        ----------
        corpus : iterable of iterable of (int, number)
            Input corpus.

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
        """Normalizes a simple count representation.

        Parameters
        ----------
        bow : list of (int, number)
            Document in BoW format.

        Returns
        -------
        list of (int, number)
            Normalized document.


        """
        vector = matutils.unitvec(bow, self.norm)
        return vector

    def __getitem__(self, bow):
        """Calls the :func:`~gensim.models.normmodel.NormModel.normalize`.

        Parameters
        ----------
        bow : list of (int, number)
            Document in BoW format.

        Returns
        -------
        list of (int, number)
            Normalized document.

        """
        return self.normalize(bow)
