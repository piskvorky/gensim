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

    Parameters
    ----------
    corpus : iterable
        Iterable of documents.
    norm : {'l1', 'l2'}
        Norm used to normalize. l1 and l2 norms are supported (l2 is default)
    bow : str
        One of the documents.

    Methods
    -------

    __init__(corpus=None, norm='l2')
        Normalizes the terms in the given corpus document-wise.
    normalize()
        Normalizes a simple count representation.
    calc_norm(corpus):
        Calculates the norm by calling matutils.unitvec with the norm parameter.
    __getitem__(bow)
        Calls the self.normalize() method.

    >>> norm_l2 = NormModel(corpus)
    >>> print(norm_l2[some_doc])
    >>> norm_l2.save('/tmp/foo.tfidf_model')

    Model persistency is achieved via its load/save methods
    """

    def __init__(self, corpus=None, norm='l2'):
        """
        Compute the 'l1' or 'l2' normalization by normalizing separately
        for each doc in a corpus.
        Formula for 'l1' norm for term 'i' in document 'j' in a corpus of 'D' documents is::

          norml1_{i, j} = (i / sum(absolute(values in j)))

        Formula for 'l2' norm for term 'i' in document 'j' in a corpus of 'D' documents is::

          norml2_{i, j} = (i / sqrt(sum(square(values in j))))
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
        vector = matutils.unitvec(bow, self.norm)
        return vector

    def __getitem__(self, bow):
        return self.normalize(bow)
