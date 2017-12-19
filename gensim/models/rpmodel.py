#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Objects of this class allow building and maintaining a model for Random Projections
(also known as Random Indexing). 

For theoretical background on RP, see:   Kanerva et al.: "Random indexing of text samples for Latent Semantic Analysis."

    The main methods are:

    1. constructor, which creates the random projection matrix
    2. the [] method, which transforms a simple count representation into the TfIdf
       space.

Model persistency is achieved via its load/save methods.


Examples:
---------
>>> from gensim.models import rpmmodel
>>> rp = RpModel(corpus)
>>> print(rp[some_doc])
>>> rp.save('/tmp/foo.rp_model')
"""

import logging

import numpy as np

from gensim import interfaces, matutils, utils


logger = logging.getLogger('gensim.models.rpmodel')


class RpModel(interfaces.TransformationABC):

    def __init__(self, corpus, id2word=None, num_topics=300):
        """
        `id2word` is a mapping from word ids (integers) to words (strings). It is
        used to determine the vocabulary size, as well as for debugging and topic
        printing. If not set, it will be determined from the corpus.


        Parameters
        ----------
        corpus : interfaces.CorpusABC
        id2word : dict of int tostring
        num_topics : int

        """
        self.id2word = id2word
        self.num_topics = num_topics
        if corpus is not None:
            self.initialize(corpus)

    def __str__(self):
        return "RpModel(num_terms=%s, num_topics=%s)" % (self.num_terms, self.num_topics)

    def initialize(self, corpus):
        """
        Initialize the random projection matrix.


        Parameters
        ----------
        corpus : :class:`~interfaces.CorpusABC`

        """
        if self.id2word is None:
            logger.info("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        else:
            self.num_terms = 1 + max([-1] + self.id2word.keys())

        shape = self.num_topics, self.num_terms
        logger.info("constructing %s random matrix", str(shape))
        # Now construct the projection matrix itself.
        # Here i use a particular form, derived in "Achlioptas: Database-friendly random projection",
        # and his (1) scenario of Theorem 1.1 in particular (all entries are +1/-1).
        randmat = 1 - 2 * np.random.binomial(1, 0.5, shape)  # convert from 0/1 to +1/-1
        # convert from int32 to floats, for faster multiplications
        self.projection = np.asfortranarray(randmat, dtype=np.float32)
        # TODO: check whether the Fortran-order shenanigans still make sense. In the original
        # code (~2010), this made a BIG difference for np BLAS implementations; perhaps now the wrappers
        # are smarter and this is no longer needed?

    def __getitem__(self, bow):
        """
        Return RP representation of the input vector and/or corpus.

        Parameters
        ----------
        bow : :class:`~interfaces.CorpusABC` (iterable of documents) or list of (int, int).

        Examples:
        -------------
        >>> rp = RpModel(corpus)
        >>> print(rp[some_doc])

        """
        # if the input vector is in fact a corpus, return a transformed corpus as result
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)

        if getattr(self, 'freshly_loaded', False):
            # This is a hack to work around a bug in np, where a FORTRAN-order array
            # unpickled from disk segfaults on using it.
            self.freshly_loaded = False
            self.projection = self.projection.copy('F')  # simply making a fresh copy fixes the broken array

        vec = matutils.sparse2full(bow, self.num_terms).reshape(self.num_terms, 1) / np.sqrt(self.num_topics)
        vec = np.asfortranarray(vec, dtype=np.float32)
        topic_dist = np.dot(self.projection, vec)  # (k, d) * (d, 1) = (k, 1)
        return [
            (topicid, float(topicvalue)) for topicid, topicvalue in enumerate(topic_dist.flat)
            if np.isfinite(topicvalue) and not np.allclose(topicvalue, 0.0)
        ]

    def __setstate__(self, state):
        """
        Sets the internal state and updates freshly_loaded to True. Called when unpicked.

        Parameters
        ----------
        state : state of the class
        """
        self.__dict__ = state
        self.freshly_loaded = True
