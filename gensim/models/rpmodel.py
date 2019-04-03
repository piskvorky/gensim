#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""Random Projections (also known as Random Indexing).

For theoretical background on Random Projections, see [1]_.


Examples
--------
.. sourcecode:: pycon

    >>> from gensim.models import RpModel
    >>> from gensim.corpora import Dictionary
    >>> from gensim.test.utils import common_texts, temporary_file
    >>>
    >>> dictionary = Dictionary(common_texts)  # fit dictionary
    >>> corpus = [dictionary.doc2bow(text) for text in common_texts]  # convert texts to BoW format
    >>>
    >>> model = RpModel(corpus, id2word=dictionary)  # fit model
    >>> result = model[corpus[3]]  # apply model to document, result is vector in BoW format
    >>>
    >>> with temporary_file("model_file") as fname:
    ...     model.save(fname)  # save model to file
    ...     loaded_model = RpModel.load(fname)  # load model


References
----------
.. [1] Kanerva et al., 2000, Random indexing of text samples for Latent Semantic Analysis,
       https://cloudfront.escholarship.org/dist/prd/content/qt5644k0w6/qt5644k0w6.pdf

"""

import logging

import numpy as np

from gensim import interfaces, matutils, utils


logger = logging.getLogger(__name__)


class RpModel(interfaces.TransformationABC):

    def __init__(self, corpus, id2word=None, num_topics=300):
        """

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Input corpus.

        id2word : {dict of (int, str), :class:`~gensim.corpora.dictionary.Dictionary`}, optional
            Mapping `token_id` -> `token`, will be determine from corpus if `id2word == None`.

        num_topics : int, optional
            Number of topics.

        """
        self.id2word = id2word
        self.num_topics = num_topics
        if corpus is not None:
            self.initialize(corpus)

    def __str__(self):
        return "RpModel(num_terms=%s, num_topics=%s)" % (self.num_terms, self.num_topics)

    def initialize(self, corpus):
        """Initialize the random projection matrix.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
          Input corpus.

        """
        if self.id2word is None:
            logger.info("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif self.id2word:
            self.num_terms = 1 + max(self.id2word)
        else:
            self.num_terms = 0

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
        """Get random-projection representation of the input vector or corpus.

        Parameters
        ----------
        bow : {list of (int, int), iterable of list of (int, int)}
            Input document or corpus.

        Returns
        -------
        list of (int, float)
            if `bow` is document OR
        :class:`~gensim.interfaces.TransformedCorpus`
            if `bow` is corpus.

        Examples
        ----------
        .. sourcecode:: pycon

            >>> from gensim.models import RpModel
            >>> from gensim.corpora import Dictionary
            >>> from gensim.test.utils import common_texts
            >>>
            >>> dictionary = Dictionary(common_texts)  # fit dictionary
            >>> corpus = [dictionary.doc2bow(text) for text in common_texts]  # convert texts to BoW format
            >>>
            >>> model = RpModel(corpus, id2word=dictionary)  # fit model
            >>>
            >>> # apply model to document, result is vector in BoW format, i.e. [(1, 0.3), ... ]
            >>> result = model[corpus[0]]

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
        """Sets the internal state and updates freshly_loaded to True, called when unpicked.

        Parameters
        ----------
        state : dict
           State of the class.

        """
        self.__dict__ = state
        self.freshly_loaded = True
