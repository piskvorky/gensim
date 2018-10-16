#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2017 Anmol Gulati <anmol01gulati@gmail.com>
# Copyright (C) 2017 Radim Rehurek <radimrehurek@seznam.cz>

"""Python wrapper around `Varembed model <https://github.com/rguthrie3/MorphologicalPriorsForWordEmbeddings>`_.
Original paper:`"Morphological Priors for Probabilistic Neural Word Embeddings" <http://arxiv.org/pdf/1608.01056.pdf>`_.

Notes
-----
* This module allows ability to obtain word vectors for out-of-vocabulary words, for the Varembed model.
* The wrapped model can not be updated with new documents for online training.

"""

import logging
import numpy as np

from gensim import utils
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Vocab

logger = logging.getLogger(__name__)


class VarEmbed(KeyedVectors):
    """Python wrapper using `Varembed <https://github.com/rguthrie3/MorphologicalPriorsForWordEmbeddings>`_.

    Warnings
    --------
    This is **only** python wrapper for `Varembed <https://github.com/rguthrie3/MorphologicalPriorsForWordEmbeddings>`_,
    this allows to load pre-trained models only.

    """
    def __init__(self):
        self.vector_size = 0
        self.vocab_size = 0

    @classmethod
    def load_varembed_format(cls, vectors, morfessor_model=None):
        """Load the word vectors into matrix from the varembed output vector files.

        Parameters
        ----------
        vectors : dict
            Pickle file containing the word vectors.
        morfessor_model : str, optional
            Path to the trained morfessor model.

        Returns
        -------
        :class:`~gensim.models.wrappers.varembed.VarEmbed`
            Ready to use instance.

        """
        result = cls()
        if vectors is None:
            raise Exception("Please provide vectors binary to load varembed model")
        d = utils.unpickle(vectors)
        word_to_ix = d['word_to_ix']
        morpho_to_ix = d['morpho_to_ix']
        word_embeddings = d['word_embeddings']
        morpho_embeddings = d['morpheme_embeddings']
        result.load_word_embeddings(word_embeddings, word_to_ix)
        if morfessor_model:
            try:
                import morfessor
                morfessor_model = morfessor.MorfessorIO().read_binary_model_file(morfessor_model)
                result.add_morphemes_to_embeddings(morfessor_model, morpho_embeddings, morpho_to_ix)
            except ImportError:
                # Morfessor Package not found.
                logger.error('Could not import morfessor. Not using morpheme embeddings')
                raise ImportError('Could not import morfessor.')

        logger.info('Loaded varembed model vectors from %s', vectors)
        return result

    def load_word_embeddings(self, word_embeddings, word_to_ix):
        """Loads the word embeddings.

        Parameters
        ----------
        word_embeddings : numpy.ndarray
            Matrix with word-embeddings.
        word_to_ix : dict of (str, int)
            Mapping word to index.

        """
        logger.info("Loading the vocabulary")
        self.vocab = {}
        self.index2word = []
        counts = {}
        for word in word_to_ix:
            counts[word] = counts.get(word, 0) + 1
        self.vocab_size = len(counts)
        self.vector_size = word_embeddings.shape[1]
        self.vectors = np.zeros((self.vocab_size, self.vector_size))
        self.index2word = [None] * self.vocab_size
        logger.info("Corpus has %i words", len(self.vocab))
        for word_id, word in enumerate(counts):
            self.vocab[word] = Vocab(index=word_id, count=counts[word])
            self.vectors[word_id] = word_embeddings[word_to_ix[word]]
            self.index2word[word_id] = word
        assert((len(self.vocab), self.vector_size) == self.vectors.shape)
        logger.info("Loaded matrix of %d size and %d dimensions", self.vocab_size, self.vector_size)

    def add_morphemes_to_embeddings(self, morfessor_model, morpho_embeddings, morpho_to_ix):
        """Include morpheme embeddings into vectors.

        Parameters
        ----------
        morfessor_model : :class:`morfessor.baseline.BaselineModel`
            Morfessor model.
        morpho_embeddings : dict
            Pickle file containing morpheme embeddings.
        morpho_to_ix : dict
            Mapping morpheme to index.

        """
        for word in self.vocab:
            morpheme_embedding = np.array(
                [
                    morpho_embeddings[morpho_to_ix.get(m, -1)]
                    for m in morfessor_model.viterbi_segment(word)[0]
                ]
            ).sum(axis=0)
            self.vectors[self.vocab[word].index] += morpheme_embedding
        logger.info("Added morphemes to word vectors")
