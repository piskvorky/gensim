#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2017 Anmol Gulati <anmol01gulati@gmail.com>
# Copyright (C) 2017 Radim Rehurek <radimrehurek@seznam.cz>

"""
Python wrapper around word representation learning from Varembed models, a library for efficient learning of word representations and sentence classification [1].

This module allows ability to obtain word vectors for out-of-vocabulary words, for the Varembed model[2].

The wrapped model can NOT be updated with new documents for online training -- use gensim's `Word2Vec` for that.

.. [1] https://github.com/rguthrie3/MorphologicalPriorsForWordEmbeddings

.. [2] http://arxiv.org/pdf/1608.01056.pdf
"""

import logging
try:
    import morfessor
    USE_MORPHEMES = 1
except ImportError:
    # Morfessor Package not found. Will only allow reading varembed vectors without morpheme embeddings.
    USE_MORPHEMES = 0

import numpy as np

from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec

# utility fnc for pickling, common scipy operations etc
from gensim import utils
from gensim.models.word2vec import Vocab

logger = logging.getLogger(__name__)


class VarEmbed(Word2Vec):
    """
    Class for word vectors using Varembed models. Contains methods to load a varembed model and implements
    functionality like `most_similar`, `similarity` by extracting vectors into numpy matrix.
    Refer to [Varembed]https://github.com/rguthrie3/MorphologicalPriorsForWordEmbeddings for
    implementation of Varembed models.
    """

    def __init__(self):
        self.wv = KeyedVectors()
        self.vector_size = 0
        self.vocab_size = 0

    @classmethod
    def load_varembed_format(cls, vectors, morfessor_model=None, use_morphemes=False):
        """
        Load the input-hidden weight matrix from the fast text output files.

        Note that due to limitations in the FastText API, you cannot continue training
        with a model loaded this way, though you can query for word similarity etc.

        'vectors' is the pickle file containing the word vectors.
        'morfessor_model' is the path to the trained morfessor model.
        'use_morphemes' False(default) use of morpheme embeddings in output.
        """
        result = cls()
        if vectors is None:
            raise Exception(
                "Please provide vectors binary to load varembed model")
        D = utils.unpickle(vectors)
        word_to_ix = D['word_to_ix']
        morpho_to_ix = D['morpho_to_ix']
        word_embeddings = D['word_embeddings']
        morpho_embeddings = D['morpheme_embeddings']
        result.load_word_embeddings(word_embeddings, word_to_ix)
        if use_morphemes:
            if USE_MORPHEMES == -1:
                logger.warning('Could not import morfessor. Not using morpheme embeddings')
            else:
                morfessor_model = morfessor.MorfessorIO().read_binary_model_file(morfessor_model)
                result.ensemble_morpheme_embeddings(morfessor_model, morpho_embeddings, morpho_to_ix)
        logger.info('Loaded varembed model vectors from %s', vectors)
        return result

    def load_word_embeddings(self, word_embeddings, word_to_ix):
        """ Loads the word embeddings """
        logger.info("Loading the vocabulary")
        self.wv.vocab = {}
        counts = {}
        for word in word_to_ix:
            counts[word] = counts.get(word, 0) + 1
        self.vocab_size = len(counts)
        self.vector_size = word_embeddings.shape[1]

        self.wv.syn0 = np.zeros((self.vocab_size, self.vector_size))
        self.wv.index2word = [None]*self.vocab_size
        logger.info("Corpus has %i words", len(self.wv.vocab))
        for word_id, word in enumerate(counts):
            self.wv.vocab[word] = Vocab(index=word_id, count=counts[word])
            self.wv.syn0[word_id] = word_embeddings[word_to_ix[word]]
            self.wv.index2word[word_id] = word
        assert((len(self.wv.vocab), self.vector_size) == self.wv.syn0.shape)
        logger.info("Loaded matrix of %d size and %d dimensions", self.vocab_size, self.vector_size)


    def ensemble_morpheme_embeddings(self, morfessor_model, morpho_embeddings, morpho_to_ix):
        for word in self.wv.vocab:
            morpheme_embedding = np.array(
                    [morpho_embeddings[morpho_to_ix.get(m, -1)] for m in
                     morfessor_model.viterbi_segment(word)[0]]).sum(axis=0)
            self.wv.syn0[self.wv.vocab[word].index] += morpheme_embedding
