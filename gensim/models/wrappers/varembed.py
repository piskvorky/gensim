#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Python wrapper around word representation learning from Varembed models, a library for efficient learning of word representations and sentence classification [1].

This module allows ability to obtain word vectors for out-of-vocabulary words, for the Varembed model[2].

The wrapped model can NOT be updated with new documents for online training -- use gensim's `Word2Vec` for that.

.. [1] https://github.com/rguthrie3/MorphologicalPriorsForWordEmbeddings

.. [2] http://arxiv.org/pdf/1608.01056.pdf
"""


import logging
import tempfile
import os
import struct
import morfessor

import numpy as np

from gensim import utils
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec

from six import string_types

logger = logging.getLogger(__name__)


# utility fnc for pickling, common scipy operations etc
from gensim import utils, matutils
from gensim.corpora.dictionary import Dictionary
from six import iteritems, itervalues, string_types
from six.moves import xrange
from types import GeneratorType
from scipy import stats

logger = logging.getLogger(__name__)

class VarEmbed(Word2Vec):
    """
    Class for word vectors using Varembed models. Contains methods to load a varembed model and 
    """
    def __init__():
        self.wv = KeyedVectors()

    @classmethod
    def load_varembed_format(cls, vectors=None, morfessor_model=None):
        result = cls()
        if vectors is None or morfessor_model is None:
            raise Exception("Please provide vectors and morfessor_model binary to load varembed model")
        D = utils.unpickle(vectors)
        word_to_ix = D["word_to_ix"]
        morpho_to_ix = D["morpho_to_ix"]
        word_embeddings = D["word_embeddings"]
        morpho_embeddings = D["morpheme_embeddings"]
        morfessor_model = morfessor.MorfessorIO().read_binary_model_file(morfessor_model)
        result.build_vocab(word_to_ix)
        result.load_embeddings(word_embeddings, morpho_embeddings, morfessor_model, word_to_ix)
        result.sort_embeddings
        return result

    def load_dict(self, word_to_ix):
        logger.info("Loading the vocabulary")
        self.wv.vocab = {}
        for word in word_to_ix:
            self.wv.vocab[word] += 1
        logger.info("Corpus has %i words", len(self.wv.vocab))

    def load_embeddings(self, word_embeddings, morpho_embeddings, morfessor_model, word_to_ix):
        for word in word_to_ix:
            embed = word_embeddings[word_to_ix[word]]
            # morpho_embed = np.array([morpho_embeddings[morpho_to_ix.get(
            #     m, -1)] for m in morfessor_model.viterbi_segment(word)[0]]).sum(axis=0)
            # embed = embed + morpho_embed
            self.wv.syn0[word] = embed

    def sort_embeddings(self):
        prev_syn0 = copy.deepcopy(self.wv.syn0)
        prev_vocab = copy.deepcopy(self.wv.vocab)
        self.wv.index2word = []

        sorted_vocab = sorted(iteritems(self.wv.vocab, key=lambda x: x[1])
        for word, count in iteritems(sorted_vocab):
            counts[word] = count
            self.wv.index2word.append(word)

        for word_id, word in enumerate(self.wv.index2word):
            self.wv.syn0[word_id] = prev_syn0[prev_vocab[word].index]
            self.wv.vocab[word].index = word_id
            self.wv.vocab[word].count = counts[word]
        
        







