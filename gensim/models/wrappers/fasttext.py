#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Python wrapper around word representation learning from FastText, a library for efficient learning
of word representations and sentence classification [1].

This module allows training a word embedding from a training corpus with the additional ability
to obtain word vectors for out-of-vocabulary words, using the fastText C implementation.

The wrapped model can NOT be updated with new documents for online training -- use gensim's
`Word2Vec` for that.

Example:

>>> model = gensim.models.wrappers.LdaMallet('/Users/kofola/fastText/fasttext', corpus_file='text8')
>>> print model[word]  # prints vector for given words

.. [1] https://github.com/facebookresearch/fastText#enriching-word-vectors-with-subword-information

"""


import logging
import tempfile
import os
import struct

import numpy as np

from gensim import utils
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec

from six import string_types

logger = logging.getLogger(__name__)


class FastTextKeyedVectors(KeyedVectors):
    pass

class FastText(Word2Vec):
    """
    Class for word vector training using FastText. Communication between FastText and Python
    takes place by working with data files on disk and calling the FastText binary with
    subprocess.call().
    Implements functionality similar to [fasttext.py](https://github.com/salestock/fastText.py),
    improving speed and scope of functionality like `most_similar`, `accuracy` by extracting vectors
    into numpy matrix.

    """

    def initialize_word_vectors(self):
        self.wv = FastTextKeyedVectors()  # wv --> word vectors

    @classmethod
    def train(cls, ft_path, corpus_file, output_file=None, model='cbow', size=100, alpha=0.025, window=5, min_count=5,
            loss='ns', sample=1e-3, negative=5, iter=5, min_n=3, max_n=6, sorted_vocab=1, threads=12):
        """
        `ft_path` is the path to the FastText executable, e.g. `/home/kofola/fastText/fasttext`.

        `corpus_file` is the filename of the text file to be used for training the FastText model.
        Expects file to contain space-separated tokens in a single line

        `model` defines the training algorithm. By default, cbow is used. Accepted values are
        cbow, skipgram.

        `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the current and predicted word within a sentence.

        `alpha` is the initial learning rate (will linearly drop to `min_alpha` as training progresses).

        `min_count` = ignore all words with total frequency lower than this.

        `loss` = defines training objective. Allowed values are `hs` (hierarchical softmax),
        `ns` (negative sampling) and `softmax`. Defaults to `ns`

        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
            default is 1e-3, useful range is (0, 1e-5).

        `negative` = the value for negative specifies how many "noise words" should be drawn
        (usually between 5-20). Default is 5. If set to 0, no negative samping is used.
        Only relevant when `loss` is set to `ns`

        `iter` = number of iterations (epochs) over the corpus. Default is 5.

        `min_n` = min length of char ngrams to be used for training word representations. Default is 1.

        `max_n` = max length of char ngrams to be used for training word representations. Set `max_n` to be
        greater than `min_n` to avoid char ngrams being used. Default is 5.

        `sorted_vocab` = if 1 (default), sort the vocabulary by descending frequency before
        assigning word indexes.

        """
        ft_path = ft_path
        output_file = output_file or os.path.join(tempfile.gettempdir(), 'ft_model')
        ft_args = {
            'input': corpus_file,
            'output': output_file,
            'lr': alpha,
            'dim': size,
            'ws': window,
            'epoch': iter,
            'minCount': min_count,
            'neg': negative,
            'loss': loss,
            'minn': min_n,
            'maxn': max_n,
            'thread': threads,
            't': sample
        }
        cmd = [ft_path, model]
        for option, value in ft_args.items():
            cmd.append("-%s" % option)
            cmd.append(str(value))

        output = utils.check_output(args=cmd)
        model = cls.load_fasttext_format(output_file)
        return model

    @classmethod
    def load_fasttext_format(cls, model_file):
        model = cls.load_word2vec_format('%s.vec' % model_file)
        model.load_binary_data('%s.bin' % model_file)
        return model

    def load_binary_data(self, model_file):
        with open(model_file, 'rb') as f:
            self.load_model_params(f)
            self.load_dict(f)
            self.load_vectors(f)

    def load_model_params(self, f):
        (dim, ws, epoch, minCount, neg, _, loss, model, bucket, minn, maxn, _, t) = self.struct_unpack(f, '@12i1d')
        self.size = dim
        self.window = ws
        self.iter = epoch
        self.min_count = minCount
        self.negative = neg
        self.loss = loss
        self.sg = model == 'skipgram'
        self.bucket = bucket
        self.min_n = minn
        self.max_n = maxn
        self.sample = t

    def load_dict(self, f):
        (dim, nwords, _) = self.struct_unpack(f, '@3i') 
        assert len(self.wv.vocab) == nwords, 'mismatch between vocab sizes'
        ntokens, = self.struct_unpack(f, '@q') 
        for i in range(nwords):
            word = ''
            char, = self.struct_unpack(f, '@c')
            char = char.decode()
            while char != '\x00':
                word += char 
                char, = self.struct_unpack(f, '@c')
                char = char.decode()
            count, _ = self.struct_unpack(f, '@ib')
            _ = self.struct_unpack(f, '@i')
            assert self.wv.vocab[word].index == i, 'mismatch between gensim word index and fastText word index'
            self.wv.vocab[word].count = count

    def load_vectors(self, f):
        num_vectors, dim = self.struct_unpack(f, '@2q')
        float_size = struct.calcsize('@f')
        if float_size == 4:
            dtype = np.dtype(np.float32)
        elif float_size == 8:
            dtype = np.dtype(np.float64)

        self.num_original_vectors = num_vectors
        self.syn0_all = np.fromstring(f.read(num_vectors * dim * float_size), dtype=dtype)
        self.syn0_all = self.syn0_all.reshape((num_vectors, dim))
        self.init_ngrams()

    def struct_unpack(self, f, fmt):
        num_bytes = struct.calcsize(fmt)
        return struct.unpack(fmt, f.read(num_bytes))

    def init_ngrams(self):
        self.ngrams = {}
        all_ngrams = []
        for w, v in self.vocab.items():
            all_ngrams += self.compute_ngrams(w, self.min_n, self.max_n)
        all_ngrams = set(all_ngrams)
        self.num_ngram_vectors = len(all_ngrams)
        ngram_indices = []
        for i, ngram in enumerate(all_ngrams):
            ngram_hash = self.ft_hash(ngram)
            ngram_indices.append((len(self.wv.vocab) + ngram_hash) % self.bucket)
            self.ngrams[ngram] = i
        self.syn0_all = self.syn0_all.take(ngram_indices, axis=0)

    @staticmethod
    def compute_ngrams(word, min_n, max_n):
        ngram_indices = []
        BOW, EOW = ('<','>')
        extended_word = BOW + word + EOW
        ngrams = set()
        for i in range(len(extended_word) - min_n + 1):
            for j in range(min_n, max(len(extended_word) - max_n, max_n + 1)):
                ngrams.add(extended_word[i:i+j])
        return ngrams

    @staticmethod
    def ft_hash(string):
        # Reproduces hash method used in fastText
        h = np.uint32(2166136261)
        for c in string:
            h = h ^ np.uint32(ord(c))
            h = h * np.uint32(16777619)
        return h

    def __getitem__(self, words):
        if isinstance(words, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.word_vector(words)

        return np.vstack([self.word_vector(word) for word in words])

    def word_vector(self, word):
        if word in self.wv.vocab:
            return self.wv[word]
        else:
            return self.oov_vector(word)

    def oov_vector(self, word):
        word_vec = np.zeros(self.size)
        ngrams = self.compute_ngrams(word, self.min_n, self.max_n)
        for ngram in ngrams:
            if ngram in self.ngrams:
                word_vec += self.syn0_all[self.ngrams[ngram]]
        if word_vec.any():
            return word_vec/len(ngrams)
        else: # No ngrams of the word are present in self.ngrams
            raise KeyError('all ngrams for word %s absent from model' % word)
