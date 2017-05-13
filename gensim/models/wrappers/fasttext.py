#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Python wrapper around word representation learning from FastText, a library for efficient learning
of word representations and sentence classification [1].

This module allows training a word embedding from a training corpus with the additional ability
to obtain word vectors for out-of-vocabulary words, using the fastText C implementation.

The wrapped model can NOT be updated with new documents for online training -- use gensim's
`Word2Vec` for that.

Example:

>>> from gensim.models.wrappers import FastText
>>> model = fasttext.FastText.train('/Users/kofola/fastText/fasttext', corpus_file='text8')
>>> print model['forests']  # prints vector for given out-of-vocabulary word

.. [1] https://github.com/facebookresearch/fastText#enriching-word-vectors-with-subword-information

"""


import logging
import tempfile
import os
import struct

import numpy as np
from numpy import float32 as REAL, sqrt, newaxis
from gensim import utils
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec

from six import string_types

logger = logging.getLogger(__name__)


class FastTextKeyedVectors(KeyedVectors):
    """
    Class to contain vectors, vocab and ngrams for the FastText training class and other methods not directly
    involved in training such as most_similar().
    Subclasses KeyedVectors to implement oov lookups, storing ngrams and other FastText specific methods

    """
    def __init__(self):
        super(FastTextKeyedVectors, self).__init__()
        self.syn0_all_norm = None
        self.ngrams = {}

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'syn0_all_norm'])
        super(FastTextKeyedVectors, self).save(*args, **kwargs)

    def word_vec(self, word, use_norm=False):
        """
        Accept a single word as input.
        Returns the word's representations in vector space, as a 1D numpy array.

        The word can be out-of-vocabulary as long as ngrams for the word are present.
        For words with all ngrams absent, a KeyError is raised.

        Example::

          >>> trained_model['office']
          array([ -1.40128313e-02, ...])

        """
        if word in self.vocab:
            return super(FastTextKeyedVectors, self).word_vec(word, use_norm)
        else:
            word_vec = np.zeros(self.syn0_all.shape[1])
            ngrams = FastText.compute_ngrams(word, self.min_n, self.max_n)
            if use_norm:
                ngram_weights = self.syn0_all_norm
            else:
                ngram_weights = self.syn0_all
            for ngram in ngrams:
                if ngram in self.ngrams:
                    word_vec += ngram_weights[self.ngrams[ngram]]
            if word_vec.any():
                return word_vec / len(ngrams)
            else: # No ngrams of the word are present in self.ngrams
                raise KeyError('all ngrams for word %s absent from model' % word)

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can only call `most_similar`, `similarity` etc.

        """
        super(FastTextKeyedVectors, self).init_sims(replace)
        if getattr(self, 'syn0_all_norm', None) is None or replace:
            logger.info("precomputing L2-norms of ngram weight vectors")
            if replace:
                for i in xrange(self.syn0_all.shape[0]):
                    self.syn0_all[i, :] /= sqrt((self.syn0_all[i, :] ** 2).sum(-1))
                self.syn0_all_norm = self.syn0_all
            else:
                self.syn0_all_norm = (self.syn0_all / sqrt((self.syn0_all ** 2).sum(-1))[..., newaxis]).astype(REAL)

    def __contains__(self, word):
        """
        Check if word is present in the vocabulary, or if any word ngrams are present. A vector for the word is
        guaranteed to exist if `__contains__` returns True.

        """
        if word in self.vocab:
            return True
        else:
            word_ngrams = set(FastText.compute_ngrams(word, self.min_n, self.max_n))
            if len(word_ngrams & set(self.ngrams.keys())):
                return True
            else:
                return False


class FastText(Word2Vec):
    """
    Class for word vector training using FastText. Communication between FastText and Python
    takes place by working with data files on disk and calling the FastText binary with
    subprocess.call().
    Implements functionality similar to [fasttext.py](https://github.com/salestock/fastText.py),
    improving speed and scope of functionality like `most_similar`, `similarity` by extracting vectors
    into numpy matrix.

    """

    def initialize_word_vectors(self):
        self.wv = FastTextKeyedVectors()

    @classmethod
    def train(cls, ft_path, corpus_file, output_file=None, model='cbow', size=100, alpha=0.025, window=5, min_count=5,
            loss='ns', sample=1e-3, negative=5, iter=5, min_n=3, max_n=6, sorted_vocab=1, threads=12):
        """
        `ft_path` is the path to the FastText executable, e.g. `/home/kofola/fastText/fasttext`.

        `corpus_file` is the filename of the text file to be used for training the FastText model.
        Expects file to contain utf-8 encoded text.

        `model` defines the training algorithm. By default, cbow is used. Accepted values are
        'cbow', 'skipgram'.

        `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the current and predicted word within a sentence.

        `alpha` is the initial learning rate.

        `min_count` = ignore all words with total occurrences lower than this.

        `loss` = defines training objective. Allowed values are `hs` (hierarchical softmax),
        `ns` (negative sampling) and `softmax`. Defaults to `ns`

        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
            default is 1e-3, useful range is (0, 1e-5).

        `negative` = the value for negative specifies how many "noise words" should be drawn
        (usually between 5-20). Default is 5. If set to 0, no negative samping is used.
        Only relevant when `loss` is set to `ns`

        `iter` = number of iterations (epochs) over the corpus. Default is 5.

        `min_n` = min length of char ngrams to be used for training word representations. Default is 3.

        `max_n` = max length of char ngrams to be used for training word representations. Set `max_n` to be
        lesser than `min_n` to avoid char ngrams being used. Default is 6.

        `sorted_vocab` = if 1 (default), sort the vocabulary by descending frequency before
        assigning word indexes.

        `threads` = number of threads to use. Default is 12.

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
        cls.delete_training_files(output_file)
        return model

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'syn0_all_norm'])
        super(FastText, self).save(*args, **kwargs)

    @classmethod
    def load_word2vec_format(cls, *args, **kwargs):
        return FastTextKeyedVectors.load_word2vec_format(*args, **kwargs)

    @classmethod
    def load_fasttext_format(cls, model_file):
        """
        Load the input-hidden weight matrix from the fast text output files.

        Note that due to limitations in the FastText API, you cannot continue training
        with a model loaded this way, though you can query for word similarity etc.

        `model_file` is the path to the FastText output files.
        FastText outputs two training files - `/path/to/train.vec` and `/path/to/train.bin`
        Expected value for this example: `/path/to/train`

        """
        model = cls()
        model.wv = cls.load_word2vec_format('%s.vec' % model_file)
        model.load_binary_data('%s.bin' % model_file)
        return model

    @classmethod
    def delete_training_files(cls, model_file):
        """Deletes the files created by FastText training"""
        try:
            os.remove('%s.vec' % model_file)
            os.remove('%s.bin' % model_file)
        except FileNotFoundError:
            logger.debug('Training files %s not found when attempting to delete', model_file)
            pass

    def load_binary_data(self, model_binary_file):
        """Loads data from the output binary file created by FastText training"""
        with utils.smart_open(model_binary_file, 'rb') as f:
            self.load_model_params(f)
            self.load_dict(f)
            self.load_vectors(f)

    def load_model_params(self, file_handle):
        (_,_,dim, ws, epoch, minCount, neg, _, loss, model, bucket, minn, maxn, _, t) = self.struct_unpack(file_handle, '@14i1d')
        # Parameters stored by [Args::save](https://github.com/facebookresearch/fastText/blob/master/src/args.cc)
        self.size = dim
        self.window = ws
        self.iter = epoch
        self.min_count = minCount
        self.negative = neg
        self.hs = loss == 1
        self.sg = model == 2
        self.bucket = bucket
        self.wv.min_n = minn
        self.wv.max_n = maxn
        self.sample = t

    def load_dict(self, file_handle):
        (vocab_size, nwords, _) = self.struct_unpack(file_handle, '@3i')
        # Vocab stored by [Dictionary::save](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)
        assert len(self.wv.vocab) == nwords, 'mismatch between vocab sizes'
        assert len(self.wv.vocab) == vocab_size, 'mismatch between vocab sizes'
        ntokens,pruneidx_size = self.struct_unpack(file_handle, '@2q')
        for i in range(nwords):
            word_bytes = b''
            char_byte = file_handle.read(1)
            # Read vocab word
            while char_byte != b'\x00':
                word_bytes += char_byte
                char_byte = file_handle.read(1)
            word = word_bytes.decode('utf8')
            count, _ = self.struct_unpack(file_handle, '@ib')
            _ = self.struct_unpack(file_handle, '@i')
            assert self.wv.vocab[word].index == i, 'mismatch between gensim word index and fastText word index'
            self.wv.vocab[word].count = count

            for j in range(pruneidx_size):
                _,_ = self.struct_unpack(file_handle,'@2i')

            _ = self.struct_unpack(file_handle,'@?')

    def load_vectors(self, file_handle):
        num_vectors, dim = self.struct_unpack(file_handle, '@2q')
        # Vectors stored by [Matrix::save](https://github.com/facebookresearch/fastText/blob/master/src/matrix.cc)
        assert self.size == dim, 'mismatch between model sizes'
        float_size = struct.calcsize('@f')
        if float_size == 4:
            dtype = np.dtype(np.float32)
        elif float_size == 8:
            dtype = np.dtype(np.float64)

        self.num_original_vectors = num_vectors
        self.wv.syn0_all = np.fromstring(file_handle.read(num_vectors * dim * float_size), dtype=dtype)
        self.wv.syn0_all = self.wv.syn0_all.reshape((num_vectors, dim))
        assert self.wv.syn0_all.shape == (self.bucket + len(self.wv.vocab), self.size), \
            'mismatch between weight matrix shape and vocab/model size'
        self.init_ngrams()

    def struct_unpack(self, file_handle, fmt):
        num_bytes = struct.calcsize(fmt)
        return struct.unpack(fmt, file_handle.read(num_bytes))

    def init_ngrams(self):
        """
        Computes ngrams of all words present in vocabulary and stores vectors for only those ngrams.
        Vectors for other ngrams are initialized with a random uniform distribution in FastText. These
        vectors are discarded here to save space.

        """
        self.wv.ngrams = {}
        all_ngrams = []
        for w, v in self.wv.vocab.items():
            all_ngrams += self.compute_ngrams(w, self.wv.min_n, self.wv.max_n)
        all_ngrams = set(all_ngrams)
        self.num_ngram_vectors = len(all_ngrams)
        ngram_indices = []
        for i, ngram in enumerate(all_ngrams):
            ngram_hash = self.ft_hash(ngram)
            ngram_indices.append((len(self.wv.vocab) + ngram_hash) % self.bucket)
            self.wv.ngrams[ngram] = i
        self.wv.syn0_all = self.wv.syn0_all.take(ngram_indices, axis=0)

    @staticmethod
    def compute_ngrams(word, min_n, max_n):
        ngram_indices = []
        BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
        extended_word = BOW + word + EOW
        ngrams = set()
        for i in range(len(extended_word) - min_n + 1):
            for j in range(min_n, max(len(extended_word) - max_n, max_n + 1)):
                ngrams.add(extended_word[i:i+j])
        return ngrams

    @staticmethod
    def ft_hash(string):
        """
        Reproduces [hash method](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)
        used in fastText.

        """
        # Runtime warnings for integer overflow are raised, this is expected behaviour. These warnings are suppressed.
        old_settings = np.seterr(all='ignore')
        h = np.uint32(2166136261)
        for c in string:
            h = h ^ np.uint32(ord(c))
            h = h * np.uint32(16777619)
        np.seterr(**old_settings)
        return h

