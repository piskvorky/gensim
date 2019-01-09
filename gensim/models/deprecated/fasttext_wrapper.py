#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayantjain1992@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Warnings
--------
.. deprecated:: 3.2.0
   Use :mod:`gensim.models.fasttext` instead.


Python wrapper around word representation learning from FastText, a library for efficient learning
of word representations and sentence classification [1].

This module allows training a word embedding from a training corpus with the additional ability
to obtain word vectors for out-of-vocabulary words, using the fastText C implementation.

The wrapped model can NOT be updated with new documents for online training -- use gensim's
`Word2Vec` for that.

Example:
.. sourcecode:: pycon

    >>> from gensim.models.wrappers import FastText
    >>> model = FastText.train('/Users/kofola/fastText/fasttext', corpus_file='text8')
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
from gensim.models.deprecated.keyedvectors import KeyedVectors, Vocab
from gensim.models.deprecated.word2vec import Word2Vec

logger = logging.getLogger(__name__)

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

FASTTEXT_FILEFORMAT_MAGIC = 793712314


class FastTextKeyedVectors(KeyedVectors):
    """
    Class to contain vectors, vocab and ngrams for the FastText training class and other methods not directly
    involved in training such as most_similar().
    Subclasses KeyedVectors to implement oov lookups, storing ngrams and other FastText specific methods

    """

    def __init__(self):
        super(FastTextKeyedVectors, self).__init__()
        self.syn0_vocab = None
        self.syn0_vocab_norm = None
        self.syn0_ngrams = None
        self.syn0_ngrams_norm = None
        self.ngrams = {}
        self.hash2index = {}
        self.ngrams_word = {}
        self.min_n = 0
        self.max_n = 0

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'syn0_vocab_norm', 'syn0_ngrams_norm'])
        super(FastTextKeyedVectors, self).save(*args, **kwargs)

    def word_vec(self, word, use_norm=False):
        """
        Accept a single word as input.
        Returns the word's representations in vector space, as a 1D numpy array.

        The word can be out-of-vocabulary as long as ngrams for the word are present.
        For words with all ngrams absent, a KeyError is raised.

        Example:

        .. sourcecode:: pycon

            >>> trained_model['office']
            array([ -1.40128313e-02, ...])

        """
        if word in self.vocab:
            return super(FastTextKeyedVectors, self).word_vec(word, use_norm)
        else:
            word_vec = np.zeros(self.syn0_ngrams.shape[1], dtype=np.float32)
            ngrams = compute_ngrams(word, self.min_n, self.max_n)
            ngrams = [ng for ng in ngrams if ng in self.ngrams]
            if use_norm:
                ngram_weights = self.syn0_ngrams_norm
            else:
                ngram_weights = self.syn0_ngrams
            for ngram in ngrams:
                word_vec += ngram_weights[self.ngrams[ngram]]
            if word_vec.any():
                return word_vec / len(ngrams)
            else:  # No ngrams of the word are present in self.ngrams
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
        if getattr(self, 'syn0_ngrams_norm', None) is None or replace:
            logger.info("precomputing L2-norms of ngram weight vectors")
            if replace:
                for i in range(self.syn0_ngrams.shape[0]):
                    self.syn0_ngrams[i, :] /= sqrt((self.syn0_ngrams[i, :] ** 2).sum(-1))
                self.syn0_ngrams_norm = self.syn0_ngrams
            else:
                self.syn0_ngrams_norm = \
                    (self.syn0_ngrams / sqrt((self.syn0_ngrams ** 2).sum(-1))[..., newaxis]).astype(REAL)

    def __contains__(self, word):
        """
        Check if `word` or any character ngrams in `word` are present in the vocabulary.
        A vector for the word is guaranteed to exist if `__contains__` returns True.
        """
        if word in self.vocab:
            return True
        else:
            char_ngrams = compute_ngrams(word, self.min_n, self.max_n)
            return any(ng in self.ngrams for ng in char_ngrams)

    @classmethod
    def load_word2vec_format(cls, *args, **kwargs):
        """Not suppported. Use gensim.models.KeyedVectors.load_word2vec_format instead."""
        raise NotImplementedError("Not supported. Use gensim.models.KeyedVectors.load_word2vec_format instead.")


class FastText(Word2Vec):
    """
    Class for word vector training using FastText. Communication between FastText and Python
    takes place by working with data files on disk and calling the FastText binary with
    subprocess.call().
    Implements functionality similar to [fasttext.py](https://github.com/salestock/fastText.py),
    improving speed and scope of functionality like `most_similar`, `similarity` by extracting vectors
    into numpy matrix.

    Warnings
    --------
    .. deprecated:: 3.2.0
       Use :class:`gensim.models.fasttext.FastText` instead of :class:`gensim.models.wrappers.fasttext.FastText`.


    """

    def initialize_word_vectors(self):
        self.wv = FastTextKeyedVectors()

    @classmethod
    def train(cls, ft_path, corpus_file, output_file=None, model='cbow', size=100, alpha=0.025, window=5, min_count=5,
              word_ngrams=1, loss='ns', sample=1e-3, negative=5, iter=5, min_n=3, max_n=6, sorted_vocab=1, threads=12):
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

        `word_ngram` = max length of word ngram

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
            'wordNgrams': word_ngrams,
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

        utils.check_output(args=cmd)
        model = cls.load_fasttext_format(output_file)
        cls.delete_training_files(output_file)
        return model

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'syn0_vocab_norm', 'syn0_ngrams_norm'])
        super(FastText, self).save(*args, **kwargs)

    @classmethod
    def load_fasttext_format(cls, model_file, encoding='utf8'):
        """
        Load the input-hidden weight matrix from the fast text output files.

        Note that due to limitations in the FastText API, you cannot continue training
        with a model loaded this way, though you can query for word similarity etc.

        `model_file` is the path to the FastText output files.
        FastText outputs two model files - `/path/to/model.vec` and `/path/to/model.bin`

        Expected value for this example: `/path/to/model` or `/path/to/model.bin`,
        as gensim requires only `.bin` file to load entire fastText model.

        """
        model = cls()
        if not model_file.endswith('.bin'):
            model_file += '.bin'
        model.file_name = model_file
        model.load_binary_data(encoding=encoding)
        return model

    @classmethod
    def load(cls, *args, **kwargs):
        model = super(FastText, cls).load(*args, **kwargs)
        if hasattr(model.wv, 'syn0_all'):
            setattr(model.wv, 'syn0_ngrams', model.wv.syn0_all)
            delattr(model.wv, 'syn0_all')
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

    def load_binary_data(self, encoding='utf8'):
        """Loads data from the output binary file created by FastText training"""

        # TODO use smart_open again when https://github.com/RaRe-Technologies/smart_open/issues/207 will be fixed
        with open(self.file_name, 'rb') as f:
            self.load_model_params(f)
            self.load_dict(f, encoding=encoding)
            self.load_vectors(f)

    def load_model_params(self, file_handle):
        magic, version = self.struct_unpack(file_handle, '@2i')
        if magic == FASTTEXT_FILEFORMAT_MAGIC:  # newer format
            self.new_format = True
            dim, ws, epoch, min_count, neg, _, loss, model, bucket, minn, maxn, _, t = \
                self.struct_unpack(file_handle, '@12i1d')
        else:  # older format
            self.new_format = False
            dim = magic
            ws = version
            epoch, min_count, neg, _, loss, model, bucket, minn, maxn, _, t = self.struct_unpack(file_handle, '@10i1d')
        # Parameters stored by [Args::save](https://github.com/facebookresearch/fastText/blob/master/src/args.cc)
        self.vector_size = dim
        self.window = ws
        self.iter = epoch
        self.min_count = min_count
        self.negative = neg
        self.hs = loss == 1
        self.sg = model == 2
        self.bucket = bucket
        self.wv.min_n = minn
        self.wv.max_n = maxn
        self.sample = t

    def load_dict(self, file_handle, encoding='utf8'):
        vocab_size, nwords, nlabels = self.struct_unpack(file_handle, '@3i')
        # Vocab stored by [Dictionary::save](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)
        if nlabels > 0:
            raise NotImplementedError("Supervised fastText models are not supported")
        logger.info("loading %s words for fastText model from %s", vocab_size, self.file_name)

        self.struct_unpack(file_handle, '@1q')  # number of tokens
        if self.new_format:
            pruneidx_size, = self.struct_unpack(file_handle, '@q')
        for i in range(vocab_size):
            word_bytes = b''
            char_byte = file_handle.read(1)
            # Read vocab word
            while char_byte != b'\x00':
                word_bytes += char_byte
                char_byte = file_handle.read(1)
            word = word_bytes.decode(encoding)
            count, _ = self.struct_unpack(file_handle, '@qb')

            self.wv.vocab[word] = Vocab(index=i, count=count)
            self.wv.index2word.append(word)

        assert len(self.wv.vocab) == nwords, (
            'mismatch between final vocab size ({} words), '
            'and expected number of words ({} words)'.format(len(self.wv.vocab), nwords))
        if len(self.wv.vocab) != vocab_size:
            # expecting to log this warning only for pretrained french vector, wiki.fr
            logger.warning(
                "mismatch between final vocab size (%s words), and expected vocab size (%s words)",
                len(self.wv.vocab), vocab_size
            )

        if self.new_format:
            for j in range(pruneidx_size):
                self.struct_unpack(file_handle, '@2i')

    def load_vectors(self, file_handle):
        if self.new_format:
            self.struct_unpack(file_handle, '@?')  # bool quant_input in fasttext.cc
        num_vectors, dim = self.struct_unpack(file_handle, '@2q')
        # Vectors stored by [Matrix::save](https://github.com/facebookresearch/fastText/blob/master/src/matrix.cc)
        assert self.vector_size == dim, (
            'mismatch between vector size in model params ({}) and model vectors ({})'
            .format(self.vector_size, dim)
        )
        float_size = struct.calcsize('@f')
        if float_size == 4:
            dtype = np.dtype(np.float32)
        elif float_size == 8:
            dtype = np.dtype(np.float64)

        self.num_original_vectors = num_vectors
        self.wv.syn0_ngrams = np.fromfile(file_handle, dtype=dtype, count=num_vectors * dim)
        self.wv.syn0_ngrams = self.wv.syn0_ngrams.reshape((num_vectors, dim))
        assert self.wv.syn0_ngrams.shape == (self.bucket + len(self.wv.vocab), self.vector_size), \
            'mismatch between actual weight matrix shape {} and expected shape {}'\
            .format(
                self.wv.syn0_ngrams.shape, (self.bucket + len(self.wv.vocab), self.vector_size)
            )

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
        self.wv.syn0 = np.zeros((len(self.wv.vocab), self.vector_size), dtype=REAL)

        for w, vocab in self.wv.vocab.items():
            all_ngrams += compute_ngrams(w, self.wv.min_n, self.wv.max_n)
            self.wv.syn0[vocab.index] += np.array(self.wv.syn0_ngrams[vocab.index])

        all_ngrams = set(all_ngrams)
        self.num_ngram_vectors = len(all_ngrams)
        ngram_indices = []
        for i, ngram in enumerate(all_ngrams):
            ngram_hash = ft_hash(ngram)
            ngram_indices.append(len(self.wv.vocab) + ngram_hash % self.bucket)
            self.wv.ngrams[ngram] = i
        self.wv.syn0_ngrams = self.wv.syn0_ngrams.take(ngram_indices, axis=0)

        ngram_weights = self.wv.syn0_ngrams

        logger.info(
            "loading weights for %s words for fastText model from %s",
            len(self.wv.vocab), self.file_name
        )

        for w, vocab in self.wv.vocab.items():
            word_ngrams = compute_ngrams(w, self.wv.min_n, self.wv.max_n)
            for word_ngram in word_ngrams:
                self.wv.syn0[vocab.index] += np.array(ngram_weights[self.wv.ngrams[word_ngram]])

            self.wv.syn0[vocab.index] /= (len(word_ngrams) + 1)
        logger.info(
            "loaded %s weight matrix for fastText model from %s",
            self.wv.syn0.shape, self.file_name
        )


def compute_ngrams(word, min_n, max_n):
    BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
    extended_word = BOW + word + EOW
    ngrams = []
    for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
        for i in range(0, len(extended_word) - ngram_length + 1):
            ngrams.append(extended_word[i:i + ngram_length])
    return ngrams


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
