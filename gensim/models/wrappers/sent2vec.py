#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Sourav Singh <ssouravsingh12@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Python wrapper around sentence representation learning from sent2vec, a library for unsupervised 
learning of short texts or sentences[1].
This module allows training a word embedding from a training corpus with the additional ability
to obtain word vectors for out-of-vocabulary words, using the fastText C implementation.
The wrapped model can NOT be updated with new documents for online training -- use gensim's
`Word2Vec` for that.

Example:
>>> from gensim.models.wrappers import FastText
>>> model = fasttext.FastText.train('/Users/kofola/fastText/fasttext', corpus_file='text8')
>>> print model['forests']  # prints vector for given out-of-vocabulary word

.. [1] https://github.com/epfml/sent2vec

"""


import logging
import tempfile
import os
import struct

import numpy as np
from numpy import float32 as REAL, sqrt, newaxis
from gensim import utils
from gensim.models.keyedvectors import KeyedVectors, Vocab
from gensim.models.word2vec import Word2Vec

from six import string_types

logger = logging.getLogger(__name__)

class Sent2VecKeyedVectors(KeyedVectors):
    def __init__(self):
        super(Sent2VecKeyedVectors, self).__init__()
        self.syn0_all_norm = None
        self.ngrams = {}

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'syn0_all_norm'])
        super(Sent2VecKeyedVectors, self).save(*args, **kwargs)

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
            return super(Sent2VecKeyedVectors, self).word_vec(word, use_norm)
        else:
            word_vec = np.zeros(self.syn0_all.shape[1])
            ngrams = Sent2Vec.compute_ngrams(word, self.min_n, self.max_n)
            ngrams = [ng for ng in ngrams if ng in self.ngrams]
            if use_norm:
                ngram_weights = self.syn0_all_norm
            else:
                ngram_weights = self.syn0_all
            for ngram in ngrams:
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
            word_ngrams = set(Sent2Vec.compute_ngrams(word, self.min_n, self.max_n))
            if len(word_ngrams & set(self.ngrams.keys())):
                return True
            else:
                return False

class Sent2Vec(Word2Vec):
    """
    Class for word vector training using Sent2Vec. Communication between Sent2Vec and Python
    takes place by working with data files on disk and calling the Sent2Vec binary with
    subprocess.call().
    """

    def initialize_word_vectors(self):
        self.wv = Sent2VecKeyedVectors()

    @classmethod
    def train(cls, ft_path, corpus_file, output_file=None, model='cbow', size=100, alpha=0.025, window=5, min_count=5,
            word_ngrams=1, loss='ns', sample=1e-3, negative=5, iter=5, min_n=3, max_n=6, sorted_vocab=1, threads=12):
        """
        `ft_path` is the path to the Sent2Vec executable, e.g. `/home/kofola/sent2vec/sent2vec`.
        `corpus_file` is the filename of the text file to be used for training the Sent2Vec model.
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

        output = utils.check_output(args=cmd)
        model = cls.load_fasttext_format(output_file)
        cls.delete_training_files(output_file)
        return model

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'syn0_all_norm'])
        super(Sent2Vec, self).save(*args, **kwargs)
