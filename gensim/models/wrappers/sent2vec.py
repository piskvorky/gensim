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

import numpy as np
from numpy import zeros, ones, vstack, sum as np_sum, empty, float32 as REAL

from gensim.models.word2vec import Word2Vec, train_sg_pair, train_cbow_pair
from gensim.models.wrappers.fasttext import FastTextKeyedVectors
from gensim.models.wrappers.fasttext import FastText as Ft_Wrapper, compute_ngrams, ft_hash

logger = logging.getLogger(__name__)


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
