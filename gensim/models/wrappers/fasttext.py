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

import numpy

from gensim import utils
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec

logger = logging.getLogger(__name__)


class FastText(Word2Vec):
    """
    Class for word vector training using FastText. Communication between FastText and Python
    takes place by working with data files on disk and calling the FastText binary with
    subprocess.call().

    """
    @classmethod
    def train(cls, ft_path, corpus_file, model='cbow', size=100, alpha=0.025, window=5, min_count=5,
            loss='ns', sample=1e-3, negative=5, iter=5, min_n=1, max_n=6, sorted_vocab=1, threads=12):
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
        model_file = os.path.join(tempfile.gettempdir(), 'ft_model')
        ft_args = {
            'input': corpus_file,
            'output': model_file,
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
        model = cls.load_word2vec_format('%s.vec' % model_file)

        return model