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
   Use :class:`gensim.models.fasttext.FastText` instead of :class:`gensim.models.wrappers.fasttext.FastText`.



Python wrapper around word representation learning from FastText, a library for efficient learning
of word representations and sentence classification [1].

This module allows training a word embedding from a training corpus with the additional ability
to obtain word vectors for out-of-vocabulary words, using the fastText C implementation.

The wrapped model can NOT be updated with new documents for online training -- use gensim's
`Word2Vec` for that.

Example:

>>> from gensim.models.wrappers import FastText
>>> model = FastText.train('/Users/kofola/fastText/fasttext', corpus_file='text8')
>>> print model['forests']  # prints vector for given out-of-vocabulary word

.. [1] https://github.com/facebookresearch/fastText#enriching-word-vectors-with-subword-information



"""
from gensim.models.deprecated.fasttext_wrapper import FastText, FastTextKeyedVectors,\
    ft_hash, compute_ngrams  # noqa:F401
