#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors: M.Cemil Guney <mcemilguneyy@gmail.com>
# Copyright (C) 2018 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Scikit-learn interface for :class:`~gensim.models.fasttext.FastText`.

Follows scikit-learn API conventions to facilitate using gensim along with scikit-learn.


Examples
--------
>>> from gensim.test.utils import common_texts
>>> from gensim.sklearn_api import FTTransformer
>>>
>>> # Create a model to represent each word by a 10 dimensional vector.
>>> model = FTTransformer(size=10, min_count=1, seed=1)
>>>
>>> # What is the vector representations of the word 'graph' and 'system'?
>>> wordvecs = model.fit(common_texts).transform(['graph', 'system'])
>>> assert wordvecs.shape == (2, 10)

Retrieve word-vector for vocab and out-of-vocab word:

>>> existent_word = "system"
>>> existent_word in model.gensim_model.wv.vocab
True
>>> existent_word_vec = model.transform(existent_word) # numpy vector of a word
>>> assert existent_word_vec.shape == (1, 10)
>>>
>>> oov_word = "sys"
>>> oov_word in model.gensim_model.wv.vocab
False
>>> oov_word_vec = model.transform(oov_word) # numpy vector of a word
>>> assert oov_word_vec.shape == (1, 10)

"""
import numpy as np
import six
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models


class FTTransformer(TransformerMixin, BaseEstimator):
    """Base FastText module, wraps :class:`~gensim.models.fasttext.FastText`.

    For more information please have a look to `Enriching Word Vectors with Subword
    Information <https://arxiv.org/abs/1607.04606>`_.

    """
    def __init__(self, sg=0, hs=0, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, word_ngrams=1, sample=1e-3, seed=1,
                 workers=3, min_alpha=0.0001, negative=5, ns_exponent=0.75,
                 cbow_mean=1, hashfxn=hash, iter=5, null_word=0, min_n=3,
                 max_n=6, sorted_vocab=1, bucket=2000000, trim_rule=None,
                 batch_words=10000):
        """

        Parameters
        ----------
        sg : {1, 0}, optional
            Training algorithm: skip-gram if `sg=1`, otherwise CBOW.
        hs : {1,0}, optional
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        size : int, optional
            Dimensionality of the word vectors.
        alpha : float, optional
            The initial learning rate.
        window : int, optional
            The maximum distance between the current and predicted word within a sentence.
        min_count : int, optional
            The model ignores all words with total frequency lower than this.
        max_vocab_size : int, optional
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        word_ngrams : {1,0}, optional
            If 1, uses enriches word vectors with subword(n-grams) information.
            If 0, this is equivalent to :class:`~gensim.models.word2vec.Word2Vec`.
        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        seed : int, optional
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
            use of the `PYTHONHASHSEED` environment variable to control hash randomization).
        workers : int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        ns_exponent : float, optional
            The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in proportion
            to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more
            than high-frequency words. The popular default value of 0.75 was chosen by the original Word2Vec paper.
            More recently, in https://arxiv.org/abs/1804.04212, Caselles-Dupré, Lesaint, & Royo-Letelier suggest that
            other values may perform better for recommendation applications.
        cbow_mean : {1,0}, optional
            If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        hashfxn : function, optional
            Hash function to use to randomly initialize weights, for increased training reproducibility.
        iter : int, optional
            Number of iterations (epochs) over the corpus.
        min_n : int, optional
            Minimum length of char n-grams to be used for training word representations.
        max_n : int, optional
            Max length of char ngrams to be used for training word representations. Set `max_n` to be
            lesser than `min_n` to avoid char ngrams being used.
        sorted_vocab : {1,0}, optional
            If 1, sort the vocabulary by descending frequency before assigning word indices.
        bucket : int, optional
            Character ngrams are hashed into a fixed number of buckets, in order to limit the
            memory usage of the model. This option specifies the number of buckets used by the model.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during
            :meth:`~gensim.models.fasttext.FastText.build_vocab` and is not stored as part of themodel.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        batch_words : int, optional
            Target size (in words) for batches of examples passed to worker threads (and
            thus cython routines).(Larger batches will be passed if individual
            texts are longer than 10000 words, but the standard cython code truncates to that maximum.)

        """
        self.gensim_model = None
        self.sg = sg
        self.hs = hs
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.word_ngrams = word_ngrams
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.min_n = min_n
        self.max_n = max_n
        self.sorted_vocab = sorted_vocab
        self.bucket = bucket
        self.trim_rule = trim_rule
        self.batch_words = batch_words

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : iterable of iterables of str
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.

        Returns
        -------
        :class:`~gensim.sklearn_api.ftmodel.FTTransformer`
            The trained model.

        """
        self.gensim_model = models.FastText(
                sentences=X, sg=self.sg, hs=self.hs, size=self.size,
                alpha=self.alpha, window=self.window, min_count=self.min_count,
                max_vocab_size=self.max_vocab_size, word_ngrams=self.word_ngrams,
                sample=self.sample, seed=self.seed, workers=self.workers,
                min_alpha=self.min_alpha, negative=self.negative,
                ns_exponent=self.ns_exponent, cbow_mean=self.cbow_mean,
                hashfxn=self.hashfxn, iter=self.iter, null_word=self.null_word,
                min_n=self.min_n, max_n=self.max_n, sorted_vocab=self.sorted_vocab,
                bucket=self.bucket, trim_rule=self.trim_rule,
                batch_words=self.batch_words
        )
        return self

    def transform(self, words):
        """Get the word vectors the input words.

        Parameters
        ----------
        words : {iterable of str, str}
            Word or a collection of words to be transformed.

        Returns
        -------
        np.ndarray of shape [`len(words)`, `size`]
            A 2D array where each row is the vector of one word.

        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        if isinstance(words, six.string_types):
            words = [words]
        vectors = [self.gensim_model[word] for word in words]
        return np.reshape(np.array(vectors), (len(words), self.size))
