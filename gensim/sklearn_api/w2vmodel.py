#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Chinmaya Pancholi <chinmayapancholi13@gmail.com>
# Copyright (C) 2017 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Scikit learn interface for :class:`~gensim.models.word2vec.Word2Vec`.

Follows scikit-learn API conventions to facilitate using gensim along with scikit-learn.


Examples
--------

>>> from gensim.test.utils import common_texts
>>> from gensim.sklearn_api import W2VTransformer
>>>
>>> # Create a model to represent each word by a 10 dimensional vector.
>>> model = W2VTransformer(size=10, min_count=1, seed=1)
>>>
>>> # What is the vector representation of the word 'graph'?
>>> wordvecs = model.fit(common_texts).transform(['graph', 'system'])
>>> assert wordvecs.shape == (2, 10)

"""


import numpy as np
import six
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models


class W2VTransformer(TransformerMixin, BaseEstimator):
    """Base Word2Vec module, wraps :class:`~gensim.models.word2vec.Word2Vec`.

    For more information on the inner workings please take a look at
    the original class.

    """
    def __init__(self, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=1e-3, seed=1,
                 workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=10000):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        Parameters
        ----------
        size : int
            Dimensionality of the feature vectors.
        alpha : float
            The initial learning rate.
        window : int
            The maximum distance between the current and predicted word within a sentence.
        min_count : int
            Ignores all words with total frequency lower than this.
        max_vocab_size : int
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        seed : int
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
            use of the `PYTHONHASHSEED` environment variable to control hash randomization).
        workers : int
            Use these many worker threads to train the model (=faster training with multicore machines).
        min_alpha : float
            Learning rate will linearly drop to `min_alpha` as training progresses.
        sg : int {1, 0}
            Defines the training algorithm. If 1, CBOW is used, otherwise, skip-gram is employed.
        hs : int {1,0}
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        negative : int
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        cbow_mean : int {1,0}
            If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        hashfxn : callable (object -> int), optional
            A hashing function. Used to create an initial random reproducible vector by hashing the random seed.
        iter : int
            Number of iterations (epochs) over the corpus.
        null_word : int {1, 0}
            If 1, a null pseudo-word will be created for padding when using concatenative L1 (run-of-words)
        trim_rule : function
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part
            of the model.
        sorted_vocab : int {1,0}
            If 1, sort the vocabulary by descending frequency before assigning word indexes.
        batch_words : int
            Target size (in words) for batches of examples passed to worker threads (and
            thus cython routines).(Larger batches will be passed if individual
            texts are longer than 10000 words, but the standard cython code truncates to that maximum.)

        """

        self.gensim_model = None
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : iterable of iterables of str
            The input corpus. X can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.

        Returns
        -------
        :class:`~gensim.sklearn_api.w2vmodel.W2VTransformer`
            The trained model.

        """
        self.gensim_model = models.Word2Vec(
            sentences=X, size=self.size, alpha=self.alpha,
            window=self.window, min_count=self.min_count, max_vocab_size=self.max_vocab_size,
            sample=self.sample, seed=self.seed, workers=self.workers, min_alpha=self.min_alpha,
            sg=self.sg, hs=self.hs, negative=self.negative, cbow_mean=self.cbow_mean,
            hashfxn=self.hashfxn, iter=self.iter, null_word=self.null_word, trim_rule=self.trim_rule,
            sorted_vocab=self.sorted_vocab, batch_words=self.batch_words
        )
        return self

    def transform(self, words):
        """Return the word vectors the input words.

        Parameters
        ----------
        words : iterable of str
            A collection of words to be transformed.

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

    def partial_fit(self, X):
        raise NotImplementedError(
            "'partial_fit' has not been implemented for W2VTransformer. "
            "However, the model can be updated with a fixed vocabulary using Gensim API call."
        )
