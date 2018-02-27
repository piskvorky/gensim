#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Scikit learn interface for :class:`~gensim.models.doc2vec.Doc2Vec`.

Follows scikit-learn API conventions to facilitate using gensim along with scikit-learn.

Examples
--------

>>> from gensim.test.utils import common_texts
>>> from gensim.sklearn_api import D2VTransformer
>>> from gensim.similarities import Similarity
>>>
>>> # Lets represent each document using a 50 dimensional vector
>>> model = D2VTransformer(min_count=1, size=5)
>>> docvecs = model.fit_transform(common_texts)
>>>
>>> # Let's use the vector representations to compute similarities with one of the documents.
>>> index = Similarity(None, docvecs, num_features=5)
>>>
>>> # Which document is most similar to the last one in the corpus? Probably itself!
>>> result = index[docvecs[8]]
>>> result.argmax()
8

"""

import numpy as np
from six import string_types
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim import models
from gensim.models import doc2vec


class D2VTransformer(TransformerMixin, BaseEstimator):
    """Base Dov2Vec module.

    Wraps :class:`~gensim.models.doc2vec.Doc2Vec`.
    For more information on the inner workings please take a look at
    the original class.

    """

    def __init__(self, dm_mean=None, dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1, docvecs=None,
                 docvecs_mapfile=None, comment=None, trim_rule=None, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001, hs=0, negative=5, cbow_mean=1,
                 hashfxn=hash, iter=5, sorted_vocab=1, batch_words=10000):
        """Sklearn api for Doc2Vec model.

        Parameters
        ----------

        dm_mean : int {1,0}, optional
            If 0, use the sum of the context word vectors. If 1, use the mean.
            Only applies when `dm` is used in non-concatenative mode.
        dm : int {1,0}, optional
            Defines the training algorithm. If `dm=1`, 'distributed memory' (PV-DM) is used.
            Otherwise, `distributed bag of words` (PV-DBOW) is employed.
        dbow_words : int {1,0}, optional
            If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW
            doc-vector training; If 0, only trains doc-vectors (faster).
        dm_concat : int {1,0}, optional
            If 1, use concatenation of context vectors rather than sum/average;
            Note concatenation results in a much-larger model, as the input
            is no longer the size of one (sampled or arithmetically combined) word vector, but the
            size of the tag(s) and all words in the context strung together.
        dm_tag_count : int, optional
            Expected constant number of document tags per document, when using
            dm_concat mode; default is 1.
        docvecs : :class:`~gensim.models.keyedvectors.Doc2VecKeyedVectors`
            A mapping from a string or int tag to its vector representation.
            Either this or `docvecs_mapfile` **MUST** be supplied.
        docvecs_mapfile : str, optional
            Path to a file containing the docvecs mapping.
            If `docvecs` is None, this file will be used to create it.
        comment : str, optional
            A model descriptive comment, used for logging and debugging purposes.
        trim_rule : callable ((str, int, int) -> int), optional
            Vocabulary trimming rule that accepts (word, count, min_count).
            Specifies whether certain words should remain in the vocabulary (:attr:`gensim.utils.RULE_KEEP`),
            be trimmed away (:attr:`gensim.utils.RULE_DISCARD`), or handled using the default
            (:attr:`gensim.utils.RULE_DEFAULT`).If None, then :func:`~gensim.utils.keep_vocab_item` will be used.
            Note: The rule, if given, is only used to prune vocabulary during build_vocab()
            and is not stored as part of the model.
        size : int, optional
            Dimensionality of the feature vectors.
        alpha : float, optional
            The initial learning rate.
        window : int, optional
            The maximum distance between the current and predicted word within a sentence.
        min_count : int, optional
            Ignores all words with total frequency lower than this.
        max_vocab_size : int, optional
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
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
            Use this many worker threads to train the model. Will yield a speedup when training with multicore machines.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        hs : int {1,0}, optional
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
        cbow_mean : int, optional
            Same as `dm_mean`, unused.
        hashfxn : callable (object -> int), optional
            A hashing function. Used to create an initial random reproducible vector by hashing the random seed.
        iter : int, optional
            Number of epochs to iterate through the corpus.
        sorted_vocab : bool, optional
            Whether the vocabulary should be sorted internally.
        batch_words : int, optional
            Number of words to be handled by each job.

        """
        self.gensim_model = None
        self.dm_mean = dm_mean
        self.dm = dm
        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        self.docvecs = docvecs
        self.docvecs_mapfile = docvecs_mapfile
        self.comment = comment
        self.trim_rule = trim_rule

        # attributes associated with gensim.models.Word2Vec
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.iter = iter
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {iterable of {:class:`~gensim.models.doc2vec.TaggedDocument`, iterable of iterable of str}
            A collection of tagged documents used for training the model.
            If these are not tagged, their order integer index will be used to tag them.

        Returns
        -------
        :class:`~gensim.sklearn_api.d2vmodel.D2VTransformer`
            The trained model.

        """
        if isinstance(X[0], doc2vec.TaggedDocument):
            d2v_sentences = X
        else:
            d2v_sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(X)]
        self.gensim_model = models.Doc2Vec(
            documents=d2v_sentences, dm_mean=self.dm_mean, dm=self.dm,
            dbow_words=self.dbow_words, dm_concat=self.dm_concat, dm_tag_count=self.dm_tag_count,
            docvecs=self.docvecs, docvecs_mapfile=self.docvecs_mapfile, comment=self.comment,
            trim_rule=self.trim_rule, size=self.size, alpha=self.alpha, window=self.window,
            min_count=self.min_count, max_vocab_size=self.max_vocab_size, sample=self.sample,
            seed=self.seed, workers=self.workers, min_alpha=self.min_alpha, hs=self.hs,
            negative=self.negative, cbow_mean=self.cbow_mean, hashfxn=self.hashfxn,
            iter=self.iter, sorted_vocab=self.sorted_vocab, batch_words=self.batch_words
        )
        return self

    def transform(self, docs):
        """Get the vector representations for the input documents.

        Parameters
        ----------
        docs : iterable of iterable of str
            The input corpus.

        Returns
        -------
        np.array of shape (`len(docs)`, `size`)
            The vector representation of the input corpus.

        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        if isinstance(docs[0], string_types):
            docs = [docs]
        vectors = [self.gensim_model.infer_vector(doc) for doc in docs]
        return np.reshape(np.array(vectors), (len(docs), self.gensim_model.vector_size))
