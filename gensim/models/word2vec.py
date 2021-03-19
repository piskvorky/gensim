#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Gensim Contributors
# Copyright (C) 2018 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Introduction
============

This module implements the word2vec family of algorithms, using highly optimized C routines,
data streaming and Pythonic interfaces.

The word2vec algorithms include skip-gram and CBOW models, using either
hierarchical softmax or negative sampling: `Tomas Mikolov et al: Efficient Estimation of Word Representations
in Vector Space <https://arxiv.org/pdf/1301.3781.pdf>`_, `Tomas Mikolov et al: Distributed Representations of Words
and Phrases and their Compositionality <https://arxiv.org/abs/1310.4546>`_.

Other embeddings
================

There are more ways to train word vectors in Gensim than just Word2Vec.
See also :class:`~gensim.models.doc2vec.Doc2Vec`, :class:`~gensim.models.fasttext.FastText`.

The training algorithms were originally ported from the C package https://code.google.com/p/word2vec/
and extended with additional functionality and
`optimizations <https://rare-technologies.com/parallelizing-word2vec-in-python/>`_ over the years.

For a tutorial on Gensim word2vec, with an interactive web app trained on GoogleNews,
visit https://rare-technologies.com/word2vec-tutorial/.

Usage examples
==============

Initialize a model with e.g.:

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_texts
    >>> from gensim.models import Word2Vec
    >>>
    >>> model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
    >>> model.save("word2vec.model")


**The training is streamed, so ``sentences`` can be an iterable**, reading input data
from the disk or network on-the-fly, without loading your entire corpus into RAM.

Note the ``sentences`` iterable must be *restartable* (not just a generator), to allow the algorithm
to stream over your dataset multiple times. For some examples of streamed iterables,
see :class:`~gensim.models.word2vec.BrownCorpus`,
:class:`~gensim.models.word2vec.Text8Corpus` or :class:`~gensim.models.word2vec.LineSentence`.

If you save the model you can continue training it later:

.. sourcecode:: pycon

    >>> model = Word2Vec.load("word2vec.model")
    >>> model.train([["hello", "world"]], total_examples=1, epochs=1)
    (0, 2)

The trained word vectors are stored in a :class:`~gensim.models.keyedvectors.KeyedVectors` instance, as `model.wv`:

.. sourcecode:: pycon

    >>> vector = model.wv['computer']  # get numpy vector of a word
    >>> sims = model.wv.most_similar('computer', topn=10)  # get other similar words

The reason for separating the trained vectors into `KeyedVectors` is that if you don't
need the full model state any more (don't need to continue training), its state can discarded,
keeping just the vectors and their keys proper.

This results in a much smaller and faster object that can be mmapped for lightning
fast loading and sharing the vectors in RAM between processes:

.. sourcecode:: pycon

    >>> from gensim.models import KeyedVectors
    >>>
    >>> # Store just the words + their trained embeddings.
    >>> word_vectors = model.wv
    >>> word_vectors.save("word2vec.wordvectors")
    >>>
    >>> # Load back with memory-mapping = read-only, shared across processes.
    >>> wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
    >>>
    >>> vector = wv['computer']  # Get numpy vector of a word

Gensim can also load word vectors in the "word2vec C format", as a
:class:`~gensim.models.keyedvectors.KeyedVectors` instance:

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> # Load a word2vec model stored in the C *text* format.
    >>> wv_from_text = KeyedVectors.load_word2vec_format(datapath('word2vec_pre_kv_c'), binary=False)
    >>> # Load a word2vec model stored in the C *binary* format.
    >>> wv_from_bin = KeyedVectors.load_word2vec_format(datapath("euclidean_vectors.bin"), binary=True)

It is impossible to continue training the vectors loaded from the C format because the hidden weights,
vocabulary frequencies and the binary tree are missing. To continue training, you'll need the
full :class:`~gensim.models.word2vec.Word2Vec` object state, as stored by :meth:`~gensim.models.word2vec.Word2Vec.save`,
not just the :class:`~gensim.models.keyedvectors.KeyedVectors`.

You can perform various NLP tasks with a trained model. Some of the operations
are already built-in - see :mod:`gensim.models.keyedvectors`.

If you're finished training a model (i.e. no more updates, only querying),
you can switch to the :class:`~gensim.models.keyedvectors.KeyedVectors` instance:

.. sourcecode:: pycon

    >>> word_vectors = model.wv
    >>> del model

to trim unneeded model state = use much less RAM and allow fast loading and memory sharing (mmap).

Embeddings with multiword ngrams
================================

There is a :mod:`gensim.models.phrases` module which lets you automatically
detect phrases longer than one word, using collocation statistics.
Using phrases, you can learn a word2vec model where "words" are actually multiword expressions,
such as `new_york_times` or `financial_crisis`:

.. sourcecode:: pycon

    >>> from gensim.models import Phrases
    >>>
    >>> # Train a bigram detector.
    >>> bigram_transformer = Phrases(common_texts)
    >>>
    >>> # Apply the trained MWE detector to a corpus, using the result to train a Word2vec model.
    >>> model = Word2Vec(bigram_transformer[common_texts], min_count=1)

Pretrained models
=================

Gensim comes with several already pre-trained models, in the
`Gensim-data repository <https://github.com/RaRe-Technologies/gensim-data>`_:

.. sourcecode:: pycon

    >>> import gensim.downloader
    >>> # Show all available models in gensim-data
    >>> print(list(gensim.downloader.info()['models'].keys()))
    ['fasttext-wiki-news-subwords-300',
     'conceptnet-numberbatch-17-06-300',
     'word2vec-ruscorpora-300',
     'word2vec-google-news-300',
     'glove-wiki-gigaword-50',
     'glove-wiki-gigaword-100',
     'glove-wiki-gigaword-200',
     'glove-wiki-gigaword-300',
     'glove-twitter-25',
     'glove-twitter-50',
     'glove-twitter-100',
     'glove-twitter-200',
     '__testing_word2vec-matrix-synopsis']
    >>>
    >>> # Download the "glove-twitter-25" embeddings
    >>> glove_vectors = gensim.downloader.load('glove-twitter-25')
    >>>
    >>> # Use the downloaded vectors as usual:
    >>> glove_vectors.most_similar('twitter')
    [('facebook', 0.948005199432373),
     ('tweet', 0.9403423070907593),
     ('fb', 0.9342358708381653),
     ('instagram', 0.9104824066162109),
     ('chat', 0.8964964747428894),
     ('hashtag', 0.8885937333106995),
     ('tweets', 0.8878158330917358),
     ('tl', 0.8778461217880249),
     ('link', 0.8778210878372192),
     ('internet', 0.8753897547721863)]

"""

from __future__ import division  # py3 "true division"

import logging
import sys
import os
import heapq
from timeit import default_timer
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from types import GeneratorType
import threading
import itertools
import copy
from queue import Queue, Empty

from numpy import float32 as REAL
import numpy as np

from gensim.utils import keep_vocab_item, call_on_class_only, deprecated
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector
from gensim import utils, matutils


logger = logging.getLogger(__name__)

try:
    from gensim.models.word2vec_inner import (  # noqa: F401
        train_batch_sg,
        train_batch_cbow,
        score_sentence_sg,
        score_sentence_cbow,
        MAX_WORDS_IN_BATCH,
        FAST_VERSION,
    )
except ImportError:
    raise utils.NO_CYTHON

try:
    from gensim.models.word2vec_corpusfile import train_epoch_sg, train_epoch_cbow, CORPUSFILE_VERSION
except ImportError:
    # file-based word2vec is not supported
    CORPUSFILE_VERSION = -1

    def train_epoch_sg(
            model, corpus_file, offset, _cython_vocab, _cur_epoch, _expected_examples, _expected_words,
            _work, _neu1, compute_loss,
        ):
        raise RuntimeError("Training with corpus_file argument is not supported")

    def train_epoch_cbow(
            model, corpus_file, offset, _cython_vocab, _cur_epoch, _expected_examples, _expected_words,
            _work, _neu1, compute_loss,
        ):
        raise RuntimeError("Training with corpus_file argument is not supported")


class Word2Vec(utils.SaveLoad):
    def __init__(
            self, sentences=None, corpus_file=None, vector_size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, epochs=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=(),
            comment=None, max_final_vocab=None,
        ):
        """Train, use and evaluate neural networks described in https://code.google.com/p/word2vec/.

        Once you're finished training a model (=no more updates, only querying)
        store and use only the :class:`~gensim.models.keyedvectors.KeyedVectors` instance in ``self.wv``
        to reduce memory.

        The full model can be stored/loaded via its :meth:`~gensim.models.word2vec.Word2Vec.save` and
        :meth:`~gensim.models.word2vec.Word2Vec.load` methods.

        The trained word vectors can also be stored/loaded from a format compatible with the
        original word2vec implementation via `self.wv.save_word2vec_format`
        and :meth:`gensim.models.keyedvectors.KeyedVectors.load_word2vec_format`.

        Parameters
        ----------
        sentences : iterable of iterables, optional
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
            See also the `tutorial on data streaming in Python
            <https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/>`_.
            If you don't supply `sentences`, the model is left uninitialized -- use if you plan to initialize it
            in some other way.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `sentences` to get performance boost. Only one of `sentences` or
            `corpus_file` arguments need to be passed (or none of them, in that case, the model is left uninitialized).
        vector_size : int, optional
            Dimensionality of the word vectors.
        window : int, optional
            Maximum distance between the current and predicted word within a sentence.
        min_count : int, optional
            Ignores all words with total frequency lower than this.
        workers : int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).
        sg : {0, 1}, optional
            Training algorithm: 1 for skip-gram; otherwise CBOW.
        hs : {0, 1}, optional
            If 1, hierarchical softmax will be used for model training.
            If 0, and `negative` is non-zero, negative sampling will be used.
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
        cbow_mean : {0, 1}, optional
            If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        alpha : float, optional
            The initial learning rate.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        seed : int, optional
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
            use of the `PYTHONHASHSEED` environment variable to control hash randomization).
        max_vocab_size : int, optional
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        max_final_vocab : int, optional
            Limits the vocab to a target vocab size by automatically picking a matching min_count. If the specified
            min_count is more than the calculated min_count, the specified min_count will be used.
            Set to `None` if not required.
        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        hashfxn : function, optional
            Hash function to use to randomly initialize weights, for increased training reproducibility.
        epochs : int, optional
            Number of iterations (epochs) over the corpus. (Formerly: `iter`)
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part of the
            model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.
        sorted_vocab : {0, 1}, optional
            If 1, sort the vocabulary by descending frequency before assigning word indexes.
            See :meth:`~gensim.models.keyedvectors.KeyedVectors.sort_by_descending_frequency()`.
        batch_words : int, optional
            Target size (in words) for batches of examples passed to worker threads (and
            thus cython routines).(Larger batches will be passed if individual
            texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
        compute_loss: bool, optional
            If True, computes and stores loss value which can be retrieved using
            :meth:`~gensim.models.word2vec.Word2Vec.get_latest_training_loss`.
        callbacks : iterable of :class:`~gensim.models.callbacks.CallbackAny2Vec`, optional
            Sequence of callbacks to be executed at specific stages during training.

        Examples
        --------
        Initialize and train a :class:`~gensim.models.word2vec.Word2Vec` model

        .. sourcecode:: pycon

            >>> from gensim.models import Word2Vec
            >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
            >>> model = Word2Vec(sentences, min_count=1)

        Attributes
        ----------
        wv : :class:`~gensim.models.keyedvectors.KeyedVectors`
            This object essentially contains the mapping between words and embeddings. After training, it can be used
            directly to query those embeddings in various ways. See the module level docstring for examples.

        """
        corpus_iterable = sentences

        self.vector_size = int(vector_size)
        self.workers = int(workers)
        self.epochs = epochs
        self.train_count = 0
        self.total_train_time = 0
        self.batch_words = batch_words

        self.sg = int(sg)
        self.alpha = float(alpha)
        self.min_alpha = float(min_alpha)

        self.window = int(window)
        self.random = np.random.RandomState(seed)

        self.hs = int(hs)
        self.negative = int(negative)
        self.ns_exponent = ns_exponent
        self.cbow_mean = int(cbow_mean)
        self.compute_loss = bool(compute_loss)
        self.running_training_loss = 0
        self.min_alpha_yet_reached = float(alpha)
        self.corpus_count = 0
        self.corpus_total_words = 0

        self.max_final_vocab = max_final_vocab
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.sample = sample
        self.sorted_vocab = sorted_vocab
        self.null_word = null_word
        self.cum_table = None  # for negative sampling
        self.raw_vocab = None

        if not hasattr(self, 'wv'):  # set unless subclass already set (eg: FastText)
            self.wv = KeyedVectors(vector_size)
        # EXPERIMENTAL lockf feature; create minimal no-op lockf arrays (1 element of 1.0)
        # advanced users should directly resize/adjust as desired after any vocab growth
        self.wv.vectors_lockf = np.ones(1, dtype=REAL)  # 0.0 values suppress word-backprop-updates; 1.0 allows

        self.hashfxn = hashfxn
        self.seed = seed
        if not hasattr(self, 'layer1_size'):  # set unless subclass already set (as for Doc2Vec dm_concat mode)
            self.layer1_size = vector_size

        self.comment = comment

        self.load = call_on_class_only

        if corpus_iterable is not None or corpus_file is not None:
            self._check_corpus_sanity(corpus_iterable=corpus_iterable, corpus_file=corpus_file, passes=(epochs + 1))
            self.build_vocab(corpus_iterable=corpus_iterable, corpus_file=corpus_file, trim_rule=trim_rule)
            self.train(
                corpus_iterable=corpus_iterable, corpus_file=corpus_file, total_examples=self.corpus_count,
                total_words=self.corpus_total_words, epochs=self.epochs, start_alpha=self.alpha,
                end_alpha=self.min_alpha, compute_loss=self.compute_loss, callbacks=callbacks)
        else:
            if trim_rule is not None:
                logger.warning(
                    "The rule, if given, is only used to prune vocabulary during build_vocab() "
                    "and is not stored as part of the model. Model initialized without sentences. "
                    "trim_rule provided, if any, will be ignored.")
            if callbacks:
                logger.warning(
                    "Callbacks are no longer retained by the model, so must be provided whenever "
                    "training is triggered, as in initialization with a corpus or calling `train()`. "
                    "The callbacks provided in this initialization without triggering train will "
                    "be ignored.")

        self.add_lifecycle_event("created", params=str(self))

    def build_vocab(
            self, corpus_iterable=None, corpus_file=None, update=False, progress_per=10000,
            keep_raw_vocab=False, trim_rule=None, **kwargs,
    ):
        """Build vocabulary from a sequence of sentences (can be a once-only generator stream).

        Parameters
        ----------
        corpus_iterable : iterable of list of str
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` module for such examples.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `sentences` to get performance boost. Only one of `sentences` or
            `corpus_file` arguments need to be passed (not both of them).
        update : bool
            If true, the new words in `sentences` will be added to model's vocab.
        progress_per : int, optional
            Indicates how many words to process before showing/updating the progress.
        keep_raw_vocab : bool, optional
            If False, the raw vocabulary will be deleted after the scaling is done to free up RAM.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during current method call and is not stored as part
            of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        **kwargs : object
            Keyword arguments propagated to `self.prepare_vocab`.

        """
        self._check_corpus_sanity(corpus_iterable=corpus_iterable, corpus_file=corpus_file, passes=1)
        total_words, corpus_count = self.scan_vocab(
            corpus_iterable=corpus_iterable, corpus_file=corpus_file, progress_per=progress_per, trim_rule=trim_rule)
        self.corpus_count = corpus_count
        self.corpus_total_words = total_words
        report_values = self.prepare_vocab(update=update, keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, **kwargs)
        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])
        self.prepare_weights(update=update)
        self.add_lifecycle_event("build_vocab", update=update, trim_rule=str(trim_rule))

    def build_vocab_from_freq(
            self, word_freq, keep_raw_vocab=False, corpus_count=None, trim_rule=None, update=False,
        ):
        """Build vocabulary from a dictionary of word frequencies.

        Parameters
        ----------
        word_freq : dict of (str, int)
            A mapping from a word in the vocabulary to its frequency count.
        keep_raw_vocab : bool, optional
            If False, delete the raw vocabulary after the scaling is done to free up RAM.
        corpus_count : int, optional
            Even if no corpus is provided, this argument can set corpus_count explicitly.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during current method call and is not stored as part
            of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        update : bool, optional
            If true, the new provided words in `word_freq` dict will be added to model's vocab.

        """
        logger.info("Processing provided word frequencies")
        # Instead of scanning text, this will assign provided word frequencies dictionary(word_freq)
        # to be directly the raw vocab
        raw_vocab = word_freq
        logger.info(
            "collected %i different raw word, with total frequency of %i",
            len(raw_vocab), sum(raw_vocab.values()),
        )

        # Since no sentences are provided, this is to control the corpus_count.
        self.corpus_count = corpus_count or 0
        self.raw_vocab = raw_vocab

        # trim by min_count & precalculate downsampling
        report_values = self.prepare_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, update=update)
        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])
        self.prepare_weights(update=update)  # build tables & arrays

    def _scan_vocab(self, sentences, progress_per, trim_rule):
        sentence_no = -1
        total_words = 0
        min_reduce = 1
        vocab = defaultdict(int)
        checked_string_types = 0
        for sentence_no, sentence in enumerate(sentences):
            if not checked_string_types:
                if isinstance(sentence, str):
                    logger.warning(
                        "Each 'sentences' item should be a list of words (usually unicode strings). "
                        "First item here is instead plain %s.",
                        type(sentence),
                    )
                checked_string_types += 1
            if sentence_no % progress_per == 0:
                logger.info(
                    "PROGRESS: at sentence #%i, processed %i words, keeping %i word types",
                    sentence_no, total_words, len(vocab)
                )
            for word in sentence:
                vocab[word] += 1
            total_words += len(sentence)

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        corpus_count = sentence_no + 1
        self.raw_vocab = vocab
        return total_words, corpus_count

    def scan_vocab(self, corpus_iterable=None, corpus_file=None, progress_per=10000, workers=None, trim_rule=None):
        logger.info("collecting all words and their counts")
        if corpus_file:
            corpus_iterable = LineSentence(corpus_file)

        total_words, corpus_count = self._scan_vocab(corpus_iterable, progress_per, trim_rule)

        logger.info(
            "collected %i word types from a corpus of %i raw words and %i sentences",
            len(self.raw_vocab), total_words, corpus_count
        )

        return total_words, corpus_count

    def prepare_vocab(
            self, update=False, keep_raw_vocab=False, trim_rule=None,
            min_count=None, sample=None, dry_run=False,
        ):
        """Apply vocabulary settings for `min_count` (discarding less-frequent words)
        and `sample` (controlling the downsampling of more-frequent words).

        Calling with `dry_run=True` will only simulate the provided settings and
        report the size of the retained vocabulary, effective corpus length, and
        estimated memory requirements. Results are both printed via logging and
        returned as a dict.

        Delete the raw vocabulary after the scaling is done to free up RAM,
        unless `keep_raw_vocab` is set.

        """
        min_count = min_count or self.min_count
        sample = sample or self.sample
        drop_total = drop_unique = 0

        # set effective_min_count to min_count in case max_final_vocab isn't set
        self.effective_min_count = min_count

        # if max_final_vocab is specified instead of min_count
        # pick a min_count which satisfies max_final_vocab as well as possible
        if self.max_final_vocab is not None:
            sorted_vocab = sorted(self.raw_vocab.keys(), key=lambda word: self.raw_vocab[word], reverse=True)
            calc_min_count = 1

            if self.max_final_vocab < len(sorted_vocab):
                calc_min_count = self.raw_vocab[sorted_vocab[self.max_final_vocab]] + 1

            self.effective_min_count = max(calc_min_count, min_count)
            self.add_lifecycle_event(
                "prepare_vocab",
                msg=(
                    f"max_final_vocab={self.max_final_vocab} and min_count={min_count} resulted "
                    f"in calc_min_count={calc_min_count}, effective_min_count={self.effective_min_count}"
                )
            )

        if not update:
            logger.info("Creating a fresh vocabulary")
            retain_total, retain_words = 0, []
            # Discard words less-frequent than min_count
            if not dry_run:
                self.wv.index_to_key = []
                # make stored settings match these applied settings
                self.min_count = min_count
                self.sample = sample
                self.wv.key_to_index = {}

            for word, v in self.raw_vocab.items():
                if keep_vocab_item(word, v, self.effective_min_count, trim_rule=trim_rule):
                    retain_words.append(word)
                    retain_total += v
                    if not dry_run:
                        self.wv.key_to_index[word] = len(self.wv.index_to_key)
                        self.wv.index_to_key.append(word)
                else:
                    drop_unique += 1
                    drop_total += v
            if not dry_run:
                # now update counts
                for word in self.wv.index_to_key:
                    self.wv.set_vecattr(word, 'count', self.raw_vocab[word])
            original_unique_total = len(retain_words) + drop_unique
            retain_unique_pct = len(retain_words) * 100 / max(original_unique_total, 1)
            self.add_lifecycle_event(
                "prepare_vocab",
                msg=(
                    f"effective_min_count={self.effective_min_count} retains {len(retain_words)} unique "
                    f"words ({retain_unique_pct}%% of original {original_unique_total}, drops {drop_unique})"
                ),
            )

            original_total = retain_total + drop_total
            retain_pct = retain_total * 100 / max(original_total, 1)
            self.add_lifecycle_event(
                "prepare_vocab",
                msg=(
                    f"effective_min_count={self.effective_min_count} leaves {retain_total} word corpus "
                    f"({retain_pct}%% of original {original_total}, drops {drop_total})"
                ),
            )
        else:
            logger.info("Updating model with new vocabulary")
            new_total = pre_exist_total = 0
            new_words = []
            pre_exist_words = []
            for word, v in self.raw_vocab.items():
                if keep_vocab_item(word, v, self.effective_min_count, trim_rule=trim_rule):
                    if self.wv.has_index_for(word):
                        pre_exist_words.append(word)
                        pre_exist_total += v
                        if not dry_run:
                            pass
                    else:
                        new_words.append(word)
                        new_total += v
                        if not dry_run:
                            self.wv.key_to_index[word] = len(self.wv.index_to_key)
                            self.wv.index_to_key.append(word)
                else:
                    drop_unique += 1
                    drop_total += v
            if not dry_run:
                # now update counts
                self.wv.allocate_vecattrs(attrs=['count'], types=[type(0)])
                for word in self.wv.index_to_key:
                    self.wv.set_vecattr(word, 'count', self.wv.get_vecattr(word, 'count') + self.raw_vocab.get(word, 0))
            original_unique_total = len(pre_exist_words) + len(new_words) + drop_unique
            pre_exist_unique_pct = len(pre_exist_words) * 100 / max(original_unique_total, 1)
            new_unique_pct = len(new_words) * 100 / max(original_unique_total, 1)
            self.add_lifecycle_event(
                "prepare_vocab",
                msg=(
                    f"added {len(new_words)} new unique words ({new_unique_pct}%% of original "
                    f"{original_unique_total}) and increased the count of {len(pre_exist_words)} "
                    f"pre-existing words ({pre_exist_unique_pct}%% of original {original_unique_total})"
                ),
            )
            retain_words = new_words + pre_exist_words
            retain_total = new_total + pre_exist_total

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + np.sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_vocab[w]
            word_probability = (np.sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.wv.set_vecattr(w, 'sample_int', np.uint32(word_probability * (2**32 - 1)))

        if not dry_run and not keep_raw_vocab:
            logger.info("deleting the raw counts dictionary of %i items", len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info("sample=%g downsamples %i most-common words", sample, downsample_unique)
        self.add_lifecycle_event(
            "prepare_vocab",
            msg=(
                f"downsampling leaves estimated {downsample_total} word corpus "
                f"({downsample_total * 100.0 / max(retain_total, 1):.1f}%% of prior {retain_total})"
            ),
        )

        # return from each step: words-affected, resulting-corpus-size, extra memory estimates
        report_values = {
            'drop_unique': drop_unique, 'retain_total': retain_total, 'downsample_unique': downsample_unique,
            'downsample_total': int(downsample_total), 'num_retained_words': len(retain_words)
        }

        if self.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input – never predicted – so count, huffman-point, etc doesn't matter
            self.add_null_word()

        if self.sorted_vocab and not update:
            self.wv.sort_by_descending_frequency()

        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()

        return report_values

    def estimate_memory(self, vocab_size=None, report=None):
        """Estimate required memory for a model using current settings and provided vocabulary size.

        Parameters
        ----------
        vocab_size : int, optional
            Number of unique tokens in the vocabulary
        report : dict of (str, int), optional
            A dictionary from string representations of the model's memory consuming members to their size in bytes.

        Returns
        -------
        dict of (str, int)
            A dictionary from string representations of the model's memory consuming members to their size in bytes.

        """
        vocab_size = vocab_size or len(self.wv)
        report = report or {}
        report['vocab'] = vocab_size * (700 if self.hs else 500)
        report['vectors'] = vocab_size * self.vector_size * np.dtype(REAL).itemsize
        if self.hs:
            report['syn1'] = vocab_size * self.layer1_size * np.dtype(REAL).itemsize
        if self.negative:
            report['syn1neg'] = vocab_size * self.layer1_size * np.dtype(REAL).itemsize
        report['total'] = sum(report.values())
        logger.info(
            "estimated required memory for %i words and %i dimensions: %i bytes",
            vocab_size, self.vector_size, report['total'],
        )
        return report

    def add_null_word(self):
        word = '\0'
        self.wv.key_to_index[word] = len(self.wv)
        self.wv.index_to_key.append(word)
        self.wv.set_vecattr(word, 'count', 1)

    def create_binary_tree(self):
        """Create a `binary Huffman tree <https://en.wikipedia.org/wiki/Huffman_coding>`_ using stored vocabulary
        word counts. Frequent words will have shorter binary codes.
        Called internally from :meth:`~gensim.models.word2vec.Word2VecVocab.build_vocab`.

        """
        _assign_binary_codes(self.wv)

    def make_cum_table(self, domain=2**31 - 1):
        """Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.

        To draw a word index, choose a random integer up to the maximum value in the table (cum_table[-1]),
        then finding that integer's sorted insertion point (as if by `bisect_left` or `ndarray.searchsorted()`).
        That insertion point is the drawn index, coming up in proportion equal to the increment at that slot.

        """
        vocab_size = len(self.wv.index_to_key)
        self.cum_table = np.zeros(vocab_size, dtype=np.uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for word_index in range(vocab_size):
            count = self.wv.get_vecattr(word_index, 'count')
            train_words_pow += count**self.ns_exponent
        cumulative = 0.0
        for word_index in range(vocab_size):
            count = self.wv.get_vecattr(word_index, 'count')
            cumulative += count**self.ns_exponent
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def prepare_weights(self, update=False):
        """Build tables and model weights based on final vocabulary settings."""
        # set initial input/projection and hidden weights
        if not update:
            self.init_weights()
        else:
            self.update_weights()

    @deprecated("Use gensim.models.keyedvectors.pseudorandom_weak_vector() directly")
    def seeded_vector(self, seed_string, vector_size):
        return pseudorandom_weak_vector(vector_size, seed_string=seed_string, hashfxn=self.hashfxn)

    def init_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        self.wv.resize_vectors(seed=self.seed)

        if self.hs:
            self.syn1 = np.zeros((len(self.wv), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = np.zeros((len(self.wv), self.layer1_size), dtype=REAL)

    def update_weights(self):
        """Copy all the existing weights, and reset the weights for the newly added vocabulary."""
        logger.info("updating layer weights")
        # Raise an error if an online update is run before initial training on a corpus
        if not len(self.wv.vectors):
            raise RuntimeError(
                "You cannot do an online vocabulary-update of a model which has no prior vocabulary. "
                "First build the vocabulary of your model with a corpus before doing an online update."
            )
        preresize_count = len(self.wv.vectors)
        self.wv.resize_vectors(seed=self.seed)
        gained_vocab = len(self.wv.vectors) - preresize_count

        if self.hs:
            self.syn1 = np.vstack([self.syn1, np.zeros((gained_vocab, self.layer1_size), dtype=REAL)])
        if self.negative:
            pad = np.zeros((gained_vocab, self.layer1_size), dtype=REAL)
            self.syn1neg = np.vstack([self.syn1neg, pad])

    @deprecated(
        "Gensim 4.0.0 implemented internal optimizations that make calls to init_sims() unnecessary. "
        "init_sims() is now obsoleted and will be completely removed in future versions. "
        "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4"
    )
    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors. Obsoleted.

        If you need a single unit-normalized vector for some key, call
        :meth:`~gensim.models.keyedvectors.KeyedVectors.get_vector` instead:
        ``word2vec_model.wv.get_vector(key, norm=True)``.

        To refresh norms after you performed some atypical out-of-band vector tampering,
        call `:meth:`~gensim.models.keyedvectors.KeyedVectors.fill_norms()` instead.

        Parameters
        ----------
        replace : bool
            If True, forget the original trained vectors and only keep the normalized ones.
            You lose information if you do this.

        """
        self.wv.init_sims(replace=replace)

    def _do_train_epoch(
            self, corpus_file, thread_id, offset, cython_vocab, thread_private_mem, cur_epoch,
            total_examples=None, total_words=None, **kwargs,
        ):
        work, neu1 = thread_private_mem

        if self.sg:
            examples, tally, raw_tally = train_epoch_sg(
                self, corpus_file, offset, cython_vocab, cur_epoch,
                total_examples, total_words, work, neu1, self.compute_loss,
            )
        else:
            examples, tally, raw_tally = train_epoch_cbow(
                self, corpus_file, offset, cython_vocab, cur_epoch,
                total_examples, total_words, work, neu1, self.compute_loss,
            )

        return examples, tally, raw_tally

    def _do_train_job(self, sentences, alpha, inits):
        """Train the model on a single batch of sentences.

        Parameters
        ----------
        sentences : iterable of list of str
            Corpus chunk to be used in this training batch.
        alpha : float
            The learning rate used in this batch.
        inits : (np.ndarray, np.ndarray)
            Each worker threads private work memory.

        Returns
        -------
        (int, int)
             2-tuple (effective word count after ignoring unknown words and sentence length trimming, total word count).

        """
        work, neu1 = inits
        tally = 0
        if self.sg:
            tally += train_batch_sg(self, sentences, alpha, work, self.compute_loss)
        else:
            tally += train_batch_cbow(self, sentences, alpha, work, neu1, self.compute_loss)
        return tally, self._raw_word_count(sentences)

    def _clear_post_train(self):
        """Clear any cached values that training may have invalidated."""
        self.wv.norms = None

    def train(
            self, corpus_iterable=None, corpus_file=None, total_examples=None, total_words=None,
            epochs=None, start_alpha=None, end_alpha=None, word_count=0,
            queue_factor=2, report_delay=1.0, compute_loss=False, callbacks=(),
            **kwargs,
        ):
        """Update the model's neural weights from a sequence of sentences.

        Notes
        -----
        To support linear learning-rate decay from (initial) `alpha` to `min_alpha`, and accurate
        progress-percentage logging, either `total_examples` (count of sentences) or `total_words` (count of
        raw words in sentences) **MUST** be provided. If `sentences` is the same corpus
        that was provided to :meth:`~gensim.models.word2vec.Word2Vec.build_vocab` earlier,
        you can simply use `total_examples=self.corpus_count`.

        Warnings
        --------
        To avoid common mistakes around the model's ability to do multiple training passes itself, an
        explicit `epochs` argument **MUST** be provided. In the common and recommended case
        where :meth:`~gensim.models.word2vec.Word2Vec.train` is only called once, you can set `epochs=self.epochs`.

        Parameters
        ----------
        corpus_iterable : iterable of list of str
            The ``corpus_iterable`` can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network, to limit RAM usage.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
            See also the `tutorial on data streaming in Python
            <https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/>`_.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `sentences` to get performance boost. Only one of `sentences` or
            `corpus_file` arguments need to be passed (not both of them).
        total_examples : int
            Count of sentences.
        total_words : int
            Count of raw words in sentences.
        epochs : int
            Number of iterations (epochs) over the corpus.
        start_alpha : float, optional
            Initial learning rate. If supplied, replaces the starting `alpha` from the constructor,
            for this one call to`train()`.
            Use only if making multiple calls to `train()`, when you want to manage the alpha learning-rate yourself
            (not recommended).
        end_alpha : float, optional
            Final learning rate. Drops linearly from `start_alpha`.
            If supplied, this replaces the final `min_alpha` from the constructor, for this one call to `train()`.
            Use only if making multiple calls to `train()`, when you want to manage the alpha learning-rate yourself
            (not recommended).
        word_count : int, optional
            Count of words already trained. Set this to 0 for the usual
            case of training on all words in sentences.
        queue_factor : int, optional
            Multiplier for size of queue (number of workers * queue_factor).
        report_delay : float, optional
            Seconds to wait before reporting progress.
        compute_loss: bool, optional
            If True, computes and stores loss value which can be retrieved using
            :meth:`~gensim.models.word2vec.Word2Vec.get_latest_training_loss`.
        callbacks : iterable of :class:`~gensim.models.callbacks.CallbackAny2Vec`, optional
            Sequence of callbacks to be executed at specific stages during training.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.models import Word2Vec
            >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
            >>>
            >>> model = Word2Vec(min_count=1)
            >>> model.build_vocab(sentences)  # prepare the model vocabulary
            >>> model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)  # train word vectors
            (1, 30)

        """
        self.alpha = start_alpha or self.alpha
        self.min_alpha = end_alpha or self.min_alpha
        self.epochs = epochs

        self._check_training_sanity(epochs=epochs, total_examples=total_examples, total_words=total_words)
        self._check_corpus_sanity(corpus_iterable=corpus_iterable, corpus_file=corpus_file, passes=epochs)

        self.add_lifecycle_event(
            "train",
            msg=(
                f"training model with {self.workers} workers on {len(self.wv)} vocabulary and "
                f"{self.layer1_size} features, using sg={self.sg} hs={self.hs} sample={self.sample} "
                f"negative={self.negative} window={self.window}"
            ),
        )

        self.compute_loss = compute_loss
        self.running_training_loss = 0.0

        for callback in callbacks:
            callback.on_train_begin(self)

        trained_word_count = 0
        raw_word_count = 0
        start = default_timer() - 0.00001
        job_tally = 0

        for cur_epoch in range(self.epochs):
            for callback in callbacks:
                callback.on_epoch_begin(self)

            if corpus_iterable is not None:
                trained_word_count_epoch, raw_word_count_epoch, job_tally_epoch = self._train_epoch(
                    corpus_iterable, cur_epoch=cur_epoch, total_examples=total_examples,
                    total_words=total_words, queue_factor=queue_factor, report_delay=report_delay,
                    callbacks=callbacks, **kwargs)
            else:
                trained_word_count_epoch, raw_word_count_epoch, job_tally_epoch = self._train_epoch_corpusfile(
                    corpus_file, cur_epoch=cur_epoch, total_examples=total_examples, total_words=total_words,
                    callbacks=callbacks, **kwargs)

            trained_word_count += trained_word_count_epoch
            raw_word_count += raw_word_count_epoch
            job_tally += job_tally_epoch

            for callback in callbacks:
                callback.on_epoch_end(self)

        # Log overall time
        total_elapsed = default_timer() - start
        self._log_train_end(raw_word_count, trained_word_count, total_elapsed, job_tally)

        self.train_count += 1  # number of times train() has been called
        self._clear_post_train()

        for callback in callbacks:
            callback.on_train_end(self)

        return trained_word_count, raw_word_count

    def _worker_loop_corpusfile(
            self, corpus_file, thread_id, offset, cython_vocab, progress_queue, cur_epoch=0,
            total_examples=None, total_words=None, **kwargs,
        ):
        """Train the model on a `corpus_file` in LineSentence format.

        This function will be called in parallel by multiple workers (threads or processes) to make
        optimal use of multicore machines.

        Parameters
        ----------
        corpus_file : str
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
        thread_id : int
            Thread index starting from 0 to `number of workers - 1`.
        offset : int
            Offset (in bytes) in the `corpus_file` for particular worker.
        cython_vocab : :class:`~gensim.models.word2vec_inner.CythonVocab`
            Copy of the vocabulary in order to access it without GIL.
        progress_queue : Queue of (int, int, int)
            A queue of progress reports. Each report is represented as a tuple of these 3 elements:
                * Size of data chunk processed, for example number of sentences in the corpus chunk.
                * Effective word count used in training (after ignoring unknown words and trimming the sentence length).
                * Total word count used in training.
        **kwargs : object
            Additional key word parameters for the specific model inheriting from this class.

        """
        thread_private_mem = self._get_thread_working_mem()

        examples, tally, raw_tally = self._do_train_epoch(
            corpus_file, thread_id, offset, cython_vocab, thread_private_mem, cur_epoch,
            total_examples=total_examples, total_words=total_words, **kwargs)

        progress_queue.put((examples, tally, raw_tally))
        progress_queue.put(None)

    def _worker_loop(self, job_queue, progress_queue):
        """Train the model, lifting batches of data from the queue.

        This function will be called in parallel by multiple workers (threads or processes) to make
        optimal use of multicore machines.

        Parameters
        ----------
        job_queue : Queue of (list of objects, float)
            A queue of jobs still to be processed. The worker will take up jobs from this queue.
            Each job is represented by a tuple where the first element is the corpus chunk to be processed and
            the second is the floating-point learning rate.
        progress_queue : Queue of (int, int, int)
            A queue of progress reports. Each report is represented as a tuple of these 3 elements:
                * Size of data chunk processed, for example number of sentences in the corpus chunk.
                * Effective word count used in training (after ignoring unknown words and trimming the sentence length).
                * Total word count used in training.

        """
        thread_private_mem = self._get_thread_working_mem()
        jobs_processed = 0
        while True:
            job = job_queue.get()
            if job is None:
                progress_queue.put(None)
                break  # no more jobs => quit this worker
            data_iterable, alpha = job

            tally, raw_tally = self._do_train_job(data_iterable, alpha, thread_private_mem)

            progress_queue.put((len(data_iterable), tally, raw_tally))  # report back progress
            jobs_processed += 1
        logger.debug("worker exiting, processed %i jobs", jobs_processed)

    def _job_producer(self, data_iterator, job_queue, cur_epoch=0, total_examples=None, total_words=None):
        """Fill the jobs queue using the data found in the input stream.

        Each job is represented by a tuple where the first element is the corpus chunk to be processed and
        the second is a dictionary of parameters.

        Parameters
        ----------
        data_iterator : iterable of list of objects
            The input dataset. This will be split in chunks and these chunks will be pushed to the queue.
        job_queue : Queue of (list of object, float)
            A queue of jobs still to be processed. The worker will take up jobs from this queue.
            Each job is represented by a tuple where the first element is the corpus chunk to be processed and
            the second is the floating-point learning rate.
        cur_epoch : int, optional
            The current training epoch, needed to compute the training parameters for each job.
            For example in many implementations the learning rate would be dropping with the number of epochs.
        total_examples : int, optional
            Count of objects in the `data_iterator`. In the usual case this would correspond to the number of sentences
            in a corpus. Used to log progress.
        total_words : int, optional
            Count of total objects in `data_iterator`. In the usual case this would correspond to the number of raw
            words in a corpus. Used to log progress.

        """
        job_batch, batch_size = [], 0
        pushed_words, pushed_examples = 0, 0
        next_alpha = self._get_next_alpha(0.0, cur_epoch)
        job_no = 0

        for data_idx, data in enumerate(data_iterator):
            data_length = self._raw_word_count([data])

            # can we fit this sentence into the existing job batch?
            if batch_size + data_length <= self.batch_words:
                # yes => add it to the current job
                job_batch.append(data)
                batch_size += data_length
            else:
                job_no += 1
                job_queue.put((job_batch, next_alpha))

                # update the learning rate for the next job
                if total_examples:
                    # examples-based decay
                    pushed_examples += len(job_batch)
                    epoch_progress = 1.0 * pushed_examples / total_examples
                else:
                    # words-based decay
                    pushed_words += self._raw_word_count(job_batch)
                    epoch_progress = 1.0 * pushed_words / total_words
                next_alpha = self._get_next_alpha(epoch_progress, cur_epoch)

                # add the sentence that didn't fit as the first item of a new job
                job_batch, batch_size = [data], data_length
        # add the last job too (may be significantly smaller than batch_words)
        if job_batch:
            job_no += 1
            job_queue.put((job_batch, next_alpha))

        if job_no == 0 and self.train_count == 0:
            logger.warning(
                "train() called with an empty iterator (if not intended, "
                "be sure to provide a corpus that offers restartable iteration = an iterable)."
            )

        # give the workers heads up that they can finish -- no more work!
        for _ in range(self.workers):
            job_queue.put(None)
        logger.debug("job loop exiting, total %i jobs", job_no)

    def _log_epoch_progress(
            self, progress_queue=None, job_queue=None, cur_epoch=0, total_examples=None,
            total_words=None, report_delay=1.0, is_corpus_file_mode=None,
        ):
        """Get the progress report for a single training epoch.

        Parameters
        ----------
        progress_queue : Queue of (int, int, int)
            A queue of progress reports. Each report is represented as a tuple of these 3 elements:
                * size of data chunk processed, for example number of sentences in the corpus chunk.
                * Effective word count used in training (after ignoring unknown words and trimming the sentence length).
                * Total word count used in training.
        job_queue : Queue of (list of object, float)
            A queue of jobs still to be processed. The worker will take up jobs from this queue.
            Each job is represented by a tuple where the first element is the corpus chunk to be processed and
            the second is the floating-point learning rate.
        cur_epoch : int, optional
            The current training epoch, needed to compute the training parameters for each job.
            For example in many implementations the learning rate would be dropping with the number of epochs.
        total_examples : int, optional
            Count of objects in the `data_iterator`. In the usual case this would correspond to the number of sentences
            in a corpus. Used to log progress.
        total_words : int, optional
            Count of total objects in `data_iterator`. In the usual case this would correspond to the number of raw
            words in a corpus. Used to log progress.
        report_delay : float, optional
            Number of seconds between two consecutive progress report messages in the logger.
        is_corpus_file_mode : bool, optional
            Whether training is file-based (corpus_file argument) or not.

        Returns
        -------
        (int, int, int)
            The epoch report consisting of three elements:
                * size of data chunk processed, for example number of sentences in the corpus chunk.
                * Effective word count used in training (after ignoring unknown words and trimming the sentence length).
                * Total word count used in training.

        """
        example_count, trained_word_count, raw_word_count = 0, 0, 0
        start, next_report = default_timer() - 0.00001, 1.0
        job_tally = 0
        unfinished_worker_count = self.workers

        while unfinished_worker_count > 0:
            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                logger.info("worker thread finished; awaiting finish of %i more threads", unfinished_worker_count)
                continue
            examples, trained_words, raw_words = report
            job_tally += 1

            # update progress stats
            example_count += examples
            trained_word_count += trained_words  # only words in vocab & sampled
            raw_word_count += raw_words

            # log progress once every report_delay seconds
            elapsed = default_timer() - start
            if elapsed >= next_report:
                self._log_progress(
                    job_queue, progress_queue, cur_epoch, example_count, total_examples,
                    raw_word_count, total_words, trained_word_count, elapsed)
                next_report = elapsed + report_delay
        # all done; report the final stats
        elapsed = default_timer() - start
        self._log_epoch_end(
            cur_epoch, example_count, total_examples, raw_word_count, total_words,
            trained_word_count, elapsed, is_corpus_file_mode)
        self.total_train_time += elapsed
        return trained_word_count, raw_word_count, job_tally

    def _train_epoch_corpusfile(
            self, corpus_file, cur_epoch=0, total_examples=None, total_words=None, callbacks=(), **kwargs,
        ):
        """Train the model for a single epoch.

        Parameters
        ----------
        corpus_file : str
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
        cur_epoch : int, optional
            The current training epoch, needed to compute the training parameters for each job.
            For example in many implementations the learning rate would be dropping with the number of epochs.
        total_examples : int, optional
            Count of objects in the `data_iterator`. In the usual case this would correspond to the number of sentences
            in a corpus, used to log progress.
        total_words : int
            Count of total objects in `data_iterator`. In the usual case this would correspond to the number of raw
            words in a corpus, used to log progress. Must be provided in order to seek in `corpus_file`.
        **kwargs : object
            Additional key word parameters for the specific model inheriting from this class.

        Returns
        -------
        (int, int, int)
            The training report for this epoch consisting of three elements:
                * Size of data chunk processed, for example number of sentences in the corpus chunk.
                * Effective word count used in training (after ignoring unknown words and trimming the sentence length).
                * Total word count used in training.

        """
        if not total_words:
            raise ValueError("total_words must be provided alongside corpus_file argument.")

        from gensim.models.word2vec_corpusfile import CythonVocab
        from gensim.models.fasttext import FastText
        cython_vocab = CythonVocab(self.wv, hs=self.hs, fasttext=isinstance(self, FastText))

        progress_queue = Queue()

        corpus_file_size = os.path.getsize(corpus_file)

        thread_kwargs = copy.copy(kwargs)
        thread_kwargs['cur_epoch'] = cur_epoch
        thread_kwargs['total_examples'] = total_examples
        thread_kwargs['total_words'] = total_words
        workers = [
            threading.Thread(
                target=self._worker_loop_corpusfile,
                args=(
                    corpus_file, thread_id, corpus_file_size / self.workers * thread_id, cython_vocab, progress_queue
                ),
                kwargs=thread_kwargs
            ) for thread_id in range(self.workers)
        ]

        for thread in workers:
            thread.daemon = True
            thread.start()

        trained_word_count, raw_word_count, job_tally = self._log_epoch_progress(
            progress_queue=progress_queue, job_queue=None, cur_epoch=cur_epoch,
            total_examples=total_examples, total_words=total_words, is_corpus_file_mode=True)

        return trained_word_count, raw_word_count, job_tally

    def _train_epoch(
            self, data_iterable, cur_epoch=0, total_examples=None, total_words=None,
            queue_factor=2, report_delay=1.0, callbacks=(),
        ):
        """Train the model for a single epoch.

        Parameters
        ----------
        data_iterable : iterable of list of object
            The input corpus. This will be split in chunks and these chunks will be pushed to the queue.
        cur_epoch : int, optional
            The current training epoch, needed to compute the training parameters for each job.
            For example in many implementations the learning rate would be dropping with the number of epochs.
        total_examples : int, optional
            Count of objects in the `data_iterator`. In the usual case this would correspond to the number of sentences
            in a corpus, used to log progress.
        total_words : int, optional
            Count of total objects in `data_iterator`. In the usual case this would correspond to the number of raw
            words in a corpus, used to log progress.
        queue_factor : int, optional
            Multiplier for size of queue -> size = number of workers * queue_factor.
        report_delay : float, optional
            Number of seconds between two consecutive progress report messages in the logger.

        Returns
        -------
        (int, int, int)
            The training report for this epoch consisting of three elements:
                * Size of data chunk processed, for example number of sentences in the corpus chunk.
                * Effective word count used in training (after ignoring unknown words and trimming the sentence length).
                * Total word count used in training.

        """
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [
            threading.Thread(
                target=self._worker_loop,
                args=(job_queue, progress_queue,))
            for _ in range(self.workers)
        ]

        workers.append(threading.Thread(
            target=self._job_producer,
            args=(data_iterable, job_queue),
            kwargs={'cur_epoch': cur_epoch, 'total_examples': total_examples, 'total_words': total_words}))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        trained_word_count, raw_word_count, job_tally = self._log_epoch_progress(
            progress_queue, job_queue, cur_epoch=cur_epoch, total_examples=total_examples,
            total_words=total_words, report_delay=report_delay, is_corpus_file_mode=False,
        )

        return trained_word_count, raw_word_count, job_tally

    def _get_next_alpha(self, epoch_progress, cur_epoch):
        """Get the correct learning rate for the next iteration.

        Parameters
        ----------
        epoch_progress : float
            Ratio of finished work in the current epoch.
        cur_epoch : int
            Number of current iteration.

        Returns
        -------
        float
            The learning rate to be used in the next training epoch.

        """
        start_alpha = self.alpha
        end_alpha = self.min_alpha
        progress = (cur_epoch + epoch_progress) / self.epochs
        next_alpha = start_alpha - (start_alpha - end_alpha) * progress
        next_alpha = max(end_alpha, next_alpha)
        self.min_alpha_yet_reached = next_alpha
        return next_alpha

    def _get_thread_working_mem(self):
        """Computes the memory used per worker thread.

        Returns
        -------
        (np.ndarray, np.ndarray)
            Each worker threads private work memory.

        """
        work = matutils.zeros_aligned(self.layer1_size, dtype=REAL)  # per-thread private work memory
        neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
        return work, neu1

    def _raw_word_count(self, job):
        """Get the number of words in a given job.

        Parameters
        ----------
        job: iterable of list of str
            The corpus chunk processed in a single batch.

        Returns
        -------
        int
            Number of raw words in the corpus chunk.

        """
        return sum(len(sentence) for sentence in job)

    def _check_corpus_sanity(self, corpus_iterable=None, corpus_file=None, passes=1):
        """Checks whether the corpus parameters make sense."""
        if corpus_file is None and corpus_iterable is None:
            raise TypeError("Either one of corpus_file or corpus_iterable value must be provided")
        if corpus_file is not None and corpus_iterable is not None:
            raise TypeError("Both corpus_file and corpus_iterable must not be provided at the same time")
        if corpus_iterable is None and not os.path.isfile(corpus_file):
            raise TypeError("Parameter corpus_file must be a valid path to a file, got %r instead" % corpus_file)
        if corpus_iterable is not None and not isinstance(corpus_iterable, Iterable):
            raise TypeError(
                "The corpus_iterable must be an iterable of lists of strings, got %r instead" % corpus_iterable)
        if corpus_iterable is not None and isinstance(corpus_iterable, GeneratorType) and passes > 1:
            raise TypeError(
                f"Using a generator as corpus_iterable can't support {passes} passes. Try a re-iterable sequence.")

    def _check_training_sanity(self, epochs=0, total_examples=None, total_words=None, **kwargs):
        """Checks whether the training parameters make sense.

        Parameters
        ----------
        epochs : int
            Number of training epochs. A positive integer.
        total_examples : int, optional
            Number of documents in the corpus. Either `total_examples` or `total_words` **must** be supplied.
        total_words : int, optional
            Number of words in the corpus. Either `total_examples` or `total_words` **must** be supplied.
        **kwargs : object
            Unused. Present to preserve signature among base and inherited implementations.

        Raises
        ------
        RuntimeError
            If one of the required training pre/post processing steps have not been performed.
        ValueError
            If the combination of input parameters is inconsistent.

        """
        if self.alpha > self.min_alpha_yet_reached:
            logger.warning("Effective 'alpha' higher than previous training cycles")

        if not self.wv.key_to_index:  # should be set by `build_vocab`
            raise RuntimeError("you must first build vocabulary before training the model")
        if not len(self.wv.vectors):
            raise RuntimeError("you must initialize vectors before training the model")

        if total_words is None and total_examples is None:
            raise ValueError(
                "You must specify either total_examples or total_words, for proper learning-rate "
                "and progress calculations. "
                "If you've just built the vocabulary using the same corpus, using the count cached "
                "in the model is sufficient: total_examples=model.corpus_count."
            )
        if epochs is None or epochs <= 0:
            raise ValueError("You must specify an explicit epochs count. The usual value is epochs=model.epochs.")

    def _log_progress(
            self, job_queue, progress_queue, cur_epoch, example_count, total_examples,
            raw_word_count, total_words, trained_word_count, elapsed
        ):
        """Callback used to log progress for long running jobs.

        Parameters
        ----------
        job_queue : Queue of (list of object, float)
            The queue of jobs still to be performed by workers. Each job is represented as a tuple containing
            the batch of data to be processed and the floating-point learning rate.
        progress_queue : Queue of (int, int, int)
            A queue of progress reports. Each report is represented as a tuple of these 3 elements:
                * size of data chunk processed, for example number of sentences in the corpus chunk.
                * Effective word count used in training (after ignoring unknown words and trimming the sentence length).
                * Total word count used in training.
        cur_epoch : int
            The current training iteration through the corpus.
        example_count : int
            Number of examples (could be sentences for example) processed until now.
        total_examples : int
            Number of all examples present in the input corpus.
        raw_word_count : int
            Number of words used in training until now.
        total_words : int
            Number of all words in the input corpus.
        trained_word_count : int
            Number of effective words used in training until now (after ignoring unknown words and trimming
            the sentence length).
        elapsed : int
            Elapsed time since the beginning of training in seconds.

        Notes
        -----
        If you train the model via `corpus_file` argument, there is no job_queue, so reported job_queue size will
        always be equal to -1.

        """
        if total_examples:
            # examples-based progress %
            logger.info(
                "EPOCH %i - PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i",
                cur_epoch + 1, 100.0 * example_count / total_examples, trained_word_count / elapsed,
                -1 if job_queue is None else utils.qsize(job_queue), utils.qsize(progress_queue)
            )
        else:
            # words-based progress %
            logger.info(
                "EPOCH %i - PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
                cur_epoch + 1, 100.0 * raw_word_count / total_words, trained_word_count / elapsed,
                -1 if job_queue is None else utils.qsize(job_queue), utils.qsize(progress_queue)
            )

    def _log_epoch_end(
            self, cur_epoch, example_count, total_examples, raw_word_count, total_words,
            trained_word_count, elapsed, is_corpus_file_mode
        ):
        """Callback used to log the end of a training epoch.

        Parameters
        ----------
        cur_epoch : int
            The current training iteration through the corpus.
        example_count : int
            Number of examples (could be sentences for example) processed until now.
        total_examples : int
            Number of all examples present in the input corpus.
        raw_word_count : int
            Number of words used in training until now.
        total_words : int
            Number of all words in the input corpus.
        trained_word_count : int
            Number of effective words used in training until now (after ignoring unknown words and trimming
            the sentence length).
        elapsed : int
            Elapsed time since the beginning of training in seconds.
        is_corpus_file_mode : bool
            Whether training is file-based (corpus_file argument) or not.

        Warnings
        --------
        In case the corpus is changed while the epoch was running.

        """
        logger.info(
            "EPOCH - %i : training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            cur_epoch + 1, raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed,
        )

        # don't warn if training in file-based mode, because it's expected behavior
        if is_corpus_file_mode:
            return

        # check that the input corpus hasn't changed during iteration
        if total_examples and total_examples != example_count:
            logger.warning(
                "EPOCH - %i : supplied example count (%i) did not equal expected count (%i)", cur_epoch + 1,
                example_count, total_examples
            )
        if total_words and total_words != raw_word_count:
            logger.warning(
                "EPOCH - %i : supplied raw word count (%i) did not equal expected count (%i)", cur_epoch + 1,
                raw_word_count, total_words
            )

    def _log_train_end(self, raw_word_count, trained_word_count, total_elapsed, job_tally):
        """Callback to log the end of training.

        Parameters
        ----------
        raw_word_count : int
            Number of words used in the whole training.
        trained_word_count : int
            Number of effective words used in training (after ignoring unknown words and trimming the sentence length).
        total_elapsed : int
            Total time spent during training in seconds.
        job_tally : int
            Total number of jobs processed during training.

        """
        self.add_lifecycle_event("train", msg=(
            f"training on {raw_word_count} raw words ({trained_word_count} effective words) "
            f"took {total_elapsed:.1f}s, {trained_word_count / total_elapsed:.0f} effective words/s"
        ))

    def score(self, sentences, total_sentences=int(1e6), chunksize=100, queue_factor=2, report_delay=1):
        """Score the log probability for a sequence of sentences.
        This does not change the fitted model in any way (see :meth:`~gensim.models.word2vec.Word2Vec.train` for that).

        Gensim has currently only implemented score for the hierarchical softmax scheme,
        so you need to have run word2vec with `hs=1` and `negative=0` for this to work.

        Note that you should specify `total_sentences`; you'll run into problems if you ask to
        score more than this number of sentences but it is inefficient to set the value too high.

        See the `article by Matt Taddy: "Document Classification by Inversion of Distributed Language Representations"
        <https://arxiv.org/pdf/1504.07295.pdf>`_ and the
        `gensim demo <https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/deepir.ipynb>`_ for examples of
        how to use such scores in document classification.

        Parameters
        ----------
        sentences : iterable of list of str
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        total_sentences : int, optional
            Count of sentences.
        chunksize : int, optional
            Chunksize of jobs
        queue_factor : int, optional
            Multiplier for size of queue (number of workers * queue_factor).
        report_delay : float, optional
            Seconds to wait before reporting progress.

        """
        logger.info(
            "scoring sentences with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s and negative=%s",
            self.workers, len(self.wv), self.layer1_size, self.sg, self.hs,
            self.sample, self.negative
        )

        if not self.wv.key_to_index:
            raise RuntimeError("you must first build vocabulary before scoring new data")

        if not self.hs:
            raise RuntimeError(
                "We have currently only implemented score for the hierarchical softmax scheme, "
                "so you need to have run word2vec with hs=1 and negative=0 for this to work."
            )

        def worker_loop():
            """Compute log probability for each sentence, lifting lists of sentences from the jobs queue."""
            work = np.zeros(1, dtype=REAL)  # for sg hs, we actually only need one memory loc (running sum)
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            while True:
                job = job_queue.get()
                if job is None:  # signal to finish
                    break
                ns = 0
                for sentence_id, sentence in job:
                    if sentence_id >= total_sentences:
                        break
                    if self.sg:
                        score = score_sentence_sg(self, sentence, work)
                    else:
                        score = score_sentence_cbow(self, sentence, work, neu1)
                    sentence_scores[sentence_id] = score
                    ns += 1
                progress_queue.put(ns)  # report progress

        start, next_report = default_timer(), 1.0
        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in range(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        sentence_count = 0
        sentence_scores = matutils.zeros_aligned(total_sentences, dtype=REAL)

        push_done = False
        done_jobs = 0
        jobs_source = enumerate(utils.grouper(enumerate(sentences), chunksize))

        # fill jobs queue with (id, sentence) job items
        while True:
            try:
                job_no, items = next(jobs_source)
                if (job_no - 1) * chunksize > total_sentences:
                    logger.warning(
                        "terminating after %i sentences (set higher total_sentences if you want more).",
                        total_sentences
                    )
                    job_no -= 1
                    raise StopIteration()
                logger.debug("putting job #%i in the queue", job_no)
                job_queue.put(items)
            except StopIteration:
                logger.info("reached end of input; waiting to finish %i outstanding jobs", job_no - done_jobs + 1)
                for _ in range(self.workers):
                    job_queue.put(None)  # give the workers heads up that they can finish -- no more work!
                push_done = True
            try:
                while done_jobs < (job_no + 1) or not push_done:
                    ns = progress_queue.get(push_done)  # only block after all jobs pushed
                    sentence_count += ns
                    done_jobs += 1
                    elapsed = default_timer() - start
                    if elapsed >= next_report:
                        logger.info(
                            "PROGRESS: at %.2f%% sentences, %.0f sentences/s",
                            100.0 * sentence_count, sentence_count / elapsed
                        )
                        next_report = elapsed + report_delay  # don't flood log, wait report_delay seconds
                else:
                    # loop ended by job count; really done
                    break
            except Empty:
                pass  # already out of loop; continue to next push

        elapsed = default_timer() - start
        self.wv.norms = None  # clear any cached lengths
        logger.info(
            "scoring %i sentences took %.1fs, %.0f sentences/s",
            sentence_count, elapsed, sentence_count / elapsed
        )
        return sentence_scores[:sentence_count]

    def predict_output_word(self, context_words_list, topn=10):
        """Get the probability distribution of the center word given context words.

        Note this performs a CBOW-style propagation, even in SG models,
        and doesn't quite weight the surrounding words the same as in
        training -- so it's just one crude way of using a trained model
        as a predictor.

        Parameters
        ----------
        context_words_list : list of str
            List of context words.
        topn : int, optional
            Return `topn` words and their probabilities.

        Returns
        -------
        list of (str, float)
            `topn` length list of tuples of (word, probability).

        """
        if not self.negative:
            raise RuntimeError(
                "We have currently only implemented predict_output_word for the negative sampling scheme, "
                "so you need to have run word2vec with negative > 0 for this to work."
            )

        if not hasattr(self.wv, 'vectors') or not hasattr(self, 'syn1neg'):
            raise RuntimeError("Parameters required for predicting the output words not found.")

        word2_indices = [self.wv.get_index(w) for w in context_words_list if w in self.wv]
        if not word2_indices:
            logger.warning("All the input context words are out-of-vocabulary for the current model.")
            return None

        l1 = np.sum(self.wv.vectors[word2_indices], axis=0)
        if word2_indices and self.cbow_mean:
            l1 /= len(word2_indices)

        # propagate hidden -> output and take softmax to get probabilities
        prob_values = np.exp(np.dot(l1, self.syn1neg.T))
        prob_values /= sum(prob_values)
        top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)
        # returning the most probable output words with their probabilities
        return [(self.wv.index_to_key[index1], prob_values[index1]) for index1 in top_indices]

    def reset_from(self, other_model):
        """Borrow shareable pre-built structures from `other_model` and reset hidden layer weights.

        Structures copied are:
            * Vocabulary
            * Index to word mapping
            * Cumulative frequency table (used for negative sampling)
            * Cached corpus length

        Useful when testing multiple models on the same corpus in parallel. However, as the models
        then share all vocabulary-related structures other than vectors, neither should then
        expand their vocabulary (which could leave the other in an inconsistent, broken state).
        And, any changes to any per-word 'vecattr' will affect both models.


        Parameters
        ----------
        other_model : :class:`~gensim.models.word2vec.Word2Vec`
            Another model to copy the internal structures from.

        """
        self.wv = KeyedVectors(self.vector_size)
        self.wv.index_to_key = other_model.wv.index_to_key
        self.wv.key_to_index = other_model.wv.key_to_index
        self.wv.expandos = other_model.wv.expandos
        self.cum_table = other_model.cum_table
        self.corpus_count = other_model.corpus_count
        self.init_weights()

    def __str__(self):
        """Human readable representation of the model's state.

        Returns
        -------
        str
            Human readable representation of the model's state, including the vocabulary size, vector size
            and learning rate.

        """
        return "%s(vocab=%s, vector_size=%s, alpha=%s)" % (
            self.__class__.__name__, len(self.wv.index_to_key), self.wv.vector_size, self.alpha,
        )

    def save(self, *args, **kwargs):
        """Save the model.
        This saved model can be loaded again using :func:`~gensim.models.word2vec.Word2Vec.load`, which supports
        online training and getting vectors for vocabulary words.

        Parameters
        ----------
        fname : str
            Path to the file.

        """
        super(Word2Vec, self).save(*args, **kwargs)

    def _save_specials(self, fname, separately, sep_limit, ignore, pickle_protocol, compress, subname):
        """Arrange any special handling for the `gensim.utils.SaveLoad` protocol."""
        # don't save properties that are merely calculated from others
        ignore = set(ignore).union(['cum_table', ])
        return super(Word2Vec, self)._save_specials(
            fname, separately, sep_limit, ignore, pickle_protocol, compress, subname)

    @classmethod
    def load(cls, *args, rethrow=False, **kwargs):
        """Load a previously saved :class:`~gensim.models.word2vec.Word2Vec` model.

        See Also
        --------
        :meth:`~gensim.models.word2vec.Word2Vec.save`
            Save model.

        Parameters
        ----------
        fname : str
            Path to the saved file.

        Returns
        -------
        :class:`~gensim.models.word2vec.Word2Vec`
            Loaded model.

        """
        try:
            model = super(Word2Vec, cls).load(*args, **kwargs)
            if not isinstance(model, Word2Vec):
                rethrow = True
                raise AttributeError("Model of type %s can't be loaded by %s" % (type(model), str(cls)))
            return model
        except AttributeError as ae:
            if rethrow:
                raise ae
            logger.error(
                "Model load error. Was model saved using code from an older Gensim Version? "
                "Try loading older model using gensim-3.8.3, then re-saving, to restore "
                "compatibility with current code.")
            raise ae

    def _load_specials(self, *args, **kwargs):
        """Handle special requirements of `.load()` protocol, usually up-converting older versions."""
        super(Word2Vec, self)._load_specials(*args, **kwargs)
        # for backward compatibility, add/rearrange properties from prior versions
        if not hasattr(self, 'ns_exponent'):
            self.ns_exponent = 0.75
        if self.negative and hasattr(self.wv, 'index_to_key'):
            self.make_cum_table()  # rebuild cum_table from vocabulary
        if not hasattr(self, 'corpus_count'):
            self.corpus_count = None
        if not hasattr(self, 'corpus_total_words'):
            self.corpus_total_words = None
        if not hasattr(self.wv, 'vectors_lockf') and hasattr(self.wv, 'vectors'):
            self.wv.vectors_lockf = np.ones(1, dtype=REAL)
        if not hasattr(self, 'random'):
            # use new instance of numpy's recommended generator/algorithm
            self.random = np.random.default_rng(seed=self.seed)
        if not hasattr(self, 'train_count'):
            self.train_count = 0
            self.total_train_time = 0
        if not hasattr(self, 'epochs'):
            self.epochs = self.iter
            del self.iter
        if not hasattr(self, 'max_final_vocab'):
            self.max_final_vocab = None
        if hasattr(self, 'vocabulary'):  # re-integrate state that had been moved
            for a in ('max_vocab_size', 'min_count', 'sample', 'sorted_vocab', 'null_word', 'raw_vocab'):
                setattr(self, a, getattr(self.vocabulary, a))
            del self.vocabulary
        if hasattr(self, 'trainables'):  # re-integrate state that had been moved
            for a in ('hashfxn', 'layer1_size', 'seed', 'syn1neg', 'syn1'):
                if hasattr(self.trainables, a):
                    setattr(self, a, getattr(self.trainables, a))
            if hasattr(self, 'syn1'):
                self.syn1 = self.syn1
                del self.syn1
            del self.trainables

    def get_latest_training_loss(self):
        """Get current value of the training loss.

        Returns
        -------
        float
            Current training loss.

        """
        return self.running_training_loss


class BrownCorpus:
    def __init__(self, dirname):
        """Iterate over sentences from the `Brown corpus <https://en.wikipedia.org/wiki/Brown_Corpus>`_
        (part of `NLTK data <https://www.nltk.org/data.html>`_).

        """
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            with utils.open(fname, 'rb') as fin:
                for line in fin:
                    line = utils.to_unicode(line)
                    # each file line is a single sentence in the Brown corpus
                    # each token is WORD/POS_TAG
                    token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                    # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                    words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                    if not words:  # don't bother sending out empty sentences
                        continue
                    yield words


class Text8Corpus:
    def __init__(self, fname, max_sentence_length=MAX_WORDS_IN_BATCH):
        """Iterate over sentences from the "text8" corpus, unzipped from http://mattmahoney.net/dc/text8.zip."""
        self.fname = fname
        self.max_sentence_length = max_sentence_length

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest = [], b''
        with utils.open(self.fname, 'rb') as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    words = utils.to_unicode(text).split()
                    sentence.extend(words)  # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(b' ')  # last token may have been split in two... keep for next iteration
                words, rest = (utils.to_unicode(text[:last_token]).split(),
                               text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= self.max_sentence_length:
                    yield sentence[:self.max_sentence_length]
                    sentence = sentence[self.max_sentence_length:]


class LineSentence:
    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        """Iterate over a file that contains sentences: one line = one sentence.
        Words must be already preprocessed and separated by whitespace.

        Parameters
        ----------
        source : string or a file-like object
            Path to the file on disk, or an already-open file object (must support `seek(0)`).
        limit : int or None
            Clip the file to the first `limit` lines. Do no clipping if `limit is None` (the default).

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> sentences = LineSentence(datapath('lee_background.cor'))
            >>> for sentence in sentences:
            ...     pass

        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                line = utils.to_unicode(line).split()
                i = 0
                while i < len(line):
                    yield line[i: i + self.max_sentence_length]
                    i += self.max_sentence_length
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.open(self.source, 'rb') as fin:
                for line in itertools.islice(fin, self.limit):
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i: i + self.max_sentence_length]
                        i += self.max_sentence_length


class PathLineSentences:
    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        """Like :class:`~gensim.models.word2vec.LineSentence`, but process all files in a directory
        in alphabetical order by filename.

        The directory must only contain files that can be read by :class:`gensim.models.word2vec.LineSentence`:
        .bz2, .gz, and text files. Any file not ending with .bz2 or .gz is assumed to be a text file.

        The format of files (either text, or compressed text files) in the path is one sentence = one line,
        with words already preprocessed and separated by whitespace.

        Warnings
        --------
        Does **not recurse** into subdirectories.

        Parameters
        ----------
        source : str
            Path to the directory.
        limit : int or None
            Read only the first `limit` lines from each file. Read all if limit is None (the default).

        """
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

        if os.path.isfile(self.source):
            logger.debug('single file given as source, rather than a directory of files')
            logger.debug('consider using models.word2vec.LineSentence for a single file')
            self.input_files = [self.source]  # force code compatibility with list of files
        elif os.path.isdir(self.source):
            self.source = os.path.join(self.source, '')  # ensures os-specific slash at end of path
            logger.info('reading directory %s', self.source)
            self.input_files = os.listdir(self.source)
            self.input_files = [self.source + filename for filename in self.input_files]  # make full paths
            self.input_files.sort()  # makes sure it happens in filename order
        else:  # not a file or a directory, then we can't do anything with it
            raise ValueError('input is neither a file nor a path')
        logger.info('files read into PathLineSentences:%s', '\n'.join(self.input_files))

    def __iter__(self):
        """iterate through the files"""
        for file_name in self.input_files:
            logger.info('reading file %s', file_name)
            with utils.open(file_name, 'rb') as fin:
                for line in itertools.islice(fin, self.limit):
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i:i + self.max_sentence_length]
                        i += self.max_sentence_length


class Word2VecVocab(utils.SaveLoad):
    """Obsolete class retained for now as load-compatibility state capture."""
    pass


class Word2VecTrainables(utils.SaveLoad):
    """Obsolete class retained for now as load-compatibility state capture."""
    pass


class Heapitem(namedtuple('Heapitem', 'count, index, left, right')):
    def __lt__(self, other):
        return self.count < other.count


def _build_heap(wv):
    heap = list(Heapitem(wv.get_vecattr(i, 'count'), i, None, None) for i in range(len(wv.index_to_key)))
    heapq.heapify(heap)
    for i in range(len(wv) - 1):
        min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(
            heap, Heapitem(count=min1.count + min2.count, index=i + len(wv), left=min1, right=min2)
        )
    return heap


def _assign_binary_codes(wv):
    """
    Appends a binary code to each vocab term.

    Parameters
    ----------
    wv : KeyedVectors
        A collection of word-vectors.

    Sets the .code and .point attributes of each node.
    Each code is a numpy.array containing 0s and 1s.
    Each point is an integer.

    """
    logger.info("constructing a huffman tree from %i words", len(wv))

    heap = _build_heap(wv)
    if not heap:
        #
        # TODO: how can we end up with an empty heap?
        #
        logger.info("built huffman tree with maximum node depth 0")
        return

    # recurse over the tree, assigning a binary code to each vocabulary word
    max_depth = 0
    stack = [(heap[0], [], [])]
    while stack:
        node, codes, points = stack.pop()
        if node[1] < len(wv):  # node[1] = index
            # leaf node => store its path from the root
            k = node[1]
            wv.set_vecattr(k, 'code', codes)
            wv.set_vecattr(k, 'point', points)
            # node.code, node.point = codes, points
            max_depth = max(len(codes), max_depth)
        else:
            # inner node => continue recursion
            points = np.array(list(points) + [node.index - len(wv)], dtype=np.uint32)
            stack.append((node.left, np.array(list(codes) + [0], dtype=np.uint8), points))
            stack.append((node.right, np.array(list(codes) + [1], dtype=np.uint8), points))

    logger.info("built huffman tree with maximum node depth %i", max_depth)


# Example: ./word2vec.py -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 \
# -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3
if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )
    logger.info("running %s", " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    from gensim.models.word2vec import Word2Vec  # noqa:F811 avoid referencing __main__ in pickle

    np.seterr(all='raise')  # don't ignore numpy errors

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", help="Use text data from file TRAIN to train the model", required=True)
    parser.add_argument("-output", help="Use file OUTPUT to save the resulting word vectors")
    parser.add_argument("-window", help="Set max skip length WINDOW between words; default is 5", type=int, default=5)
    parser.add_argument("-size", help="Set size of word vectors; default is 100", type=int, default=100)
    parser.add_argument(
        "-sample",
        help="Set threshold for occurrence of words. "
             "Those that appear with higher frequency in the training data will be randomly down-sampled;"
             " default is 1e-3, useful range is (0, 1e-5)",
        type=float, default=1e-3
    )
    parser.add_argument(
        "-hs", help="Use Hierarchical Softmax; default is 0 (not used)",
        type=int, default=0, choices=[0, 1]
    )
    parser.add_argument(
        "-negative", help="Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)",
        type=int, default=5
    )
    parser.add_argument("-threads", help="Use THREADS threads (default 12)", type=int, default=12)
    parser.add_argument("-iter", help="Run more training iterations (default 5)", type=int, default=5)
    parser.add_argument(
        "-min_count", help="This will discard words that appear less than MIN_COUNT times; default is 5",
        type=int, default=5
    )
    parser.add_argument(
        "-cbow", help="Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)",
        type=int, default=1, choices=[0, 1]
    )
    parser.add_argument(
        "-binary", help="Save the resulting vectors in binary mode; default is 0 (off)",
        type=int, default=0, choices=[0, 1]
    )
    parser.add_argument("-accuracy", help="Use questions from file ACCURACY to evaluate the model")

    args = parser.parse_args()

    if args.cbow == 0:
        skipgram = 1
    else:
        skipgram = 0

    corpus = LineSentence(args.train)

    model = Word2Vec(
        corpus, vector_size=args.size, min_count=args.min_count, workers=args.threads,
        window=args.window, sample=args.sample, sg=skipgram, hs=args.hs,
        negative=args.negative, cbow_mean=1, epochs=args.iter,
    )

    if args.output:
        outfile = args.output
        model.wv.save_word2vec_format(outfile, binary=args.binary)
    else:
        outfile = args.train
        model.save(outfile + '.model')
    if args.binary == 1:
        model.wv.save_word2vec_format(outfile + '.model.bin', binary=True)
    else:
        model.wv.save_word2vec_format(outfile + '.model.txt', binary=False)

    if args.accuracy:
        model.accuracy(args.accuracy)

    logger.info("finished running %s", program)
