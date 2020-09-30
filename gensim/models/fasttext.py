#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors: Gensim Contributors
# Copyright (C) 2018 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Introduction
------------
Learn word representations via fastText: `Enriching Word Vectors with Subword Information
<https://arxiv.org/abs/1607.04606>`_.

This module allows training word embeddings from a training corpus with the additional ability to obtain word vectors
for out-of-vocabulary words.

This module contains a fast native C implementation of fastText with Python interfaces. It is **not** only a wrapper
around Facebook's implementation.

This module supports loading models trained with Facebook's fastText implementation.
It also supports continuing training from such models.

For a tutorial see :ref:`sphx_glr_auto_examples_tutorials_run_fasttext.py`.


Usage examples
--------------

Initialize and train a model:

.. sourcecode:: pycon

    >>> # from gensim.models import FastText  # FIXME: why does Sphinx dislike this import?
    >>> from gensim.test.utils import common_texts  # some example sentences
    >>>
    >>> print(common_texts[0])
    ['human', 'interface', 'computer']
    >>> print(len(common_texts))
    9
    >>> model = FastText(vector_size=4, window=3, min_count=1)  # instantiate
    >>> model.build_vocab(sentences=common_texts)
    >>> model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)  # train

Once you have a model, you can access its keyed vectors via the `model.wv` attributes.
The keyed vectors instance is quite powerful: it can perform a wide range of NLP tasks.
For a full list of examples, see :class:`~gensim.models.keyedvectors.KeyedVectors`.

You can also pass all the above parameters to the constructor to do everything
in a single line:

.. sourcecode:: pycon

    >>> model2 = FastText(vector_size=4, window=3, min_count=1, sentences=common_texts, iter=10)

.. Important::
    This style of initialize-and-train in a single line is **deprecated**. We include it here
    for backward compatibility only.

    Please use the initialize-`build_vocab`-`train` pattern above instead, including using `epochs`
    instead of `iter`.
    The motivation is to simplify the API and resolve naming inconsistencies,
    e.g. the iter parameter to the constructor is called epochs in the train function.

The two models above are instantiated differently, but behave identically.
For example, we can compare the embeddings they've calculated for the word "computer":

.. sourcecode:: pycon

    >>> import numpy as np
    >>>
    >>> np.allclose(model.wv['computer'], model2.wv['computer'])
    True


In the above examples, we trained the model from sentences (lists of words) loaded into memory.
This is OK for smaller datasets, but for larger datasets, we recommend streaming the file,
for example from disk or the network.
In Gensim, we refer to such datasets as "corpora" (singular "corpus"), and keep them
in the format described in :class:`~gensim.models.word2vec.LineSentence`.
Passing a corpus is simple:

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> corpus_file = datapath('lee_background.cor')  # absolute path to corpus
    >>> model3 = FastText(vector_size=4, window=3, min_count=1)
    >>> model3.build_vocab(corpus_file=corpus_file)  # scan over corpus to build the vocabulary
    >>>
    >>> total_words = model3.corpus_total_words  # number of words in the corpus
    >>> model3.train(corpus_file=corpus_file, total_words=total_words, epochs=5)

The model needs the `total_words` parameter in order to
manage the training rate (alpha) correctly, and to give accurate progress estimates.
The above example relies on an implementation detail: the
:meth:`~gensim.models.fasttext.FastText.build_vocab` method
sets the `corpus_total_words` (and also `corpus_count`) model attributes.
You may calculate them by scanning over the corpus yourself, too.

If you have a corpus in a different format, then you can use it by wrapping it
in an `iterator <https://wiki.python.org/moin/Iterator>`_.
Your iterator should yield a list of strings each time, where each string should be a separate word.
Gensim will take care of the rest:

.. sourcecode:: pycon

    >>> from gensim.utils import tokenize
    >>> from gensim import utils
    >>>
    >>>
    >>> class MyIter(object):
    ...     def __iter__(self):
    ...         path = datapath('crime-and-punishment.txt')
    ...         with utils.open(path, 'r', encoding='utf-8') as fin:
    ...             for line in fin:
    ...                 yield list(tokenize(line))
    >>>
    >>>
    >>> model4 = FastText(vector_size=4, window=3, min_count=1)
    >>> model4.build_vocab(sentences=MyIter())
    >>> total_examples = model4.corpus_count
    >>> model4.train(sentences=MyIter(), total_examples=total_examples, epochs=5)

Persist a model to disk with:

.. sourcecode:: pycon

    >>> from gensim.test.utils import get_tmpfile
    >>>
    >>> fname = get_tmpfile("fasttext.model")
    >>>
    >>> model.save(fname)
    >>> model = FastText.load(fname)

Once loaded, such models behave identically to those created from scratch.
For example, you can continue training the loaded model:

.. sourcecode:: pycon

    >>> import numpy as np
    >>>
    >>> 'computation' in model.wv.vocab  # New word, currently out of vocab
    False
    >>> old_vector = np.copy(model.wv['computation'])  # Grab the existing vector
    >>> new_sentences = [
    ...     ['computer', 'aided', 'design'],
    ...     ['computer', 'science'],
    ...     ['computational', 'complexity'],
    ...     ['military', 'supercomputer'],
    ...     ['central', 'processing', 'unit'],
    ...     ['onboard', 'car', 'computer'],
    ... ]
    >>>
    >>> model.build_vocab(new_sentences, update=True)  # Update the vocabulary
    >>> model.train(new_sentences, total_examples=len(new_sentences), epochs=model.epochs)
    >>>
    >>> new_vector = model.wv['computation']
    >>> np.allclose(old_vector, new_vector, atol=1e-4)  # Vector has changed, model has learnt something
    False
    >>> 'computation' in model.wv.vocab  # Word is still out of vocab
    False

.. Important::
    Be sure to call the :meth:`~gensim.models.fasttext.FastText.build_vocab`
    method with `update=True` before the :meth:`~gensim.models.fasttext.FastText.train` method
    when continuing training.  Without this call, previously unseen terms
    will not be added to the vocabulary.

You can also load models trained with Facebook's fastText implementation:

.. sourcecode:: pycon

    >>> cap_path = datapath("crime-and-punishment.bin")
    >>> fb_model = load_facebook_model(cap_path)

Once loaded, such models behave identically to those trained from scratch.
You may continue training them on new data:

.. sourcecode:: pycon

    >>> 'computer' in fb_model.wv.vocab  # New word, currently out of vocab
    False
    >>> old_computer = np.copy(fb_model.wv['computer'])  # Calculate current vectors
    >>> fb_model.build_vocab(new_sentences, update=True)
    >>> fb_model.train(new_sentences, total_examples=len(new_sentences), epochs=model.epochs)
    >>> new_computer = fb_model.wv['computer']
    >>> np.allclose(old_computer, new_computer, atol=1e-4)  # Vector has changed, model has learnt something
    False
    >>> 'computer' in fb_model.wv.vocab  # New word is now in the vocabulary
    True

If you do not intend to continue training the model, consider using the
:func:`gensim.models.fasttext.load_facebook_vectors` function instead.
That function only loads the word embeddings (keyed vectors), consuming much less CPU and RAM:

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> cap_path = datapath("crime-and-punishment.bin")
    >>> wv = load_facebook_vectors(cap_path)
    >>>
    >>> 'landlord' in wv.vocab  # Word is out of vocabulary
    False
    >>> oov_vector = wv['landlord']
    >>>
    >>> 'landlady' in wv.vocab  # Word is in the vocabulary
    True
    >>> iv_vector = wv['landlady']

Retrieve word-vector for vocab and out-of-vocab word:

.. sourcecode:: pycon

    >>> existent_word = "computer"
    >>> existent_word in model.wv.vocab
    True
    >>> computer_vec = model.wv[existent_word]  # numpy vector of a word
    >>>
    >>> oov_word = "graph-out-of-vocab"
    >>> oov_word in model.wv.vocab
    False
    >>> oov_vec = model.wv[oov_word]  # numpy vector for OOV word

You can perform various NLP word tasks with the model, some of them are already built-in:

.. sourcecode:: pycon

    >>> similarities = model.wv.most_similar(positive=['computer', 'human'], negative=['interface'])
    >>> most_similar = similarities[0]
    >>>
    >>> similarities = model.wv.most_similar_cosmul(positive=['computer', 'human'], negative=['interface'])
    >>> most_similar = similarities[0]
    >>>
    >>> not_matching = model.wv.doesnt_match("human computer interface tree".split())
    >>>
    >>> sim_score = model.wv.similarity('computer', 'human')

Correlation with human opinion on word similarity:

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))

And on word analogies:

.. sourcecode:: pycon

    >>> analogies_result = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))

"""

import logging
import os
from collections.abc import Iterable

import numpy as np
from numpy import ones, vstack, float32 as REAL

import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim import utils
from gensim.utils import deprecated
try:
    from gensim.models.fasttext_inner import (  # noqa: F401
        train_batch_any,
        MAX_WORDS_IN_BATCH,
        compute_ngrams,
        compute_ngrams_bytes,
        ft_hash_bytes,
    )
    from gensim.models.fasttext_corpusfile import train_epoch_sg, train_epoch_cbow
except ImportError:
    raise utils.NO_CYTHON


logger = logging.getLogger(__name__)


class FastText(Word2Vec):

    def __init__(self, sentences=None, corpus_file=None, sg=0, hs=0, vector_size=100, alpha=0.025,
                 window=5, min_count=5,
                 max_vocab_size=None, word_ngrams=1, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, epochs=5, null_word=0, min_n=3, max_n=6,
                 sorted_vocab=1, bucket=2000000, trim_rule=None, batch_words=MAX_WORDS_IN_BATCH, callbacks=(),
                 max_final_vocab=None):
        """Train, use and evaluate word representations learned using the method
        described in `Enriching Word Vectors with Subword Information <https://arxiv.org/abs/1607.04606>`_,
        aka FastText.

        The model can be stored/loaded via its :meth:`~gensim.models.fasttext.FastText.save` and
        :meth:`~gensim.models.fasttext.FastText.load` methods, or loaded from a format compatible with the
        original Fasttext implementation via :func:`~gensim.models.fasttext.load_facebook_model`.

        Parameters
        ----------
        sentences : iterable of list of str, optional
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus'
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such
            examples. If you don't supply `sentences`, the model is left uninitialized -- use if you plan to
            initialize it in some other way.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `sentences` to get performance boost. Only one of `sentences` or
            `corpus_file` arguments need to be passed (or none of them, in that case, the model is left
            uninitialized).
        min_count : int, optional
            The model ignores all words with total frequency lower than this.
        size : int, optional
            Dimensionality of the word vectors.
        window : int, optional
            The maximum distance between the current and predicted word within a sentence.
        workers : int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).
        alpha : float, optional
            The initial learning rate.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        sg : {1, 0}, optional
            Training algorithm: skip-gram if `sg=1`, otherwise CBOW.
        hs : {1,0}, optional
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
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
        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
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

        sorted_vocab : {1,0}, optional
            If 1, sort the vocabulary by descending frequency before assigning word indices.
        batch_words : int, optional
            Target size (in words) for batches of examples passed to worker threads (and
            thus cython routines).(Larger batches will be passed if individual
            texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
        min_n : int, optional
            Minimum length of char n-grams to be used for training word representations.
        max_n : int, optional
            Max length of char ngrams to be used for training word representations. Set `max_n` to be
            lesser than `min_n` to avoid char ngrams being used.
        word_ngrams : int, optional
            In Facebook's FastText, "max length of word ngram" - but gensim only supports the
            default of 1 (regular unigram word handling).
        bucket : int, optional
            Character ngrams are hashed into a fixed number of buckets, in order to limit the
            memory usage of the model. This option specifies the number of buckets used by the model.
            The default value of 2000000 consumes as much memory as having 2000000 more in-vocabulary
            words in your model.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`, optional
            List of callbacks that need to be executed/run at specific stages during training.
        max_final_vocab : int, optional
            Limits the vocab to a target vocab size by automatically selecting
            ``min_count```.  If the specified ``min_count`` is more than the
            automatically calculated ``min_count``, the former will be used.
            Set to ``None`` if not required.

        Examples
        --------
        Initialize and train a `FastText` model:

        .. sourcecode:: pycon

            >>> from gensim.models import FastText
            >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
            >>>
            >>> model = FastText(sentences, min_count=1)
            >>> say_vector = model.wv['say']  # get vector for word
            >>> of_vector = model.wv['of']  # get vector for out-of-vocab word

        Attributes
        ----------
        wv : :class:`~gensim.models.fasttext.FastTextKeyedVectors`
            This object essentially contains the mapping between words and embeddings. These are similar to
            the embedding computed in the :class:`~gensim.models.word2vec.Word2Vec`, however here we also
            include vectors for n-grams. This allows the model to compute embeddings even for **unseen**
            words (that do not exist in the vocabulary), as the aggregate of the n-grams included in the word.
            After training the model, this attribute can be used directly to query those embeddings in various
            ways. Check the module level docstring for some examples.

        """
        self.load = utils.call_on_class_only
        self.load_fasttext_format = utils.call_on_class_only
        self.callbacks = callbacks
        if word_ngrams != 1:
            raise NotImplementedError("Gensim's FastText implementation does not yet support word_ngrams != 1.")
        self.word_ngrams = word_ngrams
        if max_n < min_n:
            # with no eligible char-ngram lengths, no buckets need be allocated
            bucket = 0

        self.wv = FastTextKeyedVectors(vector_size, min_n, max_n, bucket)
        self.wv.bucket = bucket

        super(FastText, self).__init__(
            sentences=sentences, corpus_file=corpus_file, workers=workers, vector_size=vector_size, epochs=epochs,
            callbacks=callbacks, batch_words=batch_words, trim_rule=trim_rule, sg=sg, alpha=alpha, window=window,
            max_vocab_size=max_vocab_size, max_final_vocab=max_final_vocab,
            min_count=min_count, sample=sample, sorted_vocab=sorted_vocab,
            null_word=null_word, ns_exponent=ns_exponent, hashfxn=hashfxn,
            seed=seed, hs=hs, negative=negative, cbow_mean=cbow_mean, min_alpha=min_alpha)

    def prepare_weights(self, update=False):
        """In addition to superclass allocations, compute ngrams of all words present in vocabulary.

        Parameters
        ----------
        update : bool
            If True, the new vocab words and their new ngrams word vectors are initialized
            with random uniform distribution and updated/added to the existing vocab word and ngram vectors.
        """
        super(FastText, self).prepare_weights(update=update)
        if not update:
            self.wv.init_ngrams_weights(self.seed)
            # EXPERIMENTAL lockf feature; create minimal no-op lockf arrays (1 element of 1.0)
            # advanced users should directly resize/adjust as necessary
            self.wv.vectors_vocab_lockf = ones(1, dtype=REAL)
            self.wv.vectors_ngrams_lockf = ones(1, dtype=REAL)
        else:
            self.wv.update_ngrams_weights(self.seed, self.old_vocab_len)
            # EXPERIMENTAL lockf feature; create minimal no-op lockf arrays (1 element of 1.0)
            # advanced users should directly resize/adjust as necessary
            self.wv.vectors_vocab_lockf = ones(1, dtype=REAL)
            self.wv.vectors_ngrams_lockf = ones(1, dtype=REAL)

    def _init_post_load(self, hidden_output):
        num_vectors = len(self.wv.vectors)
        vocab_size = len(self.wv)
        vector_size = self.wv.vector_size

        assert num_vectors > 0, 'expected num_vectors to be initialized already'
        assert vocab_size > 0, 'expected vocab_size to be initialized already'

        # EXPERIMENTAL lockf feature; create minimal no-op lockf arrays (1 element of 1.0)
        # advanced users should directly resize/adjust as necessary
        self.wv.vectors_ngrams_lockf = ones(1, dtype=REAL)
        self.wv.vectors_vocab_lockf = ones(1, dtype=REAL)

        if self.hs:
            self.syn1 = hidden_output
        if self.negative:
            self.syn1neg = hidden_output

        self.layer1_size = vector_size

    def build_vocab(self, corpus_iterable=None, corpus_file=None, update=False, progress_per=10000,
                    keep_raw_vocab=False, trim_rule=None, **kwargs):
        """Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        Parameters
        ----------
        corpus_iterable : iterable of list of str, optional
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `sentences` to get performance boost. Only one of `sentences` or
            `corpus_file` arguments need to be passed (not both of them).
        update : bool
            If true, the new words in `sentences` will be added to model's vocab.
        progress_per : int
            Indicates how many words to process before showing/updating the progress.
        keep_raw_vocab : bool
            If not true, delete the raw vocabulary after the scaling is done and free up RAM.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during
            :meth:`~gensim.models.fasttext.FastText.build_vocab` and is not stored as part of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        **kwargs
            Additional key word parameters passed to
            :meth:`~gensim.models.word2vec.Word2Vec.build_vocab`.

        Examples
        --------
        Train a model and update vocab for online training:

        .. sourcecode:: pycon

            >>> from gensim.models import FastText
            >>> sentences_1 = [["cat", "say", "meow"], ["dog", "say", "woof"]]
            >>> sentences_2 = [["dude", "say", "wazzup!"]]
            >>>
            >>> model = FastText(min_count=1)
            >>> model.build_vocab(sentences_1)
            >>> model.train(sentences_1, total_examples=model.corpus_count, epochs=model.epochs)
            >>>
            >>> model.build_vocab(sentences_2, update=True)
            >>> model.train(sentences_2, total_examples=model.corpus_count, epochs=model.epochs)

        """
        if not update:
            self.wv.init_ngrams_weights(self.seed)
        elif not len(self.wv):
            raise RuntimeError(
                "You cannot do an online vocabulary-update of a model which has no prior vocabulary. "
                "First build the vocabulary of your model with a corpus "
                "by calling the gensim.models.fasttext.FastText.build_vocab method "
                "before doing an online update."
            )
        else:
            self.old_vocab_len = len(self.wv)

        retval = super(FastText, self).build_vocab(
            corpus_iterable=corpus_iterable, corpus_file=corpus_file, update=update, progress_per=progress_per,
            keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, **kwargs)

        if update:
            self.wv.update_ngrams_weights(self.seed, self.old_vocab_len)

        return retval

    def _clear_post_train(self):
        """Clear the model's internal structures after training has finished to free up RAM."""
        self.wv.adjust_vectors()  # ensure composite-word vecs reflect latest training

    def estimate_memory(self, vocab_size=None, report=None):
        """Estimate memory that will be needed to train a model, and print the estimates to log."""
        vocab_size = vocab_size or len(self.wv)
        vec_size = self.vector_size * np.dtype(np.float32).itemsize
        l1_size = self.layer1_size * np.dtype(np.float32).itemsize
        report = report or {}
        report['vocab'] = len(self.wv) * (700 if self.hs else 500)
        report['syn0_vocab'] = len(self.wv) * vec_size
        num_buckets = self.wv.bucket
        if self.hs:
            report['syn1'] = len(self.wv) * l1_size
        if self.negative:
            report['syn1neg'] = len(self.wv) * l1_size
        if self.wv.bucket:
            report['syn0_ngrams'] = self.wv.bucket * vec_size
            num_ngrams = 0
            for word in self.wv.key_to_index:
                hashes = ft_ngram_hashes(word, self.wv.min_n, self.wv.max_n, self.wv.bucket)
                num_ngrams += len(hashes)
            # A list (64 bytes) with one np.array (100 bytes) per key, with a total of
            # num_ngrams uint32s (4 bytes) amongst them
            # Only used during training, not stored with the model
            report['buckets_word'] = 64 + (100 * len(self.wv)) + (4 * num_ngrams)  # FIXME: caching & calc sensible?
        report['total'] = sum(report.values())
        logger.info(
            "estimated required memory for %i words, %i buckets and %i dimensions: %i bytes",
            len(self.wv), num_buckets, self.vector_size, report['total'],
        )
        return report

    def _do_train_epoch(self, corpus_file, thread_id, offset, cython_vocab, thread_private_mem, cur_epoch,
                        total_examples=None, total_words=None, **kwargs):
        work, neu1 = thread_private_mem

        if self.sg:
            examples, tally, raw_tally = train_epoch_sg(self, corpus_file, offset, cython_vocab, cur_epoch,
                                                        total_examples, total_words, work, neu1)
        else:
            examples, tally, raw_tally = train_epoch_cbow(self, corpus_file, offset, cython_vocab, cur_epoch,
                                                          total_examples, total_words, work, neu1)

        return examples, tally, raw_tally

    def _do_train_job(self, sentences, alpha, inits):
        """Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.

        Parameters
        ----------
        sentences : iterable of list of str
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        alpha : float
            The current learning rate.
        inits : tuple of (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Each worker's private work memory.

        Returns
        -------
        (int, int)
            Tuple of (effective word count after ignoring unknown words and sentence length trimming, total word count)

        """
        work, neu1 = inits
        tally = train_batch_any(self, sentences, alpha, work, neu1)

        return tally, self._raw_word_count(sentences)

    def train(self, corpus_iterable=None, corpus_file=None, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None,
              word_count=0, queue_factor=2, report_delay=1.0, callbacks=(), **kwargs):
        """Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For FastText, each sentence must be a list of unicode strings.

        To support linear learning-rate decay from (initial) `alpha` to `min_alpha`, and accurate
        progress-percentage logging, either `total_examples` (count of sentences) or `total_words` (count of
        raw words in sentences) **MUST** be provided. If `sentences` is the same corpus
        that was provided to :meth:`~gensim.models.fasttext.FastText.build_vocab` earlier,
        you can simply use `total_examples=self.corpus_count`.

        To avoid common mistakes around the model's ability to do multiple training passes itself, an
        explicit `epochs` argument **MUST** be provided. In the common and recommended case
        where :meth:`~gensim.models.fasttext.FastText.train` is only called once, you can set `epochs=self.iter`.

        Parameters
        ----------
        sentences : iterable of list of str, optional
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            If you use this argument instead of `sentences`, you must provide `total_words` argument as well. Only one
            of `sentences` or `corpus_file` arguments need to be passed (not both of them).
        total_examples : int
            Count of sentences.
        total_words : int
            Count of raw words in sentences.
        epochs : int
            Number of iterations (epochs) over the corpus.
        start_alpha : float, optional
            Initial learning rate. If supplied, replaces the starting `alpha` from the constructor,
            for this one call to :meth:`~gensim.models.fasttext.FastText.train`.
            Use only if making multiple calls to :meth:`~gensim.models.fasttext.FastText.train`, when you want to manage
            the alpha learning-rate yourself (not recommended).
        end_alpha : float, optional
            Final learning rate. Drops linearly from `start_alpha`.
            If supplied, this replaces the final `min_alpha` from the constructor, for this one call to
            :meth:`~gensim.models.fasttext.FastText.train`.
            Use only if making multiple calls to :meth:`~gensim.models.fasttext.FastText.train`, when you want to manage
            the alpha learning-rate yourself (not recommended).
        word_count : int
            Count of words already trained. Set this to 0 for the usual
            case of training on all words in sentences.
        queue_factor : int
            Multiplier for size of queue (number of workers * queue_factor).
        report_delay : float
            Seconds to wait before reporting progress.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`
            List of callbacks that need to be executed/run at specific stages during training.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.models import FastText
            >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
            >>>
            >>> model = FastText(min_count=1)
            >>> model.build_vocab(sentences)
            >>> model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

        """

        if corpus_file is None and corpus_iterable is None:
            raise TypeError("Either one of corpus_file or corpus_iterable value must be provided")

        if corpus_file is not None and corpus_iterable is not None:
            raise TypeError("Both corpus_file and corpus_iterable must not be provided at the same time")

        if corpus_iterable is None and not os.path.isfile(corpus_file):
            raise TypeError("Parameter corpus_file must be a valid path to a file, got %r instead" % corpus_file)

        if corpus_iterable is not None and not isinstance(corpus_iterable, Iterable):
            raise TypeError("sentences must be an iterable of list, got %r instead" % corpus_iterable)

        super(FastText, self).train(
            corpus_iterable=corpus_iterable, corpus_file=corpus_file,
            total_examples=total_examples, total_words=total_words,
            epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, callbacks=callbacks)
        self.wv.adjust_vectors()

    @deprecated(
        "Gensim 4.0.0 implemented internal optimizations that make calls to init_sims() unnecessary. "
        "init_sims() is now obsoleted and will be completely removed in future versions. "
        "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4#init_sims"
    )
    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors. Obsoleted.

        If you need a single unit-normalized vector for some key, call
        :meth:`~gensim.models.keyedvectors.KeyedVectors.get_vector` instead:
        ``fasttext_model.wv.get_vector(key, norm=True)``.

        To refresh norms after you performed some atypical out-of-band vector tampering,
        call `:meth:`~gensim.models.keyedvectors.KeyedVectors.fill_norms()` instead.

        Parameters
        ----------
        replace : bool
            If True, forget the original trained vectors and only keep the normalized ones.
            You lose information if you do this.

        """
        self.wv.init_sims(replace=replace)

    def clear_sims(self):
        """Remove all L2-normalized word vectors from the model, to free up memory.

        You can recompute them later again using the :meth:`~gensim.models.fasttext.FastText.init_sims` method.

        """
        self._clear_post_train()

    @classmethod
    @utils.deprecated(
        'use load_facebook_vectors (to use pretrained embeddings) or load_facebook_model '
        '(to continue training with the loaded full model, more RAM) instead'
    )
    def load_fasttext_format(cls, model_file, encoding='utf8'):
        """Deprecated.

        Use :func:`gensim.models.fasttext.load_facebook_model` or
        :func:`gensim.models.fasttext.load_facebook_vectors` instead.

        """
        return load_facebook_model(model_file, encoding=encoding)

    @utils.deprecated(
        'use load_facebook_vectors (to use pretrained embeddings) or load_facebook_model '
        '(to continue training with the loaded full model, more RAM) instead'
    )
    def load_binary_data(self, encoding='utf8'):
        """Load data from a binary file created by Facebook's native FastText.

        Parameters
        ----------
        encoding : str, optional
            Specifies the encoding.

        """
        m = _load_fasttext_format(self.file_name, encoding=encoding)
        for attr, val in m.__dict__.items():
            setattr(self, attr, val)

    def save(self, *args, **kwargs):
        """Save the Fasttext model. This saved model can be loaded again using
        :meth:`~gensim.models.fasttext.FastText.load`, which supports incremental training
        and getting vectors for out-of-vocabulary words.

        Parameters
        ----------
        fname : str
            Store the model to this file.

        See Also
        --------
        :meth:`~gensim.models.fasttext.FastText.load`
            Load :class:`~gensim.models.fasttext.FastText` model.

        """
        super(FastText, self).save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a previously saved `FastText` model.

        Parameters
        ----------
        fname : str
            Path to the saved file.

        Returns
        -------
        :class:`~gensim.models.fasttext.FastText`
            Loaded model.

        See Also
        --------
        :meth:`~gensim.models.fasttext.FastText.save`
            Save :class:`~gensim.models.fasttext.FastText` model.

        """
        return super(FastText, cls).load(*args, rethrow=True, **kwargs)

    def _load_specials(self, *args, **kwargs):
        """Handle special requirements of `.load()` protocol, usually up-converting older versions."""
        super(FastText, self)._load_specials(*args, **kwargs)
        if hasattr(self, 'bucket'):
            # should only exist in one place: the wv subcomponent
            self.wv.bucket = self.bucket
            del self.bucket


class FastTextVocab(utils.SaveLoad):
    """This is a redundant class. It exists only to maintain backwards compatibility
    with older gensim versions."""


class FastTextTrainables(utils.SaveLoad):
    """Obsolete class retained for backward-compatible load()s"""


def _pad_ones(m, new_len):
    """Pad array with additional entries filled with ones."""
    if len(m) > new_len:
        raise ValueError('the new number of rows %i must be greater than old %i' % (new_len, len(m)))
    new_arr = np.ones(new_len, dtype=REAL)
    new_arr[:len(m)] = m
    return new_arr


def load_facebook_model(path, encoding='utf-8'):
    """Load the input-hidden weight matrix from Facebook's native fasttext `.bin` output file.

    Notes
    ------
    Facebook provides both `.vec` and `.bin` files with their modules.
    The former contains human-readable vectors.
    The latter contains machine-readable vectors along with other model parameters.
    This function requires you to **provide the full path to the .bin file**.
    It effectively ignores the `.vec` output file, since it is redundant.

    This function uses the smart_open library to open the path.
    The path may be on a remote host (e.g. HTTP, S3, etc).
    It may also be gzip or bz2 compressed (i.e. end in `.bin.gz` or `.bin.bz2`).
    For details, see `<https://github.com/RaRe-Technologies/smart_open>`__.

    Parameters
    ----------
    model_file : str
        Path to the FastText output files.
        FastText outputs two model files - `/path/to/model.vec` and `/path/to/model.bin`
        Expected value for this example: `/path/to/model` or `/path/to/model.bin`,
        as Gensim requires only `.bin` file to the load entire fastText model.
    encoding : str, optional
        Specifies the file encoding.

    Examples
    --------

    Load, infer, continue training:

    .. sourcecode:: pycon

        >>> from gensim.test.utils import datapath
        >>>
        >>> cap_path = datapath("crime-and-punishment.bin")
        >>> fb_model = load_facebook_model(cap_path)
        >>>
        >>> 'landlord' in fb_model.wv.vocab  # Word is out of vocabulary
        False
        >>> oov_term = fb_model.wv['landlord']
        >>>
        >>> 'landlady' in fb_model.wv.vocab  # Word is in the vocabulary
        True
        >>> iv_term = fb_model.wv['landlady']
        >>>
        >>> new_sent = [['lord', 'of', 'the', 'rings'], ['lord', 'of', 'the', 'flies']]
        >>> fb_model.build_vocab(new_sent, update=True)
        >>> fb_model.train(sentences=new_sent, total_examples=len(new_sent), epochs=5)

    Returns
    -------
    gensim.models.fasttext.FastText
        The loaded model.

    See Also
    --------
    :func:`~gensim.models.fasttext.load_facebook_vectors` loads
    the word embeddings only.  Its faster, but does not enable you to continue
    training.

    """
    return _load_fasttext_format(path, encoding=encoding, full_model=True)


def load_facebook_vectors(path, encoding='utf-8'):
    """Load word embeddings from a model saved in Facebook's native fasttext `.bin` format.

    Notes
    ------
    Facebook provides both `.vec` and `.bin` files with their modules.
    The former contains human-readable vectors.
    The latter contains machine-readable vectors along with other model parameters.
    This function requires you to **provide the full path to the .bin file**.
    It effectively ignores the `.vec` output file, since it is redundant.

    This function uses the smart_open library to open the path.
    The path may be on a remote host (e.g. HTTP, S3, etc).
    It may also be gzip or bz2 compressed.
    For details, see `<https://github.com/RaRe-Technologies/smart_open>`__.

    Parameters
    ----------
    path : str
        The location of the model file.
    encoding : str, optional
        Specifies the file encoding.

    Returns
    -------
    gensim.models.fasttext.FastTextKeyedVectors
        The word embeddings.

    Examples
    --------

    Load and infer:

        >>> from gensim.test.utils import datapath
        >>>
        >>> cap_path = datapath("crime-and-punishment.bin")
        >>> fbkv = load_facebook_vectors(cap_path)
        >>>
        >>> 'landlord' in fbkv.vocab  # Word is out of vocabulary
        False
        >>> oov_vector = fbkv['landlord']
        >>>
        >>> 'landlady' in fbkv.vocab  # Word is in the vocabulary
        True
        >>> iv_vector = fbkv['landlady']

    See Also
    --------
    :func:`~gensim.models.fasttext.load_facebook_model` loads
    the full model, not just word embeddings, and enables you to continue
    model training.

    """
    full_model = _load_fasttext_format(path, encoding=encoding, full_model=False)
    return full_model.wv


def _load_fasttext_format(model_file, encoding='utf-8', full_model=True):
    """Load the input-hidden weight matrix from Facebook's native fasttext `.bin` output files.

    Parameters
    ----------
    model_file : str
        Full path to the FastText model file.
    encoding : str, optional
        Specifies the file encoding.
    full_model : boolean, optional
        If False, skips loading the hidden output matrix. This saves a fair bit
        of CPU time and RAM, but prevents training continuation.

    Returns
    -------
    :class: `~gensim.models.fasttext.FastText`
        The loaded model.

    """
    with utils.open(model_file, 'rb') as fin:
        m = gensim.models._fasttext_bin.load(fin, encoding=encoding, full_model=full_model)

    model = FastText(
        vector_size=m.dim,
        window=m.ws,
        epochs=m.epoch,
        negative=m.neg,
        hs=int(m.loss == 1),
        sg=int(m.model == 2),
        bucket=m.bucket,
        min_count=m.min_count,
        sample=m.t,
        min_n=m.minn,
        max_n=m.maxn,
    )
    model.corpus_total_words = m.ntokens
    model.raw_vocab = m.raw_vocab
    model.nwords = m.nwords
    model.vocab_size = m.vocab_size

    #
    # This is here to fix https://github.com/RaRe-Technologies/gensim/pull/2373.
    #
    # We explicitly set min_count=1 regardless of the model's parameters to
    # ignore the trim rule when building the vocabulary.  We do this in order
    # to support loading native models that were trained with pretrained vectors.
    # Such models will contain vectors for _all_ encountered words, not only
    # those occurring more frequently than min_count.
    #
    # Native models trained _without_ pretrained vectors already contain the
    # trimmed raw_vocab, so this change does not affect them.
    #
    model.prepare_vocab(update=True, min_count=1)

    model.num_original_vectors = m.vectors_ngrams.shape[0]

    model.wv.init_post_load(m.vectors_ngrams)
    model._init_post_load(m.hidden_output)

    _check_model(model)

    logger.info("loaded %s weight matrix for fastText model from %s", m.vectors_ngrams.shape, fin.name)
    return model


def _check_model(m):
    """Model sanity checks. Run after everything has been completely initialized."""
    if m.wv.vector_size != m.wv.vectors_ngrams.shape[1]:
        raise ValueError(
            'mismatch between vector size in model params (%s) and model vectors (%s)' % (
                m.wv.vector_size, m.wv.vectors_ngrams,
            )
        )

    if hasattr(m, 'syn1neg') and m.syn1neg is not None:
        if m.wv.vector_size != m.syn1neg.shape[1]:
            raise ValueError(
                'mismatch between vector size in model params (%s) and trainables (%s)' % (
                    m.wv.vector_size, m.wv.vectors_ngrams,
                )
            )

    if len(m.wv) != m.nwords:
        raise ValueError(
            'mismatch between final vocab size (%s words), and expected number of words (%s words)' % (
                len(m.wv), m.nwords,
            )
        )

    if len(m.wv) != m.vocab_size:
        # expecting to log this warning only for pretrained french vector, wiki.fr
        logger.warning(
            "mismatch between final vocab size (%s words), and expected vocab size (%s words)",
            len(m.wv), m.vocab_size,
        )


def save_facebook_model(model, path, encoding="utf-8", lr_update_rate=100, word_ngrams=1):
    """Saves word embeddings to the Facebook's native fasttext `.bin` format.

    Notes
    ------
    Facebook provides both `.vec` and `.bin` files with their modules.
    The former contains human-readable vectors.
    The latter contains machine-readable vectors along with other model parameters.
    **This function saves only the .bin file**.

    Parameters
    ----------
    model : gensim.models.fasttext.FastText
        FastText model to be saved.
    path : str
        Output path and filename (including `.bin` extension)
    encoding : str, optional
        Specifies the file encoding. Defaults to utf-8.

    lr_update_rate : int
        This parameter is used by Facebook fasttext tool, unused by Gensim.
        It defaults to Facebook fasttext default value `100`.
        In very rare circumstances you might wish to fiddle with it.

    word_ngrams : int
        This parameter is used by Facebook fasttext tool, unused by Gensim.
        It defaults to Facebook fasttext default value `1`.
        In very rare circumstances you might wish to fiddle with it.

    Returns
    -------
    None
    """
    fb_fasttext_parameters = {"lr_update_rate": lr_update_rate, "word_ngrams": word_ngrams}
    gensim.models._fasttext_bin.save(model, path, fb_fasttext_parameters, encoding)


class FastTextKeyedVectors(KeyedVectors):
    def __init__(self, vector_size, min_n, max_n, bucket):
        """Vectors and vocab for :class:`~gensim.models.fasttext.FastText`.

        Implements significant parts of the FastText algorithm.  For example,
        the :func:`word_vec` calculates vectors for out-of-vocabulary (OOV)
        entities.  FastText achieves this by keeping vectors for ngrams:
        adding the vectors for the ngrams of an entity yields the vector for the
        entity.

        Similar to a hashmap, this class keeps a fixed number of buckets, and
        maps all ngrams to buckets using a hash function.

        Parameters
        ----------
        vector_size : int
            The dimensionality of all vectors.
        min_n : int
            The minimum number of characters in an ngram
        max_n : int
            The maximum number of characters in an ngram
        bucket : int
            The number of buckets.

        Attributes
        ----------
        vectors_vocab : np.array
            Each row corresponds to a vector for an entity in the vocabulary.
            Columns correspond to vector dimensions. When embedded in a full
            FastText model, these are the full-word-token vectors updated
            by training, whereas the inherited vectors are the actual per-word
            vectors synthesized from the full-word-token and all subword (ngram)
            vectors.
        vectors_ngrams : np.array
            A vector for each ngram across all entities in the vocabulary.
            Each row is a vector that corresponds to a bucket.
            Columns correspond to vector dimensions.
        buckets_word : list of np.array
            For each key (by its index), report bucket slots their subwords map to.

        When used in training, FastTextKeyedVectors may be decorated with
        extra attributes that closely associate with its core attributes,
        such as the experimental vectors_vocab_lockf and vectors_ngrams_lockf
        training-update-dampening factors.

        """
        super(FastTextKeyedVectors, self).__init__(vector_size=vector_size)
        self.vectors_vocab = None  # fka syn0_vocab
        self.vectors_ngrams = None  # fka syn0_ngrams
        self.buckets_word = None
        self.min_n = min_n
        self.max_n = max_n
        self.bucket = bucket  # count of buckets, fka num_ngram_vectors
        self.compatible_hash = True

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        """Load a previously saved `FastTextKeyedVectors` model.

        Parameters
        ----------
        fname : str
            Path to the saved file.

        Returns
        -------
        :class:`~gensim.models.fasttext.FastTextKeyedVectors`
            Loaded model.

        See Also
        --------
        :meth:`~gensim.models.fasttext.FastTextKeyedVectors.save`
            Save :class:`~gensim.models.fasttext.FastTextKeyedVectors` model.

        """
        return super(FastTextKeyedVectors, cls).load(fname_or_handle, **kwargs)

    def _load_specials(self, *args, **kwargs):
        """Handle special requirements of `.load()` protocol, usually up-converting older versions."""
        super(FastTextKeyedVectors, self)._load_specials(*args, **kwargs)
        if not isinstance(self, FastTextKeyedVectors):
            raise TypeError("Loaded object of type %s, not expected FastTextKeyedVectors" % type(self))
        if not hasattr(self, 'compatible_hash') or self.compatible_hash is False:
            raise TypeError(
                "Pre-gensim-3.8.x fastText models with nonstandard hashing are no longer compatible. "
                "Loading your old model into gensim-3.8.3 & re-saving may create a model compatible with gensim 4.x."
            )
        if not hasattr(self, 'vectors_vocab_lockf') and hasattr(self, 'vectors_vocab'):
            self.vectors_vocab_lockf = ones(1, dtype=REAL)
        if not hasattr(self, 'vectors_ngrams_lockf') and hasattr(self, 'vectors_ngrams'):
            self.vectors_ngrams_lockf = ones(1, dtype=REAL)
        # fixup mistakenly overdimensioned gensim-3.x lockf arrays
        if len(self.vectors_vocab_lockf.shape) > 1:
            self.vectors_vocab_lockf = ones(1, dtype=REAL)
        if len(self.vectors_ngrams_lockf.shape) > 1:
            self.vectors_ngrams_lockf = ones(1, dtype=REAL)
        if not hasattr(self, 'buckets_word') or not self.buckets_word:
            self.recalc_char_ngram_buckets()
        if not hasattr(self, 'vectors') or self.vectors is None:
            self.adjust_vectors()  # recompose full-word vectors

    def __contains__(self, word):
        """Check if `word` or any character ngrams in `word` are present in the vocabulary.
        A vector for the word is guaranteed to exist if current method returns True.

        Parameters
        ----------
        word : str
            Input word.

        Returns
        -------
        bool
            True if `word` or any character ngrams in `word` are present in the vocabulary, False otherwise.

        Note
        ----
        This method **always** returns True, because of the way FastText works.

        If you want to check if a word is an in-vocabulary term, use this instead:

        .. pycon:

            >>> from gensim.test.utils import datapath
            >>> from gensim.models import FastText
            >>> cap_path = datapath("crime-and-punishment.bin")
            >>> model = FastText.load_fasttext_format(cap_path, full_model=False)
            >>> 'steamtrain' in model.wv.key_to_index  # If False, is an OOV term
            False

        """
        return True

    def save(self, *args, **kwargs):
        """Save object.

        Parameters
        ----------
        fname : str
            Path to the output file.

        See Also
        --------
        :meth:`~gensim.models.fasttext.FastTextKeyedVectors.load`
            Load object.

        """
        super(FastTextKeyedVectors, self).save(*args, **kwargs)

    def _save_specials(self, fname, separately, sep_limit, ignore, pickle_protocol, compress, subname):
        """Arrange any special handling for the gensim.utils.SaveLoad protocol"""
        # don't save properties that are merely calculated from others
        ignore = set(ignore).union(['buckets_word', 'vectors', ])
        return super(FastTextKeyedVectors, self)._save_specials(
            fname, separately, sep_limit, ignore, pickle_protocol, compress, subname)

    def get_vector(self, word, norm=False):
        """Get `word` representations in vector space, as a 1D numpy array.

        Parameters
        ----------
        word : str
            Input word.
        norm : bool, optional
            If True, resulting vector will be L2-normalized (unit Euclidean length).

        Returns
        -------
        numpy.ndarray
            Vector representation of `word`.

        Raises
        ------
        KeyError
            If word and all its ngrams not in vocabulary.

        """
        if word in self.key_to_index:
            return super(FastTextKeyedVectors, self).get_vector(word, norm=norm)
        elif self.bucket == 0:
            raise KeyError('cannot calculate vector for OOV word without ngrams')
        else:
            word_vec = np.zeros(self.vectors_ngrams.shape[1], dtype=np.float32)
            ngram_weights = self.vectors_ngrams
            ngram_hashes = ft_ngram_hashes(word, self.min_n, self.max_n, self.bucket)
            if len(ngram_hashes) == 0:
                #
                # If it is impossible to extract _any_ ngrams from the input
                # word, then the best we can do is return a vector that points
                # to the origin.  The reference FB implementation does this,
                # too.
                #
                # https://github.com/RaRe-Technologies/gensim/issues/2402
                #
                logger.warning('could not extract any ngrams from %r, returning origin vector', word)
                return word_vec
            for nh in ngram_hashes:
                word_vec += ngram_weights[nh]
            word_vec /= len(ngram_hashes)
            if norm:
                return word_vec / np.linalg.norm(word_vec)
            else:
                return word_vec

    def init_ngrams_weights(self, seed):
        """Initialize the vocabulary and ngrams weights prior to training.

        Creates the weight matrices and initializes them with uniform random values.

        Parameters
        ----------
        seed : float
            The seed for the PRNG.

        Note
        ----
        Call this **after** the vocabulary has been fully initialized.

        """
        self.recalc_char_ngram_buckets()

        rand_obj = np.random.default_rng(seed=seed)  # use new instance of numpy's recommended generator/algorithm

        lo, hi = -1.0 / self.vector_size, 1.0 / self.vector_size
        vocab_shape = (len(self), self.vector_size)
        ngrams_shape = (self.bucket, self.vector_size)
        self.vectors_vocab = rand_obj.uniform(lo, hi, vocab_shape).astype(REAL)

        #
        # We could have initialized vectors_ngrams at construction time, but we
        # do it here for two reasons:
        #
        # 1. The constructor does not have access to the random seed
        # 2. We want to use the same rand_obj to fill vectors_vocab _and_
        #    vectors_ngrams, and vectors_vocab cannot happen at construction
        #    time because the vocab is not initialized at that stage.
        #
        self.vectors_ngrams = rand_obj.uniform(lo, hi, ngrams_shape).astype(REAL)

    def update_ngrams_weights(self, seed, old_vocab_len):
        """Update the vocabulary weights for training continuation.

        Parameters
        ----------
        seed : float
            The seed for the PRNG.
        old_vocab_length : int
            The length of the vocabulary prior to its update.

        Note
        ----
        Call this **after** the vocabulary has been updated.

        """
        self.recalc_char_ngram_buckets()

        rand_obj = np.random
        rand_obj.seed(seed)

        new_vocab = len(self) - old_vocab_len
        self.vectors_vocab = _pad_random(self.vectors_vocab, new_vocab, rand_obj)

    def init_post_load(self, fb_vectors):
        """Perform initialization after loading a native Facebook model.

        Expects that the vocabulary (self.key_to_index) has already been initialized.

        Parameters
        ----------
        fb_vectors : np.array
            A matrix containing vectors for all the entities, including words
            and ngrams.  This comes directly from the binary model.
            The order of the vectors must correspond to the indices in
            the vocabulary.

        """
        vocab_words = len(self)
        assert fb_vectors.shape[0] == vocab_words + self.bucket, 'unexpected number of vectors'
        assert fb_vectors.shape[1] == self.vector_size, 'unexpected vector dimensionality'

        #
        # The incoming vectors contain vectors for both words AND
        # ngrams.  We split them into two separate matrices, because our
        # implementation treats them differently.
        #
        self.vectors_vocab = np.array(fb_vectors[:vocab_words, :])
        self.vectors_ngrams = np.array(fb_vectors[vocab_words:, :])
        self.recalc_char_ngram_buckets()
        self.adjust_vectors()  # calculate composite full-word vectors

    def adjust_vectors(self):
        """Adjust the vectors for words in the vocabulary.

        The adjustment composes the trained full-word-token vectors with
        the vectors of the subword ngrams, matching the Facebook reference
        implementation behavior.

        """
        if self.bucket == 0:
            self.vectors = self.vectors_vocab  # no ngrams influence
            return

        self.vectors = self.vectors_vocab[:].copy()
        for i, _ in enumerate(self.index_to_key):
            ngram_buckets = self.buckets_word[i]
            for nh in ngram_buckets:
                self.vectors[i] += self.vectors_ngrams[nh]
            self.vectors[i] /= len(ngram_buckets) + 1

    def recalc_char_ngram_buckets(self):
        """
        Scan the vocabulary, calculate ngrams and their hashes, and cache the list of ngrams for each known word.

        """
        # FIXME: evaluate if precaching even necessary, compared to recalculating as needed
        if self.bucket == 0:
            self.buckets_word = [np.array([], dtype=np.uint32)] * len(self.index_to_key)
            return

        self.buckets_word = [None] * len(self.index_to_key)

        for i, word in enumerate(self.index_to_key):
            self.buckets_word[i] = np.array(
                ft_ngram_hashes(word, self.min_n, self.max_n, self.bucket),
                dtype=np.uint32,
           )


def _pad_random(m, new_rows, rand):
    """Pad a matrix with additional rows filled with random values."""
    _, columns = m.shape
    low, high = -1.0 / columns, 1.0 / columns
    suffix = rand.uniform(low, high, (new_rows, columns)).astype(REAL)
    return vstack([m, suffix])


def _unpack(m, num_rows, hash2index, seed=1, fill=None):
    """Restore the array to its natural shape, undoing the optimization.

    A packed matrix contains contiguous vectors for ngrams, as well as a hashmap.
    The hash map maps the ngram hash to its index in the packed matrix.
    To unpack the matrix, we need to do several things:

    1. Restore the matrix to its "natural" shape, where the number of rows
       equals the number of buckets.
    2. Rearrange the existing rows such that the hashmap becomes the identity
       function and is thus redundant.
    3. Fill the new rows with random values.

    Parameters
    ----------

    m : np.ndarray
        The matrix to restore.
    num_rows : int
        The number of rows that this array should have.
    hash2index : dict
        the product of the optimization we are undoing.
    seed : float, optional
        The seed for the PRNG.  Will be used to initialize new rows.
    fill : float or array or None, optional
        Value for new rows. If None (the default), randomly initialize.
    Returns
    -------
    np.array
        The unpacked matrix.

    Notes
    -----

    The unpacked matrix will reference some rows in the input matrix to save memory.
    Throw away the old matrix after calling this function, or use np.copy.

    """
    orig_rows, *more_dims = m.shape
    if orig_rows == num_rows:
        #
        # Nothing to do.
        #
        return m
    assert num_rows > orig_rows

    if fill is None:
        rand_obj = np.random
        rand_obj.seed(seed)

        #
        # Rows at the top of the matrix (the first orig_rows) will contain "packed" learned vectors.
        # Rows at the bottom of the matrix will be "free": initialized to random values.
        #
        m = _pad_random(m, num_rows - orig_rows, rand_obj)
    else:
        m = np.concatenate([m, [fill] * (num_rows - orig_rows)])

    #
    # Swap rows to transform hash2index into the identify function.
    # There are two kinds of swaps.
    # First, rearrange the rows that belong entirely within the original matrix dimensions.
    # Second, swap out rows from the original matrix dimensions, replacing them with
    # randomly initialized values.
    #
    # N.B. We only do the swap in one direction, because doing it in both directions
    # nullifies the effect.
    #
    swap = {h: i for (h, i) in hash2index.items() if h < i < orig_rows}
    swap.update({h: i for (h, i) in hash2index.items() if h >= orig_rows})
    for h, i in swap.items():
        assert h != i
        m[[h, i]] = m[[i, h]]  # swap rows i and h

    return m


#
# UTF-8 bytes that begin with 10 are subsequent bytes of a multi-byte sequence,
# as opposed to a new character.
#
_MB_MASK = 0xC0
_MB_START = 0x80


def _is_utf8_continue(b):
    return b & _MB_MASK == _MB_START


def ft_ngram_hashes(word, minn, maxn, num_buckets):
    """Calculate the ngrams of the word and hash them.

    Parameters
    ----------
    word : str
        The word to calculate ngram hashes for.
    minn : int
        Minimum ngram length
    maxn : int
        Maximum ngram length
    num_buckets : int
        The number of buckets

    Returns
    -------
        A list of hashes (integers), one per each detected ngram.

    """
    encoded_ngrams = compute_ngrams_bytes(word, minn, maxn)
    hashes = [ft_hash_bytes(n) % num_buckets for n in encoded_ngrams]
    return hashes


# BACKWARD COMPATIBILITY FOR OLDER PICKLES
from gensim.models import keyedvectors  # noqa: E402
keyedvectors.FastTextKeyedVectors = FastTextKeyedVectors
