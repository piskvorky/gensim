#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors: Shiva Manne <manneshiva@gmail.com>, Chinmaya Pancholi <chinmayapancholi13@gmail.com>
# Copyright (C) 2018 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Learn word representations via Fasttext: `Enriching Word Vectors with Subword Information
<https://arxiv.org/abs/1607.04606>`_.

This module allows training word embeddings from a training corpus with the additional ability to obtain word vectors
for out-of-vocabulary words.

This module contains a fast native C implementation of Fasttext with Python interfaces. It is **not** only a wrapper
around Facebook's implementation.

This module supports loading models trained with Facebook's fastText implementation.
It also supports continuing training from such models.

For a tutorial see `this notebook
<https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/FastText_Tutorial.ipynb>`_.

**Make sure you have a C compiler before installing Gensim, to use the optimized (compiled) Fasttext
training routines.**

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
    >>> model = FastText(size=4, window=3, min_count=1)  # instantiate
    >>> model.build_vocab(sentences=common_texts)
    >>> model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)  # train

Once you have a model, you can access its keyed vectors via the `model.wv` attributes.
The keyed vectors instance is quite powerful: it can perform a wide range of NLP tasks.
For a full list of examples, see :class:`~gensim.models.keyedvectors.FastTextKeyedVectors`.

You can also pass all the above parameters to the constructor to do everything
in a single line:

.. sourcecode:: pycon

    >>> model2 = FastText(size=4, window=3, min_count=1, sentences=common_texts, iter=10)

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
    >>> model3 = FastText(size=4, window=3, min_count=1)
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
    >>> model4 = FastText(size=4, window=3, min_count=1)
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

Implementation Notes
--------------------

These notes may help developers navigate our fastText implementation.
The implementation is split across several submodules:

- :mod:`gensim.models.fasttext`: This module. Contains FastText-specific functionality only.
- :mod:`gensim.models.keyedvectors`: Implements both generic and FastText-specific functionality.
- :mod:`gensim.models.word2vec`: Contains implementations for the vocabulary
  and the trainables for FastText.
- :mod:`gensim.models.base_any2vec`: Contains implementations for the base.
  classes, including functionality such as callbacks, logging.
- :mod:`gensim.models.utils_any2vec`: Wrapper over Cython extensions.
- :mod:`gensim.utils`: Implements model I/O (loading and saving).

Our implementation relies heavily on inheritance.
It consists of several important classes:

- :class:`~gensim.models.word2vec.Word2VecVocab`: the vocabulary.
  Keeps track of all the unique words, sometimes discarding the extremely rare ones.
  This is sometimes called the Dictionary within Gensim.
- :class:`~gensim.models.keyedvectors.FastTextKeyedVectors`: the vectors.
  Once training is complete, this class is sufficient for calculating embeddings.
- :class:`~gensim.models.fasttext.FastTextTrainables`: the underlying neural network.
  The implementation uses this class to *learn* the word embeddings.
- :class:`~gensim.models.fasttext.FastText`: ties everything together.

"""

import logging
import os

import numpy as np
from numpy import ones, vstack, float32 as REAL
import six
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import gensim.models._fasttext_bin

from gensim.models.word2vec import Word2VecVocab, Word2VecTrainables, train_sg_pair, train_cbow_pair  # noqa
from gensim.models.keyedvectors import FastTextKeyedVectors
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.utils_any2vec import ft_ngram_hashes

import gensim.utils

logger = logging.getLogger(__name__)

try:
    from gensim.models.fasttext_inner import (  # noqa: F401
        train_batch_sg,
        train_batch_cbow,
        FAST_VERSION,
        MAX_WORDS_IN_BATCH,
    )
except ImportError:
    raise gensim.utils.NO_CYTHON

try:
    from gensim.models.fasttext_corpusfile import train_epoch_sg, train_epoch_cbow
except ImportError:
    #
    # corpusfile is unavailable under Py2.7 on Windows.
    # This is not fatal, life goes on.
    #
    pass


class FastText(BaseWordEmbeddingsModel):
    """Train, use and evaluate word representations learned using the method
    described in `Enriching Word Vectors with Subword Information <https://arxiv.org/abs/1607.04606>`_, aka FastText.

    The model can be stored/loaded via its :meth:`~gensim.models.fasttext.FastText.save` and
    :meth:`~gensim.models.fasttext.FastText.load` methods, or loaded from a format compatible with the original
    Fasttext implementation via :func:`~gensim.models.fasttext.load_facebook_model`.

    Attributes
    ----------
    wv : :class:`~gensim.models.keyedvectors.FastTextKeyedVectors`
        This object essentially contains the mapping between words and embeddings. These are similar to the embeddings
        computed in the :class:`~gensim.models.word2vec.Word2Vec`, however here we also include vectors for n-grams.
        This allows the model to compute embeddings even for **unseen** words (that do not exist in the vocabulary),
        as the aggregate of the n-grams included in the word. After training the model, this attribute can be used
        directly to query those embeddings in various ways. Check the module level docstring for some examples.
    vocabulary : :class:`~gensim.models.fasttext.FastTextVocab`
        This object represents the vocabulary of the model.
        Besides keeping track of all unique words, this object provides extra functionality, such as
        constructing a huffman tree (frequent words are closer to the root), or discarding extremely rare words.
    trainables : :class:`~gensim.models.fasttext.FastTextTrainables`
        This object represents the inner shallow neural network used to train the embeddings. This is very
        similar to the network of the :class:`~gensim.models.word2vec.Word2Vec` model, but it also trains weights
        for the N-Grams (sequences of more than 1 words). The semantics of the network are almost the same as
        the one used for the :class:`~gensim.models.word2vec.Word2Vec` model.
        You can think of it as a NN with a single projection and hidden layer which we train on the corpus.
        The weights are then used as our embeddings. An important difference however between the two models, is the
        scoring function used to compute the loss. In the case of FastText, this is modified in word to also account
        for the internal structure of words, besides their concurrence counts.

    """
    def __init__(self, sentences=None, corpus_file=None, sg=0, hs=0, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, word_ngrams=1, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, min_n=3, max_n=6,
                 sorted_vocab=1, bucket=2000000, trim_rule=None, batch_words=MAX_WORDS_IN_BATCH, callbacks=(),
                 compatible_hash=True):
        """

        Parameters
        ----------
        sentences : iterable of list of str, optional
            Can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
            If you don't supply `sentences`, the model is left uninitialized -- use if you plan to initialize it
            in some other way.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `sentences` to get performance boost. Only one of `sentences` or
            `corpus_file` arguments need to be passed (or none of them, in that case, the model is left uninitialized).
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
        word_ngrams : {1,0}, optional
            If 1, uses enriches word vectors with subword(n-grams) information.
            If 0, this is equivalent to :class:`~gensim.models.word2vec.Word2Vec`.
        bucket : int, optional
            Character ngrams are hashed into a fixed number of buckets, in order to limit the
            memory usage of the model. This option specifies the number of buckets used by the model.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`, optional
            List of callbacks that need to be executed/run at specific stages during training.

        compatible_hash: bool, optional
            By default, newer versions of Gensim's FastText use a hash function
            that is 100% compatible with Facebook's FastText.
            Older versions were not 100% compatible due to a bug.
            To use the older, incompatible hash function, set this to False.

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

        """
        self.load = gensim.utils.call_on_class_only
        self.load_fasttext_format = gensim.utils.call_on_class_only
        self.callbacks = callbacks
        self.word_ngrams = int(word_ngrams)
        if self.word_ngrams <= 1 and max_n == 0:
            bucket = 0

        self.wv = FastTextKeyedVectors(size, min_n, max_n, bucket, compatible_hash)
        self.vocabulary = FastTextVocab(
            max_vocab_size=max_vocab_size, min_count=min_count, sample=sample,
            sorted_vocab=bool(sorted_vocab), null_word=null_word, ns_exponent=ns_exponent)
        self.trainables = FastTextTrainables(vector_size=size, seed=seed, bucket=bucket, hashfxn=hashfxn)
        self.trainables.prepare_weights(hs, negative, self.wv, update=False, vocabulary=self.vocabulary)
        self.wv.bucket = self.trainables.bucket

        super(FastText, self).__init__(
            sentences=sentences, corpus_file=corpus_file, workers=workers, vector_size=size, epochs=iter,
            callbacks=callbacks, batch_words=batch_words, trim_rule=trim_rule, sg=sg, alpha=alpha, window=window,
            seed=seed, hs=hs, negative=negative, cbow_mean=cbow_mean, min_alpha=min_alpha)

    @property
    @gensim.utils.deprecated("Attribute will be removed in 4.0.0, use wv.min_n instead")
    def min_n(self):
        return self.wv.min_n

    @property
    @gensim.utils.deprecated("Attribute will be removed in 4.0.0, use wv.max_n instead")
    def max_n(self):
        return self.wv.max_n

    @property
    @gensim.utils.deprecated("Attribute will be removed in 4.0.0, use trainables.bucket instead")
    def bucket(self):
        return self.trainables.bucket

    @property
    @gensim.utils.deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_vocab_lockf instead")
    def syn0_vocab_lockf(self):
        return self.trainables.vectors_vocab_lockf

    @syn0_vocab_lockf.setter
    @gensim.utils.deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_vocab_lockf instead")
    def syn0_vocab_lockf(self, value):
        self.trainables.vectors_vocab_lockf = value

    @syn0_vocab_lockf.deleter
    @gensim.utils.deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_vocab_lockf instead")
    def syn0_vocab_lockf(self):
        del self.trainables.vectors_vocab_lockf

    @property
    @gensim.utils.deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_ngrams_lockf instead")
    def syn0_ngrams_lockf(self):
        return self.trainables.vectors_ngrams_lockf

    @syn0_ngrams_lockf.setter
    @gensim.utils.deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_ngrams_lockf instead")
    def syn0_ngrams_lockf(self, value):
        self.trainables.vectors_ngrams_lockf = value

    @syn0_ngrams_lockf.deleter
    @gensim.utils.deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_ngrams_lockf instead")
    def syn0_ngrams_lockf(self):
        del self.trainables.vectors_ngrams_lockf

    @property
    @gensim.utils.deprecated("Attribute will be removed in 4.0.0, use self.wv.num_ngram_vectors instead")
    def num_ngram_vectors(self):
        return self.wv.num_ngram_vectors

    def build_vocab(self, sentences=None, corpus_file=None, update=False, progress_per=10000, keep_raw_vocab=False,
                    trim_rule=None, **kwargs):
        """Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        Parameters
        ----------
        sentences : iterable of list of str, optional
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
            :meth:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel.build_vocab`.

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
            self.wv.init_ngrams_weights(self.trainables.seed)
        elif not len(self.wv.vocab):
            raise RuntimeError(
                "You cannot do an online vocabulary-update of a model which has no prior vocabulary. "
                "First build the vocabulary of your model with a corpus "
                "by calling the gensim.models.fasttext.FastText.build_vocab method "
                "before doing an online update."
            )
        else:
            self.vocabulary.old_vocab_len = len(self.wv.vocab)

        retval = super(FastText, self).build_vocab(
            sentences=sentences, corpus_file=corpus_file, update=update, progress_per=progress_per,
            keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, **kwargs)

        if update:
            self.wv.update_ngrams_weights(self.trainables.seed, self.vocabulary.old_vocab_len)

        return retval

    def _set_train_params(self, **kwargs):
        #
        # We need the wv.buckets_word member to be initialized in order to
        # continue training. The _clear_post_train method destroys this
        # variable, so we reinitialize it here, if needed.
        #
        # The .old_vocab_len member is set only to keep the init_ngrams_weights method happy.
        #
        if self.wv.buckets_word is None:
            self.vocabulary.old_vocab_len = len(self.wv.vocab)
            self.trainables.init_ngrams_weights(self.wv, update=True, vocabulary=self.vocabulary)

    def _clear_post_train(self):
        """Clear the model's internal structures after training has finished to free up RAM."""
        self.wv.vectors_norm = None
        self.wv.vectors_vocab_norm = None
        self.wv.vectors_ngrams_norm = None
        self.wv.buckets_word = None

    def estimate_memory(self, vocab_size=None, report=None):
        vocab_size = vocab_size or len(self.wv.vocab)
        vec_size = self.vector_size * np.dtype(np.float32).itemsize
        l1_size = self.trainables.layer1_size * np.dtype(np.float32).itemsize
        report = report or {}
        report['vocab'] = len(self.wv.vocab) * (700 if self.hs else 500)
        report['syn0_vocab'] = len(self.wv.vocab) * vec_size
        num_buckets = self.trainables.bucket
        if self.hs:
            report['syn1'] = len(self.wv.vocab) * l1_size
        if self.negative:
            report['syn1neg'] = len(self.wv.vocab) * l1_size
        if self.word_ngrams > 0 and self.wv.vocab:
            num_buckets = num_ngrams = 0

            if self.trainables.bucket:
                buckets = set()
                num_ngrams = 0
                for word in self.wv.vocab:
                    hashes = ft_ngram_hashes(
                        word,
                        self.wv.min_n,
                        self.wv.max_n,
                        self.trainables.bucket,
                        self.wv.compatible_hash
                    )
                    num_ngrams += len(hashes)
                    buckets.update(hashes)
                num_buckets = len(buckets)
            report['syn0_ngrams'] = num_buckets * vec_size
            # A tuple (48 bytes) with num_ngrams_word ints (8 bytes) for each word
            # Only used during training, not stored with the model
            report['buckets_word'] = 48 * len(self.wv.vocab) + 8 * num_ngrams
        elif self.word_ngrams > 0:
            logger.warn(
                'subword information is enabled, but no vocabulary could be found, estimated required memory might be '
                'inaccurate!'
            )
        report['total'] = sum(report.values())
        logger.info(
            "estimated required memory for %i words, %i buckets and %i dimensions: %i bytes",
            len(self.wv.vocab), num_buckets, self.vector_size, report['total']
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
        tally = 0
        if self.sg:
            tally += train_batch_sg(self, sentences, alpha, work, neu1)
        else:
            tally += train_batch_cbow(self, sentences, alpha, work, neu1)

        return tally, self._raw_word_count(sentences)

    def train(self, sentences=None, corpus_file=None, total_examples=None, total_words=None,
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

        if corpus_file is None and sentences is None:
            raise TypeError("Either one of corpus_file or sentences value must be provided")

        if corpus_file is not None and sentences is not None:
            raise TypeError("Both corpus_file and sentences must not be provided at the same time")

        if sentences is None and not os.path.isfile(corpus_file):
            raise TypeError("Parameter corpus_file must be a valid path to a file, got %r instead" % corpus_file)

        if sentences is not None and not isinstance(sentences, Iterable):
            raise TypeError("sentences must be an iterable of list, got %r instead" % sentences)

        super(FastText, self).train(
            sentences=sentences, corpus_file=corpus_file, total_examples=total_examples, total_words=total_words,
            epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, callbacks=callbacks)
        self.wv.adjust_vectors()

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        Parameters
        ----------
        replace : bool
            If True, forget the original vectors and only keep the normalized ones to save RAM.

        """
        # init_sims() resides in KeyedVectors because it deals with input layer mainly, but because the
        # hidden layer is not an attribute of KeyedVectors, it has to be deleted in this class.
        # The normalizing of input layer happens inside of KeyedVectors.
        if replace and hasattr(self.trainables, 'syn1'):
            del self.trainables.syn1
        self.wv.init_sims(replace)

    def clear_sims(self):
        """Remove all L2-normalized word vectors from the model, to free up memory.

        You can recompute them later again using the :meth:`~gensim.models.fasttext.FastText.init_sims` method.

        """
        self._clear_post_train()

    @gensim.utils.deprecated("Method will be removed in 4.0.0, use self.wv.__getitem__() instead")
    def __getitem__(self, words):
        """Deprecated. Use self.wv.__getitem__() instead.

        Refer to the documentation for :meth:`gensim.models.keyedvectors.KeyedVectors.__getitem__`

        """
        return self.wv.__getitem__(words)

    @gensim.utils.deprecated("Method will be removed in 4.0.0, use self.wv.__contains__() instead")
    def __contains__(self, word):
        """Deprecated. Use self.wv.__contains__() instead.

        Refer to the documentation for :meth:`gensim.models.keyedvectors.KeyedVectors.__contains__`

        """
        return self.wv.__contains__(word)

    @classmethod
    @gensim.utils.deprecated(
        'use load_facebook_vectors (to use pretrained embeddings) or load_facebook_model '
        '(to continue training with the loaded full model, more RAM) instead'
    )
    def load_fasttext_format(cls, model_file, encoding='utf8'):
        """Deprecated.

        Use :func:`gensim.models.fasttext.load_facebook_model` or
        :func:`gensim.models.fasttext.load_facebook_vectors` instead.

        """
        return load_facebook_model(model_file, encoding=encoding)

    @gensim.utils.deprecated(
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
        for attr, val in six.iteritems(m.__dict__):
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
        kwargs['ignore'] = kwargs.get(
            'ignore', ['vectors_norm', 'vectors_vocab_norm', 'vectors_ngrams_norm', 'buckets_word'])
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
        try:
            model = super(FastText, cls).load(*args, **kwargs)

            if not hasattr(model.trainables, 'vectors_vocab_lockf') and hasattr(model.wv, 'vectors_vocab'):
                model.trainables.vectors_vocab_lockf = ones(model.wv.vectors_vocab.shape, dtype=REAL)
            if not hasattr(model.trainables, 'vectors_ngrams_lockf') and hasattr(model.wv, 'vectors_ngrams'):
                model.trainables.vectors_ngrams_lockf = ones(model.wv.vectors_ngrams.shape, dtype=REAL)

            if not hasattr(model.wv, 'bucket'):
                model.wv.bucket = model.trainables.bucket
        except AttributeError:
            logger.info('Model saved using code from earlier Gensim Version. Re-loading old model in a compatible way.')
            from gensim.models.deprecated.fasttext import load_old_fasttext
            model = load_old_fasttext(*args, **kwargs)

        gensim.models.keyedvectors._try_upgrade(model.wv)

        return model

    @gensim.utils.deprecated("Method will be removed in 4.0.0, use self.wv.accuracy() instead")
    def accuracy(self, questions, restrict_vocab=30000, most_similar=None, case_insensitive=True):
        most_similar = most_similar or FastTextKeyedVectors.most_similar
        return self.wv.accuracy(questions, restrict_vocab, most_similar, case_insensitive)


class FastTextVocab(Word2VecVocab):
    """This is a redundant class. It exists only to maintain backwards compatibility
    with older gensim versions."""
    pass


class FastTextTrainables(Word2VecTrainables):
    """Represents the inner shallow neural network used to train :class:`~gensim.models.fasttext.FastText`.

    Mostly inherits from its parent (:class:`~gensim.models.word2vec.Word2VecTrainables`).
    Adds logic for calculating and maintaining ngram weights.

    Attributes
    ----------
    hashfxn : function
        Used for randomly initializing weights. Defaults to the built-in hash()
    layer1_size : int
        The size of the inner layer of the NN. Equal to the vector dimensionality.
        Set in the :class:`~gensim.models.word2vec.Word2VecTrainables` constructor.
    seed : float
        The random generator seed used in reset_weights and update_weights.
    syn1 : numpy.array
        The inner layer of the NN. Each row corresponds to a term in the vocabulary.
        Columns correspond to weights of the inner layer.
        There are layer1_size such weights.
        Set in the reset_weights and update_weights methods, only if hierarchical sampling is used.
    syn1neg : numpy.array
        Similar to syn1, but only set if negative sampling is used.
    vectors_lockf : numpy.array
        A one-dimensional array with one element for each term in the vocab. Set in reset_weights to an array of ones.
    vectors_vocab_lockf : numpy.array
        Similar to vectors_vocab_lockf, ones(len(model.trainables.vectors), dtype=REAL)
    vectors_ngrams_lockf : numpy.array
        np.ones((self.bucket, wv.vector_size), dtype=REAL)

    """
    def __init__(self, vector_size=100, seed=1, hashfxn=hash, bucket=2000000):
        super(FastTextTrainables, self).__init__(
            vector_size=vector_size, seed=seed, hashfxn=hashfxn)
        self.bucket = int(bucket)

        #
        # There are also two "hidden" attributes that get initialized outside
        # this constructor:
        #
        #   1. vectors_vocab_lockf
        #   2. vectors_ngrams_lockf
        #
        # These are both 2D matrices of shapes equal to the shapes of
        # wv.vectors_vocab and wv.vectors_ngrams. So, each row corresponds to
        # a vector, and each column corresponds to a dimension within that
        # vector.
        #
        # Lockf stands for "lock factor": zero values suppress learning, one
        # values enable it. Interestingly, the vectors_vocab_lockf and
        # vectors_ngrams_lockf seem to be used only by the C code in
        # fasttext_inner.pyx.
        #
        # The word2vec implementation also uses vectors_lockf: in that case,
        # it's a 1D array, with a real number for each vector. The FastText
        # implementation inherits this vectors_lockf attribute but doesn't
        # appear to use it.
        #

    def prepare_weights(self, hs, negative, wv, update=False, vocabulary=None):
        super(FastTextTrainables, self).prepare_weights(hs, negative, wv, update=update, vocabulary=vocabulary)
        self.init_ngrams_weights(wv, update=update, vocabulary=vocabulary)

    def init_ngrams_weights(self, wv, update=False, vocabulary=None):
        """Compute ngrams of all words present in vocabulary and stores vectors for only those ngrams.
        Vectors for other ngrams are initialized with a random uniform distribution in FastText.

        Parameters
        ----------
        wv : :class:`~gensim.models.keyedvectors.FastTextKeyedVectors`
            Contains the mapping between the words and embeddings.
            The vectors for the computed ngrams will go here.
        update : bool
            If True, the new vocab words and their new ngrams word vectors are initialized
            with random uniform distribution and updated/added to the existing vocab word and ngram vectors.
        vocabulary : :class:`~gensim.models.fasttext.FastTextVocab`
            This object represents the vocabulary of the model.
            If update is True, then vocabulary may not be None.

        """
        if not update:
            wv.init_ngrams_weights(self.seed)
            self.vectors_vocab_lockf = ones(wv.vectors_vocab.shape, dtype=REAL)
            self.vectors_ngrams_lockf = ones(wv.vectors_ngrams.shape, dtype=REAL)
        else:
            wv.update_ngrams_weights(self.seed, vocabulary.old_vocab_len)
            self.vectors_vocab_lockf = _pad_ones(self.vectors_vocab_lockf, wv.vectors_vocab.shape)
            self.vectors_ngrams_lockf = _pad_ones(self.vectors_ngrams_lockf, wv.vectors_ngrams.shape)

    def init_post_load(self, model, hidden_output):
        num_vectors = len(model.wv.vectors)
        vocab_size = len(model.wv.vocab)
        vector_size = model.wv.vector_size

        assert num_vectors > 0, 'expected num_vectors to be initialized already'
        assert vocab_size > 0, 'expected vocab_size to be initialized already'

        self.vectors_ngrams_lockf = ones(model.wv.vectors_ngrams.shape, dtype=REAL)
        self.vectors_vocab_lockf = ones(model.wv.vectors_vocab.shape, dtype=REAL)

        if model.hs:
            self.syn1 = hidden_output
        if model.negative:
            self.syn1neg = hidden_output

        self.layer1_size = vector_size


def _pad_ones(m, new_shape):
    """Pad a matrix with additional rows filled with ones."""
    assert m.shape[0] <= new_shape[0], 'the new number of rows must be greater'
    assert m.shape[1] == new_shape[1], 'the number of columns must match'
    new_rows = new_shape[0] - m.shape[0]
    if new_rows == 0:
        return m
    suffix = ones((new_rows, m.shape[1]), dtype=REAL)
    return vstack([m, suffix])


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
    gensim.models.keyedvectors.FastTextKeyedVectors
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
    model_wrapper = _load_fasttext_format(path, encoding=encoding, full_model=False)
    return model_wrapper.wv


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
    with gensim.utils.open(model_file, 'rb') as fin:
        m = gensim.models._fasttext_bin.load(fin, encoding=encoding, full_model=full_model)

    model = FastText(
        size=m.dim,
        window=m.ws,
        iter=m.epoch,
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
    model.vocabulary.raw_vocab = m.raw_vocab
    model.vocabulary.nwords = m.nwords
    model.vocabulary.vocab_size = m.vocab_size

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
    model.vocabulary.prepare_vocab(
        model.hs, model.negative, model.wv,
        update=True, min_count=1,
    )

    model.num_original_vectors = m.vectors_ngrams.shape[0]

    model.wv.init_post_load(m.vectors_ngrams)
    model.trainables.init_post_load(model, m.hidden_output)
    _check_model(model)

    logger.info("loaded %s weight matrix for fastText model from %s", m.vectors_ngrams.shape, fin.name)
    return model


def _check_model(m):
    #
    # These checks only make sense after everything has been completely initialized.
    #
    assert m.wv.vector_size == m.wv.vectors_ngrams.shape[1], (
        'mismatch between vector size in model params ({}) and model vectors ({})'
        .format(m.wv.vector_size, m.wv.vectors_ngrams)
    )

    try:
        syn1neg = m.trainables.syn1neg
    except AttributeError:
        syn1neg = None

    if syn1neg is not None:
        assert m.wv.vector_size == m.trainables.syn1neg.shape[1], (
            'mismatch between vector size in model params ({}) and trainables ({})'
            .format(m.wv.vector_size, m.wv.vectors_ngrams)
        )

    assert len(m.wv.vocab) == m.vocabulary.nwords, (
        'mismatch between final vocab size ({} words), '
        'and expected number of words ({} words)'.format(len(m.wv.vocab), m.vocabulary.nwords)
    )

    if len(m.wv.vocab) != m.vocabulary.vocab_size:
        # expecting to log this warning only for pretrained french vector, wiki.fr
        logger.warning(
            "mismatch between final vocab size (%s words), and expected vocab size (%s words)",
            len(m.wv.vocab), m.vocabulary.vocab_size
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
