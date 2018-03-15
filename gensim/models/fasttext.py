#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors: Shiva Manne <manneshiva@gmail.com>, Chinmaya Pancholi <chinmayapancholi13@gmail.com>
# Copyright (C) 2018 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Learn word representations via fasttext's "skip-gram and CBOW models", using either
hierarchical softmax or negative sampling [1]_.

Notes
-----
There are more ways to get word vectors in Gensim than just FastText.
See wrappers for VarEmbed and WordRank or Word2Vec

This module allows training a word embedding from a training corpus with the additional ability
to obtain word vectors for out-of-vocabulary words.

For a tutorial on gensim's native fasttext, refer to the noteboook -- [2]_

**Make sure you have a C compiler before installing gensim, to use optimized (compiled) fasttext training**

.. [1] P. Bojanowski, E. Grave, A. Joulin, T. Mikolov
       Enriching Word Vectors with Subword Information. In arXiv preprint arXiv:1607.04606.
       https://arxiv.org/abs/1607.04606

.. [2] https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/FastText_Tutorial.ipynb

"""

import logging
import struct

import numpy as np
from numpy import ones, vstack, empty, float32 as REAL, sum as np_sum

from gensim.models.word2vec import Word2VecVocab, Word2VecTrainables, train_sg_pair, train_cbow_pair
from gensim.models.keyedvectors import Vocab, FastTextKeyedVectors
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.utils_any2vec import _compute_ngrams, _ft_hash

from gensim.utils import deprecated, call_on_class_only
from gensim import utils

logger = logging.getLogger(__name__)

try:
    from gensim.models.fasttext_inner import train_batch_sg, train_batch_cbow
    from gensim.models.fasttext_inner import FAST_VERSION, MAX_WORDS_IN_BATCH

except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000

    def train_batch_cbow(model, sentences, alpha, work=None, neu1=None):
        """Update CBOW model by training on a sequence of sentences.
        Each sentence is a list of string tokens, which are looked up in the model's
        vocab dictionary. Called internally from :meth:`gensim.models.fasttext.FastText.train()`.
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from fasttext_inner instead.
        Parameters
        ----------
        model : :class:`~gensim.models.fasttext.FastText`
            `FastText` instance.
        sentences : iterable of iterables
            Iterable of the sentences directly from disk/network.
        alpha : float
            Learning rate.
        work : :class:`numpy.ndarray`
            Private working memory for each worker.
        neu1 : :class:`numpy.ndarray`
            Private working memory for each worker.
        Returns
        -------
        int
            Effective number of words trained.
        """
        result = 0
        for sentence in sentences:
            word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
                           model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)
                start = max(0, pos - model.window + reduced_window)
                window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
                word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]

                vocab_subwords_indices = []
                ngrams_subwords_indices = []

                for index in word2_indices:
                    vocab_subwords_indices += [index]
                    ngrams_subwords_indices.extend(model.wv.buckets_word[index])

                l1_vocab = np_sum(model.wv.syn0_vocab[vocab_subwords_indices], axis=0)  # 1 x vector_size
                l1_ngrams = np_sum(model.wv.syn0_ngrams[ngrams_subwords_indices], axis=0)  # 1 x vector_size

                l1 = np_sum([l1_vocab, l1_ngrams], axis=0)
                subwords_indices = [vocab_subwords_indices] + [ngrams_subwords_indices]
                if (subwords_indices[0] or subwords_indices[1]) and model.cbow_mean:
                    l1 /= (len(subwords_indices[0]) + len(subwords_indices[1]))

                # train on the sliding window for target word
                train_cbow_pair(model, word, subwords_indices, l1, alpha, is_ft=True)
            result += len(word_vocabs)
        return result

    def train_batch_sg(model, sentences, alpha, work=None, neu1=None):
        """Update skip-gram model by training on a sequence of sentences.
        Each sentence is a list of string tokens, which are looked up in the model's
        vocab dictionary. Called internally from :meth:`gensim.models.fasttext.FastText.train()`.
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from fasttext_inner instead.
        Parameters
        ----------
        model : :class:`~gensim.models.fasttext.FastText`
            `FastText` instance.
        sentences : iterable of iterables
            Iterable of the sentences directly from disk/network.
        alpha : float
            Learning rate.
        work : :class:`numpy.ndarray`
            Private working memory for each worker.
        neu1 : :class:`numpy.ndarray`
            Private working memory for each worker.
        Returns
        -------
        int
            Effective number of words trained.
        """
        result = 0
        for sentence in sentences:
            word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
                           model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)

                subwords_indices = (word.index,)
                subwords_indices += model.wv.buckets_word[word.index]

                for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                    if pos2 != pos:  # don't train on the `word` itself
                        train_sg_pair(model, model.wv.index2word[word2.index], subwords_indices, alpha, is_ft=True)

            result += len(word_vocabs)
        return result

FASTTEXT_FILEFORMAT_MAGIC = 793712314


class FastText(BaseWordEmbeddingsModel):
    """Class for training, using and evaluating word representations learned using method
    described `aka Fasttext <https://arxiv.org/abs/1607.04606>`_.

    The model can be stored/loaded via its :meth:`~gensim.models.fasttext.FastText.save()` and
    :meth:`~gensim.models.fasttext.FastText.load()` methods, or loaded in a format compatible with the original
    fasttext implementation via :meth:`~gensim.models.fasttext.FastText.load_fasttext_format()`.

    """
    def __init__(self, sentences=None, sg=0, hs=0, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, word_ngrams=1, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, min_n=3, max_n=6, sorted_vocab=1,
                 bucket=2000000, trim_rule=None, batch_words=MAX_WORDS_IN_BATCH, callbacks=()):
        """Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        Parameters
        ----------
        sentences : iterable of iterables
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
            If you don't supply `sentences`, the model is left uninitialized -- use if you plan to initialize it
            in some other way.
        sg : int {1, 0}
            Defines the training algorithm. If 1, skip-gram is used, otherwise, CBOW is employed.
        size : int
            Dimensionality of the feature vectors.
        window : int
            The maximum distance between the current and predicted word within a sentence.
        alpha : float
            The initial learning rate.
        min_alpha : float
            Learning rate will linearly drop to `min_alpha` as training progresses.
        seed : int
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
            use of the `PYTHONHASHSEED` environment variable to control hash randomization).
        min_count : int
            Ignores all words with total frequency lower than this.
        max_vocab_size : int
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        workers : int
            Use these many worker threads to train the model (=faster training with multicore machines).
        hs : int {1,0}
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        negative : int
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        cbow_mean : int {1,0}
            If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        hashfxn : function
            Hash function to use to randomly initialize weights, for increased training reproducibility.
        iter : int
            Number of iterations (epochs) over the corpus.
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
        min_n : int
            Min length of char ngrams to be used for training word representations.
        max_n : int
            Max length of char ngrams to be used for training word representations. Set `max_n` to be
            lesser than `min_n` to avoid char ngrams being used.
        word_ngrams : int {1,0}
            If 1, uses enriches word vectors with subword(ngrams) information.
            If 0, this is equivalent to word2vec.
        bucket : int
            Character ngrams are hashed into a fixed number of buckets, in order to limit the
            memory usage of the model. This option specifies the number of buckets used by the model.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`
            List of callbacks that need to be executed/run at specific stages during training.

        Examples
        --------
        Initialize and train a `FastText` model

        >>> from gensim.models import FastText
        >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>>
        >>> model = FastText(sentences, min_count=1)
        >>> say_vector = model['say']  # get vector for word
        >>> of_vector = model['of']  # get vector for out-of-vocab word


        """
        self.load = call_on_class_only
        self.load_fasttext_format = call_on_class_only
        self.callbacks = callbacks
        self.word_ngrams = int(word_ngrams)
        if self.word_ngrams <= 1 and max_n == 0:
            bucket = 0

        self.wv = FastTextKeyedVectors(size, min_n, max_n)
        self.vocabulary = FastTextVocab(
            max_vocab_size=max_vocab_size, min_count=min_count, sample=sample,
            sorted_vocab=bool(sorted_vocab), null_word=null_word)
        self.trainables = FastTextTrainables(
            vector_size=size, seed=seed, bucket=bucket, hashfxn=hashfxn)
        self.wv.bucket = self.bucket

        super(FastText, self).__init__(
            sentences=sentences, workers=workers, vector_size=size, epochs=iter, callbacks=callbacks,
            batch_words=batch_words, trim_rule=trim_rule, sg=sg, alpha=alpha, window=window, seed=seed,
            hs=hs, negative=negative, cbow_mean=cbow_mean, min_alpha=min_alpha, fast_version=FAST_VERSION)

    @property
    @deprecated("Attribute will be removed in 4.0.0, use wv.min_n instead")
    def min_n(self):
        return self.wv.min_n

    @property
    @deprecated("Attribute will be removed in 4.0.0, use wv.max_n instead")
    def max_n(self):
        return self.wv.max_n

    @property
    @deprecated("Attribute will be removed in 4.0.0, use trainables.bucket instead")
    def bucket(self):
        return self.trainables.bucket

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_vocab_lockf instead")
    def syn0_vocab_lockf(self):
        return self.trainables.vectors_vocab_lockf

    @syn0_vocab_lockf.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_vocab_lockf instead")
    def syn0_vocab_lockf(self, value):
        self.trainables.vectors_vocab_lockf = value

    @syn0_vocab_lockf.deleter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_vocab_lockf instead")
    def syn0_vocab_lockf(self):
        del self.trainables.vectors_vocab_lockf

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_ngrams_lockf instead")
    def syn0_ngrams_lockf(self):
        return self.trainables.vectors_ngrams_lockf

    @syn0_ngrams_lockf.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_ngrams_lockf instead")
    def syn0_ngrams_lockf(self, value):
        self.trainables.vectors_ngrams_lockf = value

    @syn0_ngrams_lockf.deleter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_ngrams_lockf instead")
    def syn0_ngrams_lockf(self):
        del self.trainables.vectors_ngrams_lockf

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.num_ngram_vectors instead")
    def num_ngram_vectors(self):
        return self.wv.num_ngram_vectors

    def build_vocab(self, sentences, update=False, progress_per=10000, keep_raw_vocab=False, trim_rule=None, **kwargs):
        """Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        Parameters
        ----------
        sentences : iterable of iterables
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        keep_raw_vocab : bool
            If not true, delete the raw vocabulary after the scaling is done and free up RAM.
        trim_rule : function
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part
            of the model.
        progress_per : int
            Indicates how many words to process before showing/updating the progress.
        update : bool
            If true, the new words in `sentences` will be added to model's vocab.

        Example
        -------
        Train a model and update vocab for online training

        >>> from gensim.models import FastText
        >>> sentences_1 = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>> sentences_2 = [["dude", "say", "wazzup!"]]
        >>>
        >>> model = FastText(min_count=1)
        >>> model.build_vocab(sentences_1)
        >>> model.train(sentences_1, total_examples=model.corpus_count, epochs=model.iter)
        >>> model.build_vocab(sentences_2, update=True)
        >>> model.train(sentences_2, total_examples=model.corpus_count, epochs=model.iter)

        """
        if update:
            if not len(self.wv.vocab):
                raise RuntimeError(
                    "You cannot do an online vocabulary-update of a model which has no prior vocabulary. "
                    "First build the vocabulary of your model with a corpus "
                    "before doing an online update.")
            self.vocabulary.old_vocab_len = len(self.wv.vocab)
            self.trainables.old_hash2index_len = len(self.wv.hash2index)

        return super(FastText, self).build_vocab(
            sentences, update=update, progress_per=progress_per,
            keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, **kwargs)

    def _set_train_params(self, **kwargs):
        pass

    def _clear_post_train(self):
        self.wv.vectors_norm = None
        self.wv.vectors_vocab_norm = None
        self.wv.vectors_ngrams_norm = None
        self.wv.buckets_word = None

    def estimate_memory(self, vocab_size=None, report=None):
        vocab_size = vocab_size or len(self.wv.vocab)
        vec_size = self.vector_size * np.dtype(np.float32).itemsize
        l1_size = self.layer1_size * np.dtype(np.float32).itemsize
        report = report or {}
        report['vocab'] = len(self.wv.vocab) * (700 if self.hs else 500)
        report['syn0_vocab'] = len(self.wv.vocab) * vec_size
        num_buckets = self.bucket
        if self.hs:
            report['syn1'] = len(self.wv.vocab) * l1_size
        if self.negative:
            report['syn1neg'] = len(self.wv.vocab) * l1_size
        if self.word_ngrams > 0 and self.wv.vocab:
            buckets = set()
            num_ngrams = 0
            for word in self.wv.vocab:
                ngrams = _compute_ngrams(word, self.min_n, self.max_n)
                num_ngrams += len(ngrams)
                buckets.update(_ft_hash(ng) % self.bucket for ng in ngrams)
            num_buckets = len(buckets)
            report['syn0_ngrams'] = len(buckets) * vec_size
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

    def _do_train_job(self, sentences, alpha, inits):
        """Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.

        Parameters
        ----------
        sentences : iterable of iterables
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        alpha : float
            The current learning rate.
        inits : (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
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

    def train(self, sentences, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None,
              word_count=0, queue_factor=2, report_delay=1.0, callbacks=(), **kwargs):
        """Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For FastText, each sentence must be a list of unicode strings.

        To support linear learning-rate decay from (initial) alpha to min_alpha, and accurate
        progress-percentage logging, either total_examples (count of sentences) or total_words (count of
        raw words in sentences) **MUST** be provided (if the corpus is the same as was provided to
        :meth:`~gensim.models.fasttext.FastText.build_vocab()`, the count of examples in that corpus
        will be available in the model's :attr:`corpus_count` property).

        To avoid common mistakes around the model's ability to do multiple training passes itself, an
        explicit `epochs` argument **MUST** be provided. In the common and recommended case,
        where :meth:`~gensim.models.fasttext.FastText.train()` is only called once,
        the model's cached `iter` value should be supplied as `epochs` value.

        Parameters
        ----------
        sentences : iterable of iterables
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        total_examples : int
            Count of sentences.
        total_words : int
            Count of raw words in sentences.
        epochs : int
            Number of iterations (epochs) over the corpus.
        start_alpha : float
            Initial learning rate.
        end_alpha : float
            Final learning rate. Drops linearly from `start_alpha`.
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
        >>> from gensim.models import FastText
        >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>>
        >>> model = FastText(min_count=1)
        >>> model.build_vocab(sentences)
        >>> model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

        """

        super(FastText, self).train(
            sentences, total_examples=total_examples, total_words=total_words,
            epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, callbacks=callbacks)
        self.trainables.get_vocab_word_vecs(self.wv)

    def init_sims(self, replace=False):
        """
        init_sims() resides in KeyedVectors because it deals with syn0 mainly, but because syn1 is not an attribute
        of KeyedVectors, it has to be deleted in this class, and the normalizing of syn0 happens inside of KeyedVectors
        """
        if replace and hasattr(self.trainables, 'syn1'):
            del self.trainables.syn1
        return self.wv.init_sims(replace)

    def clear_sims(self):
        """
        Removes all L2-normalized vectors for words from the model.
        You will have to recompute them using init_sims method.
        """

        self._clear_post_train()

    @deprecated("Method will be removed in 4.0.0, use self.wv.__getitem__() instead")
    def __getitem__(self, words):
        """
        Deprecated. Use self.wv.__getitem__() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.__getitem__`
        """
        return self.wv.__getitem__(words)

    @deprecated("Method will be removed in 4.0.0, use self.wv.__contains__() instead")
    def __contains__(self, word):
        """
        Deprecated. Use self.wv.__contains__() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.__contains__`
        """
        return self.wv.__contains__(word)

    @classmethod
    def load_fasttext_format(cls, model_file, encoding='utf8'):
        """
        Load the input-hidden weight matrix from the fast text output files.

        Note that due to limitations in the FastText API, you cannot continue training
        with a model loaded this way, though you can query for word similarity etc.

        Parameters
        ----------
        model_file : str
            Path to the FastText output files.
            FastText outputs two model files - `/path/to/model.vec` and `/path/to/model.bin`
            Expected value for this example: `/path/to/model` or `/path/to/model.bin`,
            as gensim requires only `.bin` file to load entire fastText model.
        encoding : str
            Specifies the encoding.

        Returns
        -------
        :obj: `~gensim.models.fasttext.FastText`
            Returns the loaded model as an instance of :class: `~gensim.models.fasttext.FastText`.

        """
        model = cls()
        if not model_file.endswith('.bin'):
            model_file += '.bin'
        model.file_name = model_file
        model.load_binary_data(encoding=encoding)
        return model

    def load_binary_data(self, encoding='utf8'):
        """Loads data from the output binary file created by FastText training"""
        with utils.smart_open(self.file_name, 'rb') as f:
            self._load_model_params(f)
            self._load_dict(f, encoding=encoding)
            self._load_vectors(f)

    def _load_model_params(self, file_handle):
        magic, version = self.struct_unpack(file_handle, '@2i')
        if magic == FASTTEXT_FILEFORMAT_MAGIC:  # newer format
            self.new_format = True
            dim, ws, epoch, min_count, neg, _, loss, model, bucket, minn, maxn, _, t = \
                self.struct_unpack(file_handle, '@12i1d')
        else:  # older format
            self.new_format = False
            dim = magic
            ws = version
            epoch, min_count, neg, _, loss, model, bucket, minn, maxn, _, t = self.struct_unpack(file_handle, '@10i1d')
        # Parameters stored by [Args::save](https://github.com/facebookresearch/fastText/blob/master/src/args.cc)
        self.wv.vector_size = dim
        self.vector_size = dim
        self.window = ws
        self.epochs = epoch
        self.vocabulary.min_count = min_count
        self.negative = neg
        self.hs = loss == 1
        self.sg = model == 2
        self.trainables.bucket = bucket
        self.wv.bucket = bucket
        self.wv.min_n = minn
        self.wv.max_n = maxn
        self.vocabulary.sample = t

    def _load_dict(self, file_handle, encoding='utf8'):
        vocab_size, nwords, nlabels = self.struct_unpack(file_handle, '@3i')
        # Vocab stored by [Dictionary::save](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)
        if nlabels > 0:
            raise NotImplementedError("Supervised fastText models are not supported")
        logger.info("loading %s words for fastText model from %s", vocab_size, self.file_name)

        self.struct_unpack(file_handle, '@1q')  # number of tokens
        if self.new_format:
            pruneidx_size, = self.struct_unpack(file_handle, '@q')
        for i in range(vocab_size):
            word_bytes = b''
            char_byte = file_handle.read(1)
            # Read vocab word
            while char_byte != b'\x00':
                word_bytes += char_byte
                char_byte = file_handle.read(1)
            word = word_bytes.decode(encoding)
            count, _ = self.struct_unpack(file_handle, '@qb')

            self.wv.vocab[word] = Vocab(index=i, count=count)
            self.wv.index2word.append(word)

        assert len(self.wv.vocab) == nwords, (
            'mismatch between final vocab size ({} words), '
            'and expected number of words ({} words)'.format(len(self.wv.vocab), nwords))
        if len(self.wv.vocab) != vocab_size:
            # expecting to log this warning only for pretrained french vector, wiki.fr
            logger.warning(
                "mismatch between final vocab size (%s words), and expected vocab size (%s words)",
                len(self.wv.vocab), vocab_size
            )

        if self.new_format:
            for j in range(pruneidx_size):
                self.struct_unpack(file_handle, '@2i')

    def _load_vectors(self, file_handle):
        if self.new_format:
            self.struct_unpack(file_handle, '@?')  # bool quant_input in fasttext.cc
        num_vectors, dim = self.struct_unpack(file_handle, '@2q')
        # Vectors stored by [Matrix::save](https://github.com/facebookresearch/fastText/blob/master/src/matrix.cc)
        assert self.wv.vector_size == dim, (
            'mismatch between vector size in model params ({}) and model vectors ({})'
            .format(self.wv.vector_size, dim)
        )
        float_size = struct.calcsize('@f')
        if float_size == 4:
            dtype = np.dtype(np.float32)
        elif float_size == 8:
            dtype = np.dtype(np.float64)

        self.num_original_vectors = num_vectors
        self.wv.vectors_ngrams = np.fromfile(file_handle, dtype=dtype, count=num_vectors * dim)
        self.wv.vectors_ngrams = self.wv.vectors_ngrams.reshape((num_vectors, dim))
        assert self.wv.vectors_ngrams.shape == (
            self.trainables.bucket + len(self.wv.vocab), self.wv.vector_size), \
            'mismatch between actual weight matrix shape {} and expected shape {}'\
            .format(
                self.wv.vectors_ngrams.shape, (self.trainables.bucket + len(self.wv.vocab), self.wv.vector_size)
            )

        self.trainables.init_ngrams_post_load(self.file_name, self.wv)
        self._clear_post_train()

    def struct_unpack(self, file_handle, fmt):
        num_bytes = struct.calcsize(fmt)
        return struct.unpack(fmt, file_handle.read(num_bytes))

    def save(self, *args, **kwargs):
        """Save the model. This saved model can be loaded again using :func:`~gensim.models.fasttext.FastText.load`,
        which supports online training and getting vectors for out-of-vocabulary words.

        Parameters
        ----------
        fname : str
            Path to the file.

        """
        kwargs['ignore'] = kwargs.get(
            'ignore', ['vectors_norm', 'vectors_vocab_norm', 'vectors_ngrams_norm', 'buckets_word'])
        super(FastText, self).save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        """Loads a previously saved `FastText` model. Also see `save()`.

        Parameters
        ----------
        fname : str
            Path to the saved file.

        Returns
        -------
        :obj: `~gensim.models.fasttext.FastText`
            Returns the loaded model as an instance of :class: `~gensim.models.fasttext.FastText`.
        """
        try:
            model = super(FastText, cls).load(*args, **kwargs)
            if not hasattr(model.trainables, 'vectors_vocab_lockf') and hasattr(model.wv, 'vectors_vocab'):
                model.trainables.vectors_vocab_lockf = ones(len(model.trainables.vectors), dtype=REAL)
            if not hasattr(model.trainables, 'vectors_ngrams_lockf') and hasattr(model.wv, 'vectors_ngrams'):
                model.trainables.vectors_ngrams_lockf = ones(len(model.trainables.vectors), dtype=REAL)
            return model
        except AttributeError:
            logger.info('Model saved using code from earlier Gensim Version. Re-loading old model in a compatible way.')
            from gensim.models.deprecated.fasttext import load_old_fasttext
            return load_old_fasttext(*args, **kwargs)

    @deprecated("Method will be removed in 4.0.0, use self.wv.accuracy() instead")
    def accuracy(self, questions, restrict_vocab=30000, most_similar=None, case_insensitive=True):
        most_similar = most_similar or FastTextKeyedVectors.most_similar
        return self.wv.accuracy(questions, restrict_vocab, most_similar, case_insensitive)


class FastTextVocab(Word2VecVocab):
    def __init__(self, max_vocab_size=None, min_count=5, sample=1e-3, sorted_vocab=True, null_word=0):
        super(FastTextVocab, self).__init__(
            max_vocab_size=max_vocab_size, min_count=min_count, sample=sample,
            sorted_vocab=sorted_vocab, null_word=null_word)

    def prepare_vocab(self, hs, negative, wv, update=False, keep_raw_vocab=False, trim_rule=None,
                      min_count=None, sample=None, dry_run=False):
        report_values = super(FastTextVocab, self).prepare_vocab(
            hs, negative, wv, update=update, keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule,
            min_count=min_count, sample=sample, dry_run=dry_run)
        return report_values


class FastTextTrainables(Word2VecTrainables):
    def __init__(self, vector_size=100, seed=1, hashfxn=hash, bucket=2000000):
        super(FastTextTrainables, self).__init__(
            vector_size=vector_size, seed=seed, hashfxn=hashfxn)
        self.bucket = int(bucket)

    def prepare_weights(self, hs, negative, wv, update=False, vocabulary=None):
        super(FastTextTrainables, self).prepare_weights(hs, negative, wv, update=update, vocabulary=vocabulary)
        self.init_ngrams_weights(wv, update=update, vocabulary=vocabulary)

    def init_ngrams_weights(self, wv, update=False, vocabulary=None):
        """Compute ngrams of all words present in vocabulary and stores vectors for only those ngrams.
        Vectors for other ngrams are initialized with a random uniform distribution in FastText.

        Parameters
        ----------
        update : bool
            If True, the new vocab words and their new ngrams word vectors are initialized
            with random uniform distribution and updated/added to the existing vocab word and ngram vectors.

        """
        if not update:
            wv.vectors_vocab = empty((len(wv.vocab), wv.vector_size), dtype=REAL)
            self.vectors_vocab_lockf = ones((len(wv.vocab), wv.vector_size), dtype=REAL)

            wv.vectors_ngrams = empty((self.bucket, wv.vector_size), dtype=REAL)
            self.vectors_ngrams_lockf = ones((self.bucket, wv.vector_size), dtype=REAL)

            wv.hash2index = {}
            wv.buckets_word = {}
            ngram_indices = []
            for word, vocab in wv.vocab.items():
                buckets = []
                for ngram in _compute_ngrams(word, wv.min_n, wv.max_n):
                    ngram_hash = _ft_hash(ngram) % self.bucket
                    if ngram_hash not in wv.hash2index:
                        wv.hash2index[ngram_hash] = len(ngram_indices)
                        ngram_indices.append(ngram_hash)
                    buckets.append(wv.hash2index[ngram_hash])
                wv.buckets_word[vocab.index] = tuple(buckets)
            wv.num_ngram_vectors = len(ngram_indices)

            logger.info("Total number of ngrams is %d", wv.num_ngram_vectors)

            wv.vectors_ngrams = wv.vectors_ngrams.take(ngram_indices, axis=0)
            self.vectors_ngrams_lockf = self.vectors_ngrams_lockf.take(ngram_indices, axis=0)
            self.reset_ngrams_weights(wv)
        else:
            wv.buckets_word = {}
            num_new_ngrams = 0
            for word, vocab in wv.vocab.items():
                buckets = []
                for ngram in _compute_ngrams(word, wv.min_n, wv.max_n):
                    ngram_hash = _ft_hash(ngram) % self.bucket
                    if ngram_hash not in wv.hash2index:
                        wv.hash2index[ngram_hash] = num_new_ngrams + self.old_hash2index_len
                        num_new_ngrams += 1
                    buckets.append(wv.hash2index[ngram_hash])
                wv.buckets_word[vocab.index] = tuple(buckets)

            wv.num_ngram_vectors += num_new_ngrams
            logger.info("Number of new ngrams is %d", num_new_ngrams)

            rand_obj = np.random
            rand_obj.seed(self.seed)
            new_vocab_rows = rand_obj.uniform(
                -1.0 / wv.vector_size, 1.0 / wv.vector_size,
                (len(wv.vocab) - vocabulary.old_vocab_len, wv.vector_size)
            ).astype(REAL)
            new_vocab_lockf_rows = ones(
                (len(wv.vocab) - vocabulary.old_vocab_len, wv.vector_size), dtype=REAL)
            new_ngram_rows = rand_obj.uniform(
                -1.0 / wv.vector_size, 1.0 / wv.vector_size,
                (len(wv.hash2index) - self.old_hash2index_len, wv.vector_size)
            ).astype(REAL)
            new_ngram_lockf_rows = ones(
                (len(wv.hash2index) - self.old_hash2index_len, wv.vector_size), dtype=REAL)

            wv.vectors_vocab = vstack([wv.vectors_vocab, new_vocab_rows])
            self.vectors_vocab_lockf = vstack([self.vectors_vocab_lockf, new_vocab_lockf_rows])
            wv.vectors_ngrams = vstack([wv.vectors_ngrams, new_ngram_rows])
            self.vectors_ngrams_lockf = vstack([self.vectors_ngrams_lockf, new_ngram_lockf_rows])

    def reset_ngrams_weights(self, wv):
        """Reset all projection weights to an initial (untrained) state,
        but keep the existing vocabulary and their ngrams.

        """
        rand_obj = np.random
        rand_obj.seed(self.seed)
        for index in range(len(wv.vocab)):
            wv.vectors_vocab[index] = rand_obj.uniform(
                -1.0 / wv.vector_size, 1.0 / wv.vector_size, wv.vector_size
            ).astype(REAL)
        for index in range(len(wv.hash2index)):
            wv.vectors_ngrams[index] = rand_obj.uniform(
                -1.0 / wv.vector_size, 1.0 / wv.vector_size, wv.vector_size
            ).astype(REAL)

    def get_vocab_word_vecs(self, wv):
        """Calculate vectors for words in vocabulary and stores them in `vectors`."""
        for w, v in wv.vocab.items():
            word_vec = np.copy(wv.vectors_vocab[v.index])
            ngrams = _compute_ngrams(w, wv.min_n, wv.max_n)
            ngram_weights = wv.vectors_ngrams
            for ngram in ngrams:
                word_vec += ngram_weights[wv.hash2index[_ft_hash(ngram) % self.bucket]]
            word_vec /= (len(ngrams) + 1)
            wv.vectors[v.index] = word_vec

    def init_ngrams_post_load(self, file_name, wv):
        """
        Computes ngrams of all words present in vocabulary and stores vectors for only those ngrams.
        Vectors for other ngrams are initialized with a random uniform distribution in FastText. These
        vectors are discarded here to save space.

        """
        wv.vectors = np.zeros((len(wv.vocab), wv.vector_size), dtype=REAL)

        for w, vocab in wv.vocab.items():
            wv.vectors[vocab.index] += np.array(wv.vectors_ngrams[vocab.index])

        ngram_indices = []
        wv.num_ngram_vectors = 0
        for word in wv.vocab.keys():
            for ngram in _compute_ngrams(word, wv.min_n, wv.max_n):
                ngram_hash = _ft_hash(ngram) % self.bucket
                if ngram_hash in wv.hash2index:
                    continue
                wv.hash2index[ngram_hash] = len(ngram_indices)
                ngram_indices.append(len(wv.vocab) + ngram_hash)
        wv.num_ngram_vectors = len(ngram_indices)
        wv.vectors_ngrams = wv.vectors_ngrams.take(ngram_indices, axis=0)

        ngram_weights = wv.vectors_ngrams

        logger.info(
            "loading weights for %s words for fastText model from %s",
            len(wv.vocab), file_name
        )

        for w, vocab in wv.vocab.items():
            word_ngrams = _compute_ngrams(w, wv.min_n, wv.max_n)
            for word_ngram in word_ngrams:
                vec_idx = wv.hash2index[_ft_hash(word_ngram) % self.bucket]
                wv.vectors[vocab.index] += np.array(ngram_weights[vec_idx])

            wv.vectors[vocab.index] /= (len(word_ngrams) + 1)
        logger.info(
            "loaded %s weight matrix for fastText model from %s",
            wv.vectors.shape, file_name
        )
