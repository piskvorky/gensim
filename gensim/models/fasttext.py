#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Chinmaya Pancholi <chinmayapancholi13@gmail.com>, Shiva Manne <s.manne@rare-technologies.com>
# Copyright (C) 2017 RaRe Technologies s.r.o.

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

import numpy as np
from numpy import zeros, ones, vstack, random, sum as np_sum, empty, float32 as REAL

# from gensim.models.word2vec import Word2Vec, train_sg_pair, train_cbow_pair
from gensim.models.word2vec import Word2VecVocab, Word2VecTrainables
from gensim.models.keyedvectors import FastTextKeyedVectors, Vocab
from gensim.models.wrappers.fasttext import FastText as Ft_Wrapper, compute_ngrams, ft_hash
from gensim.models.base_any2vec import BaseAny2VecModel, BaseVocabBuilder, BaseModelTrainables

from types import GeneratorType

logger = logging.getLogger(__name__)

try:
    from gensim.models.fasttext_inner import train_batch_sg, train_batch_cbow
    from gensim.models.fasttext_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
    logger.debug('Fast version of Fasttext is being used')

except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    raise RuntimeError("Support for Python/Numpy implementations has been continued.")


class FastText(BaseAny2VecModel):
    """Class for training, using and evaluating word representations learned using method
    described in [1]_ aka Fasttext.

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
            Defines the training algorithm. If 1, CBOW is used, otherwise, skip-gram is employed.
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
        self.callbacks = callbacks

        if FAST_VERSION == -1:
            logger.warning('Slow version of %s is being used', __name__)
        else:
            logger.debug('Fast version of %s is being used', __name__)

        self.sg = int(sg)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.min_alpha_yet_reached = float(alpha)  # To warn user if alpha increases
        self.window = int(window)
        self.random = random.RandomState(seed)
        self.min_alpha = float(min_alpha)
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.running_training_loss = 0
        if self.word_ngrams <= 1 and self.max_n == 0:
            bucket = 0
        self.word_ngrams = word_ngrams

        self.wv = FastTextKeyedVectors()
        self.vocabulary = FastTextVocab(
            max_vocab_size=max_vocab_size, min_count=min_count, sample=sample,
            sorted_vocab=sorted_vocab, null_word=null_word, min_n=min_n, max_n=max_n)
        self.trainables = FastTextTrainables(
            vector_size=size, seed=seed, alpha=alpha, min_alpha=min_alpha, hs=hs, negative=negative,
            hashfxn=hashfxn, bucket=bucket)

        super(FastText, self).__init__(
            workers=workers, vector_size=size, epochs=iter, callbacks=callbacks, batch_words=batch_words)

        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")
            self.build_vocab(sentences, trim_rule=trim_rule)
            self.train(
                sentences, total_examples=self.corpus_count, epochs=self.iter,
                start_alpha=self.alpha, end_alpha=self.min_alpha)
        else:
            if trim_rule is not None:
                logger.warning(
                    "The rule, if given, is only used to prune vocabulary during build_vocab() "
                    "and is not stored as part of the model. Model initialized without sentences. "
                    "trim_rule provided, if any, will be ignored.")

    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False):
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
        update: bool
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
            self.vocabulary.old_vocab_len = len(self.vocabulary.vocab)
            self.trainables.old_hash2index_len = len(self.trainables.hash2index)

        super(FastText, self).build_vocab(
            sentences, keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, progress_per=progress_per, update=update)
        self.init_ngrams(update=update)

    def reset_ngram_weights(self):
        """Reset all projection weights to an initial (untrained) state,
        but keep the existing vocabulary and their ngrams.

        """
        rand_obj = np.random
        rand_obj.seed(self.seed)
        for index in range(len(self.wv.vocab)):
            self.wv.syn0_vocab[index] = rand_obj.uniform(
                -1.0 / self.vector_size, 1.0 / self.vector_size, self.vector_size
            ).astype(REAL)
        for index in range(len(self.wv.hash2index)):
            self.wv.syn0_ngrams[index] = rand_obj.uniform(
                -1.0 / self.vector_size, 1.0 / self.vector_size, self.vector_size
            ).astype(REAL)

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
              word_count=0, queue_factor=2, report_delay=1.0):
        """Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For FastText, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)

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

        Examples
        --------
        >>> from gensim.models import FastText
        >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>>
        >>> model = FastText(min_count=1)
        >>> model.build_vocab(sentences)
        >>> model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

        """

        FastText.train(
            self, sentences, total_examples=self.corpus_count, epochs=self.iter,
            start_alpha=self.alpha, end_alpha=self.min_alpha)
        self.get_vocab_word_vecs()

    @classmethod
    def load_fasttext_format(cls, *args, **kwargs):
        """Load a :class:`~gensim.models.fasttext.FastText` model from a format compatible with
        the original fasttext implementation.

        Parameters
        ----------
        fname : str
            Path to the file.

        """
        return Ft_Wrapper.load_fasttext_format(*args, **kwargs)

    def save(self, *args, **kwargs):
        """Save the model. This saved model can be loaded again using :func:`~gensim.models.fasttext.FastText.load`,
        which supports online training and getting vectors for out-of-vocabulary words.

        Parameters
        ----------
        fname : str
            Path to the file.

        """
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_norm', 'vectors_vocab_norm', 'vectors_ngrams_norm'])
        super(FastText, self).save(*args, **kwargs)


class FastTextVocab(Word2VecVocab):
    def __init__(self, max_vocab_size=None, min_count=5, sample=1e-3, sorted_vocab=True, null_word=0,
                 min_n=3, max_n=6):
        super(FastTextVocab, self).__init(
            max_vocab_size=max_vocab_size, min_count=min_count, sample=sample,
            sorted_vocab=sorted_vocab, null_word=null_word)
        self.min_n = min_n
        self.max_n = max_n

    def prepare_vocab(self, update=False, keep_raw_vocab=False, trim_rule=None, min_count=None,
                      sample=None, dry_run=False):
        super(FastTextVocab, self).prepare_vocab(
            update=update, keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, min_count=min_count,
            sample=sample, dry_run=dry_run)
        self.build_ngrams(update=update)

    def build_ngrams(self, update=False):
        if not update:
            self.ngrams_word = {}
            for w, v in self.vocab.items():
                self.ngrams_word[w] = compute_ngrams(w, self.min_n, self.max_n)
        else:
            for w, v in self.vocab.items():
                self.ngrams_word[w] = compute_ngrams(w, self.min_n, self.max_n)


class FastTextTrainables(Word2VecTrainables):
    def __init__(self, vector_size=100, seed=1, alpha=0.025, min_alpha=0.0001, hs=0, negative=5,
                 hashfxn=hash, bucket=2000000):
        super(FastTextVocab, self).__init(
            vector_size=vector_size, seed=seed, alpha=alpha, min_alpha=min_alpha, hs=hs,
            negative=negative, hashfxn=hashfxn)
        self.bucket = bucket

    def prepare_weights(self, update=False, vocabulary=None):
        super(FastTextVocab, self).prepare_weights(update=update, vocabulary=vocabulary)
        self.init_ngrams(update=update, vocabulary=vocabulary)

    def init_ngrams_weights(self, update=False, vocabulary=None):
        """Compute ngrams of all words present in vocabulary and stores vectors for only those ngrams.
        Vectors for other ngrams are initialized with a random uniform distribution in FastText.

        Parameters
        ----------
        update : bool
            If True, the new vocab words and their new ngrams word vectors are initialized
            with random uniform distribution and updated/added to the existing vocab word and ngram vectors.

        """
        if not update:
            self.ngrams = {}
            self.vectors_vocab = empty((len(vocabulary.vocab), self.vector_size), dtype=REAL)
            self.vectors_vocab_lockf = ones((len(vocabulary.vocab), self.vector_size), dtype=REAL)

            self.vocabulary_ngrams = empty((self.bucket, self.vector_size), dtype=REAL)
            self.vocabulary_ngrams_lockf = ones((self.bucket, self.vector_size), dtype=REAL)

            all_ngrams = []
            for w, ngrams in vocabulary.ngrams_word.items():
                all_ngrams += ngrams

            all_ngrams = list(set(all_ngrams))
            self.num_ngram_vectors = len(all_ngrams)
            logger.info("Total number of ngrams is %d", len(all_ngrams))

            self.hash2index = {}
            ngram_indices = []
            new_hash_count = 0
            for i, ngram in enumerate(all_ngrams):
                ngram_hash = ft_hash(ngram) % self.bucket
                if ngram_hash in self.wv.hash2index:
                    self.ngrams[ngram] = self.hash2index[ngram_hash]
                else:
                    ngram_indices.append(ngram_hash % self.bucket)
                    self.hash2index[ngram_hash] = new_hash_count
                    self.ngrams[ngram] = self.hash2index[ngram_hash]
                    new_hash_count = new_hash_count + 1

            self.vectors_ngrams = self.vectors_ngrams.take(ngram_indices, axis=0)
            self.vectors_ngrams_lockf = self.vectors_ngrams_lockf.take(ngram_indices, axis=0)
            self.reset_ngram_weights()
        else:
            new_ngrams = []
            for w, ngrams in vocabulary.ngrams_word.items():
                new_ngrams += [ng for ng in ngrams if ng not in self.ngrams]

            new_ngrams = list(set(new_ngrams))
            logger.info("Number of new ngrams is %d", len(new_ngrams))
            new_hash_count = 0
            for i, ngram in enumerate(new_ngrams):
                ngram_hash = ft_hash(ngram) % self.bucket
                if ngram_hash not in self.hash2index:
                    self.hash2index[ngram_hash] = new_hash_count + self.old_hash2index_len
                    self.ngrams[ngram] = self.hash2index[ngram_hash]
                    new_hash_count = new_hash_count + 1
                else:
                    self.ngrams[ngram] = self.hash2index[ngram_hash]

            rand_obj = np.random
            rand_obj.seed(self.seed)
            new_vocab_rows = rand_obj.uniform(
                -1.0 / self.vector_size, 1.0 / self.vector_size,
                (len(vocabulary.vocab) - vocabulary.old_vocab_len, self.vector_size)
            ).astype(REAL)
            new_vocab_lockf_rows = ones((len(vocabulary.vocab) - vocabulary.old_vocab_len,
                self.vector_size), dtype=REAL)
            new_ngram_rows = rand_obj.uniform(
                -1.0 / self.vector_size, 1.0 / self.vector_size,
                (len(self.hash2index) - self.old_hash2index_len, self.vector_size)
            ).astype(REAL)
            new_ngram_lockf_rows = ones(
                (len(self.hash2index) - self.old_hash2index_len,
                self.vector_size),
                dtype=REAL)

            self.vectors_vocab = vstack([self.vectors_vocab, new_vocab_rows])
            self.vectors_vocab_lockf = vstack([self.vectors_vocab_lockf, new_vocab_lockf_rows])
            self.vectors_ngrams = vstack([self.vectors_ngrams, new_ngram_rows])
            self.vectors_ngrams_lockf = vstack([self.vectors_ngrams_lockf, new_ngram_lockf_rows])
