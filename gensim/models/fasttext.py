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
from numpy import zeros, ones, vstack, sum as np_sum, empty, float32 as REAL

from gensim.models.word2vec import Word2Vec, train_sg_pair, train_cbow_pair
from gensim.models.wrappers.fasttext import FastTextKeyedVectors
from gensim.models.wrappers.fasttext import FastText as Ft_Wrapper, compute_ngrams, ft_hash

logger = logging.getLogger(__name__)

try:
    from gensim.models.fasttext_inner import train_batch_sg, train_batch_cbow
    from gensim.models.fasttext_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
    logger.debug('Fast version of Fasttext is being used')

except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    logger.warning('Slow version of Fasttext is being used')
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
                           model.wv.vocab[w].sample_int > model.random.rand() * 2**32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)
                start = max(0, pos - model.window + reduced_window)
                window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
                word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]

                word2_subwords = []
                vocab_subwords_indices = []
                ngrams_subwords_indices = []

                for index in word2_indices:
                    vocab_subwords_indices += [index]
                    word2_subwords += model.wv.ngrams_word[model.wv.index2word[index]]

                for subword in word2_subwords:
                    ngrams_subwords_indices.append(model.wv.ngrams[subword])

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
                           model.wv.vocab[w].sample_int > model.random.rand() * 2**32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)

                subwords_indices = [word.index]
                word2_subwords = model.wv.ngrams_word[model.wv.index2word[word.index]]

                for subword in word2_subwords:
                    subwords_indices.append(model.wv.ngrams[subword])

                for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                    if pos2 != pos:  # don't train on the `word` itself
                        train_sg_pair(model, model.wv.index2word[word2.index], subwords_indices, alpha, is_ft=True)

            result += len(word_vocabs)
        return result


class FastText(Word2Vec):
    """Class for training, using and evaluating word representations learned using method
    described in [1]_ aka Fasttext.

    The model can be stored/loaded via its :meth:`~gensim.models.fasttext.FastText.save()` and
    :meth:`~gensim.models.fasttext.FastText.load()` methods, or loaded in a format compatible with the original
    fasttext implementation via :meth:`~gensim.models.fasttext.FastText.load_fasttext_format()`.

    """
    def __init__(
            self, sentences=None, sg=0, hs=0, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, word_ngrams=1, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, min_n=3, max_n=6, sorted_vocab=1,
            bucket=2000000, trim_rule=None, batch_words=MAX_WORDS_IN_BATCH):
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
        # fastText specific params
        self.bucket = bucket
        self.word_ngrams = word_ngrams
        self.min_n = min_n
        self.max_n = max_n
        if self.word_ngrams <= 1 and self.max_n == 0:
            self.bucket = 0

        super(FastText, self).__init__(
            sentences=sentences, size=size, alpha=alpha, window=window, min_count=min_count,
            max_vocab_size=max_vocab_size, sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,
            sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, hashfxn=hashfxn, iter=iter, null_word=null_word,
            trim_rule=trim_rule, sorted_vocab=sorted_vocab, batch_words=batch_words)

    def initialize_word_vectors(self):
        """Initializes FastTextKeyedVectors instance to store all vocab/ngram vectors for the model."""
        self.wv = FastTextKeyedVectors()
        self.wv.min_n = self.min_n
        self.wv.max_n = self.max_n

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
            self.old_vocab_len = len(self.wv.vocab)
            self.old_hash2index_len = len(self.wv.hash2index)

        super(FastText, self).build_vocab(
            sentences, keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, progress_per=progress_per, update=update)
        self.init_ngrams(update=update)

    def init_ngrams(self, update=False):
        """Compute ngrams of all words present in vocabulary and stores vectors for only those ngrams.
        Vectors for other ngrams are initialized with a random uniform distribution in FastText.

        Parameters
        ----------
        update : bool
            If True, the new vocab words and their new ngrams word vectors are initialized
            with random uniform distribution and updated/added to the existing vocab word and ngram vectors.

        """
        if not update:
            self.wv.ngrams = {}
            self.wv.syn0_vocab = empty((len(self.wv.vocab), self.vector_size), dtype=REAL)
            self.syn0_vocab_lockf = ones((len(self.wv.vocab), self.vector_size), dtype=REAL)

            self.wv.syn0_ngrams = empty((self.bucket, self.vector_size), dtype=REAL)
            self.syn0_ngrams_lockf = ones((self.bucket, self.vector_size), dtype=REAL)

            all_ngrams = []
            for w, v in self.wv.vocab.items():
                self.wv.ngrams_word[w] = compute_ngrams(w, self.min_n, self.max_n)
                all_ngrams += self.wv.ngrams_word[w]

            all_ngrams = list(set(all_ngrams))
            self.num_ngram_vectors = len(all_ngrams)
            logger.info("Total number of ngrams is %d", len(all_ngrams))

            self.wv.hash2index = {}
            ngram_indices = []
            new_hash_count = 0
            for i, ngram in enumerate(all_ngrams):
                ngram_hash = ft_hash(ngram)
                if ngram_hash in self.wv.hash2index:
                    self.wv.ngrams[ngram] = self.wv.hash2index[ngram_hash]
                else:
                    ngram_indices.append(ngram_hash % self.bucket)
                    self.wv.hash2index[ngram_hash] = new_hash_count
                    self.wv.ngrams[ngram] = self.wv.hash2index[ngram_hash]
                    new_hash_count = new_hash_count + 1

            self.wv.syn0_ngrams = self.wv.syn0_ngrams.take(ngram_indices, axis=0)
            self.syn0_ngrams_lockf = self.syn0_ngrams_lockf.take(ngram_indices, axis=0)
            self.reset_ngram_weights()
        else:
            new_ngrams = []
            for w, v in self.wv.vocab.items():
                self.wv.ngrams_word[w] = compute_ngrams(w, self.min_n, self.max_n)
                new_ngrams += [ng for ng in self.wv.ngrams_word[w] if ng not in self.wv.ngrams]

            new_ngrams = list(set(new_ngrams))
            logger.info("Number of new ngrams is %d", len(new_ngrams))
            new_hash_count = 0
            for i, ngram in enumerate(new_ngrams):
                ngram_hash = ft_hash(ngram)
                if ngram_hash not in self.wv.hash2index:
                    self.wv.hash2index[ngram_hash] = new_hash_count + self.old_hash2index_len
                    self.wv.ngrams[ngram] = self.wv.hash2index[ngram_hash]
                    new_hash_count = new_hash_count + 1
                else:
                    self.wv.ngrams[ngram] = self.wv.hash2index[ngram_hash]

            rand_obj = np.random
            rand_obj.seed(self.seed)
            new_vocab_rows = rand_obj.uniform(
                -1.0 / self.vector_size, 1.0 / self.vector_size,
                (len(self.wv.vocab) - self.old_vocab_len, self.vector_size)
            ).astype(REAL)
            new_vocab_lockf_rows = ones((len(self.wv.vocab) - self.old_vocab_len, self.vector_size), dtype=REAL)
            new_ngram_rows = rand_obj.uniform(
                -1.0 / self.vector_size, 1.0 / self.vector_size,
                (len(self.wv.hash2index) - self.old_hash2index_len, self.vector_size)
            ).astype(REAL)
            new_ngram_lockf_rows = ones(
                (len(self.wv.hash2index) - self.old_hash2index_len,
                self.vector_size),
                dtype=REAL)

            self.wv.syn0_vocab = vstack([self.wv.syn0_vocab, new_vocab_rows])
            self.syn0_vocab_lockf = vstack([self.syn0_vocab_lockf, new_vocab_lockf_rows])
            self.wv.syn0_ngrams = vstack([self.wv.syn0_ngrams, new_ngram_rows])
            self.syn0_ngrams_lockf = vstack([self.syn0_ngrams_lockf, new_ngram_lockf_rows])

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
        self.neg_labels = []
        if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.

        Word2Vec.train(
            self, sentences, total_examples=self.corpus_count, epochs=self.iter,
            start_alpha=self.alpha, end_alpha=self.min_alpha)
        self.get_vocab_word_vecs()

    def __getitem__(self, word):
        """Get `word` representations in vector space, as a 1D numpy array.

        Parameters
        ----------
        word : str
            A single word whose vector needs to be returned.

        Returns
        -------
        :class:`numpy.ndarray`
            The word's representations in vector space, as a 1D numpy array.

        Raises
        ------
        KeyError
            For words with all ngrams absent, a KeyError is raised.

        Example
        -------
        >>> from gensim.models import FastText
        >>> from gensim.test.utils import datapath
        >>>
        >>> trained_model = FastText.load_fasttext_format(datapath('lee_fasttext'))
        >>> meow_vector = trained_model['hello']  # get vector for word

        """
        return self.word_vec(word)

    def get_vocab_word_vecs(self):
        """Calculate vectors for words in vocabulary and stores them in `wv.syn0`."""
        for w, v in self.wv.vocab.items():
            word_vec = np.copy(self.wv.syn0_vocab[v.index])
            ngrams = self.wv.ngrams_word[w]
            ngram_weights = self.wv.syn0_ngrams
            for ngram in ngrams:
                word_vec += ngram_weights[self.wv.ngrams[ngram]]
            word_vec /= (len(ngrams) + 1)
            self.wv.syn0[v.index] = word_vec

    def word_vec(self, word, use_norm=False):
        """Get the word's representations in vector space, as a 1D numpy array.

        Parameters
        ----------
        word : str
            A single word whose vector needs to be returned.
        use_norm : bool
            If True, returns normalized vector.

        Returns
        -------
        :class:`numpy.ndarray`
            The word's representations in vector space, as a 1D numpy array.

        Raises
        ------
        KeyError
            For words with all ngrams absent, a KeyError is raised.

        Example
        -------
        >>> from gensim.models import FastText
        >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>>
        >>> model = FastText(sentences, min_count=1)
        >>> meow_vector = model.word_vec('meow')  # get vector for word

        """
        return FastTextKeyedVectors.word_vec(self.wv, word, use_norm=use_norm)

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
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'syn0_vocab_norm', 'syn0_ngrams_norm'])
        super(FastText, self).save(*args, **kwargs)
