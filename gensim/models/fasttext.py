#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Chinmaya Pancholi <chinmayapancholi13@gmail.com>, Shiva Manne <s.manne@rare-technologies.com>
# Copyright (C) 2017 RaRe Technologies s.r.o.

"""Learn word representations via fasttext's "skip-gram and CBOW models", using either
hierarchical softmax or negative sampling [1]_.

NOTE: There are more ways to get word vectors in Gensim than just FastText.
See wrappers for VarEmbed and WordRank or Word2Vec

This module allows training a word embedding from a training corpus with the additional ability
to obtain word vectors for out-of-vocabulary words.

For a tutorial on gensim's native fasttext, refer to the noteboook --
https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/FastText_Tutorial.ipynb

**Make sure you have a C compiler before installing gensim, to use optimized (compiled) fasttext training**

.. [1] P. Bojanowski, E. Grave, A. Joulin, T. Mikolov
       Enriching Word Vectors with Subword Information. In arXiv preprint arXiv:1607.04606.

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
        vocab dictionary. Called internally from `FastText.train()`.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from fasttext_inner instead.

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
        vocab dictionary. Called internally from `FastText.train()`.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from fasttext_inner instead.

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
    """Class for training, using and evaluating word representations learned using method described in
    `Enriching Word Vectors with Subword Information` aka Fasttext.

    The model can be stored/loaded via its `save()` and `load()` methods, or loaded in a format
    compatible with the original fasttext implementation via `load_fasttext_format()`.

    """
    def __init__(
            self, sentences=None, sg=0, hs=0, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, word_ngrams=1, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, min_n=3, max_n=6, sorted_vocab=1,
            bucket=2000000, trim_rule=None, batch_words=MAX_WORDS_IN_BATCH):
        """Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :Word2Vec:`BrownCorpus`, :Word2Vec:`Text8Corpus` or :Word2Vec:`LineSentence` in
        this module for such examples.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `sg` defines the training algorithm. By default (`sg=0`), CBOW is used.
        Otherwise (`sg=1`), skip-gram is employed.

        `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the current and predicted word within a sentence.

        `alpha` is the initial learning rate (will linearly drop to `min_alpha` as training progresses).

        `seed` = for the random number generator. Initial vectors for each
        word are seeded with a hash of the concatenation of word + str(seed).
        Note that for a fully deterministically-reproducible run, you must also limit the model to
        a single worker thread, to eliminate ordering jitter from OS thread scheduling. (In Python
        3, reproducibility between interpreter launches also requires use of the PYTHONHASHSEED
        environment variable to control hash randomization.)

        `min_count` = ignore all words with total frequency lower than this.

        `max_vocab_size` = limit RAM during vocabulary building; if there are more unique
        words than this, then prune the infrequent ones. Every 10 million word types
        need about 1GB of RAM. Set to `None` for no limit (default).

        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
            default is 1e-3, useful range is (0, 1e-5).

        `workers` = use this many worker threads to train the model (=faster training with multicore machines).

        `hs` = if 1, hierarchical softmax will be used for model training.
        If set to 0 (default), and `negative` is non-zero, negative sampling will be used.

        `negative` = if > 0, negative sampling will be used, the int for negative
        specifies how many "noise words" should be drawn (usually between 5-20).
        Default is 5. If set to 0, no negative samping is used.

        `cbow_mean` = if 0, use the sum of the context word vectors. If 1 (default), use the mean.
        Only applies when cbow is used.

        `hashfxn` = hash function to use to randomly initialize weights, for increased
        training reproducibility. Default is Python's rudimentary built in hash function.

        `iter` = number of iterations (epochs) over the corpus. Default is 5.

        `trim_rule` = vocabulary trimming rule, specifies whether certain words should remain
        in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count).
        Can be None (min_count will be used), or a callable that accepts parameters (word, count, min_count) and
        returns either `utils.RULE_DISCARD`, `utils.RULE_KEEP` or `utils.RULE_DEFAULT`.
        Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part
        of the model.

        `sorted_vocab` = if 1 (default), sort the vocabulary by descending frequency before
        assigning word indexes.

        `batch_words` = target size (in words) for batches of examples passed to worker threads (and
        thus cython routines). Default is 10000. (Larger batches will be passed if individual
        texts are longer than 10000 words, but the standard cython code truncates to that maximum.)

        `min_n` = min length of char ngrams to be used for training word representations. Default is 3.

        `max_n` = max length of char ngrams to be used for training word representations. Set `max_n` to be
        lesser than `min_n` to avoid char ngrams being used. Default is 6.

        `bucket` = Word and character ngram features are hashed into a fixed number of buckets, in order to limit the
        memory usage of the model. This option specifies the number of buckets used by the model. Default: 2000000

        Example
        -------
        Initialize and train a `FastText` model:

        >>> model = FastText(sg=1, hs=0,window=2, negative=5, iter=4)
        >>> model.build_vocab(sentences)
        >>> model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

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
        self.wv = FastTextKeyedVectors()
        self.wv.min_n = self.min_n
        self.wv.max_n = self.max_n

    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False):
        """Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.
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
        """Computes ngrams of all words present in vocabulary and stores vectors for only those ngrams.
        Vectors for other ngrams are initialized with a random uniform distribution in FastText.
        If `update` is `True`, the new vocab words and their new ngrams word vectors are initialized
        with random uniform distribution and updated/added to the exisiting vocab word and ngram vectors.
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
        """Reset all projection weights to an initial (untrained) state, but keep the
        existing vocabulary and their ngrams."""
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
        progres-percentage logging, either total_examples (count of sentences) or total_words (count of
        raw words in sentences) MUST be provided. (If the corpus is the same as was provided to
        `build_vocab()`, the count of examples in that corpus will be available in the model's
        `corpus_count` property.)

        To avoid common mistakes around the model's ability to do multiple training passes itself, an
        explicit `epochs` argument MUST be provided. In the common and recommended case, where `train()`
        is only called once, the model's cached `iter` value should be supplied as `epochs` value.
        """
        self.neg_labels = []
        if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.

        Word2Vec.train(self, sentences, total_examples=self.corpus_count, epochs=self.iter,
            start_alpha=self.alpha, end_alpha=self.min_alpha)
        self.get_vocab_word_vecs()

    def __getitem__(self, word):
        return self.word_vec(word)

    def get_vocab_word_vecs(self):
        for w, v in self.wv.vocab.items():
            word_vec = np.copy(self.wv.syn0_vocab[v.index])
            ngrams = self.wv.ngrams_word[w]
            ngram_weights = self.wv.syn0_ngrams
            for ngram in ngrams:
                word_vec += ngram_weights[self.wv.ngrams[ngram]]
            word_vec /= (len(ngrams) + 1)
            self.wv.syn0[v.index] = word_vec

    def word_vec(self, word, use_norm=False):
        """Accept a single word as input.
        Returns the word's representations in vector space, as a 1D numpy array.

        The word can be out-of-vocabulary as long as ngrams for the word are present.
        For words with all ngrams absent, a KeyError is raised.

        Example
        -------

          >>> trained_model['office']
          array([ -1.40128313e-02, ...])

        """
        return FastTextKeyedVectors.word_vec(self.wv, word, use_norm=use_norm)

    @classmethod
    def load_fasttext_format(cls, *args, **kwargs):
        return Ft_Wrapper.load_fasttext_format(*args, **kwargs)

    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'syn0_vocab_norm', 'syn0_ngrams_norm'])
        super(FastText, self).save(*args, **kwargs)
