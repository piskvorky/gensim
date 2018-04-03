#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Produce sentence vectors with deep learning via sent2vec model using negative sampling [1]_.

The training algorithms were originally ported from the C package [2]_. and extended with additional functionality.


Examples
--------
Initialize a model with e.g.

>>> from gensim.models import Sent2Vec
>>> from gensim.test.utils import common_texts
>>>
>>> model = Sent2Vec(common_texts, size=100, min_count=1)

Or

>>> model = Sent2Vec(size=100, min_count=1)
>>> model.build_vocab(common_texts)
>>> model.train(common_texts)
145

The sentence vectors are stored in a numpy array

>>> vector = model[['computer', 'interface']] # vector of a sentence

You can perform the NLP similarity task with the model

>>> similarity = model.similarity(['graph', 'minors', 'trees'], ['eps', 'user', 'interface', 'system'])


References
----------
.. [1] Matteo Pagliardini, Prakhar Gupta, Martin Jaggi.
       Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features.
       https://arxiv.org/abs/1703.02507
.. [2] https://github.com/epfml/sent2vec

"""
from __future__ import division
import logging
import numpy as np
from numpy import dot
from gensim import utils, matutils
from gensim.utils import tokenize
from random import randint
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from scipy.special import expit
import os

try:
    from gensim.models.sent2vec_inner import _do_train_job_fast, FAST_VERSION
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1

logger = logging.getLogger(__name__)


class Entry(object):
    """Class for populating Sent2Vec's dictionary."""

    def __init__(self, word=None, count=0):
        """

        Parameters
        ----------
        word : str, optional
            Actual vocabulary word.
        count : int
            Number of times the word occurs in the vocabulary.

        """
        self.word = word
        self.count = count


class Sent2VecVocab(object):
    """Class for maintaining Sent2Vec vocbulary. Provides functionality for storing and training
    word and character ngrams.

    """

    def __init__(self, sample, bucket, minn, maxn, max_vocab_size, min_count=5, max_line_size=1024):
        """

        Parameters
        ----------
        sample : float
            Threshold for configuring which higher-frequency words are randomly downsampled.
        bucket : int
            Number of hash buckets for vocabulary.
        minn : int
            Min length of char ngrams.
        maxn : int
            Max length of char ngrams.
        max_vocab_size : int
            Limit RAM during vocabulary building; if there are more unique words than this,
            then prune the infrequent ones.
        max_line_size : int, optional
            Maximum number of characters in a sentence.

        """
        self.max_vocab_size = max_vocab_size
        self.max_line_size = max_line_size
        self.words = []
        self.word2int = [-1] * max_vocab_size
        self.pdiscard = []
        self.ntokens = 0
        self.size = 0
        self.sample = sample
        self.bucket = bucket
        self.maxn = maxn
        self.minn = minn
        self.min_count = min_count

    @staticmethod
    def hash_(word):
        """Compute hash of given word.

        Parameters
        ----------
        word : str
            Actual vocabulary word.
        Returns
        -------
        int
            Hash of the given word.

        """
        h = 2166136261
        for i in range(len(word)):
            h = h ^ ord(word[i])
            h = h * 16777619
        return h

    def find(self, word):
        """Find hash of given word. The word may or may not be present in the vocabulary.

        Parameters
        ----------
        word : str
            Actual vocabulary word.

        Returns
        -------
        int
            Hash of the given word.

        """
        h = self.hash_(word) % self.max_vocab_size
        while self.word2int[h] != -1 and self.words[self.word2int[h]].word != word:
            h = (h + 1) % self.max_vocab_size
        return h

    def add(self, word):
        """Add given word to vocabulary.

        Parameters
        ----------
        word : str
            Actual vocabulary word.

        """
        h = self.find(word)
        self.ntokens += 1
        if self.word2int[h] == -1:
            e = Entry(word=word, count=1)
            self.words.append(e)
            self.word2int[h] = self.size
            self.size += 1
        else:
            self.words[self.word2int[h]].count += 1

    def read(self, sentences, min_count):
        """Process all words present in sentences.
        Initialize discard table to downsampled higher frequency words according to given sampling threshold.
        Also initialize character ngrams for all words and threshold lower frequency words if their count
        is less than a given value `min_count`.

        Parameters
        ----------
        sentences : iterable of iterable of str
            Stream of sentences, see :class:`~gensim.models.sent2vec.TorontoCorpus` in this module for such examples.
        min_count : int
            Value for thresholding lower frequency words.

        """
        min_threshold = 1
        for sentence_no, sentence in enumerate(sentences):
            for word in sentence:
                self.add(word)
                if self.ntokens % 1000000 == 0:
                    logger.info("Read %.2f M words", self.ntokens / 1000000)
                if self.size > 0.75 * self.max_vocab_size:
                    min_threshold += 1
                    self.threshold(min_threshold)

        self.threshold(min_count)
        self.init_table_discard()
        logger.info("Read %.2f M words", self.ntokens / 1000000)
        if self.size == 0:
            raise RuntimeError("Empty vocabulary. Try a smaller min_count value.")
        return sentence_no + 1

    def threshold(self, t):
        """Remove words from vocabulary having count lower than `t`.

        Parameters
        ----------
        t : int
            Value for thresholding lower frequency words.

        """
        self.words = [entry for entry in self.words if entry.count > t]
        self.size = 0
        self.word2int = [-1] * self.max_vocab_size
        for entry in self.words:
            h = self.find(entry.word)
            self.word2int[h] = self.size
            self.size += 1

    def init_table_discard(self):
        """Downsample higher frequency words. Initializing discard table according to given sampling threshold."""

        for i in range(self.size):
            f = self.words[i].count / self.ntokens
            self.pdiscard.append(((self.sample / f) ** 0.5) + (self.sample / f))

    def add_ngrams_train(self, context, n, k):
        """
        Training word ngrams for a given context and target word.
        Parameters
        ----------
        context : list
            List of word ids.
        n : int
            Number of word ngrams.
        k : int
            Number of word ngrams dropped while
        training a Sent2Vec model.
        Returns
        -------
        line : list
            List of word and word ngram ids.
        """

        line = list(context)
        num_discarded = 0
        line_size = len(line)
        discard = [False] * line_size
        while num_discarded < k and line_size - num_discarded > 2:
            token_to_discard = randint(0, line_size - 1)
            if discard[token_to_discard] is False:
                discard[token_to_discard] = True
                num_discarded += 1
        for i in range(line_size):
            if discard[i] is True:
                continue
            h = line[i]
            for j in range(i + 1, line_size):
                if j >= i + n or discard[j] is True:
                    break
                h = h * 116049371 + line[j]
                line.append(self.size + (h % self.bucket))
        return line

    def add_ngrams(self, context, n):
        """Computing word ngrams for given sentence while inferring sentence vector.

        Parameters
        ----------
        context : list of int
            List of word ids.
        n : int
            Number of word ngrams.

        Returns
        -------
        list of int
            List of word and word ngram ids.

        """

        line = list(context)
        line_size = len(context)
        for i in range(line_size):
            h = line[i]
            for j in range(i + 1, line_size):
                if j >= i + n:
                    break
                h = h * 116049371 + line[j]
                line.append(self.size + (h % self.bucket))
        return line

    def get_line(self, sentence):
        """Converting sentence to a list of word ids inferred from the dictionary.

        Parameters
        ----------
        sentence : list of str
            List of words.

        Returns
        -------
        ntokens : int
            Number of tokens processed in given sentence.
        words : list of int
            List of word ids.

        """

        words = []
        ntokens = 0
        for word in sentence:
            h = self.find(word)
            wid = self.word2int[h]
            if wid < 0:
                continue
            ntokens += 1
            words.append(wid)
            if ntokens > self.max_line_size:
                break
        return ntokens, words


class Sent2Vec(BaseWordEmbeddingsModel):
    """Class for training and using neural networks described in [1]_"""

    def __init__(self, sentences=None, size=100, alpha=0.01, epochs=5, min_count=5, negative=10,
                 word_ngrams=2, bucket=2000000, sample=0.0001, minn=3, maxn=6, dropout_k=2, seed=42,
                 min_alpha=0.001, batch_words=10000, workers=3, max_vocab_size=30000000,
                 compute_loss=False, callbacks=()):
        """

        Parameters
        ----------
        sentences : iterable of iterable of str, optional
            Stream of sentences, see :class:`~gensim.models.sent2vec.TorontoCorpus` in this module for such examples.
        size : int, optional
            Dimensionality of the feature vectors.
        alpha : float, optional
            Initial learning rate.
        epochs : int, optional
            Number of iterations (epochs) over the corpus.
        min_count : int, optional
            Ignore all words with total frequency lower than this.
        negative : int, optional
            Specifies how many "noise words" should be drawn (usually between 5-20).
        word_ngrams : int, optional
            Max length of word ngram.
        bucket : int, optional
            Number of hash buckets for vocabulary.
        sample : float, optional
            Threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
        minn : int, optional
            Min length of char ngrams.
        maxn : int, optional
            Max length of char ngrams.
        dropout_k : int, optional
            Number of ngrams dropped when training a model.
        seed : int, optional
            For the random number generator for reproducible reasons.
        min_alpha : float, optional
            Minimal learning rate.
        batch_words : int, optional
            Target size (in words) for batches of examples passed to worker threads (and thus cython routines).
            Larger batches will be passed if individual texts are longer than 10000 words, but the standard cython code
            truncates to that maximum.
        workers : int, optional
            Use this many worker threads to train the model (=faster training with multicore machines).
        max_vocab_size : int, optional
            Limit RAM during vocabulary building,
            if there are more unique words than this, then prune the infrequent ones.
        compute_loss: bool
            If True, computes and stores loss value which can be retrieved using `model.get_latest_training_loss()`.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`
            List of callbacks that need to be executed/run at specific stages during training.

        """
        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.negpos = 1
        self.loss = 0.0
        self.negative_table_size = 10000000
        self.negatives = []
        self.vocabulary = Sent2VecVocab(sample=sample, bucket=bucket, maxn=maxn,
                                        minn=minn, max_vocab_size=max_vocab_size)
        self.min_count = min_count
        self.word_ngrams = word_ngrams
        self.bucket = bucket
        self.sample = sample
        self.minn = minn
        self.maxn = maxn
        self.dropout_k = dropout_k
        self.max_vocab_size = max_vocab_size
        self.corpus_count = 0
        self.callbacks = callbacks

        super(Sent2Vec, self).__init__(
            sentences=sentences, workers=workers, vector_size=size, epochs=epochs,
            batch_words=batch_words, alpha=alpha, seed=seed, negative=negative,
            min_alpha=min_alpha, fast_version=FAST_VERSION, compute_loss=compute_loss,
            callbacks=callbacks)

    def _set_train_params(self, **kwargs):
        if 'compute_loss' in kwargs:
            self.compute_loss = kwargs['compute_loss']

    def _clear_post_train(self):
        """Resets certain properties of the model, post training."""
        # Avoid NotImplementedError

    def _check_training_sanity(self, epochs=None, total_examples=None, total_words=None, **kwargs):
        """Check that the training parameters provided make sense. e.g. raise error if `epochs` not provided."""
        if self.alpha > self.min_alpha_yet_reached:
            logger.warning("Effective 'alpha' higher than previous training cycles")

        if self.vocabulary.ntokens == 0:  # should be set by `build_vocab`
            raise RuntimeError("you must first build vocabulary before training the model")
        if not len(self.wi):
            raise RuntimeError("you must initialize vectors before training the model")

        if not hasattr(self, 'corpus_count'):
            raise ValueError(
                "The number of examples in the training corpus is missing. "
                "Please make sure this is set inside `build_vocab` function."
                "Call the `build_vocab` function before calling `train`."
            )

        if total_words is None and total_examples is None:
            raise ValueError(
                "You must specify either total_examples or total_words, for proper job parameters updation"
                "and progress calculations. "
                "The usual value is total_examples=model.corpus_count."
            )
        if epochs is None:
            raise ValueError("You must specify an explict epochs count. The usual value is epochs=model.epochs.")
        logger.info(
            "training model with %i workers on %i vocabulary and %i features, "
            "using sample=%s negative=%s",
            self.workers, self.vocabulary.size, self.vector_size, self.sample, self.negative)

    def _get_thread_working_mem(self):
        """Get private working memory per thread."""
        hidden = np.zeros(self.vector_size, dtype=np.float32)  # per-thread private work memory
        grad = np.zeros(self.vector_size, dtype=np.float32)
        return hidden, grad

    def _negative_sampling(self, target, lr):
        """
        Get loss using negative sampling.
        Pararmeters
        -----------
        target : int
            Word id of target word.
        lr : float
            current learning rate.
        Returns
        -------
        loss : float
            Negative sampling loss.
        """

        loss = 0.0
        self.grad = np.zeros(self.vector_size)
        for i in range(self.negative + 1):
            if i == 0:
                loss += self._binarylogistic(target, True, lr)
            else:
                loss += self._binarylogistic(self._get_negative(target), False, lr)
        return loss

    def _binarylogistic(self, target, label, lr):
        """
        Compute loss for given target, label and learning rate using binary logistic regression.
        Parameters
        ----------
        target : int
            Target word id.
        label : bool
            True if no negative is sampled, False otherwise.
        lr : float
            Current learning rate.
        Returns
        -------
        float
            Binary logistic regression loss.
        """

        score = expit(np.dot(self.wo[target], self.hidden))
        alpha = lr * (float(label) - score)
        self.grad += self.wo[target] * alpha
        self.wo[target] += self.hidden * alpha
        if label is True:
            return -np.log(score)
        else:
            return -np.log(1.0 - score)

    def _init_table_negatives(self, counts, update):
        """Initialise table of negatives for negative sampling.

        Parameters
        ----------
        counts : list of int
            List of counts of all words in the vocabulary.

        """
        if update:
            self.negatives = list(self.negatives)
        z = 0.0
        for i in range(len(counts)):
            z += counts[i] ** 0.5
        for i in range(len(counts)):
            c = counts[i] ** 0.5
            for j in range(int(c * self.negative_table_size / z) + 1):
                self.negatives.append(i)
        self.random.shuffle(self.negatives)
        self.negatives = np.array(self.negatives)

    def _get_negative(self, target):
        """
        Get a negative from the list of negatives for caluculating nagtive sampling loss.
        Parameter
        ---------
        target : int
            Target word id.
        Returns
        -------
        int
            Word id of negative sample.
        """

        while True:
            negative = self.negatives[self.negpos]
            self.negpos = (self.negpos + 1) % len(self.negatives)
            if target != negative:
                break
        return negative

    def _update(self, input_, target, lr):
        """
        Update model's neural weights for given context, target word and learning rate.
        Parameters
        ----------
        input_ : list
            List of word ids of context words.
        target : int
            Word id of target word.
        lr : float
            Current Learning rate.
        """

        assert(target >= 0)
        assert(target < self.vocabulary.size)
        if len(input_) == 0:
            return
        self.hidden = np.zeros(self.vector_size)
        for i in input_:
            self.hidden += self.wi[i]
        self.hidden *= (1.0 / len(input_))
        self.loss += self._negative_sampling(target, lr)
        self.nexamples += 1
        self.grad *= (1.0 / len(input_))
        for i in input_:
            self.wi[i] += self.grad

    def _do_train_job_slow(self, sentences):
        """
        Train on a batch of input sentences with plain python/numpy.
        """
        ntokens = self.vocabulary.ntokens
        local_token_count = 0
        self.token_count = 0
        self.nexamples = 0
        progress = 0
        for i in range(self.epochs):
            logger.info("Epoch %i :", i)
            for sentence in sentences:
                progress = self.token_count / (self.epochs * ntokens)
                if progress >= 1:
                    break
                lr = self.alpha * (1.0 - progress)
                ntokens_temp, words = self.vocabulary.get_line(sentence)
                local_token_count += ntokens_temp
                if len(words) > 1:
                    for i in range(len(words)):
                        if self.random.uniform(0, 1) > self.vocabulary.pdiscard[words[i]]:
                            continue
                        context = list(words)
                        context[i] = 0
                        context = self.vocabulary.add_ngrams_train(
                                context=context, n=self.word_ngrams, k=self.dropout_k)
                        self._update(input_=context, target=words[i], lr=lr)
                if local_token_count > self.batch_words:
                    self.token_count += local_token_count
                    local_token_count = 0
                if self.token_count >= self.epochs * ntokens:
                    break
            if self.compute_loss is True:
                logger.info("Progress: %.2f, lr: %.4f, loss: %.4f",
                            progress * 100, lr, self.loss / self.nexamples)
            else:
                logger.info("Progress: %.2f, lr: %.4f", progress * 100, lr)
        return self.token_count

    def _do_train_job(self, sentences, alpha, inits):
        """Train on a batch of input `sentences`

        Parameters
        ----------
        sentences : iterable of iterable of str
            Input sentences.
        alpha : float
            Learning rate for given batch of input sentences.
        hidden : numpy.ndarray
            Hidden vector for neural network computation.
        grad : numpy.ndarray
            Gradient vector for neural network computation.

        Returns
        -------
        local_token_count : int
            Number of tokens processed for given training batch.
        nexamples : int
            Number of examples processed in given training batch.
        loss : float
            Loss for given training batch.
        """
        hidden, grad = inits
        local_token_count, nexamples, loss = _do_train_job_fast(self, sentences, alpha, hidden, grad)
        return nexamples, local_token_count

    def build_vocab(self, sentences, update=False, trim_rule=None):
        """Build vocab from `sentences`

        Parameters
        ----------
        sentences : iterable of iterable of str
            Input sentences.
        update : boolean
            Update existing vocabulary using input sentences if True
        """
        if not update:
            logger.info("Creating dictionary...")
            self.corpus_count = self.vocabulary.read(sentences=sentences, min_count=self.min_count)
            logger.info("Dictionary created, dictionary size: %i, tokens read: %i",
                        self.vocabulary.size, self.vocabulary.ntokens)
            counts = [entry.count for entry in self.vocabulary.words]
            self.wi = self.random.uniform((-1 / self.vector_size), ((-1 / self.vector_size) + 1),
                                          (self.vocabulary.size + self.bucket, self.vector_size)
                                          ).astype(np.float32)
            self.wo = np.zeros((self.vocabulary.size, self.vector_size), dtype=np.float32)
            self._init_table_negatives(counts=counts, update=update)
        else:
            logger.info("Updating dictionary...")
            if self.vocabulary.size == 0:
                raise RuntimeError(
                    "You cannot do an online vocabulary-update of a model which has no prior vocabulary. "
                    "First build the vocabulary of your model with a corpus "
                    "before doing an online update.")
            prev_dict_size = self.vocabulary.size
            self.corpus_count = self.vocabulary.read(sentences=sentences, min_count=self.min_count)
            logger.info("Dictionary updated, dictionary size: %i, tokens read: %i",
                        self.vocabulary.size, self.vocabulary.ntokens)
            counts = [entry.count for entry in self.vocabulary.words]
            new_wi = self.random.uniform((-1 / self.vector_size), ((-1 / self.vector_size) + 1),
                                              (self.vocabulary.size - prev_dict_size + self.bucket,
                                               self.vector_size)).astype(np.float32)
            new_wo = np.zeros((self.vocabulary.size - prev_dict_size, self.vector_size), dtype=np.float32)
            self.wi = np.append(self.wi, new_wi, axis=0)
            self.wo = np.append(self.wo, new_wo, axis=0)
            self._init_table_negatives(counts=counts, update=update)

    def train(self, sentences, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None, word_count=0,
              queue_factor=2, report_delay=1.0, compute_loss=False, callbacks=()):
        """Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For Sent2Vec, each sentence must be a list of unicode strings.
        To support linear learning-rate decay from (initial) alpha to min_alpha, and accurate
        progress-percentage logging, either total_examples (count of sentences) or total_words (count of
        raw words in sentences) **MUST** be provided (if the corpus is the same as was provided to
        :meth:`~gensim.models.sent2vec.Sent2Vec.build_vocab()`, the count of examples in that corpus
        will be available in the model's :attr:`corpus_count` property).
        To avoid common mistakes around the model's ability to do multiple training passes itself, an
        explicit `epochs` argument **MUST** be provided. In the common and recommended case,
        where :meth:`~gensim.models.sent2vec.Sent2Vec.train()` is only called once,
        the model's cached `epochs` value should be supplied as `epochs` value.

        Parameters
        ----------
        sentences : iterable of iterables
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
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
        compute_loss: bool
            If True, computes and stores loss value which can be retrieved using `model.get_latest_training_loss()`.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`
            List of callbacks that need to be executed/run at specific stages during training.

        Examples
        --------
        >>> from gensim.models import Sent2Vec
        >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>>
        >>> model = Sent2Vec(min_count=1)
        >>> model.build_vocab(sentences)
        >>> model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
        """

        if FAST_VERSION < 0:
            return self._do_train_job_slow(sentences)
        return super(Sent2Vec, self).train(
            sentences, total_examples=total_examples, total_words=total_words,
            epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, compute_loss=compute_loss, callbacks=callbacks)

    def __getitem__(self, sentence):
        """Get sentence vector for an input sentence.

        Parameters
        ----------
        sentence : list of str
            List of words.

        Returns
        -------
        numpy.ndarray
            Sentence vector for input sentence.

        """

        ntokens_temp, words = self.vocabulary.get_line(sentence)
        sent_vec = np.zeros(self.vector_size)
        line = self.vocabulary.add_ngrams(context=words, n=self.word_ngrams)
        for word_vec in line:
            sent_vec += self.wi[word_vec]
        if len(line) > 0:
            sent_vec *= (1.0 / len(line))
        return sent_vec

    @classmethod
    def load(cls, *args, **kwargs):
        return super(BaseWordEmbeddingsModel, cls).load(*args, **kwargs)

    def similarity(self, sent1, sent2):
        """Function to compute cosine similarity between two sentences.

        Parameters
        ----------
        sent1 : list of str
            List of words.
        sent2 : list of str
            List of words.

        Returns
        -------
        float
            Cosine similarity score between two sentence vectors.

        """

        return dot(matutils.unitvec(self[sent1]), matutils.unitvec(self[sent2]))


class TorontoCorpus(object):
    """Iterate over sentences from the Toronto Book Corpus."""

    def __init__(self, dirname):
        """

        Parameters
        ----------
        dirname : str
            Name of the directory where the dataset is located.

        """
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname) or ".txt" not in fname:
                continue
            for line in utils.smart_open(fname):
                if line not in ['\n', '\r\n']:
                    sentence = list(tokenize(line))
                if not sentence:  # don't bother sending out empty sentences
                    continue
                yield sentence
