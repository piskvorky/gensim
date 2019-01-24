#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Produce sentence vectors with deep learning via sent2vec model using negative sampling.

The training algorithms were originally ported from the C package and extended with additional functionality.


Examples
--------
Initialize a model with e.g.

.. sourcecode:: pycon

    >>> from gensim.models import Sent2Vec
    >>> from gensim.test.utils import common_texts
    >>>
    >>> model = Sent2Vec(common_texts, size=100, min_count=1)

Or

.. sourcecode:: pycon

    >>> model = Sent2Vec(size=100, min_count=1)
    >>> model.build_vocab(common_texts)
    >>> model.train(common_texts)

The sentence vectors are stored in a numpy array

.. sourcecode:: pycon

    >>> vector = model[['computer', 'interface']]  # vector of a sentence

You can perform the NLP similarity task with the model

.. sourcecode:: pycon

    >>> similarity = model.similarity(['graph', 'minors', 'trees'], ['eps', 'user', 'interface', 'system'])

See also
--------
`Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features <https://arxiv.org/abs/1703.02507>`_
`Sent2Vec C++ implementation. <https://github.com/epfml/sent2vec>`_

"""
from __future__ import division
import logging
import numpy as np
from numpy import dot
from gensim import utils, matutils
from gensim.utils import tokenize
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.utils_any2vec import ft_hash_bytes as _ft_hash
from scipy.stats import logistic
import os

try:
    from gensim.models.sent2vec_inner import _do_train_job_fast, FAST_VERSION
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1

    def do_train_job_slow(model, sentences):
        """Train on a batch of input sentences with plain python/numpy."""

        ntokens = model.vocabulary.ntokens
        token_count = 0
        nexamples = 0
        progress = 0
        for i in range(model.epochs):
            logger.info("Epoch %i :", i)
            for sentence in sentences:
                progress = token_count / (model.epochs * ntokens)
                lr = model.alpha * (1.0 - progress)
                ntokens_temp, words = model.vocabulary.get_line(sentence)
                token_count += ntokens_temp
                if len(words) > 1:
                    for i in range(len(words)):
                        if model.random.uniform(0, 1) > model.vocabulary.pdiscard[words[i]]:
                            continue
                        context = list(words)
                        context[i] = 0
                        context = model.vocabulary.add_word_ngrams_train(
                                context=context, n=model.word_ngrams, k=model.dropout_k, random=model.random)
                        model._update(input_=context, target=words[i], lr=lr)
                        nexamples += 1
                if token_count >= model.epochs * ntokens:
                    break
            loss = model.loss
            if nexamples > 0:
                loss = model.loss / nexamples
            if model.compute_loss is True:
                logger.info(
                    "Progress: %.2f, lr: %.4f, loss: %.4f", progress * 100, lr, loss
                    )
            else:
                logger.info("Progress: %.2f, lr: %.4f", progress * 100, lr)
        return token_count

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
    """Class for maintaining Sent2Vec vocbulary. Provides functionality for storing and training word ngrams."""

    def __init__(self, sample, bucket, max_vocab_size, min_count=5, max_line_size=1024):
        """

        Parameters
        ----------
        sample : float
            Threshold for configuring which higher-frequency words are randomly downsampled.
        bucket : int
            Number of hash buckets for vocabulary.
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
        self.min_count = min_count

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
        h = _ft_hash(word.encode("utf-8")) % self.max_vocab_size
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
        Threshold lower frequency words if their count is less than a given value `min_count`.

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
        self.words = [entry for entry in self.words if entry.count >= t]
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

    def add_word_ngrams_train(self, context, n, k, random):
        """Training word ngrams for a given context and target word.

        Parameters
        ----------
        context : list
            List of word ids.
        n : int
            Number of word ngrams.
        k : int
            Number of word ngrams dropped while training a Sent2Vec model.
        random : np.random.RandomState
            Model random state seeded with a particular value.
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
            token_to_discard = random.randint(0, line_size - 1)
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

    def add_word_ngrams(self, context, n):
        """Computing word ngrams for given sentence while inferring sentence vector.
        The ngrams computed are continuous sequence of `n` items starting from a particular target word.

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
    """Sent2Vec Model."""

    def __init__(self, sentences=None, input_streams=None, size=100, alpha=0.01, epochs=5, min_count=5, negative=10,
                 word_ngrams=2, bucket=2000000, sample=0.0001, dropout_k=2, seed=42,
                 min_alpha=0.001, batch_words=10000, workers=3, max_vocab_size=30000000,
                 compute_loss=False, callbacks=(), corpus_file=None):
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
            Max length of word ngram in number of words.
        bucket : int, optional
            Number of hash buckets for vocabulary.
        sample : float, optional
            Threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
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
        self.negatives = []
        self.vocabulary = Sent2VecVocab(sample=sample, bucket=bucket, max_vocab_size=max_vocab_size)
        self.min_count = min_count
        self.word_ngrams = word_ngrams
        self.bucket = bucket
        self.sample = sample
        self.dropout_k = dropout_k
        self.max_vocab_size = max_vocab_size
        self.corpus_count = 0
        self.callbacks = callbacks

        if(corpus_file is not None):
            raise RuntimeError(
                "Aruguement corpus_file is not None.")

        super(Sent2Vec, self).__init__(
            sentences=sentences, input_streams=input_streams, workers=workers, vector_size=size, epochs=epochs,
            batch_words=batch_words, alpha=alpha, seed=seed, negative=negative,
            min_alpha=min_alpha, fast_version=FAST_VERSION, compute_loss=compute_loss,
            callbacks=callbacks, corpus_file=corpus_file)

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
        """Get loss using negative sampling.

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
        """Compute loss for given target, label and learning rate using binary logistic regression.

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
        score = logistic.cdf(np.dot(self.wo[target], self.hidden))
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
        negative_table_size = min(3 * len(counts), 10000000)
        for i in range(len(counts)):
            z += counts[i] ** 0.5
        for i in range(len(counts)):
            c = counts[i] ** 0.5
            for j in range(int(c * negative_table_size / z) + 1):
                self.negatives.append(i)
        self.random.shuffle(self.negatives)
        self.negatives = np.array(self.negatives)

    def _get_negative(self, target):
        """Get a negative from the list of negatives for caluculating nagtive sampling loss.

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
        """Update model's neural weights for given context, target word and learning rate.

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
        self.hidden = np.mean(self.wi[input_], axis=0)
        self.loss += self._negative_sampling(target, lr)
        self.grad *= (1.0 / len(input_))
        for i in input_:
            self.wi[i] += self.grad

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

    def build_vocab(self, sentences, input_streams=None, update=False, trim_rule=None, corpus_file=None):
        """Build vocab from `sentences`.

        Parameters
        ----------
        sentences : iterable of iterable of str
            Input sentences.
        update : boolean
            Update existing vocabulary using input sentences if True

        """
        if(corpus_file is not None):
            raise RuntimeError(
                "Aruguement corpus_file is not None.")
        if not update:
            logger.info("Creating dictionary...")
            self.corpus_count = self.vocabulary.read(sentences=sentences, min_count=self.min_count)
            logger.info(
                "Dictionary created, dictionary size: %i, tokens read: %i",
                self.vocabulary.size, self.vocabulary.ntokens
                )
            counts = [entry.count for entry in self.vocabulary.words]
            self.wi = self.random.uniform((-1 / self.vector_size), ((-1 / self.vector_size) + 1),
                                          (self.vocabulary.size + self.bucket, self.vector_size)
                                          ).astype(np.float32)
            self.wo = np.zeros((self.vocabulary.size, self.vector_size), dtype=np.float32)
            self._init_table_negatives(counts=counts, update=update)
            return
        logger.info("Updating dictionary...")
        if self.vocabulary.size == 0:
            raise RuntimeError(
                "You cannot do an online vocabulary-update of a model which has no prior vocabulary. "
                "First build the vocabulary of your model with a corpus "
                "before doing an online update.")
        prev_dict_size = self.vocabulary.size
        self.corpus_count = self.vocabulary.read(sentences=sentences, min_count=self.min_count)
        logger.info(
            "Dictionary updated, dictionary size: %i, tokens read: %i",
            self.vocabulary.size, self.vocabulary.ntokens
            )
        counts = [entry.count for entry in self.vocabulary.words]
        new_wi = self.random.uniform((-1 / self.vector_size), ((-1 / self.vector_size) + 1),
                                          (self.vocabulary.size - prev_dict_size + self.bucket,
                                           self.vector_size)).astype(np.float32)
        new_wo = np.zeros((self.vocabulary.size - prev_dict_size, self.vector_size), dtype=np.float32)
        self.wi = np.append(self.wi, new_wi, axis=0)
        self.wo = np.append(self.wo, new_wo, axis=0)
        self._init_table_negatives(counts=counts, update=update)

    def train(self, sentences, input_streams=None, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None, word_count=0, corpus_file=None,
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
            If True, computes and stores loss value which can be retrieved using
            :meth:`~gensim.models.sent2vec.Sent2Vec.get_latest_training_loss`.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`
            List of callbacks that need to be executed/run at specific stages during training.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.models import Sent2Vec
            >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
            >>>
            >>> model = Sent2Vec(min_count=1)
            >>> model.build_vocab(sentences)
            >>> model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

        """
        if(corpus_file is not None):
            raise RuntimeError(
                "Aruguement corpus_file is not None.")
        if FAST_VERSION < 0:
            return do_train_job_slow(self, sentences)
        return super(Sent2Vec, self).train(
            sentences, total_examples=total_examples, total_words=total_words,
            epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, compute_loss=compute_loss,
            callbacks=callbacks, corpus_file=corpus_file)

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
        line = self.vocabulary.add_word_ngrams(context=words, n=self.word_ngrams)
        if len(line) > 0:
            sent_vec = np.mean(self.wi[line], axis=0)
        return sent_vec

    @classmethod
    def load(cls, *args, **kwargs):
        return super(BaseWordEmbeddingsModel, cls).load(*args, **kwargs)

    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['hidden', 'grad'])
        return super(Sent2Vec, self).save(*args, **kwargs)

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
                if not sentence:
                    continue
                yield sentence
