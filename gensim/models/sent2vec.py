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

>>> vector = model.sentence_vectors(['computer', 'interface']) # vector of a sentence

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
import sys
from random import randint
from gensim.utils import SaveLoad, tokenize
from types import GeneratorType
import os
import threading
from timeit import default_timer
from six.moves import xrange

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

logger = logging.getLogger(__name__)

try:
    from gensim.models.sent2vec_inner import _do_train_job_fast
    from gensim.models.word2vec_inner import FAST_VERSION
    logger.debug('Fast version of %s is being used', __name__)
except ImportError:
    # failed... fall back to plain numpy
    # TODO remove this later (i.e. stay only cython version)
    FAST_VERSION = -1
    logger.warning('Slow version of %s is being used', __name__)


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


class ModelDictionary(object):
    """Class for maintaining Sent2Vec vocbulary. Provides functionality for storing and training
    word and character ngrams.

    """

    def __init__(self, t, bucket, minn, maxn, max_vocab_size, max_line_size=1024):
        """

        Parameters
        ----------
        t : float
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
        self.t = t
        self.bucket = bucket
        self.maxn = maxn
        self.minn = minn

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
        for sentence in sentences:
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
            self.pdiscard.append(((self.t / f) ** 0.5) + (self.t / f))

    def add_ngrams_train(self, context, n, k):
        """Training word ngrams for a given context and target word.

        Parameters
        ----------
        context : list of int
            List of word ids.
        n : int
            Number of word ngrams.
        k : int
            Number of word ngrams dropped while training a Sent2Vec model.

        Returns
        -------
        list of int
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


class Sent2Vec(SaveLoad):
    """Class for training and using neural networks described in [1]_"""

    def __init__(self, sentences=None, size=100, lr=0.2, lr_update_rate=100, epochs=5, min_count=5, neg=10,
                 word_ngrams=2, bucket=2000000, t=0.0001, minn=3, maxn=6, dropoutk=2, seed=42,
                 min_lr=0.001, batch_words=10000, workers=3, max_vocab_size=30000000):
        """

        Parameters
        ----------
        sentences : iterable of iterable of str, optional
            Stream of sentences, see :class:`~gensim.models.sent2vec.TorontoCorpus` in this module for such examples.
        size : int, optional
            Dimensionality of the feature vectors.
        lr : float, optional
            Initial learning rate.
        lr_update_rate : int, optional
            Change the rate of updates for the learning rate.
        epochs : int, optional
            Number of iterations (epochs) over the corpus.
        min_count : int, optional
            Ignore all words with total frequency lower than this.
        neg : int, optional
            Specifies how many "noise words" should be drawn (usually between 5-20).
        word_ngrams : int, optional
            Max length of word ngram.
        bucket : int, optional
            Number of hash buckets for vocabulary.
        t : float, optional
            Threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
        minn : int, optional
            Min length of char ngrams.
        maxn : int, optional
            Max length of char ngrams.
        dropoutk : int, optional
            Number of ngrams dropped when training a model.
        seed : int, optional
            For the random number generator for reproducible reasons.
        min_lr : float, optional
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

        """
        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.negpos = 1
        self.loss = 0.0
        self.negative_table_size = 10000000
        self.negatives = []
        self.vector_size = size
        self.lr = lr
        self.lr_update_rate = lr_update_rate
        self.epochs = epochs
        self.min_count = min_count
        self.neg = neg
        self.word_ngrams = word_ngrams
        self.bucket = bucket
        self.t = t
        self.minn = minn
        self.maxn = maxn
        self.dropoutk = dropoutk
        self.dict = None
        self.min_lr = min_lr
        self.min_lr_yet_reached = lr
        self.batch_words = batch_words
        self.train_count = 0
        self.workers = workers
        self.total_train_time = 0
        self.max_vocab_size = max_vocab_size
        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")
            self.build_vocab(sentences)
            self.train(sentences)

    def negative_sampling(self, target, lr):
        """Get loss using negative sampling.

        Parameters
        ----------
        target : int
            Word id of target word.
        lr : float
            Current learning rate.

        Returns
        -------
        float
            Negative sampling loss.

        """

        loss = 0.0
        self.grad = np.zeros(self.vector_size)
        for i in range(self.neg + 1):
            if i == 0:
                loss += self.binary_logistic(target, True, lr)
            else:
                loss += self.binary_logistic(self.get_negative(target), False, lr)
        return loss

    @staticmethod
    def sigmoid(val):
        """Compute sigmoid of a particular value.

        Parameters
        ----------
        val : {float, numpy.ndarray}
            Value for which sigmoid has to be calculated.

        Returns
        -------
        float
            Sigmoid of `val`.

        """

        return 1.0 / (1.0 + np.exp(-val))

    def binary_logistic(self, target, label, lr):
        """Compute loss for given target, label and learning rate using binary logistic regression.

        Parameters
        ----------
        target : int
            Target word id.
        label : bool
            If True - no negative is sampled, False otherwise.
        lr : float
            Current learning rate.

        Returns
        -------
        float
            Binary logistic regression loss.

        """

        score = self.sigmoid(np.dot(self.wo[target], self.hidden))
        alpha = lr * (float(label) - score)
        self.grad += self.wo[target] * alpha
        self.wo[target] += self.hidden * alpha
        if label is True:
            return -np.log(score)
        else:
            return -np.log(1.0 - score)

    def init_table_negatives(self, counts):
        """Initialise table of negatives for negative sampling.

        Parameters
        ----------
        counts : list of int
            List of counts of all words in the vocabulary.

        """

        z = 0.0
        for i in range(len(counts)):
            z += counts[i] ** 0.5
        for i in range(len(counts)):
            c = counts[i] ** 0.5
            for j in range(int(c * self.negative_table_size / z) + 1):
                self.negatives.append(i)
        self.random.shuffle(self.negatives)
        self.negatives = np.array(self.negatives)

    def get_negative(self, target):
        """Get a negative from the list of negatives for calculating negative sampling loss.

        Parameters
        ----------
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

    def _do_train_job(self, sentences, lr, hidden, grad):
        if FAST_VERSION == -1:
            local_token_count = 0
            nexamples = 0
            loss = 0.0
            for sentence in sentences:
                ntokens_temp, words = self.dict.get_line(sentence)
                local_token_count += ntokens_temp
                if len(words) > 1:
                    for i in range(len(words)):
                        if self.random.uniform(0, 1) > self.dict.pdiscard[words[i]]:
                            continue
                        nexamples += 1
                        context = list(words)
                        context[i] = 0
                        context = self.dict.add_ngrams_train(context=context, n=self.word_ngrams, k=self.dropoutk)
                        loss += self.update(input_=context, target=words[i], lr=lr)
            return local_token_count, nexamples, loss
        else:
            local_token_count, nexamples, loss = _do_train_job_fast(self, sentences, lr, hidden, grad)
            return local_token_count, nexamples, loss

    def update(self, input_, target, lr):
        """Update model's neural weights for given context, target word and learning rate.

        Parameters
        ----------
        input_ : list of int
            Word ids of context words.
        target : int
            Word id of target word.
        lr : float
            Current Learning rate.

        """
        assert target >= 0
        assert target < self.dict.size
        if len(input_) == 0:
            return

        self.hidden = np.zeros(self.vector_size)

        for i in input_:
            self.hidden += self.wi[i]

        self.hidden *= (1.0 / len(input_))
        loss = self.negative_sampling(target, lr)
        self.grad *= (1.0 / len(input_))

        for i in input_:
            self.wi[i] += self.grad
        return loss

    def build_vocab(self, sentences):
        logger.info("Creating dictionary...")
        self.dict = ModelDictionary(t=self.t, bucket=self.bucket, maxn=self.maxn,
                                    minn=self.minn, max_vocab_size=self.max_vocab_size)
        self.dict.read(sentences=sentences, min_count=self.min_count)
        logger.info("Dictionary created, dictionary size: %i, tokens read: %i", self.dict.size, self.dict.ntokens)
        counts = [entry.count for entry in self.dict.words]
        self.wi = self.random.uniform((-1 / self.vector_size), ((-1 / self.vector_size) + 1),
                                      (self.dict.size + self.bucket, self.vector_size)).astype(np.float32)
        self.wo = np.zeros((self.dict.size, self.vector_size), dtype=np.float32)
        self.init_table_negatives(counts=counts)

    def train(self, sentences, queue_factor=2, report_delay=1.0):
        if FAST_VERSION < 0:
            logger.warning(
                "C extension not loaded for Word2Vec, training will be slow. "
                "Install a C compiler and reinstall gensim for fast training."
            )

        logger.info(
            "training model with %i workers on %i vocabulary and %i features",
            self.workers, self.dict.size, self.vector_size)

        if not self.dict:
            raise RuntimeError("you must first build vocabulary before training the model")

        start_lr = self.lr
        end_lr = self.min_lr

        job_tally = 0
        nexamples = 0

        if self.epochs > 1:
            total_words = self.dict.ntokens * self.epochs
            sentences = utils.RepeatCorpusNTimes(sentences, self.epochs)

        def worker_loop():
            """Train the model, lifting lists of sentences from the job_queue."""
            hidden = np.zeros(self.vector_size, dtype=np.float32)  # per-thread private work memory
            grad = np.zeros(self.vector_size, dtype=np.float32)
            jobs_processed = 0
            while True:
                job = job_queue.get()
                if job is None:
                    progress_queue.put(None)
                    break  # no more jobs => quit this worker
                sentences, lr = job
                tally, nexamples, loss = self._do_train_job(sentences, lr, hidden, grad)
                progress_queue.put((np.sum(len(sentence) for sentence in sentences), tally,
                                    nexamples, loss))
                jobs_processed += 1
            logger.debug("worker exiting, processed %i jobs", jobs_processed)

        def job_producer():
            """Fill jobs queue using the input `sentences` iterator."""
            job_batch, batch_size = [], 0
            pushed_words = 0
            next_lr = start_lr
            if next_lr > self.min_lr_yet_reached:
                logger.warning("Effective learning rate higher than previous training cycles")
            self.min_lr_yet_reached = next_lr
            job_no = 0

            for sent_idx, sentence in enumerate(sentences):
                sentence_length = len(sentence)

                # can we fit this sentence into the existing job batch?
                if batch_size + sentence_length <= self.batch_words:
                    # yes => add it to the current job
                    job_batch.append(sentence)
                    batch_size += sentence_length
                else:
                    # no => submit the existing job
                    logger.debug(
                        "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                        job_no, batch_size, len(job_batch), next_lr
                    )
                    job_no += 1
                    job_queue.put((job_batch, next_lr))

                    # update the learning rate for the next job
                    if end_lr < next_lr:
                        pushed_words += len(job_batch)
                        progress = 1.0 * pushed_words / total_words
                        next_lr = start_lr - (start_lr - end_lr) * progress
                        next_lr = max(end_lr, next_lr)

                    # add the sentence that didn't fit as the first item of a new job
                    job_batch, batch_size = [sentence], sentence_length

            # add the last job too (may be significantly smaller than batch_words)
            if job_batch:
                logger.debug(
                    "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                    job_no, batch_size, len(job_batch), next_lr
                )
                job_no += 1
                job_queue.put((job_batch, next_lr))

            if job_no == 0 and self.train_count == 0:
                logger.warning(
                    "train() called with an empty iterator (if not intended, "
                    "be sure to provide a corpus that offers restartable iteration = an iterable)."
                )

            # give the workers heads up that they can finish -- no more work!
            for _ in xrange(self.workers):
                job_queue.put(None)
            logger.debug("job loop exiting, total %i jobs", job_no)

        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        unfinished_worker_count = len(workers)
        workers.append(threading.Thread(target=job_producer))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        trained_word_count, raw_word_count, self.loss = 0, 0, 0.0
        start, next_report = default_timer() - 0.00001, 1.0

        while unfinished_worker_count > 0:
            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                logger.info("worker thread finished; awaiting finish of %i more threads", unfinished_worker_count)
                continue
            raw_words, trained_words, nexamples_temp, loss_temp = report
            job_tally += 1

            # update progress stats
            trained_word_count += trained_words  # only words in vocab
            raw_word_count += raw_words
            nexamples += nexamples_temp
            self.loss += loss_temp

            # log progress once every report_delay seconds
            elapsed = default_timer() - start
            if elapsed >= next_report:
                # words-based progress %
                logger.info(
                "PROGRESS: at %.2f%% words, %.0f words/s",
                100.0 * raw_word_count / total_words, trained_word_count / elapsed)
                next_report = elapsed + report_delay

        # all done; report the final stats
        elapsed = default_timer() - start
        logger.info(
            "training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed
        )
        if job_tally < 10 * self.workers:
            logger.warning("under 10 jobs per worker: consider setting a \
            smaller `batch_words' for smoother alpha decay")

        # check that the input corpus hasn't changed during iteration
        if total_words != raw_word_count:
            logger.warning("supplied raw word count (%i) did not equal expected count (%i)",
                           raw_word_count, total_words)

        self.train_count += 1  # number of times train() has been called
        self.total_train_time += elapsed
        return trained_word_count

    def sentence_vectors(self, sentence):
        """Function for getting sentence vector for an input sentence.

        Parameters
        ----------
        sentence : list of unicode strings
            List of words.

        Returns
        -------
        numpy array
            Sentence vector for input sentence.

        """

        ntokens_temp, words = self.dict.get_line(sentence)
        sent_vec = np.zeros(self.vector_size)
        line = self.dict.add_ngrams(context=words, n=self.word_ngrams)
        for word_vec in line:
            sent_vec += self.wi[word_vec]
        if len(line) > 0:
            sent_vec *= (1.0 / len(line))
        return sent_vec

    def similarity(self, sent1, sent2):
        """Function to compute cosine similarity between two sentences.

        Parameters
        ----------
        sent1, sent2 : list of unicode strings
            List of words.
        Returns
        -------
        float
            Cosine similarity score between two sentence vectors.

        """

        return dot(matutils.unitvec(self.sentence_vectors(sent1)), matutils.unitvec(self.sentence_vectors(sent2)))


class TorontoCorpus():
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
            if not os.path.isfile(fname):
                continue
            for line in utils.smart_open(fname):
                if line not in ['\n', '\r\n']:
                    sentence = list(tokenize(line))
                if not sentence:  # don't bother sending out empty sentences
                    continue
                yield sentence
