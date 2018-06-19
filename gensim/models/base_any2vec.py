#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>
# Copyright (C) 2018 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Contains base classes required for implementing any2vec algorithms."""
from gensim import utils
import logging
from timeit import default_timer
import threading
from six.moves import xrange
from six import itervalues
from gensim import matutils
from numpy import float32 as REAL, ones, random, dtype, zeros
from types import GeneratorType
from gensim.utils import deprecated
import warnings
import itertools

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

logger = logging.getLogger(__name__)


class BaseAny2VecModel(utils.SaveLoad):
    """Base class for training, using and evaluating any2vec model.
    Contains implementation for multi-threaded training.

    """

    def __init__(self, workers=3, vector_size=100, epochs=5, callbacks=(), batch_words=10000):
        """Initialize model parameters.

        A subclass should initialize the following attributes:
        - self.kv (instance of concrete implementation of `BaseKeyedVectors` interface)
        - self.vocabulary (instance of concrete implementation of `BaseVocabBuilder` abstract class)
        - self.trainables (instance of concrete implementation of `BaseTrainables` abstract class)

        """
        self.vector_size = int(vector_size)
        self.workers = int(workers)
        self.epochs = epochs
        self.train_count = 0
        self.total_train_time = 0
        self.batch_words = batch_words
        self.model_trimmed_post_training = False
        self.callbacks = callbacks

    def _get_job_params(self, cur_epoch):
        """Get job parameters required for each batch."""
        raise NotImplementedError()

    def _set_train_params(self, **kwargs):
        """Set model parameters required for training"""
        raise NotImplementedError()

    def _update_job_params(self, job_params, epoch_progress, cur_epoch):
        """Get updated job parameters based on the epoch_progress and cur_epoch."""
        raise NotImplementedError()

    def _get_thread_working_mem(self):
        """Get private working memory per thread."""
        raise NotImplementedError()

    def _raw_word_count(self, job):
        """Get the number of words in a given job."""
        raise NotImplementedError()

    def _clear_post_train(self):
        """Resets certain properties of the model post training. eg. `keyedvectors.vectors_norm`."""
        raise NotImplementedError()

    def _do_train_job(self, data_iterable, job_parameters, thread_private_mem):
        """Train a single batch. Return 2-tuple `(effective word count, total word count)`."""
        raise NotImplementedError()

    def _check_training_sanity(self, epochs=None, total_examples=None, total_words=None, **kwargs):
        """Check that the training parameters provided make sense. e.g. raise error if `epochs` not provided."""
        raise NotImplementedError()

    def _check_input_data_sanity(self, data_iterable=None, data_iterables=None):
        """Check that only one argument is not None."""
        if not ((data_iterable is not None) ^ (data_iterables is not None)):
            raise ValueError("You must provide only one of singlestream or multistream arguments.")

    def _worker_loop(self, job_queue, progress_queue):
        """Train the model, lifting lists of data from the job_queue."""
        thread_private_mem = self._get_thread_working_mem()
        jobs_processed = 0
        while True:
            job = job_queue.get()
            if job is None:
                progress_queue.put(None)
                break  # no more jobs => quit this worker
            data_iterable, job_parameters = job

            for callback in self.callbacks:
                callback.on_batch_begin(self)

            tally, raw_tally = self._do_train_job(data_iterable, job_parameters, thread_private_mem)

            for callback in self.callbacks:
                callback.on_batch_end(self)

            progress_queue.put((len(data_iterable), tally, raw_tally))  # report back progress
            jobs_processed += 1
        logger.debug("worker exiting, processed %i jobs", jobs_processed)

    def _job_producer(self, data_iterator, job_queue, cur_epoch=0, total_examples=None, total_words=None):
        """Fill jobs queue using the input `data_iterator`."""
        job_batch, batch_size = [], 0
        pushed_words, pushed_examples = 0, 0
        next_job_params = self._get_job_params(cur_epoch)
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
                job_queue.put((job_batch, next_job_params))

                # update the learning rate for the next job
                if total_examples:
                    # examples-based decay
                    pushed_examples += len(job_batch)
                    epoch_progress = 1.0 * pushed_examples / total_examples
                else:
                    # words-based decay
                    pushed_words += self._raw_word_count(job_batch)
                    epoch_progress = 1.0 * pushed_words / total_words
                next_job_params = self._update_job_params(next_job_params, epoch_progress, cur_epoch)

                # add the sentence that didn't fit as the first item of a new job
                job_batch, batch_size = [data], data_length
        # add the last job too (may be significantly smaller than batch_words)
        if job_batch:
            job_no += 1
            job_queue.put((job_batch, next_job_params))

        if job_no == 0 and self.train_count == 0:
            logger.warning(
                "train() called with an empty iterator (if not intended, "
                "be sure to provide a corpus that offers restartable iteration = an iterable)."
            )

        # give the workers heads up that they can finish -- no more work!
        for _ in xrange(self.workers):
            job_queue.put(None)
        logger.debug("job loop exiting, total %i jobs", job_no)

    def _log_progress(self, job_queue, progress_queue, cur_epoch, example_count, total_examples,
                      raw_word_count, total_words, trained_word_count, elapsed):
        raise NotImplementedError()

    def _log_epoch_end(self, cur_epoch, example_count, total_examples, raw_word_count, total_words,
                       trained_word_count, elapsed):
        raise NotImplementedError()

    def _log_train_end(self, raw_word_count, trained_word_count, total_elapsed, job_tally):
        raise NotImplementedError()

    def _log_epoch_progress(self, progress_queue, job_queue, cur_epoch=0, total_examples=None, total_words=None,
                            report_delay=1.0):
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
            trained_word_count, elapsed)
        self.total_train_time += elapsed
        return trained_word_count, raw_word_count, job_tally

    def _train_epoch(self, data_iterable=None, data_iterables=None, cur_epoch=0, total_examples=None,
                     total_words=None, queue_factor=2, report_delay=1.0):
        """Train one epoch."""
        self._check_input_data_sanity(data_iterable, data_iterables)
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [
            threading.Thread(
                target=self._worker_loop,
                args=(job_queue, progress_queue,))
            for _ in xrange(self.workers)
        ]

        # Chain all input streams into one, because multistream training is not supported yet.
        if data_iterables is not None:
            data_iterable = itertools.chain(*data_iterable)
        workers.append(threading.Thread(
            target=self._job_producer,
            args=(data_iterable, job_queue),
            kwargs={'cur_epoch': cur_epoch, 'total_examples': total_examples, 'total_words': total_words}))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        trained_word_count, raw_word_count, job_tally = self._log_epoch_progress(
            progress_queue, job_queue, cur_epoch=cur_epoch, total_examples=total_examples, total_words=total_words,
            report_delay=report_delay)

        return trained_word_count, raw_word_count, job_tally

    def train(self, data_iterable=None, data_iterables=None, epochs=None, total_examples=None,
              total_words=None, queue_factor=2, report_delay=1.0, callbacks=(), **kwargs):
        """Handle multi-worker training."""
        self._set_train_params(**kwargs)
        if callbacks:
            self.callbacks = callbacks
        self.epochs = epochs
        self._check_training_sanity(
            epochs=epochs,
            total_examples=total_examples,
            total_words=total_words, **kwargs)

        for callback in self.callbacks:
            callback.on_train_begin(self)

        trained_word_count = 0
        raw_word_count = 0
        start = default_timer() - 0.00001
        job_tally = 0

        for cur_epoch in range(self.epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(self)

            trained_word_count_epoch, raw_word_count_epoch, job_tally_epoch = self._train_epoch(
                data_iterable=data_iterable, data_iterables=data_iterables, cur_epoch=cur_epoch,
                total_examples=total_examples, total_words=total_words, queue_factor=queue_factor,
                report_delay=report_delay)
            trained_word_count += trained_word_count_epoch
            raw_word_count += raw_word_count_epoch
            job_tally += job_tally_epoch

            for callback in self.callbacks:
                callback.on_epoch_end(self)

        # Log overall time
        total_elapsed = default_timer() - start
        self._log_train_end(raw_word_count, trained_word_count, total_elapsed, job_tally)

        self.train_count += 1  # number of times train() has been called
        self._clear_post_train()

        for callback in self.callbacks:
            callback.on_train_end(self)
        return trained_word_count, raw_word_count

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        return super(BaseAny2VecModel, cls).load(fname_or_handle, **kwargs)

    def save(self, fname_or_handle, **kwargs):
        super(BaseAny2VecModel, self).save(fname_or_handle, **kwargs)


class BaseWordEmbeddingsModel(BaseAny2VecModel):
    """
    Base class containing common methods for training, using & evaluating word embeddings learning models.
    For example - `Word2Vec`, `FastText`, etc.

    """

    def _clear_post_train(self):
        raise NotImplementedError()

    def _do_train_job(self, data_iterable, job_parameters, thread_private_mem):
        raise NotImplementedError()

    def _set_train_params(self, **kwargs):
        raise NotImplementedError()

    def __init__(self, sentences=None, input_streams=None, workers=3, vector_size=100, epochs=5, callbacks=(),
                 batch_words=10000, trim_rule=None, sg=0, alpha=0.025, window=5, seed=1, hs=0, negative=5, cbow_mean=1,
                 min_alpha=0.0001, compute_loss=False, fast_version=0, **kwargs):
        self.sg = int(sg)
        if vector_size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.window = int(window)
        self.random = random.RandomState(seed)
        self.min_alpha = float(min_alpha)
        self.hs = int(hs)
        self.negative = int(negative)
        self.cbow_mean = int(cbow_mean)
        self.compute_loss = bool(compute_loss)
        self.running_training_loss = 0
        self.min_alpha_yet_reached = float(alpha)
        self.corpus_count = 0

        super(BaseWordEmbeddingsModel, self).__init__(
            workers=workers, vector_size=vector_size, epochs=epochs, callbacks=callbacks, batch_words=batch_words)

        if fast_version < 0:
            warnings.warn(
                "C extension not loaded, training will be slow. "
                "Install a C compiler and reinstall gensim for fast training."
            )
            self.neg_labels = []
            if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.

        if sentences is not None or input_streams is not None:
            self._check_input_data_sanity(data_iterable=sentences, data_iterables=input_streams)
            if input_streams is not None:
                if not isinstance(input_streams, (tuple, list)):
                    raise TypeError("You must pass tuple or list as the input_streams argument.")
                if any(isinstance(stream, GeneratorType) for stream in input_streams):
                    raise TypeError("You can't pass a generator as any of input streams. Try an iterator.")
            elif isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")

            self.build_vocab(sentences=sentences, input_streams=input_streams, trim_rule=trim_rule)
            self.train(
                sentences=sentences, input_streams=input_streams, total_examples=self.corpus_count, epochs=self.epochs,
                start_alpha=self.alpha, end_alpha=self.min_alpha, compute_loss=compute_loss)
        else:
            if trim_rule is not None:
                logger.warning(
                    "The rule, if given, is only used to prune vocabulary during build_vocab() "
                    "and is not stored as part of the model. Model initialized without sentences. "
                    "trim_rule provided, if any, will be ignored.")

    # for backward compatibility (aliases pointing to corresponding variables in trainables, vocabulary)
    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.epochs instead")
    def iter(self):
        return self.epochs

    @iter.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.epochs instead")
    def iter(self, value):
        self.epochs = value

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.syn1 instead")
    def syn1(self):
        return self.trainables.syn1

    @syn1.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.syn1 instead")
    def syn1(self, value):
        self.trainables.syn1 = value

    @syn1.deleter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.syn1 instead")
    def syn1(self):
        del self.trainables.syn1

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.syn1neg instead")
    def syn1neg(self):
        return self.trainables.syn1neg

    @syn1neg.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.syn1neg instead")
    def syn1neg(self, value):
        self.trainables.syn1neg = value

    @syn1neg.deleter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.syn1neg instead")
    def syn1neg(self):
        del self.trainables.syn1neg

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_lockf instead")
    def syn0_lockf(self):
        return self.trainables.vectors_lockf

    @syn0_lockf.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_lockf instead")
    def syn0_lockf(self, value):
        self.trainables.vectors_lockf = value

    @syn0_lockf.deleter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.vectors_lockf instead")
    def syn0_lockf(self):
        del self.trainables.vectors_lockf

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.layer1_size instead")
    def layer1_size(self):
        return self.trainables.layer1_size

    @layer1_size.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.layer1_size instead")
    def layer1_size(self, value):
        self.trainables.layer1_size = value

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.hashfxn instead")
    def hashfxn(self):
        return self.trainables.hashfxn

    @hashfxn.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.trainables.hashfxn instead")
    def hashfxn(self, value):
        self.trainables.hashfxn = value

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.vocabulary.sample instead")
    def sample(self):
        return self.vocabulary.sample

    @sample.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.vocabulary.sample instead")
    def sample(self, value):
        self.vocabulary.sample = value

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.vocabulary.min_count instead")
    def min_count(self):
        return self.vocabulary.min_count

    @min_count.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.vocabulary.min_count instead")
    def min_count(self, value):
        self.vocabulary.min_count = value

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.vocabulary.cum_table instead")
    def cum_table(self):
        return self.vocabulary.cum_table

    @cum_table.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.vocabulary.cum_table instead")
    def cum_table(self, value):
        self.vocabulary.cum_table = value

    @cum_table.deleter
    @deprecated("Attribute will be removed in 4.0.0, use self.vocabulary.cum_table instead")
    def cum_table(self):
        del self.vocabulary.cum_table

    def __str__(self):
        return "%s(vocab=%s, size=%s, alpha=%s)" % (
            self.__class__.__name__, len(self.wv.index2word), self.vector_size, self.alpha
        )

    def build_vocab(self, sentences=None, input_streams=None, workers=None, update=False, progress_per=10000,
                    keep_raw_vocab=False, trim_rule=None, **kwargs):
        """Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence is a iterable of iterables (can simply be a list of unicode strings too).

        Parameters
        ----------
        sentences : iterable of iterables
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        input_streams : list or tuple of iterable of iterables
            The tuple or list of `sentences`-like arguments. Use it if you have multiple input streams. It is possible
            to process streams in parallel, using `workers` parameter.
        workers : int
            Used if `input_streams` is passed. Determines how many processes to use for vocab building.
            Actual number of workers is determined by `min(len(input_streams), workers)`.
        update : bool
            If true, the new words in `sentences` will be added to model's vocab.
        progress_per : int
            Indicates how many words to process before showing/updating the progress.

        """
        workers = workers or self.workers
        total_words, corpus_count = self.vocabulary.scan_vocab(
            sentences=sentences, input_streams=input_streams, progress_per=progress_per, trim_rule=trim_rule,
            workers=workers)
        self.corpus_count = corpus_count
        report_values = self.vocabulary.prepare_vocab(
            self.hs, self.negative, self.wv, update=update, keep_raw_vocab=keep_raw_vocab,
            trim_rule=trim_rule, **kwargs)
        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])
        self.trainables.prepare_weights(self.hs, self.negative, self.wv, update=update, vocabulary=self.vocabulary)

    def build_vocab_from_freq(self, word_freq, keep_raw_vocab=False, corpus_count=None, trim_rule=None, update=False):
        """Build vocabulary from a dictionary of word frequencies.
        Build model vocabulary from a passed dictionary that contains (word,word count).
        Words must be of type unicode strings.

        Parameters
        ----------
        word_freq : dict
            Word,Word_Count dictionary.
        keep_raw_vocab : bool
            If not true, delete the raw vocabulary after the scaling is done and free up RAM.
        corpus_count : int
            Even if no corpus is provided, this argument can set corpus_count explicitly.
        trim_rule : function
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part
            of the model.
        update : bool
            If true, the new provided words in `word_freq` dict will be added to model's vocab.

        Examples
        --------
        >>> from gensim.models import Word2Vec
        >>>
        >>> model= Word2Vec()
        >>> model.build_vocab_from_freq({"Word1": 15, "Word2": 20})

        """
        logger.info("Processing provided word frequencies")
        # Instead of scanning text, this will assign provided word frequencies dictionary(word_freq)
        # to be directly the raw vocab
        raw_vocab = word_freq
        logger.info(
            "collected %i different raw word, with total frequency of %i",
            len(raw_vocab), sum(itervalues(raw_vocab))
        )

        # Since no sentences are provided, this is to control the corpus_count
        self.corpus_count = corpus_count or 0
        self.vocabulary.raw_vocab = raw_vocab

        # trim by min_count & precalculate downsampling
        report_values = self.vocabulary.prepare_vocab(
            self.hs, self.negative, self.wv, keep_raw_vocab=keep_raw_vocab,
            trim_rule=trim_rule, update=update)
        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])
        self.trainables.prepare_weights(
            self.hs, self.negative, self.wv, update=update, vocabulary=self.vocabulary)  # build tables & arrays

    def estimate_memory(self, vocab_size=None, report=None):
        """Estimate required memory for a model using current settings and provided vocabulary size."""
        vocab_size = vocab_size or len(self.wv.vocab)
        report = report or {}
        report['vocab'] = vocab_size * (700 if self.hs else 500)
        report['vectors'] = vocab_size * self.vector_size * dtype(REAL).itemsize
        if self.hs:
            report['syn1'] = vocab_size * self.trainables.layer1_size * dtype(REAL).itemsize
        if self.negative:
            report['syn1neg'] = vocab_size * self.trainables.layer1_size * dtype(REAL).itemsize
        report['total'] = sum(report.values())
        logger.info(
            "estimated required memory for %i words and %i dimensions: %i bytes",
            vocab_size, self.vector_size, report['total']
        )
        return report

    def train(self, sentences=None, input_streams=None, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None, word_count=0,
              queue_factor=2, report_delay=1.0, compute_loss=False, callbacks=()):

        self.alpha = start_alpha or self.alpha
        self.min_alpha = end_alpha or self.min_alpha
        self.compute_loss = compute_loss
        self.running_training_loss = 0.0
        return super(BaseWordEmbeddingsModel, self).train(
            data_iterable=sentences, data_iterables=input_streams, total_examples=total_examples,
            total_words=total_words, epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, compute_loss=compute_loss, callbacks=callbacks)

    def _get_job_params(self, cur_epoch):
        """Get the parameter required for each batch."""
        alpha = self.alpha - ((self.alpha - self.min_alpha) * float(cur_epoch) / self.epochs)
        return alpha

    def _update_job_params(self, job_params, epoch_progress, cur_epoch):
        start_alpha = self.alpha
        end_alpha = self.min_alpha
        progress = (cur_epoch + epoch_progress) / self.epochs
        next_alpha = start_alpha - (start_alpha - end_alpha) * progress
        next_alpha = max(end_alpha, next_alpha)
        self.min_alpha_yet_reached = next_alpha
        return next_alpha

    def _get_thread_working_mem(self):
        work = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL)  # per-thread private work memory
        neu1 = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL)
        return work, neu1

    def _raw_word_count(self, job):
        """Get the number of words in a given job."""
        return sum(len(sentence) for sentence in job)

    def _check_training_sanity(self, epochs=None, total_examples=None, total_words=None, **kwargs):
        if self.alpha > self.min_alpha_yet_reached:
            logger.warning("Effective 'alpha' higher than previous training cycles")
        if self.model_trimmed_post_training:
            raise RuntimeError("Parameters for training were discarded using model_trimmed_post_training method")

        if not self.wv.vocab:  # should be set by `build_vocab`
            raise RuntimeError("you must first build vocabulary before training the model")
        if not len(self.wv.vectors):
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
            "using sg=%s hs=%s sample=%s negative=%s window=%s",
            self.workers, len(self.wv.vocab), self.trainables.layer1_size, self.sg,
            self.hs, self.vocabulary.sample, self.negative, self.window
        )

    @classmethod
    def load(cls, *args, **kwargs):
        model = super(BaseWordEmbeddingsModel, cls).load(*args, **kwargs)
        if model.negative and hasattr(model.wv, 'index2word'):
            model.vocabulary.make_cum_table(model.wv)  # rebuild cum_table from vocabulary
        if not hasattr(model, 'corpus_count'):
            model.corpus_count = None
        if not hasattr(model.trainables, 'vectors_lockf') and hasattr(model.wv, 'vectors'):
            model.trainables.vectors_lockf = ones(len(model.wv.vectors), dtype=REAL)
        if not hasattr(model, 'random'):
            model.random = random.RandomState(model.trainables.seed)
        if not hasattr(model, 'train_count'):
            model.train_count = 0
            model.total_train_time = 0
        return model

    def _log_progress(self, job_queue, progress_queue, cur_epoch, example_count, total_examples,
                      raw_word_count, total_words, trained_word_count, elapsed):
        if total_examples:
            # examples-based progress %
            logger.info(
                "EPOCH %i - PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i",
                cur_epoch + 1, 100.0 * example_count / total_examples, trained_word_count / elapsed,
                utils.qsize(job_queue), utils.qsize(progress_queue)
            )
        else:
            # words-based progress %
            logger.info(
                "EPOCH %i - PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
                cur_epoch + 1, 100.0 * raw_word_count / total_words, trained_word_count / elapsed,
                utils.qsize(job_queue), utils.qsize(progress_queue)
            )

    def _log_epoch_end(self, cur_epoch, example_count, total_examples, raw_word_count, total_words,
                       trained_word_count, elapsed):
        logger.info(
            "EPOCH - %i : training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            cur_epoch + 1, raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed
        )

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
        logger.info(
            "training on a %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, total_elapsed, trained_word_count / total_elapsed
        )
        if job_tally < 10 * self.workers:
            logger.warning(
                "under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay"
            )

    # for backward compatibility
    @deprecated("Method will be removed in 4.0.0, use self.wv.most_similar() instead")
    def most_similar(self, positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None):
        """
        Deprecated. Use self.wv.most_similar() instead.
        Refer to the documentation for `gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.most_similar`
        """
        return self.wv.most_similar(positive, negative, topn, restrict_vocab, indexer)

    @deprecated("Method will be removed in 4.0.0, use self.wv.wmdistance() instead")
    def wmdistance(self, document1, document2):
        """
        Deprecated. Use self.wv.wmdistance() instead.
        Refer to the documentation for `gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.wmdistance`
        """
        return self.wv.wmdistance(document1, document2)

    @deprecated("Method will be removed in 4.0.0, use self.wv.most_similar_cosmul() instead")
    def most_similar_cosmul(self, positive=None, negative=None, topn=10):
        """
        Deprecated. Use self.wv.most_similar_cosmul() instead.
        Refer to the documentation for `gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.most_similar_cosmul`
        """
        return self.wv.most_similar_cosmul(positive, negative, topn)

    @deprecated("Method will be removed in 4.0.0, use self.wv.similar_by_word() instead")
    def similar_by_word(self, word, topn=10, restrict_vocab=None):
        """
        Deprecated. Use self.wv.similar_by_word() instead.
        Refer to the documentation for `gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similar_by_word`
        """
        return self.wv.similar_by_word(word, topn, restrict_vocab)

    @deprecated("Method will be removed in 4.0.0, use self.wv.similar_by_vector() instead")
    def similar_by_vector(self, vector, topn=10, restrict_vocab=None):
        """
        Deprecated. Use self.wv.similar_by_vector() instead.
        Refer to the documentation for `gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similar_by_vector`
        """
        return self.wv.similar_by_vector(vector, topn, restrict_vocab)

    @deprecated("Method will be removed in 4.0.0, use self.wv.doesnt_match() instead")
    def doesnt_match(self, words):
        """
        Deprecated. Use self.wv.doesnt_match() instead.
        Refer to the documentation for `gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.doesnt_match`
        """
        return self.wv.doesnt_match(words)

    @deprecated("Method will be removed in 4.0.0, use self.wv.similarity() instead")
    def similarity(self, w1, w2):
        """
        Deprecated. Use self.wv.similarity() instead.
        Refer to the documentation for `gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similarity`
        """
        return self.wv.similarity(w1, w2)

    @deprecated("Method will be removed in 4.0.0, use self.wv.n_similarity() instead")
    def n_similarity(self, ws1, ws2):
        """
        Deprecated. Use self.wv.n_similarity() instead.
        Refer to the documentation for `gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.n_similarity`
        """
        return self.wv.n_similarity(ws1, ws2)

    @deprecated("Method will be removed in 4.0.0, use self.wv.evaluate_word_pairs() instead")
    def evaluate_word_pairs(self, pairs, delimiter='\t', restrict_vocab=300000,
                            case_insensitive=True, dummy4unknown=False):
        """
        Deprecated. Use self.wv.evaluate_word_pairs() instead.
        Refer to the documentation for `gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.evaluate_word_pairs`
        """
        return self.wv.evaluate_word_pairs(pairs, delimiter, restrict_vocab, case_insensitive, dummy4unknown)
