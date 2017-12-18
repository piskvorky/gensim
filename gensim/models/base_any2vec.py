#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from abc import ABCMeta, abstractmethod
from gensim import utils
import logging
from timeit import default_timer
import threading
from six import string_types
from collections import defaultdict

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

logger = logging.getLogger(__name__)


# Public Interfaces
class BaseAny2VecModel(utils.SaveLoad):

    def __init__(self, data_iterable, workers=3, vector_size=100, epochs=5,
                 callbacks=(), seed=1, batch_words=10000, **kwargs):
        """Initialize model parameters.
        Subclass should initialize the following attributes:
        - self.kv (instance of concrete implementation of `BaseKeyedVectors` interface)
        - self.vocabulary (instance of concrete implementation of `BaseVocabBuilder` abstract class)
        - self.trainables (instance of concrete implementation of `BaseTrainables` abstract clas)
        """
        self.vector_size = int(vector_size)
        self.seed = seed
        self.workers = int(workers)
        self.epochs = epochs
        self.train_count = 0
        self.total_train_time = 0
        self.batch_words = batch_words
        self.model_trimmed_post_training = False
        self.callbacks = callbacks

    def build_vocab(self, data_iterable, update=False, progress_per=10000, **kwargs):
        """Scan through all the data and create/update vocabulary.
        Should also initialize/reset/update vectors for new vocab entities.
        """
        self.vocabulary.corpus_count, self.vocabulary.raw_vocab = self.vocabulary.scan_vocab(
            data_iterable, progress_per=progress_per, **kwargs)
        self.vocabulary.prepare_vocab(update=update, **kwargs)
        self.trainables.prepare_weights(self.vocabulary)
        self._set_keyedvectors()

    def _get_job_params(self):
        """Return job parameters required for each batch"""
        raise NotImplementedError

    def _set_train_params(self, **kwargs):
        """Set model parameters required for training"""
        raise NotImplementedError

    def _update_job_params(self, job_params, epoch_progress, cur_epoch):
        """Return updated job parameters based on the epoch_progress and cur_epoch"""
        raise NotImplementedError

    def _get_thread_working_mem(self):
        """Return private working memory per thread"""
        raise NotImplementedError

    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        raise NotImplementedError

    def _clear_post_train(self):
        """Resets certain properties of the model post training. eg. `kv.vectors_norm`"""
        raise NotImplementedError

    def _do_train_job(self, batch_data, job_parameters, thread_private_mem):
        """Train a single batch. Return 2-tuple `(effective word count, total word count)`."""
        raise NotImplementedError

    def _check_training_sanity(self, epochs=None, total_examples=None, total_words=None, **kwargs):
        """Check that the training parameters provided make sense. e.g. raise error if `epochs` not provided"""
        raise NotImplementedError

    def _set_keyedvectors(self):
        """Point `keyedvectors` attributes to corresponding `trainables`"""
        self.kv.vectors = self.trainables.vectors
        self.kv.vector_size = self.trainables.vector_size
        self.kv.vocab = self.vocabulary.vocab
        self.kv.index2entity = self.vocabulary.index2word

    def _worker_loop(self, job_queue, progress_queue):
        """Train the model, lifting lists of data from the job_queue."""
        thread_private_mem = self._get_thread_working_mem()
        jobs_processed = 0
        while True:
            job = job_queue.get()
            if job is None:
                progress_queue.put(None)
                break  # no more jobs => quit this worker
            sentences, job_parameters = job

            for callback in self.callbacks:
                callback.on_batch_begin(self)

            tally, raw_tally = self._do_train_job(sentences, job_parameters, thread_private_mem)

            for callback in self.callbacks:
                callback.on_batch_end(self)

            progress_queue.put((len(sentences), tally, raw_tally))  # report back progress
            jobs_processed += 1
        logger.debug("worker exiting, processed %i jobs", jobs_processed)

    def _job_producer(self, data_iterator, job_queue, cur_epoch=0, total_examples=None, total_words=None):
            """Fill jobs queue using the input `data_iterator`."""
            job_batch, batch_size = [], 0
            pushed_words, pushed_examples = 0, 0
            next_job_params = self._get_job_params()
            job_no = 0

            for data_idx, data in enumerate(data_iterator):
                sentence_length = self._raw_word_count([data])

                # can we fit this sentence into the existing job batch?
                if batch_size + sentence_length <= self.batch_words:
                    # yes => add it to the current job
                    job_batch.append(data)
                    batch_size += sentence_length
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
                    job_batch, batch_size = [data], sentence_length
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

    def _log_epoch_progress(
        self, progress_queue, job_queue, cur_epoch=0, total_examples=None, total_words=None, report_delay=1.0):
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
                if total_examples:
                    # examples-based progress %
                    logger.info(
                        "EPOCH %i - PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i",
                        cur_epoch, 100.0 * example_count / total_examples, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue)
                    )
                else:
                    # words-based progress %
                    logger.info(
                        "EPOCH %i - PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
                        cur_epoch, 100.0 * raw_word_count / total_words, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue)
                    )
                next_report = elapsed + report_delay
        # all done; report the final stats
        elapsed = default_timer() - start
        logger.info(
            "EPOCH - %i : training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            cur_epoch, raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed
        )
        if job_tally < 10 * self.workers:
            logger.warning(
                "under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay"
            )

        # check that the input corpus hasn't changed during iteration
        if total_examples and total_examples != example_count:
            logger.warning(
                "EPOCH - %i : supplied example count (%i) did not equal expected count (%i)", cur_epoch,
                example_count, total_examples
            )
        if total_words and total_words != raw_word_count:
            logger.warning(
                "EPOCH - %i : supplied raw word count (%i) did not equal expected count (%i)", cur_epoch,
                raw_word_count, total_words
            )
        self.total_train_time += elapsed
        return trained_word_count, raw_word_count

    def _train_epoch(self, data_iterable, cur_epoch=0, total_examples=None,
              total_words=None, queue_factor=2, report_delay=1.0):
        """Train one epoch."""
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [
        threading.Thread(
            target=self._worker_loop,
            args=(job_queue, progress_queue,))
        for _ in xrange(self.workers)
        ]

        workers.append(threading.Thread(
            target=self._job_producer,
            args=(data_iterable, job_queue),
            kwargs={'cur_epoch': cur_epoch, 'total_examples': total_examples, 'total_words': total_words}))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        trained_word_count = self._log_epoch_progress(
            progress_queue, job_queue, cur_epoch=cur_epoch, total_examples=total_examples, total_words=total_words,
            report_delay=report_delay)

        self._set_keyedvectors()

        return trained_word_count

    def train(self, data_iterable, epochs=None, total_examples=None,
              total_words=None, queue_factor=2, callbacks=None, report_delay=1.0, **kwargs):
        """Handle multi-worker training."""
        self._set_train_params(**kwargs)
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

        for cur_epoch in self.epochs:
            for callback in callbacks:
                callback.on_epoch_begin(self)

            trained_word_count_epoch, raw_word_count_epoch = self._train_epoch(data_iterable, cur_epoch=cur_epoch,
                total_examples=total_examples, total_words=total_words, queue_factor=queue_factor,
                report_delay=report_delay)
            trained_word_count += trained_word_count_epoch
            raw_word_count += raw_word_count_epoch

            for callback in callbacks:
                callback.on_epoch_end(self)

        # Log overall time
        total_elapsed = default_timer() - start
        logger.info(
            "training on a %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, total_elapsed, trained_word_count / total_elapsed
        )

        self.train_count += 1  # number of times train() has been called
        self._clear_post_train()

        for callback in self.callbacks:
            callback.on_train_end(self)
        return trained_word_count, raw_word_count

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        model = super(BaseAny2VecModel, cls).load(fname_or_handle, **kwargs)
        return model

    def save(self, fname_or_handle, **kwargs):
        super(BaseAny2VecModel, self).save(fname_or_handle, **kwargs)


class BaseKeyedVectors(utils.SaveLoad):

    def __init__(self):
        self.vectors = []
        self.vocab = {}
        self.index2entity = []
        self.vector_size = None

    def save(self, fname_or_handle, **kwargs):
        super(BaseKeyedVectors, self).save(fname_or_handle, **kwargs)

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        return super(BaseKeyedVectors, cls).load(fname_or_handle, **kwargs)

    def similarity(self, entity1, entity2):
        """Compute cosine similarity between entities, specified by int index or
        string tag.
        """
        raise NotImplementedError

    def most_similar(self, entity1, **kwargs):
        """Find the top-N most similar entities.
        Possibly have `positive` and `negative` list of entities in `**kwargs`.
        """
        return NotImplementedError

    def distance(self, entity1, entity2):
        """Compute distance between vectors of two input entities, specified by int index or
        string tag.
        """
        raise NotImplementedError

    def distances(self, entity1, other_entities=()):
        """Compute distances from given entity (string tag or index) to all entities in `other_entity`.
        If `other_entities` is empty, return distance between `entity1` and all entities in vocab.
        """
        raise NotImplementedError

    def get_vector(self, entity):
        """Accept a single entity as input, specified by string tag or index.
        Returns the entity's representations in vector space, as a 1D numpy array.
        """
        raise NotImplementedError

    def __getitem__(self, entities):
        """
        Accept a single entity (int or string tag) or list of entities as input.

        If a single string or int, return designated tag's vector
        representation, as a 1D numpy array.

        If a list, return designated tags' vector representations as a
        2D numpy array: #tags x #vector_size.
        """
        if isinstance(entities, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.get_vector(entities)

        return vstack([self.get_vector(entity) for entity in entities])

    def most_similar_to_given(self, entity1, entities_list):
        """Return the entity from entities_list most similar to entity1."""
        raise NotImplementedError

    def entities_closer_than(self, entity1, entity2):
        """Returns all entities that are closer to `entity1` than `entity2` is to `entity1`."""
        raise NotImplementedError

    def rank(self, entity1, entity2):
        """Rank of the distance of `entity2` from `entity1`, in relation to distances of all entities from `entity1`."""
        raise NotImplementedError


class Callback(object):
    """Abstract base class used to build new callbacks."""

    def __init__(self):
        self.model = None

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        pass

    def on_batch_begin(self, model):
        pass

    def on_batch_end(self, model):
        pass

    def on_train_begin(self, model):
        pass

    def on_train_end(self, model):
        pass


class VocabItem(object):
    """A single vocabulary item, used for collecting per-word frequency, and it's mapped index."""

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)


class BaseVocabBuilder(utils.SaveLoad):
    """Class for managing vocabulary of a model. Takes care of building, pruning and updating vocabulary."""
    def __init__(self):
        self.vocab = {}
        self.index2word = []

    def scan_vocab(self, data_iterable, progress_per=10000, **kwargs):
        """Do an initial scan of all words appearing in data_iterable.
        Return num_examples(total examples in data_iterable),
        raw_vocab(collections.defaultdict(int) mapping str vocab elements to their counts)"""
        raise NotImplementedError

    def prepare_vocab(self, update=False, **kwargs):
        raise NotImplementedError


class BaseModelTrainables(utils.SaveLoad):
    """Class for storing and initializing/updating the trainable weights of a model. Should also include
    tables required for training weights. """
    def __init__(self):
        self.vectors = []
        self.vector_size = None

    def prepare_weights(self, vocab):
        raise NotImplementedError

    def reset_weights(self, vocab):
        """Reset all trainable weights to an initial (untrained) state, but keep the existing vocabulary."""
        raise NotImplementedError

    def update_weights(self, vocab):
        """
        Copy all the existing weights, and reset the weights for the newly
        added vocabulary.
        """
        raise NotImplementedError

    def seeded_vector(self, seed_string):
        """Create one 'random' vector (but deterministic by seed_string)"""
        raise NotImplementedError
