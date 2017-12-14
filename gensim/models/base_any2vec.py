#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from abc import ABCMeta, abstractmethod
from gensim import utils
import logging
from timeit import default_timer
import threading

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

logger = logging.getLogger(__name__)


# Public Interfaces
class BaseAny2VecModel(utils.SaveLoad):

    def __init__(self, data_iterable, workers=3, vector_size=100, epochs=5,
                 callbacks=None, seed=1, batch_words=10000, **kwargs):
        """Initialize model parameters."""
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
        Should also initialize/reset vectors for new vocab entities.
        Sets self.kv, self.corpus_count
        """
        raise NotImplementedError

    def _get_job_params(self):
        """Return the paramter required for each batch"""
        raise NotImplementedError

    def _update_job_params(self, job_params, progress):
        raise NotImplementedError

    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return sum(len(sentence) for sentence in job)

    def _clear_post_train(self):
        """Resets certain properties of the model, post training. eg. `kv.syn0norm`"""
        raise NotImplementedError

    def _do_train_job(self, sentences, job_parameters, thread_private_mem):
        raise NotImplementedError

    def train(self, data_iterable, epochs=None, total_examples=None,
              total_words=None, queue_factor=2, callbacks=None, report_delay=1.0, **kwargs):
        """Handle multiworker training."""
        if self.model_trimmed_post_training:
            raise RuntimeError("Parameters for training were discarded using model_trimmed_post_training method")

        if not self.kv.vocab:  # should be set by `build_vocab`
            raise RuntimeError("you must first build vocabulary before training the model")
        if not len(self.kv.syn0):
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
        job_params = self._get_job_params()

        job_tally = 0

        for callback in self.callbacks:
            callback.on_train_begin(self)

        if epochs > 1:
            sentences = utils.RepeatCorpusNTimes(data_iterable, epochs)
            total_words = total_words and total_words * epochs
            total_examples = total_examples and total_examples * epochs

        def worker_loop():
            """Train the model, lifting lists of sentences from the job_queue."""
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

        def job_producer():
            """Fill jobs queue using the input `sentences` iterator."""
            job_batch, batch_size = [], 0
            pushed_words, pushed_examples = 0, 0
            # next_alpha = start_alpha
            next_job_params = job_params
            job_no = 0

            for sent_idx, sentence in enumerate(sentences):
                sentence_length = self._raw_word_count([sentence])

                # can we fit this sentence into the existing job batch?
                if batch_size + sentence_length <= self.batch_words:
                    # yes => add it to the current job
                    job_batch.append(sentence)
                    batch_size += sentence_length
                else:
                    job_no += 1
                    job_queue.put((job_batch, next_job_params))

                    # update the learning rate for the next job
                    if total_examples:
                        # examples-based decay
                        pushed_examples += len(job_batch)
                        progress = 1.0 * pushed_examples / total_examples
                    else:
                        # words-based decay
                        pushed_words += self._raw_word_count(job_batch)
                        progress = 1.0 * pushed_words / total_words
                    next_job_params = self._update_job_params(job_params, progress)

                    # add the sentence that didn't fit as the first item of a new job
                    job_batch, batch_size = [sentence], sentence_length

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

        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        unfinished_worker_count = len(workers)
        workers.append(threading.Thread(target=job_producer))

        # for first epoch
        for callback in self.callbacks:
            callback.on_epoch_begin()

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        example_count, trained_word_count, raw_word_count = 0, 0, 0
        start, next_report = default_timer() - 0.00001, 1.0

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

                if total_examples:
                    if not ((100 * example_count) / total_examples) % (100 / self.epochs):
                        for callback in self.callbacks:
                            callback.on_epoch_end(self)
                            callback.on_epoch_begin(self)
                else:
                    if not ((100 * raw_word_count) / total_words) % (100 / self.epochs):
                        for callback in self.callbacks:
                            callback.on_epoch_end(self)
                            callback.on_epoch_begin(self)

                # log progress once every report_delay seconds
                elapsed = default_timer() - start
                if elapsed >= next_report:
                    if total_examples:
                        # examples-based progress %
                        logger.info(
                            "PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i",
                            100.0 * example_count / total_examples, trained_word_count / elapsed,
                            utils.qsize(job_queue), utils.qsize(progress_queue)
                        )
                    else:
                        # words-based progress %
                        logger.info(
                            "PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
                            100.0 * raw_word_count / total_words, trained_word_count / elapsed,
                            utils.qsize(job_queue), utils.qsize(progress_queue)
                        )
                    next_report = elapsed + report_delay

        # all done; report the final stats
        elapsed = default_timer() - start
        logger.info(
            "training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed
        )
        if job_tally < 10 * self.workers:
            logger.warning(
                "under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay"
            )

        # check that the input corpus hasn't changed during iteration
        if total_examples and total_examples != example_count:
            logger.warning(
                "supplied example count (%i) did not equal expected count (%i)", example_count, total_examples
            )
        if total_words and total_words != raw_word_count:
            logger.warning(
                "supplied raw word count (%i) did not equal expected count (%i)", raw_word_count, total_words
            )

        self.train_count += 1  # number of times train() has been called
        self.total_train_time += elapsed
        self._clear_post_train()

        for callback in self.callbacks:
            callback.on_train_end(self)
        return trained_word_count

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        model = super(BaseAny2VecModel, cls).load(fname_or_handle, **kwargs)
        return model

    def save(self, fname_or_handle, **kwargs):
        super(BaseAny2VecModel, self).save(fname_or_handle, **kwargs)


class BaseKeyedVectors(utils.SaveLoad):

    def __init__(self):
        self.syn0 = []
        self.vocab = {}
        self.index2entity = []
        self.vector_size = None

    def save(self, fname_or_handle, **kwargs):
        super(BaseKeyedVectors, self).save(fname_or_handle, **kwargs)

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        return super(BaseKeyedVectors, cls).load(fname_or_handle, **kwargs)

    def similarity(self, e1, e2):
        """Compute similarity between vectors of two input entities (words, documents, sentences etc.).
        To be implemented by child class.
        """
        raise NotImplementedError

    def most_similar(self, e1, **kwargs):
        """Find the top-N most similar entities.
        Possibly have `postive` and `negative` list of entities in `**kwargs`."""
        return NotImplementedError

    def distance(self, e1, e2):  # inconsistency in API
        """Compute distance between vectors of two input words.
        To be implemented by child class.
        """
        raise NotImplementedError

    def distances(self, entity_or_vector, other_entities=()):
        """Compute distances from given entity or vector to all entities in `other_entity`.
        If `other_entities` is empty, return distance between `entity_or_vectors` and all entities in vocab.
        To be implemented by child class.
        """
        raise NotImplementedError

    def get_vector(self, entity):  # __getitem__
        """Accept a single entity as input.
        Returns the entity's representations in vector space, as a 1D numpy array.
        """
        raise NotImplementedError

    def most_similar_to_given(self, e1, entities_list):
        """Return the entity from entities_list most similar to e1."""
        raise NotImplementedError

    def entities_closer_than(self, e1, e2):
        """Returns all entities that are closer to `e1` than `e2` is to `e1`."""
        raise NotImplementedError

    def rank(self, e1, e2):
        """Rank of the distance of `e2` from `e1`, in relation to distances of all entities from `e1`."""
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
    """A single vocabulary item, used internally for collecting per-entity frequency, and it's mapped index."""

    def __init__(self, count, index):
        self.count = count
        self.index = index

# class BaseVocabBuilder(object):
#     """Base class to handle building and updating vocabulary (of any entity)."""

#     __metaclass__ = ABCMeta

#     @abstractmethod
#     def scan_vocab(self, entites, **kwargs):
#         """Do an initial scan of all entities appearing in iterator."""
#         pass

#     def reset_weights(self):
#         """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
#         return

#     def update_weights(self):
#         """Copy all the existing weights, and reset the weights for the newly added vocabulary."""
#         return

#     def sort_vocab(self):
#         """Sort the vocabulary so the most frequent words have the lowest indexes."""
#         return

#     def scale_vocab(self, **kwargs):
#         """Apply vocabulary settings for `min_count` (discarding less-frequent entities)
#         and `sample` (controlling the downsampling of more-frequent entities).
#         """
#         return

#     def finalize_vocab(self, **kwargs):
#         """Initialize model weights based on final vocabulary settings."""
#         return

#     def build_vocab(self, entites, **kwargs):
#         """Build vocabulary. Internally calls `scan_vocab`, `scale_vocab` and `finalize_vocab`."""
#         return


# class BaseVectorTrainer(object):
#     """Base class for training any2vec model. This handles feeding data into queues and
#     multi workers learning of vectors.
#     """
#     __metaclass__ = ABCMeta

#     @abstractmethod
#     def _do_train_job(self, *args):
#         """Train single batch of entities."""
#         return

#     def train(self, *ars, **kwargs):
#         """Provide implementation for learning vectors using job producer and `worker_loop`."""
#         return


# class BaseWord2VecTypeModel(BaseVocabBuilder, BaseVectorTrainer):
#     """Base Class for all "Word2Vec like" algorithms -- which use sg, cbow architecture with neg/hs loss.
#     Should contain all repeated/reused code for presently implemented algorithms -- Doc2Vec, FastText, Word2Vec.
#     """
#     def __init__(self, *args, **kwargs):
#         """Initialize common parameters."""
#         return

#     def train_sg_pair(*args, **kwagrs):
#         return

#     def train_cbow_pair(*args, **kwagrs):
#         return
