#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim import utils
import logging
from timeit import default_timer
import threading
import numpy as np
from six import string_types
from numpy import vstack
from gensim import matutils
from numpy import float32 as REAL, ones, random, argmax
from types import GeneratorType

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

logger = logging.getLogger(__name__)


# Public Interfaces
class BaseAny2VecModel(utils.SaveLoad):

    def __init__(self, workers=3, vector_size=100, epochs=5,
                 callbacks=(), batch_words=10000):
        """Initialize model parameters.
        Subclass should initialize the following attributes:
        - self.kv (instance of concrete implementation of `BaseKeyedVectors` interface)
        - self.vocabulary (instance of concrete implementation of `BaseVocabBuilder` abstract class)
        - self.trainables (instance of concrete implementation of `BaseTrainables` abstract clas)
        """
        self.vector_size = int(vector_size)
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
        raise NotImplementedError

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

    def _do_train_job(self, data_iterable, job_parameters, thread_private_mem):
        """Train a single batch. Return 2-tuple `(effective word count, total word count)`."""
        raise NotImplementedError

    def _check_training_sanity(self, epochs=None, total_examples=None, total_words=None, **kwargs):
        """Check that the training parameters provided make sense. e.g. raise error if `epochs` not provided"""
        raise NotImplementedError

    def _set_keyedvectors(self):
        raise NotImplementedError

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
        """Fill jobs queue using the input `sentences` iterator."""
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
        raise NotImplementedError

    def _log_epoch_end(self, cur_epoch, example_count, total_examples, raw_word_count, total_words,
                       trained_word_count, elapsed, job_tally):
        raise NotImplementedError

    def _log_train_end(self, raw_word_count, trained_word_count, total_elapsed):
        raise NotImplementedError

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
            trained_word_count, elapsed, job_tally)
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
              total_words=None, queue_factor=2, report_delay=1.0, **kwargs):
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

        for cur_epoch in range(self.epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(self)

            trained_word_count_epoch, raw_word_count_epoch = self._train_epoch(data_iterable, cur_epoch=cur_epoch,
                total_examples=total_examples, total_words=total_words, queue_factor=queue_factor,
                report_delay=report_delay)
            trained_word_count += trained_word_count_epoch
            raw_word_count += raw_word_count_epoch

            for callback in self.callbacks:
                callback.on_epoch_end(self)

        # Log overall time
        total_elapsed = default_timer() - start
        self._log_train_end(raw_word_count, trained_word_count, total_elapsed)

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


class BaseKeyedVectors(utils.SaveLoad):

    def __init__(self):
        self.vectors = []
        self.vocab = {}
        self.vector_size = None
        self.index2entity = []

    def save(self, fname_or_handle, **kwargs):
        super(BaseKeyedVectors, self).save(fname_or_handle, **kwargs)

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        return super(BaseKeyedVectors, cls).load(fname_or_handle, **kwargs)

    def similarity(self, entity1, entity2):
        """Compute cosine similarity between entities, specified by string tag.
        """
        raise NotImplementedError

    def most_similar(self, **kwargs):
        """Find the top-N most similar entities.
        Possibly have `positive` and `negative` list of entities in `**kwargs`.
        """
        return NotImplementedError

    def distance(self, entity1, entity2):
        """Compute distance between vectors of two input entities, specified by string tag.
        """
        raise NotImplementedError

    def distances(self, entity1, other_entities=()):
        """Compute distances from given entity (string tag) to all entities in `other_entity`.
        If `other_entities` is empty, return distance between `entity1` and all entities in vocab.
        """
        raise NotImplementedError

    def get_vector(self, entity):
        """Accept a single entity as input, specified by string tag.
        Returns the entity's representations in vector space, as a 1D numpy array.
        """
        if entity in self.vocab:
            result = self.vectors[self.vocab[entity].index]
            result.setflags(write=False)
            return result
        else:
            raise KeyError("'%s' not in vocabulary" % entity)

    def __getitem__(self, entities):
        """
        Accept a single entity (string tag) or list of entities as input.

        If a single string or int, return designated tag's vector
        representation, as a 1D numpy array.

        If a list, return designated tags' vector representations as a
        2D numpy array: #tags x #vector_size.
        """
        if isinstance(entities, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.get_vector(entities)

        return vstack([self.get_vector(entity) for entity in entities])

    def __contains__(self, entity):
        return entity in self.vocab

    def most_similar_to_given(self, entity1, entities_list):
        """Return the entity from entities_list most similar to entity1."""
        return entities_list[argmax([self.similarity(entity1, entity) for entity in entities_list])]

    def closer_than(self, entity1, entity2):
        """Returns all entities that are closer to `entity1` than `entity2` is to `entity1`."""
        all_distances = self.distances(entity1)
        e1_index = self.vocab[entity1].index
        e2_index = self.vocab[entity2].index
        closer_node_indices = np.where(all_distances < all_distances[e2_index])[0]
        return [self.index2entity[index] for index in closer_node_indices if index != e1_index]

    def rank(self, entity1, entity2):
        """Rank of the distance of `entity2` from `entity1`, in relation to distances of all entities from `entity1`."""
        return len(self.closer_than(entity1, entity2)) + 1


class BaseVocabBuilder(utils.SaveLoad):
    """Class for managing vocabulary of a model. Takes care of building, pruning and updating vocabulary."""
    def __init__(self):
        self.vocab = {}
        self.index2word = []

    def scan_vocab(self, data_iterable, progress_per=10000, **kwargs):
        """Do an initial scan of all words appearing in data_iterable.
        Sets num_examples(total examples in data_iterable) and
        raw_vocab(collections.defaultdict(int) mapping str vocab elements to their counts for all vocab words)"""
        raise NotImplementedError

    def prepare_vocab(self, update=False, **kwargs):
        raise NotImplementedError


class BaseModelTrainables(utils.SaveLoad):
    """Class for storing and initializing/updating the trainable weights of a model. Also includes
    tables required for training weights. """
    def __init__(self, vector_size=None, seed=1):
        self.vectors = []
        self.vector_size = int(vector_size)
        self.seed = seed

    def prepare_weights(self, update=False, vocabulary=None):
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


class BaseWordEmbedddingsModel(BaseAny2VecModel):
    def __init__(self, sentences=None, workers=3, vector_size=100, epochs=5, callbacks=(), batch_words=10000,
                 trim_rule=None, sg=0, alpha=0.025, window=5, seed=1, hs=0, negative=5, cbow_mean=1,
                 min_alpha=0.0001, compute_loss=False, **kwargs):
        self.sg = int(sg)
        if vector_size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.window = int(window)
        self.random = random.RandomState(seed)
        self.min_alpha = float(min_alpha)
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.compute_loss = compute_loss
        self.running_training_loss = 0
        self.min_alpha_yet_reached = float(alpha)

        super(BaseWordEmbedddingsModel, self).__init__(
            workers=workers, vector_size=vector_size, epochs=epochs, callbacks=callbacks, batch_words=batch_words)
        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")
            self.build_vocab(sentences, trim_rule=trim_rule)
            self.train(
                sentences, total_examples=self.corpus_count, epochs=self.epochs, start_alpha=self.alpha,
                end_alpha=self.min_alpha, compute_loss=compute_loss)
        else:
            if trim_rule is not None:
                logger.warning(
                    "The rule, if given, is only used to prune vocabulary during build_vocab() "
                    "and is not stored as part of the model. Model initialized without sentences. "
                    "trim_rule provided, if any, will be ignored.")

    # for backward compatibility (aliases pointing to corresponding variables in trainables, vocabulary)
    @property
    def iter(self):
        return self.epochs

    def build_vocab(self, sentences, update=False, progress_per=10000, **kwargs):
        """Scan through all the data and create/update vocabulary.
        Should also initialize/reset/update vectors for new vocab entities.
        """
        total_words, corpus_count = self.vocabulary.scan_vocab(sentences, progress_per=progress_per, **kwargs)
        self.corpus_count = corpus_count
        self.vocabulary.prepare_vocab(len(self.trainables.vectors), self.hs, self.negative, update=update, **kwargs)
        self.trainables.prepare_weights(self.hs, self.negative, update=update, vocabulary=self.vocabulary)
        self._set_keyedvectors()

    def train(self, sentences, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None, word_count=0,
              queue_factor=2, report_delay=1.0, compute_loss=False):
        self.alpha = start_alpha or self.alpha
        self.min_alpha = end_alpha or self.min_alpha
        self.epochs = epochs or self.epochs
        self.compute_loss = compute_loss
        self.running_training_loss = 0.0
        return super(BaseWordEmbedddingsModel, self).train(
            sentences, total_examples=total_examples, total_words=total_words,
            epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, compute_loss=compute_loss)

    def _get_job_params(self, cur_epoch):
        """Return the paramter required for each batch."""
        alpha_ = self.alpha - ((self.alpha - self.min_alpha) * float(cur_epoch) / self.epochs)
        return alpha_

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
        """Return the number of words in a given job."""
        return sum(len(sentence) for sentence in job)

    def _check_training_sanity(self, epochs=None, total_examples=None, total_words=None, **kwargs):
            if self.alpha > self.min_alpha_yet_reached:
                logger.warning("Effective 'alpha' higher than previous training cycles")
            if len(self.wv.vocab) > 0:
                self._set_params_from_kv()
            if self.model_trimmed_post_training:
                raise RuntimeError("Parameters for training were discarded using model_trimmed_post_training method")

            if not self.vocabulary.vocab:  # should be set by `build_vocab`
                raise RuntimeError("you must first build vocabulary before training the model")
            if not len(self.trainables.vectors):
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

    @classmethod
    def load(cls, *args, **kwargs):
        model = super(BaseWordEmbedddingsModel, cls).load(*args, **kwargs)
        if model.negative and hasattr(model.trainables, 'index2word'):
            model.trainables.make_cum_table(vocabulary=model.vocabulary)  # rebuild cum_table from vocabulary
        if not hasattr(model, 'corpus_count'):
            model.corpus_count = None
        if not hasattr(model.trainables, 'vectors_lockf') and hasattr(model.trainables, 'vectors'):
            model.trainables.vectors_lockf = ones(len(model.trainables.vectors), dtype=REAL)
        if not hasattr(model, 'random'):
            model.random = random.RandomState(model.trainables.seed)
        if not hasattr(model, 'train_count'):
            model.train_count = 0
            model.total_train_time = 0
        return model

    def _load_specials(self, *args, **kwargs):
        super(BaseWordEmbedddingsModel, self)._load_specials(*args, **kwargs)
        # loading from a pre-KeyedVectors word2vec model
        if not hasattr(self, 'wv'):
            self.wv = self._get_keyedvector_instance()
            try:
                self._set_keyedvectors()
            except AttributeError:
                # load model saved with previous Gensim version
                raise RuntimeError(
                    "You might be trying to load a Gensim model saved using an older Gensim version."
                    "Current Gensim version does not support loading old models.")

    def _get_keyedvector_instance(self):
        raise NotImplementedError

    def _set_params_from_kv(self):
        self.trainables.vectors = self.wv.__dict__.get('vectors', [])
        self.trainables.vectors_norm = self.wv.__dict__.get('vectors_norm', None)
        self.trainables.vector_size = self.wv.__dict__.get('vector_size', None)
        self.vocabulary.vocab = self.wv.__dict__.get('vocab', {})
        self.vocabulary.index2word = self.wv.__dict__.get('index2word', [])

    def _set_keyedvectors(self):
        self.wv.vectors = self.trainables.__dict__.get('vectors', [])
        self.wv.vectors_norm = self.trainables.__dict__.get('vectors_norm', None)
        self.wv.vector_size = self.trainables.__dict__.get('vector_size', None)
        self.wv.vocab = self.vocabulary.__dict__.get('vocab', {})
        self.wv.index2word = self.vocabulary.__dict__.get('index2word', [])

    def _log_progress(self, job_queue, progress_queue, cur_epoch, example_count, total_examples,
                      raw_word_count, total_words, trained_word_count, elapsed):
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

    def _log_epoch_end(self, cur_epoch, example_count, total_examples, raw_word_count, total_words,
                       trained_word_count, elapsed, job_tally):
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

    def _log_train_end(self, raw_word_count, trained_word_count, total_elapsed):
        logger.info(
            "training on a %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, total_elapsed, trained_word_count / total_elapsed
        )
