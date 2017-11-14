#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jan Zikes, Radim Rehurek
# Copyright (C) 2014 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Latent Dirichlet Allocation (LDA) in Python, using all CPU cores to parallelize and
speed up model training.

The parallelization uses multiprocessing; in case this doesn't work for you for
some reason, try the :class:`gensim.models.ldamodel.LdaModel` class which is an
equivalent, but more straightforward and single-core implementation.

The training algorithm:

* is **streamed**: training documents may come in sequentially, no random access required,
* runs in **constant memory** w.r.t. the number of documents: size of the
  training corpus does not affect memory footprint, can process corpora larger than RAM

Wall-clock `performance on the English Wikipedia <http://radimrehurek.com/gensim/wiki.html>`_
(2G corpus positions, 3.5M documents, 100K features, 0.54G non-zero entries in the final
bag-of-words matrix), requesting 100 topics:


====================================================== ==============
 algorithm                                             training time
====================================================== ==============
 LdaMulticore(workers=1)                               2h30m
 LdaMulticore(workers=2)                               1h24m
 LdaMulticore(workers=3)                               1h6m
 old LdaModel()                                        3h44m
 simply iterating over input corpus = I/O overhead     20m
====================================================== ==============

(Measured on `this i7 server <http://www.hetzner.de/en/hosting/produkte_rootserver/ex40ssd>`_
with 4 physical cores, so that optimal `workers=3`, one less than the number of cores.)

This module allows both LDA model estimation from a training corpus and inference of topic
distribution on new, unseen documents. The model can also be updated with new documents
for online training.

The core estimation code is based on the `onlineldavb.py` script by M. Hoffman [1]_, see
**Hoffman, Blei, Bach: Online Learning for Latent Dirichlet Allocation, NIPS 2010.**

.. [1] http://www.cs.princeton.edu/~mdhoffma
"""

import logging

import numpy as np

from gensim import utils
from gensim.models.ldamodel import LdaModel, LdaState

import six
from six.moves import queue, xrange
from multiprocessing import Pool, Queue, cpu_count

logger = logging.getLogger(__name__)


class LdaMulticore(LdaModel):
    """
    The constructor estimates Latent Dirichlet Allocation model parameters based
    on a training corpus:

    >>> lda = LdaMulticore(corpus, num_topics=10)

    You can then infer topic distributions on new, unseen documents, with

    >>> doc_lda = lda[doc_bow]

    The model can be updated (trained) with new documents via

    >>> lda.update(other_corpus)

    Model persistency is achieved through its `load`/`save` methods.

    """

    def __init__(self, corpus=None, num_topics=100, id2word=None, workers=None,
                 chunksize=2000, passes=1, batch=False, alpha='symmetric',
                 eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50,
                 gamma_threshold=0.001, random_state=None, minimum_probability=0.01,
                 minimum_phi_value=0.01, per_word_topics=False, dtype=np.float32):
        """
        If given, start training from the iterable `corpus` straight away. If not given,
        the model is left untrained (presumably because you want to call `update()` manually).

        `num_topics` is the number of requested latent topics to be extracted from
        the training corpus.

        `id2word` is a mapping from word ids (integers) to words (strings). It is
        used to determine the vocabulary size, as well as for debugging and topic
        printing.

        `workers` is the number of extra processes to use for parallelization. Uses
        all available cores by default: `workers=cpu_count()-1`. **Note**: for
        hyper-threaded CPUs, `cpu_count()` returns a useless number -- set `workers`
        directly to the number of your **real** cores (not hyperthreads) minus one,
        for optimal performance.

        If `batch` is not set, perform online training by updating the model once
        every `workers * chunksize` documents (online training). Otherwise,
        run batch LDA, updating model only once at the end of each full corpus pass.

        `alpha` and `eta` are hyperparameters that affect sparsity of the document-topic
        (theta) and topic-word (lambda) distributions. Both default to a symmetric
        1.0/num_topics prior.

        `alpha` can be set to an explicit array = prior of your choice. It also
        support special values of 'asymmetric' and 'auto': the former uses a fixed
        normalized asymmetric 1.0/topicno prior, the latter learns an asymmetric
        prior directly from your data.

        `eta` can be a scalar for a symmetric prior over topic/word
        distributions, or a matrix of shape num_topics x num_words,
        which can be used to impose asymmetric priors over the word
        distribution on a per-topic basis. This may be useful if you
        want to seed certain topics with particular words by boosting
        the priors for those words.

        Calculate and log perplexity estimate from the latest mini-batch once every
        `eval_every` documents. Set to `None` to disable perplexity estimation (faster),
        or to `0` to only evaluate perplexity once, at the end of each corpus pass.

        `decay` and `offset` parameters are the same as Kappa and Tau_0 in
        Hoffman et al, respectively.

        `random_state` can be a numpy.random.RandomState object or the seed for one

        Example:

        >>> lda = LdaMulticore(corpus, id2word=id2word, num_topics=100)  # train model
        >>> print(lda[doc_bow]) # get topic probability distribution for a document
        >>> lda.update(corpus2) # update the LDA model with additional documents
        >>> print(lda[doc_bow])

        """
        self.workers = max(1, cpu_count() - 1) if workers is None else workers
        self.batch = batch

        if isinstance(alpha, six.string_types) and alpha == 'auto':
            raise NotImplementedError("auto-tuning alpha not implemented in multicore LDA; use plain LdaModel.")

        super(LdaMulticore, self).__init__(
            corpus=corpus, num_topics=num_topics,
            id2word=id2word, chunksize=chunksize, passes=passes, alpha=alpha, eta=eta,
            decay=decay, offset=offset, eval_every=eval_every, iterations=iterations,
            gamma_threshold=gamma_threshold, random_state=random_state, minimum_probability=minimum_probability,
            minimum_phi_value=minimum_phi_value, per_word_topics=per_word_topics, dtype=dtype
        )

    def update(self, corpus, chunks_as_numpy=False):
        """
        Train the model with new documents, by EM-iterating over `corpus` until
        the topics converge (or until the maximum number of allowed iterations
        is reached). `corpus` must be an iterable (repeatable stream of documents),

        The E-step is distributed into the several processes.

        This update also supports updating an already trained model (`self`)
        with new documents from `corpus`; the two models are then merged in
        proportion to the number of old vs. new documents. This feature is still
        experimental for non-stationary input streams.

        For stationary input (no topic drift in new documents), on the other hand,
        this equals the online update of Hoffman et al. and is guaranteed to
        converge for any `decay` in (0.5, 1.0>.

        """
        try:
            lencorpus = len(corpus)
        except TypeError:
            logger.warning("input corpus stream has no len(); counting documents")
            lencorpus = sum(1 for _ in corpus)
        if lencorpus == 0:
            logger.warning("LdaMulticore.update() called with an empty corpus")
            return

        self.state.numdocs += lencorpus

        if not self.batch:
            updatetype = "online"
            updateafter = self.chunksize * self.workers
        else:
            updatetype = "batch"
            updateafter = lencorpus
        evalafter = min(lencorpus, (self.eval_every or 0) * updateafter)

        updates_per_pass = max(1, lencorpus / updateafter)
        logger.info(
            "running %s LDA training, %s topics, %i passes over the supplied corpus of %i documents, "
            "updating every %i documents, evaluating every ~%i documents, iterating %ix with a convergence threshold of %f",
            updatetype, self.num_topics, self.passes, lencorpus, updateafter, evalafter, self.iterations, self.gamma_threshold
        )

        if updates_per_pass * self.passes < 10:
            logger.warning(
                "too few updates, training might not converge; "
                "consider increasing the number of passes or iterations to improve accuracy"
            )

        job_queue = Queue(maxsize=2 * self.workers)
        result_queue = Queue()

        # rho is the "speed" of updating; TODO try other fncs
        # pass_ + num_updates handles increasing the starting t for each pass,
        # while allowing it to "reset" on the first pass of each update
        def rho():
            return pow(self.offset + pass_ + (self.num_updates / self.chunksize), -self.decay)

        logger.info("training LDA model using %i processes", self.workers)
        pool = Pool(self.workers, worker_e_step, (job_queue, result_queue,))
        for pass_ in xrange(self.passes):
            queue_size, reallen = [0], 0
            other = LdaState(self.eta, self.state.sstats.shape)

            def process_result_queue(force=False):
                """
                Clear the result queue, merging all intermediate results, and update the
                LDA model if necessary.

                """
                merged_new = False
                while not result_queue.empty():
                    other.merge(result_queue.get())
                    queue_size[0] -= 1
                    merged_new = True
                if (force and merged_new and queue_size[0] == 0) or (not self.batch and (other.numdocs >= updateafter)):
                    self.do_mstep(rho(), other, pass_ > 0)
                    other.reset()
                    if self.eval_every is not None and ((force and queue_size[0] == 0) or (self.eval_every != 0 and (self.num_updates / updateafter) % self.eval_every == 0)):
                        self.log_perplexity(chunk, total_docs=lencorpus)

            chunk_stream = utils.grouper(corpus, self.chunksize, as_numpy=chunks_as_numpy)
            for chunk_no, chunk in enumerate(chunk_stream):
                reallen += len(chunk)  # keep track of how many documents we've processed so far

                # put the chunk into the workers' input job queue
                chunk_put = False
                while not chunk_put:
                    try:
                        job_queue.put((chunk_no, chunk, self), block=False, timeout=0.1)
                        chunk_put = True
                        queue_size[0] += 1
                        logger.info(
                            "PROGRESS: pass %i, dispatched chunk #%i = documents up to #%i/%i, outstanding queue size %i",
                            pass_, chunk_no, chunk_no * self.chunksize + len(chunk), lencorpus, queue_size[0]
                        )
                    except queue.Full:
                        # in case the input job queue is full, keep clearing the
                        # result queue, to make sure we don't deadlock
                        process_result_queue()

                process_result_queue()
            # endfor single corpus pass

            # wait for all outstanding jobs to finish
            while queue_size[0] > 0:
                process_result_queue(force=True)

            if reallen != lencorpus:
                raise RuntimeError("input corpus size changed during training (don't use generators as input)")
        # endfor entire update

        pool.terminate()


def worker_e_step(input_queue, result_queue):
    """
    Perform E-step for each (chunk_no, chunk, model) 3-tuple from the
    input queue, placing the resulting state into the result queue.

    """
    logger.debug("worker process entering E-step loop")
    while True:
        logger.debug("getting a new job")
        chunk_no, chunk, worker_lda = input_queue.get()
        logger.debug("processing chunk #%i of %i documents", chunk_no, len(chunk))
        worker_lda.state.reset()
        worker_lda.do_estep(chunk)  # TODO: auto-tune alpha?
        del chunk
        logger.debug("processed chunk, queuing the result")
        result_queue.put(worker_lda.state)
        del worker_lda  # free up some memory
        logger.debug("result put")
