#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from gensim import utils
from gensim.models.ldamodel import LdaModel, LdaState
from Queue import Full
from multiprocessing import Pool, Queue, cpu_count

logger = logging.getLogger(__name__)


class LdaModelMulticore(LdaModel):
    def __init__(self, *args, **kwargs):
        """
        `workers` is the number of extra processes used for parallelization. Default
        is `max(1, multiprocessing.cpu_count() - 1)`.

        if `update_every` is set, perform M-step once every `workers * chunksize`
        documents. Otherwise, run batch LDA, updating model only once at the end
        of each full corpus pass.

        For all other arguments, see LdaModel. `distributed` and `alpha='auto'` are
        not supported.

        Example:

        >>> lda = LdaModelMulticore(corpus, id2word=id2word, num_topics=100)  # train model
        >>> print(lda[doc_bow]) # get topic probability distribution for a document
        >>> lda.update(corpus2) # update the LDA model with additional documents
        >>> print(lda[doc_bow])

        """
        # TODO: full explicit init params, incl. defaults
        self.workers = kwargs.pop("workers", max(1, cpu_count() - 1))
        super(LdaModelMulticore, self).__init__(*args, **kwargs)


    def update(self, corpus):
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
        # rho is the "speed" of updating, decelerating over time
        rho = lambda: pow(1.0 + self.num_updates / self.chunksize, -self.decay)

        try:
            lencorpus = len(corpus)
        except:
            logger.warning("input corpus stream has no len(); counting documents")
            lencorpus = sum(1 for _ in corpus)
        if lencorpus == 0:
            logger.warning("LdaModelMulticore.update() called with an empty corpus")
            return

        self.state.numdocs += lencorpus

        if self.update_every:
            updatetype = "online"
            updateafter = self.chunksize * self.workers
        else:
            updatetype = "batch"
            updateafter = lencorpus
        evalafter = min(lencorpus, (self.eval_every * updateafter or 0))

        updates_per_pass = max(1, lencorpus / updateafter)
        logger.info("running %s LDA training, %s topics, %i passes over the"
            " supplied corpus of %i documents, updating every %i documents,"
            " evaluating every %i documents, iterating %ix with a convergence threshold of %f",
            updatetype, self.num_topics, self.passes, lencorpus, updateafter, evalafter,
            self.iterations, self.gamma_threshold)

        if updates_per_pass * self.passes < 10:
            logger.warning("too few updates, training might not converge; consider "
                "increasing the number of passes or iterations to improve accuracy")

        def worker_e_step(input_queue, result_queue):
            """
            Perform E-step for each (chunk_no, chunk, model) 3-tuple from the
            input queue, placing the resulting state into the result queue.

            """
            logger.info("worker process entering E-step loop")
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

        job_queue = Queue(maxsize=2 * self.workers)
        result_queue = Queue()

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
                if (force and merged_new) or (self.update_every and (other.numdocs >= updateafter)):
                    self.do_mstep(rho(), other)
                    other.reset()
                    if self.eval_every is not None and self.eval_every != 0 and ((force and queue_size[0] == 0) or (self.num_updates / updateafter) % self.eval_every == 0):
                        self.log_perplexity(chunk, total_docs=lencorpus)

            chunk_stream = utils.grouper(corpus, self.chunksize, as_numpy=True)
            for chunk_no, chunk in enumerate(chunk_stream):
                reallen += len(chunk)  # keep track of how many documents we've processed so far

                # add chunk to the workers' job queue
                chunk_put = False
                while not chunk_put:
                    try:
                        job_queue.put((chunk_no, chunk, self), block=False, timeout=0.1)
                        chunk_put = True
                        queue_size[0] += 1
                        logger.info('PROGRESS: pass %i, dispatched chunk #%i = '
                            'documents up to #%i/%i, outstanding queue size %i',
                            pass_, chunk_no, chunk_no * self.chunksize + len(chunk), lencorpus, queue_size[0])
                    except Full:
                        # in case the job queue is full, check the result queue to make sure we don't deadlock
                        process_result_queue()

                process_result_queue()

            while queue_size[0] > 0:
                process_result_queue(force=True)
            #endfor single corpus pass

            if reallen != lencorpus:
                raise RuntimeError("input corpus size changed during training (don't use generators as input)")
        #endfor entire corpus update

        pool.close()
