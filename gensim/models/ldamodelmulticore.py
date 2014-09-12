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
        self.workers = kwargs.pop("workers", cpu_count())
        super(LdaModelMulticore, self).__init__(*args, **kwargs)


    def update(self, corpus):
        """
        Train the model with new documents, by EM-iterating over `corpus` until
        the topics converge (or until the maximum number of allowed iterations
        is reached). `corpus` must be an iterable (repeatable stream of documents),

        In distributed mode, the E step is distributed over a cluster of machines.

        This update also supports updating an already trained model (`self`)
        with new documents from `corpus`; the two models are then merged in
        proportion to the number of old vs. new documents. This feature is still
        experimental for non-stationary input streams.

        For stationary input (no topic drift in new documents), on the other hand,
        this equals the online update of Hoffman et al. and is guaranteed to
        converge for any `decay` in (0.5, 1.0>.

        """

        # rho is the "speed" of updating; TODO try other fncs
        rho = lambda: pow(1.0 + self.num_updates / self.chunksize, -self.decay)

        try:
            lencorpus = len(corpus)
        except:
            logger.warning("input corpus stream has no len(); counting documents")
            lencorpus = sum(1 for _ in corpus)
        if lencorpus == 0:
            logger.warning("LdaModel.update() called with an empty corpus")
            return

        self.state.numdocs += lencorpus

        if self.update_every:
            updatetype = "online"
            updateafter = min(lencorpus, self.update_every * self.chunksize)
        else:
            updatetype = "batch"
            updateafter = lencorpus
        evalafter = min(lencorpus, (self.eval_every or 0) * self.chunksize)

        updates_per_pass = max(1, lencorpus / updateafter)
        logger.info("running %s LDA training, %s topics, %i passes over "
                    "the supplied corpus of %i documents, updating model once "
                    "every %i documents, evaluating perplexity every %i documents, "
                    "iterating %ix with a convergence threshold of %f",
                    updatetype, self.num_topics, self.passes, lencorpus,
                    updateafter, evalafter, self.iterations,
                    self.gamma_threshold)

        if updates_per_pass * self.passes < 10:
            logger.warning("too few updates, training might not converge; consider "
                           "increasing the number of passes or iterations to improve accuracy")


        def worker_e_step(input_queue, result_queue):
            """
            Perform E-step for each (model, chunk) pair from the input queue, placing
            the resulting state into the result queue.

            """
            logger.info("worker process entering E-step loop")
            while True:
                logger.debug("getting a new job")
                chunk_no, worker_lda, chunk = input_queue.get()
                logger.info("processing chunk #%i of %i documents", chunk_no, len(chunk))
                worker_lda.state.reset()
                worker_lda.do_estep(chunk)
                del chunk
                logger.info("processed chunk, queuing the result")
                result_queue.put(worker_lda.state)
                del worker_lda  # free up some memory
                logger.info("result put")


        logger.info("training LDA model using %i processes", self.workers)
        job_queue = Queue(maxsize=2 * self.workers)
        result_queue = Queue()
        pool = Pool(self.workers, worker_e_step, (job_queue, result_queue,))

        for pass_ in xrange(self.passes):
            queue_size, reallen = [0], 0
            other = LdaState(self.eta, self.state.sstats.shape)     #TODO optimization do not create zero matrix

            def process_result_queue(force=False):
                """
                """
                while not result_queue.empty():
                    other.merge(result_queue.get())
                    logger.info("document merged, current numdocs = %i", other.numdocs)
                    queue_size[0] -= 1
                if force or (other.numdocs >= self.chunksize * self.workers):
                    self.do_mstep(rho(), other)
                    other.reset()

            # setting other to None for case of the batch version
            chunk_stream = utils.grouper(corpus, self.chunksize, as_numpy=True)
            for chunk_no, chunk in enumerate(chunk_stream):
                reallen += len(chunk)  # keep track of how many documents we've processed so far

                if self.eval_every and ((reallen == lencorpus) or ((chunk_no + 1) % self.eval_every == 0)):
                    self.log_perplexity(chunk, total_docs=lencorpus)

                # Add the data to the queue for the E step
                chunk_put = False
                while not chunk_put:
                    try:
                        job_queue.put((chunk_no, self, chunk), block=False, timeout=0.1)
                        chunk_put = True
                        queue_size[0] += 1
                        logger.info('PROGRESS: pass %i, dispatched chunk #%i = documents up to #%i/%i',
                            pass_, chunk_no, chunk_no * self.chunksize + len(chunk), lencorpus)
                        del chunk
                    except Full:
                        process_result_queue()

                process_result_queue()

            while queue_size[0] > 0:
                process_result_queue(force=True)

                # FIXME solve what to do in case when self.optimize_alpha?

            #endfor single corpus pass
            if reallen != lencorpus:
                raise RuntimeError("input corpus size changed during training (don't use generators as input)")
        #endfor entire corpus update

        pool.close()
