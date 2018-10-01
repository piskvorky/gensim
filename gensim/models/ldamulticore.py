#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jan Zikes, Radim Rehurek
# Copyright (C) 2014 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Online Latent Dirichlet Allocation (LDA) in Python, using all CPU cores to parallelize and speed up model training.

The parallelization uses multiprocessing; in case this doesn't work for you for some reason,
try the :class:`gensim.models.ldamodel.LdaModel` class which is an equivalent, but more straightforward and single-core
implementation.

The training algorithm:

* is **streamed**: training documents may come in sequentially, no random access required,
* runs in **constant memory** w.r.t. the number of documents: size of the
  training corpus does not affect memory footprint, can process corpora larger than RAM

Wall-clock `performance on the English Wikipedia <http://radimrehurek.com/gensim/wiki.html>`_ (2G corpus positions,
3.5M documents, 100K features, 0.54G non-zero entries in the final bag-of-words matrix), requesting 100 topics:


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

This module allows both LDA model estimation from a training corpus and inference of topic distribution on new,
unseen documents. The model can also be updated with new documents for online training.

The core estimation code is based on the `onlineldavb.py script
<https://github.com/blei-lab/onlineldavb/blob/master/onlineldavb.py>`_, by `Hoffman, Blei, Bach:
Online Learning for Latent Dirichlet Allocation, NIPS 2010 <http://www.cs.princeton.edu/~mdhoffma>`_.

Usage examples
--------------
The constructor estimates Latent Dirichlet Allocation model parameters based on a training corpus

>>> from gensim.test.utils import common_corpus, common_dictionary
>>>
>>> lda = LdaMulticore(common_corpus, id2word=common_dictionary, num_topics=10)

Save a model to disk, or reload a pre-trained model

>>> from gensim.test.utils import datapath
>>>
>>> # Save model to disk.
>>> temp_file = datapath("model")
>>> lda.save(temp_file)
>>>
>>> # Load a potentially pretrained model from disk.
>>> lda = LdaModel.load(temp_file)

Query, or update the model using new, unseen documents

>>> other_texts = [
...     ['computer', 'time', 'graph'],
...     ['survey', 'response', 'eps'],
...     ['human', 'system', 'computer']
... ]
>>> other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
>>>
>>> unseen_doc = other_corpus[0]
>>> vector = lda[unseen_doc] # get topic probability distribution for a document
>>>
>>> # Update the model by incrementally training on the new corpus.
>>> lda.update(other_corpus) # update the LDA model with additional documents

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
    """An optimized implementation of the LDA algorithm, able to harness the power of multicore CPUs.
    Follows the similar API as the parent class :class:`~gensim.models.ldamodel.LdaModel`.

    """
    def __init__(self, corpus=None, num_topics=100, id2word=None, workers=None,
                 chunksize=2000, passes=1, batch=False, alpha='symmetric',
                 eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50,
                 gamma_threshold=0.001, random_state=None, minimum_probability=0.01,
                 minimum_phi_value=0.01, per_word_topics=False, dtype=np.float32):
        """

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            Stream of document vectors or sparse matrix of shape (`num_terms`, `num_documents`).
            If not given, the model is left untrained (presumably because you want to call
            :meth:`~gensim.models.ldamodel.LdaModel.update` manually).
        num_topics : int, optional
            The number of requested latent topics to be extracted from the training corpus.
        id2word : {dict of (int, str),  :class:`gensim.corpora.dictionary.Dictionary`}
            Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for
            debugging and topic printing.
        workers : int, optional
            Number of workers processes to be used for parallelization. If None all available cores
            (as estimated by `workers=cpu_count()-1` will be used. **Note** however that for
            hyper-threaded CPUs, this estimation returns a too high number -- set `workers`
            directly to the number of your **real** cores (not hyperthreads) minus one, for optimal performance.
        chunksize :  int, optional
            Number of documents to be used in each training chunk.
        passes : int, optional
            Number of passes through the corpus during training.
        alpha : {np.ndarray, str}, optional
            Can be set to an 1D array of length equal to the number of expected topics that expresses
            our a-priori belief for the each topics' probability.
            Alternatively default prior selecting strategies can be employed by supplying a string:

                * 'asymmetric': Uses a fixed normalized asymmetric prior of `1.0 / topicno`.
                * 'auto': Learns an asymmetric prior from the corpus.
        eta : {float, np.array, str}, optional
            A-priori belief on word probability, this can be:

                * scalar for a symmetric prior over topic/word probability,
                * vector of length num_words to denote an asymmetric user defined probability for each word,
                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination,
                * the string 'auto' to learn the asymmetric prior from the data.
        decay : float, optional
            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten
            when each new document is examined. Corresponds to Kappa from
            `Matthew D. Hoffman, David M. Blei, Francis Bach:
            "Online Learning for Latent Dirichlet Allocation NIPS'10" <https://www.di.ens.fr/~fbach/mdhnips2010.pdf>`_.
        offset : float, optional
            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.
            Corresponds to Tau_0 from `Matthew D. Hoffman, David M. Blei, Francis Bach:
            "Online Learning for Latent Dirichlet Allocation NIPS'10" <https://www.di.ens.fr/~fbach/mdhnips2010.pdf>`_.
        eval_every : int, optional
            Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.
        iterations : int, optional
            Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.
        gamma_threshold : float, optional
            Minimum change in the value of the gamma parameters to continue iterating.
        minimum_probability : float, optional
            Topics with a probability lower than this threshold will be filtered out.
        random_state : {np.random.RandomState, int}, optional
            Either a randomState object or a seed to generate one. Useful for reproducibility.
        minimum_phi_value : float, optional
            if `per_word_topics` is True, this represents a lower bound on the term probabilities.
        per_word_topics : bool
            If True, the model also computes a list of topics, sorted in descending order of most likely topics for
            each word, along with their phi values multiplied by the feature length (i.e. word count).
        dtype : {numpy.float16, numpy.float32, numpy.float64}, optional
            Data-type to use during calculations inside model. All inputs are also converted.

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
        """Train the model with new documents, by EM-iterating over `corpus` until the topics converge
        (or until the maximum number of allowed iterations is reached).

        Train the model with new documents, by EM-iterating over the corpus until the topics converge, or until
        the maximum number of allowed iterations is reached. `corpus` must be an iterable. The E step is distributed
        into the several processes.

        Notes
        -----
        This update also supports updating an already trained model (`self`)
        with new documents from `corpus`; the two models are then merged in
        proportion to the number of old vs. new documents. This feature is still
        experimental for non-stationary input streams.

        For stationary input (no topic drift in new documents), on the other hand,
        this equals the online update of Hoffman et al. and is guaranteed to
        converge for any `decay` in (0.5, 1.0>.

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            Stream of document vectors or sparse matrix of shape (`num_terms`, `num_documents`) used to update the
            model.
        chunks_as_numpy : bool
            Whether each chunk passed to the inference step should be a np.ndarray or not. Numpy can in some settings
            turn the term IDs into floats, these will be converted back into integers in inference, which incurs a
            performance hit. For distributed computing it may be desirable to keep the chunks as `numpy.ndarray`.

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
            "updating every %i documents, evaluating every ~%i documents, "
            "iterating %ix with a convergence threshold of %f",
            updatetype, self.num_topics, self.passes, lencorpus, updateafter,
            evalafter, self.iterations, self.gamma_threshold
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
                    if self.eval_every is not None and \
                            ((force and queue_size[0] == 0) or
                                 (self.eval_every != 0 and (self.num_updates / updateafter) % self.eval_every == 0)):
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
                            "PROGRESS: pass %i, dispatched chunk #%i = documents up to #%i/%i, "
                            "outstanding queue size %i",
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
    """Perform E-step for each job.

    Parameters
    ----------
    input_queue : queue of (int, list of (int, float), :class:`~gensim.models.lda_worker.Worker`)
        Each element is a job characterized by its ID, the corpus chunk to be processed in BOW format and the worker
        responsible for processing it.
    result_queue : queue of :class:`~gensim.models.ldamodel.LdaState`
        After the worker finished the job, the state of the resulting (trained) worker model is appended to this queue.

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
