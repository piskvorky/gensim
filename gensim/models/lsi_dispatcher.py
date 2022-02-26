#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Dispatcher process which orchestrates distributed :class:`~gensim.models.lsimodel.LsiModel` computations.
Run this script only once, on any node in your cluster.

Notes
-----
The dispatcher expects to find worker scripts already running. Make sure you run as many workers as you like on
your machines **before** launching the dispatcher.


How to use distributed LSI
--------------------------

#. Install needed dependencies (Pyro4) ::

    pip install gensim[distributed]

#. Setup serialization (on each machine) ::

    export PYRO_SERIALIZERS_ACCEPTED=pickle
    export PYRO_SERIALIZER=pickle

#. Run nameserver ::

    python -m Pyro4.naming -n 0.0.0.0 &

#. Run workers (on each machine) ::

    python -m gensim.models.lsi_worker &

#. Run dispatcher ::

    python -m gensim.models.lsi_dispatcher &

#. Run :class:`~gensim.models.lsimodel.LsiModel` in distributed mode:

    .. sourcecode:: pycon

        >>> from gensim.test.utils import common_corpus, common_dictionary
        >>> from gensim.models import LsiModel
        >>>
        >>> model = LsiModel(common_corpus, id2word=common_dictionary, distributed=True)

Command line arguments
----------------------

.. program-output:: python -m gensim.models.lsi_dispatcher --help
   :ellipsis: 0, -5

"""

import os
import sys
import logging
import argparse
import threading
import time
from queue import Queue

import Pyro4

from gensim import utils


logger = logging.getLogger(__name__)

# How many jobs (=chunks of N documents) to keep "pre-fetched" in a queue?
# A small number is usually enough, unless iteration over the corpus is very very
# slow (slower than the actual computation of LSI), in which case you can override
# this value from command line. ie. run "python ./lsi_dispatcher.py 100"
MAX_JOBS_QUEUE = 10

# timeout for the Queue object put/get blocking methods.
# it should really be infinity, but then keyboard interrupts don't work.
# so this is really just a hack, see http://bugs.python.org/issue1360
HUGE_TIMEOUT = 365 * 24 * 60 * 60  # one year


class Dispatcher:
    """Dispatcher object that communicates and coordinates individual workers.

    Warnings
    --------
    There should never be more than one dispatcher running at any one time.

    """
    def __init__(self, maxsize=0):
        """Partly initialize the dispatcher.

        A full initialization (including initialization of the workers) requires a call to
        :meth:`~gensim.models.lsi_dispatcher.Dispatcher.initialize`

        Parameters
        ----------
        maxsize : int, optional
            Maximum number of jobs to be kept pre-fetched in the queue.

        """
        self.maxsize = maxsize
        self.workers = {}
        self.callback = None  # a pyro proxy to this object (unknown at init time, but will be set later)

    @Pyro4.expose
    def initialize(self, **model_params):
        """Fully initialize the dispatcher and all its workers.

        Parameters
        ----------
        **model_params
            Keyword parameters used to initialize individual workers
            (gets handed all the way down to :meth:`gensim.models.lsi_worker.Worker.initialize`).
            See :class:`~gensim.models.lsimodel.LsiModel`.

        Raises
        ------
        RuntimeError
            When no workers are found (the :mod:`gensim.model.lsi_worker` script must be ran beforehand).

        """
        self.jobs = Queue(maxsize=self.maxsize)
        self.lock_update = threading.Lock()
        self._jobsdone = 0
        self._jobsreceived = 0

        # locate all available workers and store their proxies, for subsequent RMI calls
        self.workers = {}
        with utils.getNS() as ns:
            self.callback = Pyro4.Proxy('PYRONAME:gensim.lsi_dispatcher')  # = self
            for name, uri in ns.list(prefix='gensim.lsi_worker').items():
                try:
                    worker = Pyro4.Proxy(uri)
                    workerid = len(self.workers)
                    # make time consuming methods work asynchronously
                    logger.info("registering worker #%i from %s", workerid, uri)
                    worker.initialize(workerid, dispatcher=self.callback, **model_params)
                    self.workers[workerid] = worker
                except Pyro4.errors.PyroError:
                    logger.exception("unresponsive worker at %s, deleting it from the name server", uri)
                    ns.remove(name)

        if not self.workers:
            raise RuntimeError('no workers found; run some lsi_worker scripts on your machines first!')

    @Pyro4.expose
    def getworkers(self):
        """Get pyro URIs of all registered workers.

        Returns
        -------
        list of URIs
            The pyro URIs for each worker.

        """
        return [worker._pyroUri for worker in self.workers.values()]

    @Pyro4.expose
    def getjob(self, worker_id):
        """Atomically pop a job from the queue.

        Parameters
        ----------
        worker_id : int
            The worker that requested the job.

        Returns
        -------
        iterable of iterable of (int, float)
            The corpus in BoW format.

        """
        logger.info("worker #%i requesting a new job", worker_id)
        job = self.jobs.get(block=True, timeout=1)
        logger.info("worker #%i got a new job (%i left)", worker_id, self.jobs.qsize())
        return job

    @Pyro4.expose
    def putjob(self, job):
        """Atomically add a job to the queue.

        Parameters
        ----------
        job : iterable of list of (int, float)
            The corpus in BoW format.

        """
        self._jobsreceived += 1
        self.jobs.put(job, block=True, timeout=HUGE_TIMEOUT)
        logger.info("added a new job (len(queue)=%i items)", self.jobs.qsize())

    @Pyro4.expose
    def getstate(self):
        """Merge projections from across all workers and get the final projection.

        Returns
        -------
        :class:`~gensim.models.lsimodel.Projection`
            The current projection of the total model.

        """
        logger.info("end of input, assigning all remaining jobs")
        logger.debug("jobs done: %s, jobs received: %s", self._jobsdone, self._jobsreceived)
        while self._jobsdone < self._jobsreceived:
            time.sleep(0.5)  # check every half a second

        # TODO: merge in parallel, so that we're done in `log_2(workers)` merges,
        # and not `workers - 1` merges!
        # but merging only takes place once, after all input data has been processed,
        # so the overall effect would be small... compared to the amount of coding :-)
        logger.info("merging states from %i workers", len(self.workers))
        workers = list(self.workers.items())
        result = workers[0][1].getstate()
        for workerid, worker in workers[1:]:
            logger.info("pulling state from worker %s", workerid)
            result.merge(worker.getstate())
        logger.info("sending out merged projection")
        return result

    @Pyro4.expose
    def reset(self):
        """Re-initialize all workers for a new decomposition."""
        for workerid, worker in self.workers.items():
            logger.info("resetting worker %s", workerid)
            worker.reset()
            worker.requestjob()
        self._jobsdone = 0
        self._jobsreceived = 0

    @Pyro4.expose
    @Pyro4.oneway
    @utils.synchronous('lock_update')
    def jobdone(self, workerid):
        """A worker has finished its job. Log this event and then asynchronously transfer control back to the worker.

        Callback used by workers to notify when their job is done.

        The job done event is logged and then control is asynchronously transfered back to the worker
        (who can then request another job). In this way, control flow basically oscillates between
        :meth:`gensim.models.lsi_dispatcher.Dispatcher.jobdone` and :meth:`gensim.models.lsi_worker.Worker.requestjob`.

        Parameters
        ----------
        workerid : int
            The ID of the worker that finished the job (used for logging).

        """
        self._jobsdone += 1
        logger.info("worker #%s finished job #%i", workerid, self._jobsdone)
        worker = self.workers[workerid]
        worker.requestjob()  # tell the worker to ask for another job, asynchronously (one-way)

    def jobsdone(self):
        """Wrap :attr:`~gensim.models.lsi_dispatcher.Dispatcher._jobsdone`, needed for remote access through proxies.

        Returns
        -------
        int
            Number of jobs already completed.

        """
        return self._jobsdone

    @Pyro4.oneway
    def exit(self):
        """Terminate all registered workers and then the dispatcher."""
        for workerid, worker in self.workers.items():
            logger.info("terminating worker %s", workerid)
            worker.exit()
        logger.info("terminating dispatcher")
        os._exit(0)  # exit the whole process (not just this thread ala sys.exit())


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__[:-135], formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'maxsize',
        nargs='?',
        type=int,
        help='Maximum number of jobs to be kept pre-fetched in the queue.',
        default=MAX_JOBS_QUEUE,
    )
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    utils.pyro_daemon('gensim.lsi_dispatcher', Dispatcher(maxsize=args.maxsize))
    logger.info("finished running %s", parser.prog)
