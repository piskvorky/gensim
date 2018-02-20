#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

""":class:`~gensim.models.lda_dispatcher.Dispatcher` process which orchestrates
distributed :class:`~gensim.models.ldamodel.LdaModel` computations.
Run this script only once, on the master node in your cluster.

Notes
-----
The dispatches expects to find worker scripts already running. Make sure
you run as many workers as you like on your machines **before** launching
the dispatcher.

Warnings
--------
Requires installed `Pyro4 <https://pythonhosted.org/Pyro4/>`_.
Distributed version works only in local network.


How to use distributed :class:`~gensim.models.ldamodel.LdaModel`
----------------------------------------------------------------


#. Install needed dependencies (Pyro4) ::

    pip install gensim[distributed]

#. Setup serialization (on each machine) ::

    export PYRO_SERIALIZERS_ACCEPTED=pickle
    export PYRO_SERIALIZER=pickle

#. Run nameserver ::

    python -m Pyro4.naming -n 0.0.0.0 &

#. Run workers (on each machine) ::

    python -m gensim.models.lda_worker &

#. Run dispatcher ::

    python -m gensim.models.lda_dispatcher &

#. Run :class:`~gensim.models.ldamodel.LdaModel` in distributed mode ::

    >>> from gensim.test.utils import common_corpus, common_dictionary
    >>> from gensim.models import LdaModel
    >>>
    >>> model = LdaModel(common_corpus, id2word=common_dictionary,
                         distributed=True)

#. You can then infer topic distributions on new, unseen documents, with

    >>> doc_lda = model[doc_bow]
    The model can be updated (trained) with new documents via
    >>> lda.update(other_corpus)


Command line arguments
----------------------

.. program-output:: python -m gensim.models.lda_dispatcher --help
   :ellipsis: 0, -5

"""


from __future__ import with_statement
import argparse
import os
import sys
import logging
import threading
import time
from six import iteritems, itervalues

try:
    from Queue import Queue
except ImportError:
    from queue import Queue
import Pyro4
from gensim import utils
from gensim.models.lda_worker import LDA_WORKER_PREFIX


logger = logging.getLogger("gensim.models.lda_dispatcher")


# How many jobs (=chunks of N documents) to keep "pre-fetched" in a queue?
# A small number is usually enough, unless iteration over the corpus is
# very very slow (slower than the actual computation of LDA), in which case
# you can override this value from command line. ie.
# run "python ./lda_dispatcher.py 100"
MAX_JOBS_QUEUE = 10

# timeout for the Queue object put/get blocking methods.
# it should theoretically be infinity, but then keyboard interrupts don't work.
# so this is really just a hack, see http://bugs.python.org/issue1360
HUGE_TIMEOUT = 365 * 24 * 60 * 60  # one year

LDA_DISPATCHER_PREFIX = 'gensim.lda_dispatcher'


class Dispatcher(object):
    """Dispatcher object that communicates and coordinates individual workers.

    Attributes
    ----------
    callback : :class: `~Pyro4.core.Proxy`
        A proxy for some remote object.Intercepts method calls and \
        dispatches them to the remote object.
    jobs : :class: `~Queue.Queue`
        Constructs a FIFO queue.
    lock_update : :class: `~threading.Lock`
        This class implements primitive lock objects. Once a thread has \
        acquired a lock, subsequent attempts to acquire it block, until it is \
        released; any thread may release it.

    Warnings
    --------
    There should never be more than one dispatcher running at any one time.

    """

    def __init__(self, maxsize=MAX_JOBS_QUEUE, ns_conf=None):
        """Partly initializes the dispatcher.

        A full initialization (including initialization of the workers)
        requires a call to
        :meth:`~gensim.models.lda_dispatcher.Dispatcher.initialize`

        Parameters
        ----------
        maxsize : int, optional
                Maximum number of jobs to be kept pre-fetched in the queue.
        ns_conf : dict of {str:(str,optional),str:(int,optional), \
                            str:(bool:optional),str:(str,optional)},optional
            Sets up the name server configuration for the pyro daemon server \
            of dispatcher.This also helps to keep track of your objects in \
            your netword by using logical object names instead of exact \
            object name(or id) and its location.
        workers : dict of { int : :class: `~Pyro4.core.Proxy` }
            Locates all available workers and store their proxies, for \
            subsequent RMI calls.
        }

        """
        self.maxsize = maxsize
        self.callback = None
        self.ns_conf = ns_conf if ns_conf is not None else {}

    @Pyro4.expose
    def initialize(self, **model_params):
        """Fully initializes the dispatcher and all its workers.

        Parameters
        ----------
        **model_params
            Keyword parameters used to initialize individual workers,
            see:class:`~gensim.models.ldamodel.LdaModel`.

        Raises
        ------
        RuntimeError
            No workers found.Need to have atleast one worker running.

        """
        self.jobs = Queue(maxsize=self.maxsize)
        self.lock_update = threading.Lock()
        self._jobsdone = 0
        self._jobsreceived = 0

        self.workers = {}
        with utils.getNS(**self.ns_conf) as ns:
            self.callback = Pyro4.Proxy(ns.list(
                            prefix=LDA_DISPATCHER_PREFIX)
                            [LDA_DISPATCHER_PREFIX])
            for name, uri in iteritems(ns.list(prefix=LDA_WORKER_PREFIX)):
                try:
                    worker = Pyro4.Proxy(uri)
                    workerid = len(self.workers)
                    # make time consuming methods work asynchronously
                    logger.info("registering worker #%i at %s", workerid,
                                uri)
                    worker.initialize(workerid, dispatcher=self.callback,
                                      **model_params)
                    self.workers[workerid] = worker
                except Pyro4.errors.PyroError:
                    logger.warning("unresponsive worker at %s,deleting it"
                                   " from the name server", uri)
                    ns.remove(name)

        if not self.workers:
            raise RuntimeError('no workers found; run some lda_worker '
                               'scripts on your machines first!')

    @Pyro4.expose
    def getworkers(self):
        """Return pyro URIs of all registered workers.

        Returns
        -------
        list of URIs
            The pyro URIs for each worker.

        """
        return [worker._pyroUri for worker in itervalues(self.workers)]

    @Pyro4.expose
    def getjob(self, worker_id):
        """Atomically pops a job from the queue.

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
        logger.info("worker #%i got a new job (%i left)", worker_id,
                    self.jobs.qsize())
        return job

    @Pyro4.expose
    def putjob(self, job):
        """Atomically add a job to the queue.

        Parameters
        ----------
        job : iterable of iterable of (int, float)
            The corpus in BoW format.

        """
        self._jobsreceived += 1
        self.jobs.put(job, block=True, timeout=HUGE_TIMEOUT)
        logger.info("added a new job (len(queue)=%i items)",
                    self.jobs.qsize())

    @Pyro4.expose
    def getstate(self):
        """
        Merge states from across all workers and return the result.

        Returns
        -------
        :class:`~gensim.models.ldamodel.LdaState`
            Merged resultant state

        """
        logger.info("end of input, assigning all remaining jobs")
        logger.debug("jobs done: %s, jobs received: %s",
                     self._jobsdone, self._jobsreceived)
        i = 0
        count = 10
        while self._jobsdone < self._jobsreceived:
            time.sleep(0.5)  # check every half a second
            i += 1
            if i > count:
                i = 0
                for workerid, worker in iteritems(self.workers):
                    logger.info("checking aliveness for worker %s", workerid)
                    worker.ping()

        logger.info("merging states from %i workers", len(self.workers))
        workers = list(self.workers.values())
        result = workers[0].getstate()
        for worker in workers[1:]:
            result.merge(worker.getstate())

        logger.info("sending out merged state")
        return result

    @Pyro4.expose
    def reset(self, state):
        """Reinitializes all workers for a new EM iteration.

        Parameters
        ----------
        state : :class:`~gensim.models.ldamodel.LdaState`
            Encapsulates information for distributed computation
            of LdaModel objects.

        """
        for workerid, worker in iteritems(self.workers):
            logger.info("resetting worker %s", workerid)
            worker.reset(state)
            worker.requestjob()
        self._jobsdone = 0
        self._jobsreceived = 0

    @Pyro4.expose
    @Pyro4.oneway
    @utils.synchronous('lock_update')
    def jobdone(self, workerid):
        """Workers use callback to notify when their job is done.

        The job done event is logged and then control is asynchronously
        transfered back to the worker(who can then request another job).
        In this way, control flow basically oscillates between
        :meth:`gensim.models.lda_dispatcher.Dispatcher.jobdone` and
        :meth:`gensim.models.lda_worker.Worker.requestjob`.

        Parameters
        ----------
        workerid : int
            The ID of the worker that finished the job (used for logging).

        """
        self._jobsdone += 1
        logger.info("worker #%s finished job #%i", workerid, self._jobsdone)
        self.workers[workerid].requestjob()  # tell the worker to ask for
        # another job, asynchronously (one-way)

    def jobsdone(self):
        """Wrap :attr:`~gensim.models.lda_dispatcher.Dispatcher._jobsdone`,
        needed for remote access through proxies.

        Returns
        -------
        int
            Number of jobs already completed.

        """
        return self._jobsdone

    @Pyro4.oneway
    def exit(self):
        """Terminate all registered workers and then the dispatcher."""
        for workerid, worker in iteritems(self.workers):
            logger.info("terminating worker %s", workerid)
            worker.exit()
        logger.info("terminating dispatcher")
        os._exit(0)  # exit the whole process (not just this thread ala
        # sys.exit())
# endclass Dispatcher


def main():
    """Set up argument parser,logger and launches pyro daemon."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxsize", help="How many jobs (=chunks of N "
                        "documents) to keep 'pre-fetched' in a queue "
                        "(default: %(default)s)", type=int,
                        default=MAX_JOBS_QUEUE)
    parser.add_argument("--host", help="Nameserver hostname (default: "
                        "%(default)s)", default=None)
    parser.add_argument("--port", help="Nameserver port (default: "
                        "%(default)s)", default=None, type=int)
    parser.add_argument("--no-broadcast", help="Disable broadcast (default"
                        ": %(default)s)", action='store_const', default=True,
                        const=False)
    parser.add_argument("--hmac", help="Nameserver hmac key (default: "
                        "%(default)s)", default=None)
    parser.add_argument('-v', '--verbose', help='Verbose flag',
                        action='store_const', dest="loglevel",
                        const=logging.INFO, default=logging.WARNING)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=args.loglevel)
    logger.info("running %s", " ".join(sys.argv))

    ns_conf = {
        "broadcast": args.no_broadcast,
        "host": args.host,
        "port": args.port,
        "hmac_key": args.hmac
    }
    utils.pyro_daemon(LDA_DISPATCHER_PREFIX, Dispatcher(
                      maxsize=args.maxsize, ns_conf=ns_conf),
                      ns_conf=ns_conf)
    logger.info("finished running %s", " ".join(sys.argv))


if __name__ == '__main__':
    main()
