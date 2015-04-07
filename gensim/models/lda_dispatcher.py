#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s SIZE_OF_JOBS_QUEUE

    Dispatcher process which orchestrates distributed LDA computations. Run this \
script only once, on any node in your cluster.

Example: python -m gensim.models.lda_dispatcher
"""


from __future__ import with_statement
import os, sys, logging, threading, time
try:
    from Queue import Queue
except ImportError:
    from queue import Queue
import Pyro4
from gensim import utils


logger = logging.getLogger("gensim.models.lda_dispatcher")


# How many jobs (=chunks of N documents) to keep "pre-fetched" in a queue?
# A small number is usually enough, unless iteration over the corpus is very very
# slow (slower than the actual computation of LDA), in which case you can override
# this value from command line. ie. run "python ./lda_dispatcher.py 100"
MAX_JOBS_QUEUE = 10

# timeout for the Queue object put/get blocking methods.
# it should theoretically be infinity, but then keyboard interrupts don't work.
# so this is really just a hack, see http://bugs.python.org/issue1360
HUGE_TIMEOUT = 365 * 24 * 60 * 60 # one year



class Dispatcher(object):
    """
    Dispatcher object that communicates and coordinates individual workers.

    There should never be more than one dispatcher running at any one time.
    """

    def __init__(self, maxsize=MAX_JOBS_QUEUE):
        """
        Note that the constructor does not fully initialize the dispatcher;
        use the `initialize()` function to populate it with workers etc.
        """
        self.maxsize = maxsize
        self.callback = None # a pyro proxy to this object (unknown at init time, but will be set later)


    def initialize(self, **model_params):
        """
        `model_params` are parameters used to initialize individual workers (gets
        handed all the way down to `worker.initialize()`).
        """
        self.jobs = Queue(maxsize=self.maxsize)
        self.lock_update = threading.Lock()
        self._jobsdone = 0
        self._jobsreceived = 0

        # locate all available workers and store their proxies, for subsequent RMI calls
        self.workers = {}
        import Pyro4
        with utils.getNS() as ns:
            self.callback = Pyro4.Proxy('PYRONAME:gensim.lda_dispatcher') # = self
            for name, uri in ns.list(prefix='gensim.lda_worker').iteritems():
                try:
                    worker = Pyro4.Proxy(uri)
                    workerid = len(self.workers)
                    # make time consuming methods work asynchronously
                    logger.info("registering worker #%i at %s" % (workerid, uri))
                    worker.initialize(workerid, dispatcher=self.callback, **model_params)
                    self.workers[workerid] = worker
                except Pyro4.errors.PyroError:
                    logger.warning("unresponsive worker at %s, deleting it from the name server" % uri)
                    ns.remove(name)

        if not self.workers:
            raise RuntimeError('no workers found; run some lda_worker scripts on your machines first!')


    def getworkers(self):
        """
        Return pyro URIs of all registered workers.
        """
        return [worker._pyroUri for worker in self.workers.itervalues()]


    def getjob(self, worker_id):
        logger.info("worker #%i requesting a new job" % worker_id)
        job = self.jobs.get(block=True, timeout=1)
        logger.info("worker #%i got a new job (%i left)" % (worker_id, self.jobs.qsize()))
        return job


    def putjob(self, job):
        self._jobsreceived += 1
        self.jobs.put(job, block=True, timeout=HUGE_TIMEOUT)
        logger.info("added a new job (len(queue)=%i items)" % self.jobs.qsize())


    def getstate(self):
        """
        Merge states from across all workers and return the result.
        """
        logger.info("end of input, assigning all remaining jobs")
        logger.debug("jobs done: %s, jobs received: %s" % (self._jobsdone, self._jobsreceived))
        while self._jobsdone < self._jobsreceived:
            time.sleep(0.5) # check every half a second

        logger.info("merging states from %i workers" % len(self.workers))
        workers = self.workers.values()
        result = workers[0].getstate()
        for worker in workers[1:]:
            result.merge(worker.getstate())

        logger.info("sending out merged state")
        return result


    def reset(self, state):
        """
        Initialize all workers for a new EM iterations.
        """
        for workerid, worker in self.workers.iteritems():
            logger.info("resetting worker %s" % workerid)
            worker.reset(state)
            worker.requestjob()
        self._jobsdone = 0
        self._jobsreceived = 0


    @Pyro4.oneway
    @utils.synchronous('lock_update')
    def jobdone(self, workerid):
        """
        A worker has finished its job. Log this event and then asynchronously
        transfer control back to the worker.

        In this way, control flow basically oscillates between `dispatcher.jobdone()`
        and `worker.requestjob()`.
        """
        self._jobsdone += 1
        logger.info("worker #%s finished job #%i" % (workerid, self._jobsdone))
        self.workers[workerid].requestjob() # tell the worker to ask for another job, asynchronously (one-way)


    def jobsdone(self):
        """Wrap self._jobsdone, needed for remote access through Pyro proxies"""
        return self._jobsdone


    @Pyro4.oneway
    def exit(self):
        """
        Terminate all registered workers and then the dispatcher.
        """
        for workerid, worker in self.workers.iteritems():
            logger.info("terminating worker %s" % workerid)
            worker.exit()
        logger.info("terminating dispatcher")
        os._exit(0) # exit the whole process (not just this thread ala sys.exit())
#endclass Dispatcher



def main():
    logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))

    program = os.path.basename(sys.argv[0])
    # make sure we have enough cmd line parameters
    if len(sys.argv) < 1:
        print(globals()["__doc__"] % locals())
        sys.exit(1)

    if len(sys.argv) < 2:
        maxsize = MAX_JOBS_QUEUE
    else:
        maxsize = int(sys.argv[1])
    utils.pyro_daemon('gensim.lda_dispatcher', Dispatcher(maxsize=maxsize))

    logger.info("finished running %s" % program)



if __name__ == '__main__':
    main()
