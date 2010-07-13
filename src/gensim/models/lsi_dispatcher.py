#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s SIZE_OF_JOBS_QUEUE

    Dispatcher process which orchestrates distributed LSI computations. Run this \
script only once, on any node in your cluster.

Example: python lsi_dispatcher.py
"""


from __future__ import with_statement
import os, sys, logging, threading
from Queue import Queue

import Pyro

from gensim.utils import synchronous


logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger("dispatcher")
logger.setLevel(logging.DEBUG)


# How many jobs (=chunks of N documents) to keep "pre-fetched" in a queue?
# A small number is usually enough, unless iteration over the corpus is very very 
# slow (slower than the actual computation of LSI), in which case you can override 
# this value from command line. ie. run "python ./lsi_dispatcher.py 100" 
MAX_JOBS_QUEUE = 10

# timeout for the Queue object put/get blocking methods.
# it should really be infinity, but then keyboard interrupts don't work.
# so this is really just a hack, see http://bugs.python.org/issue1360
HUGE_TIMEOUT = 365 * 24 * 60 * 60 # one year



class Dispatcher(object):
    """
    Dispatcher object that communicates and coordinates individual workers.
    
    There should never be more than one dispatcher running at any one time.
    """
    
    def __init__(self, maxsize = 100):
        """
        Note that the constructor does not fully initialize the dispatcher;
        use the `initialize()` function to populate it with workers etc.
        """
        self.maxsize = maxsize
        self.callback = None # a pyro proxy to this object (unknown at init time, but will be set later)
    
    
    def initialize(self, **model_params):
        """
        `model_params` are parameters used to initialize individual workers (gets
        handed all the way down to worker.initialize()).
        """
        Dispatcher.worker_id = 0
        self.jobs = Queue(maxsize = maxsize)
        self.lock_collect = threading.Lock()
        self.update_no = 0
        self.callback._pyroOneway.add("collect_result") # make sure workers transfer control back to dispatcher asynchronously

        # locate all available workers and store their proxies, for subsequent RMI calls
        self.workers = {}
        with Pyro.naming.locateNS() as ns:
            for name, uri in ns.list(prefix = 'gensim.worker').iteritems():
                if not self.register_worker(uri, **model_params):
                    logger.warning("unresponsive worker at %s, deleting it from name server" % uri)
                    ns.remove(name)
        
        logger.info("found %i workers" % len(self.workers))
        if len(self.workers) == 0:
            raise RuntimeError('no workers found; run some worker scripts on your machines first!')
    
    
    def register_worker(self, uri, **model_params):
        """
        Register a single worker, located at `uri`.
        
        This includes resetting its internal state and initializing it.
        """
        try:
            worker = Pyro.core.Proxy(uri)
            worker.initialize(Dispatcher.worker_id, dispatcher = self.callback, **model_params)
            
            # make time consuming methods work asynchronously
            worker._pyroOneway.add("request_job")
            worker._pyroOneway.add("add_update")
            
            logger.debug("registering worker #%i at %s" % (Dispatcher.worker_id, uri))
            self.workers[Dispatcher.worker_id] = worker
            Dispatcher.worker_id += 1
            worker.request_job()
            return True
        except Pyro.errors.PyroError, err:
            logger.warning("worker %s failed with: %s" % (uri, err))


    def get_job(self, worker_id):
        logger.debug("worker #%i requesting a new job" % worker_id)
        return self.jobs.get(block = True, timeout = HUGE_TIMEOUT)


    def put_job(self, job):
        self.jobs.put(job, block = True, timeout = HUGE_TIMEOUT)
        logging.debug("added a new job (len(queue)=%i items)" % self.jobs.qsize())


    def broadcast_update(self, update):
        """
        Broadcast an update across all workers.
        """
        update_id, result = update
        logger.debug("broadcasting update #%i to %i workers" % (update_id, len(self.workers)))
        for worker in self.workers.itervalues():
            worker.add_update(update)

    
    def get_state(self):
        """
        Return the state of an arbitrary worker.
        
        The states of all workers should be equivalent, so just pick any).
        """
        worker_id, worker = self.workers.items()[0]
        logger.info("pulling state from worker %s" % worker_id)
        return worker.get_state()
        
        
    def updates_done(self):
        return self.update_no


    @synchronous('lock_collect')
    def collect_result(self, worker_id):
        """
        A worker has finished its job, so pull the results to the dispatcher and
        distribute them amongst all workers. Once this is done, tell the same 
        worker to request another job, to keep it busy.
        
        In this way, control flow basically oscillates between this function and
        worker.request_job() in an asynchronous, non-blocking way.
        """
        logger.info("collecting result from worker #%s" % worker_id)
        worker = self.workers[worker_id]
        result = worker.get_result() # retrieve the result
        update = (self.update_no, result)
        self.broadcast_update(update) # distribute the update to all workers
        self.update_no += 1
        worker.request_job() # tell the worker to ask for another job; asynchronous call (one-way)
#endclass Dispatcher



def main(maxsize):
    with Pyro.naming.locateNS() as ns:
        with Pyro.core.Daemon() as daemon:
            dispatcher = Dispatcher(maxsize = maxsize)
            uri = daemon.register(dispatcher)
            # prepare callback object for the workers
            dispatcher.callback = Pyro.core.Proxy(uri)
            
            name = 'gensim.dispatcher'
            ns.remove(name)
            ns.register(name, uri)
            logger.info("dispatcher is ready at URI %s" % uri)
            daemon.requestLoop()



if __name__ == '__main__':
    logger.info("running %s" % " ".join(sys.argv))

    program = os.path.basename(sys.argv[0])
    # make sure we have enough cmd line parameters
    if len(sys.argv) < 1:
        print globals()["__doc__"] % locals()
        sys.exit(1)
    
    if len(sys.argv) < 2:
        maxsize = MAX_JOBS_QUEUE
    else:
        maxsize = int(sys.argv[1])
    
    main(maxsize)
    
    logger.info("finished running %s" % program)
