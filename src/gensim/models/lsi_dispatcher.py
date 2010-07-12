#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
"""


from __future__ import with_statement
import os, sys, logging, threading
import itertools
from Queue import Queue
import time

import Pyro
from gensim.utils import synchronous


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger("lsi_dispatcher")
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
    worker_id = 0
    
    def __init__(self, maxsize = 100):
        self.jobs = Queue(maxsize = maxsize)
        self.lock_collect = threading.Lock()
    
    
    def initialize(self, callback, prefix = 'gensim.worker'):
        """
        Callback is a pyro proxy object for this same Dispatcher. It is used 
        for callbacks from workers, so that Workers and Dispatchers can 
        communicate both ways.
        """
        self.update_no = 0

        # locate all available workers and store their proxies, for subsequent RMI calls
        self.workers = {}
        with Pyro.naming.locateNS() as ns:
            for name, uri in ns.list(prefix = prefix).iteritems():
                if not self.register_worker(uri, callback = callback):
                    logger.warning("unresponsive worker at %s, deleting it from name server" % uri)
                    ns.remove(name)
        
        logger.info("found %i workers" % len(self.workers))
        if len(self.workers) == 0:
            raise RuntimeError('no workers found; run some worker scripts on your machines first!')
    
    
    def register_worker(self, uri, callback = None):
        try:
            worker = Pyro.core.Proxy(uri)
            worker.initialize(Dispatcher.worker_id, callback = callback)
            
            # make time consuming methods work asynchronously
            worker._pyroOneway.add("request_job")
            
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
        logging.debug("adding a new job (new queue size %i items)" % (1 + self.jobs.qsize()))
        self.jobs.put(job, block = True, timeout = HUGE_TIMEOUT)


    def broadcast_update(self, update):
        """
        Broadcast an update across all workers. This call is blocking.
        """
        updateId, result = update
        logger.debug("broadcasting update #%i to %i workers" % (updateId, len(self.workers)))
        for worker in self.workers.itervalues():
            worker.add_update(update)


    def save_state(self, fname):
        logger.info("saving intermediate state to %s" % fname)


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
        if self.save_every and self.update_no % self.save_every == 0:
            self.save_state("intermediate.%i" % self.update_no)
        update = (self.update_no, result)
        self.broadcast_update(update) # distribute the update to all workers
        self.update_no += 1
        worker.request_job() # tell the worker to ask for another job; asynchronous call (one-way)


    def process(self, corpus, chunks = 100, save_every = 100, save_to = "/tmp/"):
        """
        Process whole corpus, in chunks of `chunks` documents. 
        
        Save intermediate state to `save_to`/intermediate.xy after every `save_every`
        chunks.
        
        Note that after this function has returned, all workers will have finished their
        jobs, but not necessarily applied all updates (the queue of pending updates
        may be non-empty). Call worker.process_updates() to finalize the computation,
        if needed.
        """
        self.save_every = save_every
        chunker = itertools.groupby(enumerate(corpus), key = lambda val: val[0] / chunks)
        for chunkNo, (key, group) in enumerate(chunker):
            logger.info("creating job #%i" % chunkNo)
            job = [doc for docNo, doc in group]
            self.put_job(job) # put jobs into queue; this will eventually block, because the queue has a finite (small) size
        
        while self.update_no < chunkNo:
            print 'waiting...'
            time.sleep(1)
        print 'done!'
#endclass Dispatcher



def main(maxsize):
    ns = Pyro.naming.locateNS()
    with Pyro.core.Daemon() as daemon:
        dispatcher = Dispatcher(maxsize = maxsize)
        uri = daemon.register(dispatcher)
        
        # prepare callback object for the workers
        callback = Pyro.core.Proxy(uri)
        callback._pyroOneway.add("collect_result") # make sure we transfer control back to dispatcher asynchronously
        callback._pyroOneway.add("process") # not needed, but why not...
        dispatcher.initialize(callback)
        
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
