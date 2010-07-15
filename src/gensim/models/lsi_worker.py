#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s

    Worker ("slave") process used in computing distributed LSI. Run this script \
on every node in your cluster. If you wish, you may even run it multiple times \
on a single machine, to make better use of multiple cores (just beware that \
memory footprint increases accordingly).

Example: python lsi_worker.py
"""


from __future__ import with_statement
import os, sys, logging
import threading

import Pyro
import Pyro.config

from gensim.models import lsimodel
from gensim import utils

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger('lsi_worker')
logger.setLevel(logging.DEBUG)



class Worker(object):
    def __init__(self):
        self.model = None

    
    def initialize(self, my_id, dispatcher, **model_params):
        self.lock_update = threading.Lock()
        self.state = 0 # state of this worker (=number of updates applied so far)
        self.jobs_done = 0 # how many jobs has this worker completed itself
        self.my_id = my_id # id of this worker in the dispatcher; just a convenience var for easy access TODO remove?
        self.dispatcher = dispatcher
        logger.info("initializing worker #%s" % my_id)
        self.model = lsimodel.LsiModel(**model_params)
        self.model.projection_on(False)
        self.update = self.model.empty_state() # initialize pending updates to empty
    
    
    def alive(self):
        return True
    
    
    def getid(self):
        return self.my_id
    
    
    def add_workers(self, uris):
        logger.debug("worker #%s taking note of %i fellow workers" % 
                     (self.my_id, len(uris) - 1))
        self.workers = {}
        for uri in uris:
            worker = Pyro.core.Proxy(uri)
            worker._pyroOneway.add("request_job")
            worker._pyroOneway.add("add_update")
            self.workers[worker.getid()] = worker
        

    def broadcast_update(self, update_id):
        """
        Broadcast an update across all workers.
        """
        logger.info("worker #%s broadcasting update #%i from state #%s" % 
                    (self.my_id, update_id, self.state))
        update = update_id, self.result
        for worker_id, worker in self.workers.iteritems():
            if worker_id != self.my_id: # don't send to self, less elegant but saves some bandwidth :-)
                worker.add_update(update)
        self.add_update(update) # update self directly instead
        self.result = None


    @utils.synchronous('lock_update')
    def add_update(self, update):
        update_id, lsi_update = update
        if update_id != self.state + self.update.pending:
            raise ValueError("received state update #%i, expected #%i" %
                             (update_id, self.state))
        logger.debug("enqueuing update #%i" % update_id)
        self.update.add_update(lsi_update)


    @utils.synchronous('lock_update')
    def apply_update(self):
        """
        Make sure all pending updates (from other workers) have been applied.
        
        This ensures that the internal state stays reasonably up-to-date and 
        synchronized across all workers (jobs are out of sync only by the amount
        of data currently in the pipeline, i.e. in other workers).
        
        Return True if state was updated, False otherwise.
        """
        if self.update.pending > 0:
            logger.debug("worker #%s is applying %i pending updates since state %i" % 
                         (self.my_id, self.update.pending, self.state))
            self.state += self.update.pending
            self.model.update_state(self.update)
            self.update = None # give Python some (slim) chance to reclaim the memory before reallocating it, to avoid having two copies in memory at the same time
            self.update = self.model.empty_state() # reset updates to empty
            logger.debug("worker #%s now at state %s" % (self.my_id, self.state))
    
    
    def request_job(self):
        """
        Request a new job from the dispatcher and process it asynchronously.
        
        Once the job is finished, the dispatcher is notified that it should collect
        the results, broadcast them to other workers etc., and eventually tell
        this worker to request another job by calling this function again.
        """
        if self.model is None:
            raise RuntimeError("worker must be initialized before receiving jobs")
        job = self.dispatcher.get_job(self.my_id) # blocks until a new job is available from the dispatcher
        self.apply_update() # try to stay as up-to-date as possible
        logger.debug("worker #%s received a new job" % self.my_id)
        self.result = self.model.compute_update(job)
        self.jobs_done += 1
        self.dispatcher.collect_result(self.my_id) # must be asynchronous (one-way)

    
    def get_state(self):
        self.apply_update() # make sure all pending updates have been applied
        logger.info("worker #%i returning its state after %s updates" % 
                    (self.my_id, self.state))
        self.model.projection_on(True)
        return self.model.projection
#endclass Worker


    

def main():
    Pyro.config.HOST = utils.get_ip()
    
    with Pyro.naming.locateNS() as ns:
        with Pyro.core.Daemon() as daemon:
            worker = Worker()
            uri = daemon.register(worker)
            name = 'gensim.worker.' + str(uri)
            ns.remove(name)
            ns.register(name, uri)
            logger.info("worker is ready at URI %s" % uri)
            daemon.requestLoop()



if __name__ == '__main__':
    logger.info("running %s" % " ".join(sys.argv))

    program = os.path.basename(sys.argv[0])
    # make sure we have enough cmd line parameters
    if len(sys.argv) < 1:
        print globals()["__doc__"] % locals()
        sys.exit(1)
    
    main()
    
    logger.info("finished running %s" % program)
