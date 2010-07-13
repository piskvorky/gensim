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
from Queue import Queue
import threading

import Pyro

from gensim.utils import synchronous
from gensim.models import LsiModel

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger('lsi_worker')
logger.setLevel(logging.DEBUG)



class Worker(object):
    def __init__(self):
        self.model = None

    
    def initialize(self, my_id, dispatcher, **model_params):
        self.lock_update = threading.Lock()
        self.updates = Queue() # infinite size queue for keeping updates
        self.state = 0 # state of this worker (=number of updates applied so far)
        self.jobs_done = 0 # how many jobs has this worker completed itself
        self.my_id = my_id # id of this worker in the dispatcher; just a convenience var for easy access TODO remove?
        self.dispatcher = dispatcher
        logger.info("initializing worker #%s" % my_id)
        self.model = LsiModel(**model_params)
    

    def get_result(self):
        logger.info("worker #%s finished job from state %s" % (self.my_id, self.state))
        result, self.result = self.result, None
        return result


    @synchronous('lock_update')
    def add_update(self, update):
        logger.debug("enqueueing update #%i, len(queue)=%i items" % 
                     (update[0], 1 + self.updates.qsize()))
        self.updates.put(update)


    def apply_update(self, update):
        logger.debug("worker #%s applying update %i" % (self.my_id, self.state))
        self.model.apply_update(update)
        self.state += 1
    
    
    @synchronous('lock_update')
    def process_updates(self):
        """
        Make sure all pending updates (from other workers) have been applied.
        
        This ensures that the internal state stays reasonably up-to-date and 
        synchronized across all workers (jobs are out of sync only by the amount
        of data currently in the pipeline, i.e. in other workers).
        
        Return True if state was updated, False otherwise.
        """
        old_state = self.state
        while not self.updates.empty():
            update_id, update = self.updates.get(block = False)
            if update_id != self.state:
                raise ValueError("received state update #%i, expected #%i" %
                                 (update_id, self.state))
            self.apply_update(update)
        result = old_state != self.state
        if result:
            logger.debug("worker #%s now at state %s" % (self.my_id, self.state))
        return result


    def request_job(self):
        """
        Request a new job from the dispatcher and process it asynchronously.
        
        Once the job is finished, the dispatcher is notified that it should collect
        the results, broadcast them to other workers etc., and eventually tell
        this worker to request another job by calling this function again.
        """
        if self.model is None:
            raise RuntimeError("worker must be initialized before receiving jobs")
        self.process_updates()
        job = self.dispatcher.get_job(self.my_id) # blocks until a new job is available from the dispatcher
        self.process_updates() # call process_updates again, to stay as up-to-date as possible, in case the get_job() call blocked for a long period of time
        logger.debug("worker #%s received a new job" % self.my_id)
        self.result = self.model.compute_update(job)
        self.jobs_done += 1
        self.dispatcher.collect_result(self.my_id) # must be asynchronous (one-way)

    
    def get_state(self):
        self.process_updates() # make sure all pending updates have been applied
        logger.info("worker #%i returning its state after %s updates" % 
                    (self.my_id, self.state))
        self.model.projection_on(True)
        return self.model.projection
#endclass Worker


def main():
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
