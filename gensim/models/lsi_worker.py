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

Example: python -m gensim.models.lsi_worker
"""


from __future__ import with_statement
import os, sys, logging
import threading
import tempfile

import Pyro4

from gensim.models import lsimodel
from gensim import utils

logger = logging.getLogger('gensim.models.lsi_worker')


SAVE_DEBUG = 0 # save intermediate models after every SAVE_DEBUG updates (0 for never)



class Worker(object):
    def __init__(self):
        self.model = None


    def initialize(self, myid, dispatcher, **model_params):
        self.lock_update = threading.Lock()
        self.jobsdone = 0 # how many jobs has this worker completed?
        self.myid = myid # id of this worker in the dispatcher; just a convenience var for easy access/logging TODO remove?
        self.dispatcher = dispatcher
        logger.info("initializing worker #%s" % myid)
        self.model = lsimodel.LsiModel(**model_params)


    def requestjob(self):
        """
        Request jobs from the dispatcher in an infinite loop. The requests are
        blocking, so if there are no jobs available, the thread will wait.
        """
        if self.model is None:
            raise RuntimeError("worker must be initialized before receiving jobs")
        job = self.dispatcher.getjob(self.myid) # blocks until a new job is available from the dispatcher
        logger.info("worker #%s received job #%i" % (self.myid, self.jobsdone))
        self.processjob(job)
        self.dispatcher.jobdone(self.myid)


    @utils.synchronous('lock_update')
    def processjob(self, job):
        self.model.add_documents(job)
        self.jobsdone += 1
        if SAVE_DEBUG and self.jobsdone % SAVE_DEBUG == 0:
            fname = os.path.join(tempfile.gettempdir(), 'lsi_worker.pkl')
            self.model.save(fname)


    @utils.synchronous('lock_update')
    def getstate(self):
        logger.info("worker #%i returning its state after %s jobs" %
                    (self.myid, self.jobsdone))
        assert isinstance(self.model.projection, lsimodel.Projection)
        result = self.model.projection
        self.model.projection = self.model.projection.empty_like()
        return result


    def exit(self):
        logger.info("terminating worker #%i" % self.myid)
        os._exit(0)
#endclass Worker



def main():
    logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s')
    logger.info("running %s" % " ".join(sys.argv))

    program = os.path.basename(sys.argv[0])
    # make sure we have enough cmd line parameters
    if len(sys.argv) < 1:
        print globals()["__doc__"] % locals()
        sys.exit(1)

    with Pyro4.naming.locateNS() as ns:
        with Pyro4.core.Daemon() as daemon:
            worker = Worker()
            uri = daemon.register(worker)
            name = 'gensim.lsi_worker.' + str(uri)
            ns.remove(name)
            ns.register(name, uri)
            logger.info("worker is ready at URI %s" % uri)
            daemon.requestLoop()

    logger.info("finished running %s" % program)



if __name__ == '__main__':
    main()
