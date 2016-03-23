#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s

    Worker ("slave") process used in computing distributed LDA. Run this script \
on every node in your cluster. If you wish, you may even run it multiple times \
on a single machine, to make better use of multiple cores (just beware that \
memory footprint increases accordingly).

Example: python -m gensim.models.lda_worker
"""


from __future__ import with_statement
import os, sys, logging
import threading
import tempfile
try:
    import Queue
except ImportError:
    import queue as Queue
import Pyro4
from gensim.models import ldamodel
from gensim import utils

logger = logging.getLogger('gensim.models.lda_worker')


# periodically save intermediate models after every SAVE_DEBUG updates (0 for never)
SAVE_DEBUG = 0



class Worker(object):
    def __init__(self):
        self.model = None


    def initialize(self, myid, dispatcher, **model_params):
        self.lock_update = threading.Lock()
        self.jobsdone = 0 # how many jobs has this worker completed?
        self.myid = myid # id of this worker in the dispatcher; just a convenience var for easy access/logging TODO remove?
        self.dispatcher = dispatcher
        self.finished = False
        logger.info("initializing worker #%s" % myid)
        self.model = ldamodel.LdaModel(**model_params)


    @Pyro4.oneway
    def requestjob(self):
        """
        Request jobs from the dispatcher, in a perpetual loop until `getstate()` is called.
        """
        if self.model is None:
            raise RuntimeError("worker must be initialized before receiving jobs")

        job = None
        while job is None and not self.finished:
            try:
                job = self.dispatcher.getjob(self.myid)
            except Queue.Empty:
                # no new job: try again, unless we're finished with all work
                continue
        if job is not None:
            logger.info("worker #%s received job #%i" % (self.myid, self.jobsdone))
            self.processjob(job)
            self.dispatcher.jobdone(self.myid)
        else:
            logger.info("worker #%i stopping asking for jobs" % self.myid)


    @utils.synchronous('lock_update')
    def processjob(self, job):
        logger.debug("starting to process job #%i" % self.jobsdone)
        self.model.do_estep(job)
        self.jobsdone += 1
        if SAVE_DEBUG and self.jobsdone % SAVE_DEBUG == 0:
            fname = os.path.join(tempfile.gettempdir(), 'lda_worker.pkl')
            self.model.save(fname)
        logger.info("finished processing job #%i" % (self.jobsdone - 1))


    @utils.synchronous('lock_update')
    def getstate(self):
        logger.info("worker #%i returning its state after %s jobs" %
                    (self.myid, self.jobsdone))
        result = self.model.state
        assert isinstance(result, ldamodel.LdaState)
        self.model.clear() # free up mem in-between two EM cycles
        self.finished = True
        return result


    @utils.synchronous('lock_update')
    def reset(self, state):
        assert state is not None
        logger.info("resetting worker #%i" % self.myid)
        self.model.state = state
        self.model.sync_state()
        self.model.state.reset()
        self.finished = False


    @Pyro4.oneway
    def exit(self):
        logger.info("terminating worker #%i" % self.myid)
        os._exit(0)
#endclass Worker



def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))

    program = os.path.basename(sys.argv[0])
    # make sure we have enough cmd line parameters
    if len(sys.argv) < 1:
        print(globals()["__doc__"] % locals())
        sys.exit(1)

    utils.pyro_daemon('gensim.lda_worker', Worker(), random_suffix=True)

    logger.info("finished running %s" % program)



if __name__ == '__main__':
    main()
