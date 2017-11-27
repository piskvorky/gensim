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
import os
import sys
import logging
import threading
import tempfile
try:
    import Queue
except ImportError:
    import queue as Queue
import Pyro4
from gensim.models import lsimodel
from gensim import utils

logger = logging.getLogger('gensim.models.lsi_worker')


SAVE_DEBUG = 0  # save intermediate models after every SAVE_DEBUG updates (0 for never)


class Worker(object):
    def __init__(self):
        self.model = None

    @Pyro4.expose
    def initialize(self, myid, dispatcher, **model_params):
        self.lock_update = threading.Lock()
        self.jobsdone = 0  # how many jobs has this worker completed?
        # id of this worker in the dispatcher; just a convenience var for easy access/logging TODO remove?
        self.myid = myid
        self.dispatcher = dispatcher
        self.finished = False
        logger.info("initializing worker #%s", myid)
        self.model = lsimodel.LsiModel(**model_params)

    @Pyro4.expose
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
            logger.info("worker #%s received job #%i", self.myid, self.jobsdone)
            self.processjob(job)
            self.dispatcher.jobdone(self.myid)
        else:
            logger.info("worker #%i stopping asking for jobs", self.myid)

    @utils.synchronous('lock_update')
    def processjob(self, job):
        self.model.add_documents(job)
        self.jobsdone += 1
        if SAVE_DEBUG and self.jobsdone % SAVE_DEBUG == 0:
            fname = os.path.join(tempfile.gettempdir(), 'lsi_worker.pkl')
            self.model.save(fname)

    @Pyro4.expose
    @utils.synchronous('lock_update')
    def getstate(self):
        logger.info("worker #%i returning its state after %s jobs", self.myid, self.jobsdone)
        assert isinstance(self.model.projection, lsimodel.Projection)
        self.finished = True
        return self.model.projection

    @Pyro4.expose
    @utils.synchronous('lock_update')
    def reset(self):
        logger.info("resetting worker #%i", self.myid)
        self.model.projection = self.model.projection.empty_like()
        self.finished = False

    @Pyro4.oneway
    def exit(self):
        logger.info("terminating worker #%i", self.myid)
        os._exit(0)
# endclass Worker


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s", " ".join(sys.argv))

    program = os.path.basename(sys.argv[0])
    # make sure we have enough cmd line parameters
    if len(sys.argv) < 1:
        print(globals()["__doc__"] % locals())
        sys.exit(1)

    utils.pyro_daemon('gensim.lsi_worker', Worker(), random_suffix=True)

    logger.info("finished running %s", program)


if __name__ == '__main__':
    main()
