#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Worker ("slave") process used in computing distributed LDA. Run this script \
on every node in your cluster. If you wish, you may even run it multiple times \
on a single machine, to make better use of multiple cores (just beware that \
memory footprint increases accordingly).

Example: python -m gensim.models.lda_worker
"""


from __future__ import with_statement
import os
import sys
import logging
import threading
import tempfile
import argparse

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

LDA_WORKER_PREFIX = 'gensim.lda_worker'


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
        self.model = ldamodel.LdaModel(**model_params)

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
        logger.debug("starting to process job #%i", self.jobsdone)
        self.model.do_estep(job)
        self.jobsdone += 1
        if SAVE_DEBUG and self.jobsdone % SAVE_DEBUG == 0:
            fname = os.path.join(tempfile.gettempdir(), 'lda_worker.pkl')
            self.model.save(fname)
        logger.info("finished processing job #%i", self.jobsdone - 1)

    @Pyro4.expose
    def ping(self):
        return True

    @Pyro4.expose
    @utils.synchronous('lock_update')
    def getstate(self):
        logger.info("worker #%i returning its state after %s jobs", self.myid, self.jobsdone)
        result = self.model.state
        assert isinstance(result, ldamodel.LdaState)
        self.model.clear()  # free up mem in-between two EM cycles
        self.finished = True
        return result

    @Pyro4.expose
    @utils.synchronous('lock_update')
    def reset(self, state):
        assert state is not None
        logger.info("resetting worker #%i", self.myid)
        self.model.state = state
        self.model.sync_state()
        self.model.state.reset()
        self.finished = False

    @Pyro4.oneway
    def exit(self):
        logger.info("terminating worker #%i", self.myid)
        os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help="Nameserver hostname (default: %(default)s)", default=None)
    parser.add_argument("--port", help="Nameserver port (default: %(default)s)", default=None, type=int)
    parser.add_argument(
        "--no-broadcast", help="Disable broadcast (default: %(default)s)", action='store_const',
        default=True, const=False
    )
    parser.add_argument("--hmac", help="Nameserver hmac key (default: %(default)s)", default=None)
    parser.add_argument(
        '-v', '--verbose', help='Verbose flag', action='store_const', dest="loglevel",
        const=logging.INFO, default=logging.WARNING
    )
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=args.loglevel)
    logger.info("running %s", " ".join(sys.argv))

    ns_conf = {
        "broadcast": args.no_broadcast,
        "host": args.host,
        "port": args.port,
        "hmac_key": args.hmac
    }
    utils.pyro_daemon(LDA_WORKER_PREFIX, Worker(), random_suffix=True, ns_conf=ns_conf)
    logger.info("finished running %s", " ".join(sys.argv))


if __name__ == '__main__':
    main()
