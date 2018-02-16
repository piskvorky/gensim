#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Worker ("slave") process used in computing distributed LSI. Run this script
on every node in your cluster. If you wish, you may even run it multiple times
on a single machine, to make better use of multiple cores (just beware that
memory footprint increases accordingly).

How to use
----------

#. Launch a worker instance on a node of your cluster

    python -m gensim.models.lsi_worker


Command line arguments
----------------------
    .. program-output:: python -m gensim.models.lsi_worker --help

    :ellipsis: 0, -5

"""


from __future__ import with_statement
import os
import sys
import logging
import argparse
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
        """Partly initializes the model.

        A full initialization requires a call to `self.initialize` as well.

        """
        self.model = None

    @Pyro4.expose
    def initialize(self, myid, dispatcher, **model_params):
        """Fully initializes the worker.

        Parameters
        ----------
        myid : int
            An ID number used to identify this worker in the dispatcher object.
        dispatcher : :class:`~gensim.models.lsi_dispatcher.Dispatcher`
            The dispatcher responsible for scheduling this worker.
        **model_params
            Keyword parameters to initialize the inner LSI model.

        """
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
        """Request jobs from the dispatcher, in a perpetual loop until `self.getstate()` is called."""
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
        """Incrementally proccesses the job and potentially logs progress.

        Parameters
        ----------
        job : iterable of iterable of (int, float)
            The corpus to be used for further training the LSI model.

        """
        self.model.add_documents(job)
        self.jobsdone += 1
        if SAVE_DEBUG and self.jobsdone % SAVE_DEBUG == 0:
            fname = os.path.join(tempfile.gettempdir(), 'lsi_worker.pkl')
            self.model.save(fname)

    @Pyro4.expose
    @utils.synchronous('lock_update')
    def getstate(self):
        """Logs and returns the LSI model's current projection.

        Returns
        -------
        :class:`~gensim.models.lsimodel.Projection`
            The current projection.

        """
        logger.info("worker #%i returning its state after %s jobs", self.myid, self.jobsdone)
        assert isinstance(self.model.projection, lsimodel.Projection)
        self.finished = True
        return self.model.projection

    @Pyro4.expose
    @utils.synchronous('lock_update')
    def reset(self):
        """Resets the worker by deleting its current projection."""
        logger.info("resetting worker #%i", self.myid)
        self.model.projection = self.model.projection.empty_like()
        self.finished = False

    @Pyro4.oneway
    def exit(self):
        """Terminates the worker. """
        logger.info("terminating worker #%i", self.myid)
        os._exit(0)


def main():
    """The main script. """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__[:-135], formatter_class=argparse.RawTextHelpFormatter)
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))

    utils.pyro_daemon('gensim.lsi_worker', Worker(), random_suffix=True)
    logger.info("finished running %s", parser.prog)


if __name__ == '__main__':
    main()
