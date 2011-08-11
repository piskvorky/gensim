#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s DATA_DIRECTORY

    Start a sample similarity server, register it with Pyro and leave it running \
as a daemon. Assumes Pyro nameserve is already running.

Example: ./run_simserver.py /tmp/simserver

"""

from __future__ import with_statement

import logging
import os
import sys
import tempfile
import random

import gensim


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(funcName)s(%(threadName)s) : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s" % ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) < 2:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    basename = sys.argv[1]

    server = gensim.similarities.SimServer(basename)

    import Pyro4 # register server for remote access
    with Pyro4.locateNS() as ns:
        with Pyro4.Daemon() as daemon:
            uri = daemon.register(server)
            name = 'gensim.testserver'
            ns.remove(name)
            ns.register(name, uri)
            logging.info("server is ready at URI '%s', or under nameserver as %r" % (uri, name))
            daemon.requestLoop()

    logging.info("finished running %s" % program)
