#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s DATA_DIRECTORY

    Start a sample similarity server, register it with Pyro and leave it running \
as a daemon. Assumes Pyro nameserver is already running.

Example:
    python -m Pyro4.naming -n 0.0.0.0 &              # run Pyro naming server
    python -m gensim.test.run_simserver /tmp/server  # create SessionServer and register it with Pyro
"""

from __future__ import with_statement

import logging
import os
import sys

import gensim


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(module)s:%(lineno)d : %(funcName)s(%(threadName)s) : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s" % ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) < 2:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    basename = sys.argv[1]
    server = gensim.similarities.SessionServer(basename)

    import Pyro4 # don't import too early because Pyro messes up logging
    with Pyro4.locateNS() as ns:
        with Pyro4.Daemon() as daemon:
            # register server for remote access
            uri = daemon.register(server)
            name = 'gensim.testserver'
            ns.remove(name)
            ns.register(name, uri)
            logging.info("server is ready at URI '%s', or under nameserver as %r" % (uri, name))
            daemon.requestLoop()

    logging.info("finished running %s" % program)
