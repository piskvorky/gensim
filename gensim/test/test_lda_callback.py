#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Allenyl <allen7575@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking visdom API
"""
from __future__ import print_function

import unittest

from gensim.models import ldamodel
from gensim.test.utils import datapath, common_texts
from gensim.corpora import mmcorpus, Dictionary
from gensim.models.callbacks import CoherenceMetric

from visdom import Visdom
# import numpy as np
import subprocess
import time

DEFAULT_PORT = 8097
DEFAULT_HOSTNAME = "http://localhost"

##########
# use argparse to parse commandline arguments
##########
# import argparse

# parser = argparse.ArgumentParser(description='Demo arguments')
# parser.add_argument('-port', metavar='port', type=int, default=DEFAULT_PORT,
#                     help='port the visdom server is running on.')
# parser.add_argument('-server', metavar='server', type=str,
#                     default=DEFAULT_HOSTNAME,
#                     help='Server address of the target to run the demo on.')
# FLAGS = parser.parse_args()

##########
# use easydict if argparse doesn't work
##########
# import easydict

# FLAGS = easydict.EasyDict({
#         'port': DEFAULT_PORT,
#         'server': DEFAULT_HOSTNAME
# })


class ARGS():
    def __init__(self, port=DEFAULT_PORT, server=DEFAULT_HOSTNAME):
        self.port = port
        self.server = server


FLAGS = ARGS()


dictionary = Dictionary(common_texts)


class TestLdaCallback(unittest.TestCase):

    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.ch_umass = CoherenceMetric(corpus=self.corpus, 
                                        coherence="u_mass", 
                                        logger="visdom", 
                                        title="Coherence (u_mass)")
        self.callback = [self.ch_umass]
        self.class_ = ldamodel.LdaModel
        self.model = self.class_(id2word=dictionary, num_topics=2, passes=10, callbacks=self.callback)

    def testCallbackUpdateGraph(self):

        # create a new process for visdom.server
        proc = subprocess.Popen(['python', '-m', 'visdom.server'])

        # wait for visdom server startup
        time.sleep(3)

        # create visdom object
        print(FLAGS.port)
        print(FLAGS.server)
        viz = Visdom(port=FLAGS.port, server=FLAGS.server)

        # check connection
        assert viz.check_connection()

        # clear screen
        viz.close()

        # ## create a window
        # win=viz.line(
        #     X=np.array([0,1]),
        #     Y=np.array([0,1]),
        #     opts=dict(
        #         xtickmin=-2,
        #         xtickmax=10,
        #         xtickstep=1,
        #         ytickmin=-1,
        #         ytickmax=10,
        #         ytickstep=1,
        #         markersymbol='dot',
        #         markersize=5,
        #     ),
        #     name="1"
        # )

        # # loop for update line
        # for a in range(10):
        #     try:
        #         # #old API
        #         # viz.updateTrace(
        #         #     X=np.array([a]),
        #         #     Y=np.array([a]),
        #         #     win=win,
        #         #     name="1"
        #         # )

        #         # new API
        #         viz.line(
        #             X=np.array([a]),
        #             Y=np.array([a]),
        #             win=win,
        #             name="1",
        #             update = 'append'
        #         )

        #         time.sleep(1)

        #     except Exception as e:
        #         print(e)
        #         proc.kill()
        #         print('server killed')
        #         self.assertTrue(0)

        # test callback's update graph function
        try:
            self.model.update(self.corpus)
            # raise AttributeError("test")
        except Exception as e:
            print(e)
            # kill visdom.server
            proc.kill()
            print('server killed')
            self.assertTrue(0)

        # # kill visdom.server
        # try:
        #     proc.wait(timeout=3)
        # except subprocess.TimeoutExpired:
        #     proc.kill()
        #     print('server killed')

        # kill visdom.server
        time.sleep(3)
        proc.kill()
        print('server killed')


if __name__ == '__main__':
    unittest.main()