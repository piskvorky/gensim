#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Allenyl <allen7575@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking visdom API
"""
import unittest
import subprocess
import time

from gensim.models import LdaModel
from gensim.test.utils import datapath, common_dictionary
from gensim.corpora import MmCorpus
from gensim.models.callbacks import CoherenceMetric

try:
    from visdom import Visdom
    VISDOM_INSTALLED = True
except ImportError:
    VISDOM_INSTALLED = False


@unittest.skipIf(VISDOM_INSTALLED is False, "Visdom not installed")
class TestLdaCallback(unittest.TestCase):

    def setUp(self):
        self.corpus = MmCorpus(datapath('testcorpus.mm'))
        self.ch_umass = CoherenceMetric(corpus=self.corpus, coherence="u_mass", logger="visdom", title="Coherence")
        self.callback = [self.ch_umass]
        self.model = LdaModel(id2word=common_dictionary, num_topics=2, passes=10, callbacks=self.callback)

        self.host = "http://localhost"
        self.port = 8097

    def testCallbackUpdateGraph(self):

        # Popen have no context-manager in 2.7, for this reason - try/finally.
        try:
            # spawn visdom.server
            proc = subprocess.Popen(['python', '-m', 'visdom.server', '-port', str(self.port)])

            # wait for visdom server startup (any better way?)
            time.sleep(3)

            viz = Visdom(server=self.host, port=self.port)
            assert viz.check_connection()

            # clear screen
            viz.close()

            self.model.update(self.corpus)
        finally:
            proc.kill()


if __name__ == '__main__':
    unittest.main()
