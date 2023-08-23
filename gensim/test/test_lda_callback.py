#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Allenyl <allen7575@gmail.com>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

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

    def test_callback_update_graph(self):
        with subprocess.Popen(['python', '-m', 'visdom.server', '-port', str(self.port)]) as proc:
            # wait for visdom server startup (any better way?)
            viz = Visdom(server=self.host, port=self.port)
            for attempt in range(5):
                time.sleep(1.0)  # seconds
                if viz.check_connection():
                    break
            assert viz.check_connection()
            viz.close()
            self.model.update(self.corpus)
            proc.kill()


if __name__ == '__main__':
    unittest.main()
