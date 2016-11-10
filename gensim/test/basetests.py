#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""

import six
import numpy as np

class TestBaseTopicModel(object):
    def testPrintTopic(self):
        topics = self.model.show_topics(formatted=True)
        for topic_no, topic in topics:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(topic, str) or isinstance(topic, unicode))

    def testPrintTopics(self):
        topics = self.model.print_topics()

        for topic_no, topic in topics:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(topic, str) or isinstance(topic, unicode))

    def testShowTopic(self):
        topic = self.model.show_topic(1)

        for k, v in topic:
            self.assertTrue(isinstance(k, six.string_types))
            self.assertTrue(isinstance(v, (np.floating, float)))

    def testShowTopics(self):
        topics = self.model.show_topics(formatted=False)

        for topic_no, topic in topics:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(topic, list))
            for k, v in topic:
                self.assertTrue(isinstance(k, six.string_types))
                self.assertTrue(isinstance(v, (np.floating, float)))
