#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""

import numpy as np
import six


class TestBaseTopicModel(object):
    def test_print_topic(self):
        topics = self.model.show_topics(formatted=True)
        for topic_no, topic in topics:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(topic, str) or isinstance(topic, unicode))  # noqa:F821

    def test_print_topics(self):
        topics = self.model.print_topics()

        for topic_no, topic in topics:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(topic, str) or isinstance(topic, unicode))  # noqa:F821

    def test_show_topic(self):
        topic = self.model.show_topic(1)

        for k, v in topic:
            self.assertTrue(isinstance(k, six.string_types))
            self.assertTrue(isinstance(v, (np.floating, float)))

    def test_show_topics(self):
        topics = self.model.show_topics(formatted=False)

        for topic_no, topic in topics:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(topic, list))
            for k, v in topic:
                self.assertTrue(isinstance(k, six.string_types))
                self.assertTrue(isinstance(v, (np.floating, float)))

    def test_get_topics(self):
        topics = self.model.get_topics()
        vocab_size = len(self.model.id2word)
        for topic in topics:
            self.assertTrue(isinstance(topic, np.ndarray))
            # Note: started moving to np.float32 as default
            # self.assertEqual(topic.dtype, np.float64)
            self.assertEqual(vocab_size, topic.shape[0])
            self.assertAlmostEqual(np.sum(topic), 1.0, 5)
