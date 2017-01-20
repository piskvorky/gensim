#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Python wrapper
TODO: write desciption

"""


import logging
import random
import tempfile
import os

import numpy

import zipfile

from six import iteritems
from smart_open import smart_open

from gensim import utils, matutils
from gensim.utils import check_output
from gensim.models import basemodel
import os
logger = logging.getLogger(__name__)


class dl4jWrapper(utils.SaveLoad, basemodel.BaseTopicModel):
    """

    """
    def __init__(self, file_path, minWordFrequency, iterations, layerSize, seed, windowSize):
        """
        params

        """
        self.file_path = file_path
        self.minWordFrequency = minWordFrequency
        self.iterations = iterations
        self.layerSize = layerSize
        self.seed = seed
        self.windowSize = windowSize

    def train(self):
        cmd = "java -cp ./dl4j-examples/target/dl4j-examples-*-bin.jar org.deeplearning4j.examples.nlp.word2vec.Word2VecRawTextExample %s %s %s %s %s %s"
        cmd = cmd % (self.file_path, self.minWordFrequency, self.iterations, self.layerSize, self.seed, self.windowSize)
        os.system(cmd)
        # check_output(cmd, shell=True)

    def __getitem__(self, bow, iterations=100):
        pass
