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

logger = logging.getLogger(__name__)


class dl4jWrapper(utils.SaveLoad, basemodel.BaseTopicModel):
    """

    """
    def __init__(self):
        """
 	params

        """
 

    def train(self, corpus):


    def __getitem__(self, bow, iterations=100):




