#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Mohit Rathore <mrmohitrathoremr@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
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
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import os
logger = logging.getLogger(__name__)


class dl4jWrapper(KeyedVectors):
    """

    """

    @classmethod
    def train(cls, jar_file, corpus_file, minWordFrequency, iterations, layerSize, seed, windowSize, output_file=None):
        output_file = output_file or os.path.join(tempfile.gettempdir(), 'dl4j_model')
        wr_args = {
            'corpus_file': corpus_file,
            'minWordFrequency': minWordFrequency,
            'iterations': iterations,
            'layerSize': layerSize,
            'seed': seed,
            'windowSize': windowSize
        }

        cmd = ['java', '-cp', 'lib/*:' + jar_file + ':.', 'org.deeplearning4j.examples.nlp.word2vec.Word2VecRawTextExample']
        for option, value in wr_args.items():
            #TODO pass options along with value
            cmd.append(str(value))

        output = utils.check_output(args=cmd)
        model = cls.load_dl4j_w2v_format(output_file)
        return model

    @classmethod
    def load_dl4j_w2v_format(cls, model_file):
        glove2word2vec(model_file, model_file+'.w2vformat')
        model = cls.load_word2vec_format(model_file+'.w2vformat')
        return model
