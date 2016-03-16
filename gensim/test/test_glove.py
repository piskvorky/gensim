#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Manas Ranjan Kar <manasrkar91@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Test for gensim.scripts.glove2word2vec.py."""


import os
import gensim
from gensim.utils import check_output

module_path = os.path.dirname(gensim.__file__)
datapath = os.path.join(module_path, 'test_data', 'testglove.txt')  # Sample data files are located in the same folder
output_file = 'sample_word2vec_out.txt'

output = check_output(['python', 'glove2word2vec.py', '-i', datapath, '-o', output_file])
