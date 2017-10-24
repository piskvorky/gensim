#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2016 Manas Ranjan Kar <manasrkar91@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE:
    $ python -m gensim.scripts.glove2word2vec --input <GloVe vector file> --output <Word2vec vector file>

Where:

* <GloVe vector file>: Input GloVe .txt file.
* <Word2vec vector file>: Desired name of output Word2vec .txt file.

This script is used to convert GloVe vectors in text format into the word2vec text format.
The only difference between the two formats is an extra header line in word2vec,
which contains the number of vectors and their dimensionality (two integers).
"""

import sys
import logging
import argparse

from smart_open import smart_open

logger = logging.getLogger(__name__)


def get_glove_info(glove_file_name):
    """Return the number of vectors and dimensions in a file in GloVe format."""
    with smart_open(glove_file_name) as f:
        num_lines = sum(1 for _ in f)
    with smart_open(glove_file_name) as f:
        num_dims = len(f.readline().split()) - 1
    return num_lines, num_dims


def glove2word2vec(glove_input_file, word2vec_output_file):
    """Convert `glove_input_file` in GloVe format into `word2vec_output_file` in word2vec format."""
    num_lines, num_dims = get_glove_info(glove_input_file)
    logger.info("converting %i vectors from %s to %s", num_lines, glove_input_file, word2vec_output_file)
    with smart_open(word2vec_output_file, 'wb') as fout:
        fout.write("{0} {1}\n".format(num_lines, num_dims).encode('utf-8'))
        with smart_open(glove_input_file, 'rb') as fin:
            for line in fin:
                fout.write(line)
    return num_lines, num_dims


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s", ' '.join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input file, in gloVe format (read-only).")
    parser.add_argument("-o", "--output", required=True, help="Output file, in word2vec text format (will be overwritten).")
    args = parser.parse_args()

    # do the actual conversion
    num_lines, num_dims = glove2word2vec(args.input, args.output)
    logger.info('Converted model with %i vectors and %i dimensions', num_lines, num_dims)
