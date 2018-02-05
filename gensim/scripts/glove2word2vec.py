#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2016 Manas Ranjan Kar <manasrkar91@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""This script allows to convert GloVe vectors into the word2vec. Both files are
presented in text format and almost identical except that word2vec includes
number of vectors and its dimension which is only difference regard to GloVe.

This script uses `smart_open <https://github.com/RaRe-Technologies/smart_open>`_
library for reading and writing files.

.. program-output:: python -m gensim.scripts.glove2word2vec --help

"""

import sys
import logging
import argparse

from smart_open import smart_open

logger = logging.getLogger(__name__)


def get_glove_info(glove_file_name):
    """Returns number of vectors in provided `glove_file_name` and dimension of
    vectors. Please note it assumes given file is correct and then only
    dimension of the first vector taken.
    
    Parameters
    ----------
    glove_file_name : str
        Filename.
    
    Returns
    -------
    tuple of int
        Number of vectors (lines) of given file and its dimension.

    """
    with smart_open(glove_file_name) as f:
        num_lines = sum(1 for _ in f)
    with smart_open(glove_file_name) as f:
        num_dims = len(f.readline().split()) - 1
    return num_lines, num_dims


def glove2word2vec(glove_input_file, word2vec_output_file):
    """Converts `glove_input_file` in GloVe format to word2vec format with
    filename specified in `word2vec_output_file`. Returns Returns number of
    vectors `glove_input_file` and its dimension.
    
    Parameters
    ----------
    glove_file_name : str
        Input file in GloVe format.
    word2vec_output_file: str
        Output file in word2vec format.
    
    Returns
    -------
    tuple of int
        Number of vectors (lines) of input file and its dimension.

    """
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

    parser = argparse.ArgumentParser(
        description="This script converts GloVe file to word2vec format.")
    parser.add_argument("-i", "--input", required=True,
                        help="Input file, in GloVe format (read-only).")
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output file, in word2vec text format (will be overwritten)."
    )
    args = parser.parse_args()

    # do the actual conversion
    num_lines, num_dims = glove2word2vec(args.input, args.output)
    logger.info('Converted model with %i vectors and %i dimensions', num_lines, num_dims)
