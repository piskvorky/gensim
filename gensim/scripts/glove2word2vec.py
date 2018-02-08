#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2016 Manas Ranjan Kar <manasrkar91@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
This script allows to convert GloVe vectors into the word2vec. Both files are
presented in text format and almost identical except that word2vec includes
number of vectors and its dimension which is only difference regard to GloVe.

Notes
-----

GloVe format (real example can be founded `on Stanford size <https://nlp.stanford.edu/projects/glove/>`_) ::

    word1 0.123 0.134 0.532 0.152
    word2 0.934 0.412 0.532 0.159
    word3 0.334 0.241 0.324 0.188
    ...
    word9 0.334 0.241 0.324 0.188


Word2Vec format (real example can be founded `on w2v old repository <https://code.google.com/archive/p/word2vec/>`_) ::

    9 4
    word1 0.123 0.134 0.532 0.152
    word2 0.934 0.412 0.532 0.159
    word3 0.334 0.241 0.324 0.188
    ...
    word9 0.334 0.241 0.324 0.188


Command line arguments

.. program-output:: python -m gensim.scripts.glove2word2vec --help
   :ellipsis: 0, -6

"""
import sys
import logging
import argparse

from smart_open import smart_open

logger = logging.getLogger(__name__)


def get_glove_info(glove_file_name):
    """Get number of vectors in provided `glove_file_name` and dimension of vectors.

    Parameters
    ----------
    glove_file_name : str
        Path to file in GloVe format.

    Returns
    -------
    (int, int)
        Number of vectors (lines) of input file and its dimension.

    """
    with smart_open(glove_file_name) as f:
        num_lines = sum(1 for _ in f)
    with smart_open(glove_file_name) as f:
        num_dims = len(f.readline().split()) - 1
    return num_lines, num_dims


def glove2word2vec(glove_input_file, word2vec_output_file):
    """Convert `glove_input_file` in GloVe format to word2vec format and write it to `word2vec_output_file`.

    Parameters
    ----------
    glove_input_file : str
        Path to file in GloVe format.
    word2vec_output_file: str
        Path to output file.

    Returns
    -------
    (int, int)
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
    parser = argparse.ArgumentParser(description=__doc__[:-115], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="Path to input file in GloVe format")
    parser.add_argument("-o", "--output", required=True, help="Path to output file")
    args = parser.parse_args()

    logger.info("running %s", ' '.join(sys.argv))
    num_lines, num_dims = glove2word2vec(args.input, args.output)
    logger.info('Converted model with %i vectors and %i dimensions', num_lines, num_dims)
