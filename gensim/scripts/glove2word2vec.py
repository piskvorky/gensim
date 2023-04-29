#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2016 Manas Ranjan Kar <manasrkar91@gmail.com>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html


"""This script allows to convert GloVe vectors into the word2vec. Both files are
presented in text format and almost identical except that word2vec includes
number of vectors and its dimension which is only difference regard to GloVe.

Notes
-----

GloVe format (a real example can be found on the
`Stanford site <https://nlp.stanford.edu/projects/glove/>`_) ::

    word1 0.123 0.134 0.532 0.152
    word2 0.934 0.412 0.532 0.159
    word3 0.334 0.241 0.324 0.188
    ...
    word9 0.334 0.241 0.324 0.188


Word2Vec format (a real example can be found in the
`old w2v repository <https://code.google.com/archive/p/word2vec/>`_) ::

    9 4
    word1 0.123 0.134 0.532 0.152
    word2 0.934 0.412 0.532 0.159
    word3 0.334 0.241 0.324 0.188
    ...
    word9 0.334 0.241 0.324 0.188


How to use
----------

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath, get_tmpfile
    >>> from gensim.models import KeyedVectors
    >>> from gensim.scripts.glove2word2vec import glove2word2vec
    >>>
    >>> glove_file = datapath('test_glove.txt')
    >>> tmp_file = get_tmpfile("test_word2vec.txt")
    >>>
    >>> _ = glove2word2vec(glove_file, tmp_file)
    >>>
    >>> model = KeyedVectors.load_word2vec_format(tmp_file)

Command line arguments
----------------------

.. program-output:: python -m gensim.scripts.glove2word2vec --help
   :ellipsis: 0, -5

"""
import sys
import logging
import argparse

from gensim import utils
from gensim.utils import deprecated
from gensim.models.keyedvectors import KeyedVectors

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
    with utils.open(glove_file_name, 'rb') as f:
        num_lines = sum(1 for _ in f)
    with utils.open(glove_file_name, 'rb') as f:
        num_dims = len(f.readline().split()) - 1
    return num_lines, num_dims


@deprecated("KeyedVectors.load_word2vec_format(.., binary=False, no_header=True) loads GLoVE text vectors.")
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
    glovekv = KeyedVectors.load_word2vec_format(glove_input_file, binary=False, no_header=True)

    num_lines, num_dims = len(glovekv), glovekv.vector_size
    logger.info("converting %i vectors from %s to %s", num_lines, glove_input_file, word2vec_output_file)
    glovekv.save_word2vec_format(word2vec_output_file, binary=False)
    return num_lines, num_dims


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__[:-135], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="Path to input file in GloVe format")
    parser.add_argument("-o", "--output", required=True, help="Path to output file")
    args = parser.parse_args()

    logger.info("running %s", ' '.join(sys.argv))
    num_lines, num_dims = glove2word2vec(args.input, args.output)
    logger.info('Converted model with %i vectors and %i dimensions', num_lines, num_dims)
