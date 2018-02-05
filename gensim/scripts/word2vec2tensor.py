#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Loreto Parisi <loretoparisi@gmail.com>
# Copyright (C) 2016 Silvio Olivastri <silvio.olivastri@gmail.com>
# Copyright (C) 2016 Radim Rehurek <radim@rare-technologies.com>


"""This script helps to convert data in word2vec format into Tensorflow 2D
tensor and metadata formats for Embedding Visualization.

To use the generated TSV 2D tensor and metadata file in the Projector 
Visualizer, please follow next steps:

#. Open http://projector.tensorflow.org/
#. Choose "Load Data" from the left menu.
#. Select "Choose file" in "Load a TSV file of vectors." and choose you local "_tensor.tsv" file.
#. Select "Choose file" in "Load a TSV file of metadata." and choose you local "_metadata.tsv" file.

For more information about TensorBoard TSV format please visit:
https://www.tensorflow.org/versions/master/how_tos/embedding_viz/

.. program-output:: python -m gensim.scripts.word2vec2tensor --help

"""

import os
import sys
import logging
import argparse

import gensim

logger = logging.getLogger(__name__)


def word2vec2tensor(word2vec_model_path, tensor_filename, binary=False):
    """Converts Word2Vec model and writes two files 2D tensor TSV file (ends 
    with _tensor.tsv) and metadata file (ends with _metadata.tsv).

    Parameters
    ----------
    word2vec_model_path : str
        Path to input Word2Vec file.
    tensor_filename : str 
        Prefix for output files. 
    binary : bool, optional
        True if input file in binary format.

    """
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=binary)
    outfiletsv = tensor_filename + '_tensor.tsv'
    outfiletsvmeta = tensor_filename + '_metadata.tsv'

    with open(outfiletsv, 'w+') as file_vector:
        with open(outfiletsvmeta, 'w+') as file_metadata:
            for word in model.index2word:
                file_metadata.write(gensim.utils.to_utf8(word) + gensim.utils.to_utf8('\n'))
                vector_row = '\t'.join(str(x) for x in model[word])
                file_vector.write(vector_row + '\n')

    logger.info("2D tensor file saved to %s", outfiletsv)
    logger.info("Tensor metadata file saved to %s", outfiletsvmeta)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s", ' '.join(sys.argv))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
        This script helps to convert data in word2vec format to
        Tensorflow 2D tensor and metadata formats for Embedding
        Visualization. It produces 2 files:
            * <TENSOR_PREFIX>_tensor.tsv - 2D tensor file.
            * <TENSOR_PREFIX>_metadata.tsv - Word Embedding metadata file.""")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to input word2vec file")
    parser.add_argument("-o", "--output", required=True,
                        help="Prefix for output files")
    parser.add_argument(
        "-b", "--binary", required=False,
        help="Set True if word2vec model in binary format, optional."
    )
    args = parser.parse_args()

    # word2vec2tensor(args.input, args.output, args.binary)
    print(args.input, args.output, args.binary)

    logger.info("finished running %s", os.path.basename(sys.argv[0]))
