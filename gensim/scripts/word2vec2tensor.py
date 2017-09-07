#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Loreto Parisi <loretoparisi@gmail.com>
# Copyright (C) 2016 Silvio Olivastri <silvio.olivastri@gmail.com>
# Copyright (C) 2016 Radim Rehurek <radim@rare-technologies.com>

"""
USAGE: $ python -m gensim.scripts.word2vec2tensor --input <Word2Vec model file> --output <TSV tensor filename prefix> [--binary] <Word2Vec binary flag>
Where:
    <Word2Vec model file>: Input Word2Vec model
    <TSV tensor filename prefix>: 2D tensor TSV output file name prefix
    <Word2Vec binary flag>: Set True if Word2Vec model is binary. Defaults to False.
Output:
    The script will create two TSV files. A 2d tensor format file, and a Word Embedding metadata file. Both files will
    us the --output file name as prefix
This script is used to convert the word2vec format to Tensorflow 2D tensor and metadata formats for Embedding Visualization
To use the generated TSV 2D tensor and metadata file in the Projector Visualizer, please
1) Open http://projector.tensorflow.org/.
2) Choose "Load Data" from the left menu.
3) Select "Choose file" in "Load a TSV file of vectors." and choose you local "_tensor.tsv" file
4) Select "Choose file" in "Load a TSV file of metadata." and choose you local "_metadata.tsv" file

For more information about TensorBoard TSV format please visit:
https://www.tensorflow.org/versions/master/how_tos/embedding_viz/

"""

import os
import sys
import logging
import argparse

import gensim

logger = logging.getLogger(__name__)


def word2vec2tensor(word2vec_model_path, tensor_filename, binary=False):
    """
    Convert Word2Vec mode to 2D tensor TSV file and metadata file
    Args:
        word2vec_model_path (str): word2vec model file path
        tensor_filename (str): filename prefix
        binary (bool): set True to use a binary Word2Vec model, defaults to False
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input word2vec model")
    parser.add_argument("-o", "--output", required=True, help="Output tensor file name prefix")
    parser.add_argument("-b", "--binary", required=False, help="If word2vec model in binary format, set True, else False")
    args = parser.parse_args()

    word2vec2tensor(args.input, args.output, args.binary)

    logger.info("finished running %s", os.path.basename(sys.argv[0]))
