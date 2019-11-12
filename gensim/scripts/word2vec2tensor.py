#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Vimig Socrates <vimig.socrates@gmail.com>
# Copyright (C) 2016 Loreto Parisi <loretoparisi@gmail.com>
# Copyright (C) 2016 Silvio Olivastri <silvio.olivastri@gmail.com>
# Copyright (C) 2016 Radim Rehurek <radim@rare-technologies.com>


"""This script allows converting word-vectors from word2vec format into Tensorflow 2D tensor and metadata format.
This script used for word-vector visualization on `Embedding Visualization <http://projector.tensorflow.org/>`_.


How to use
----------
#. Convert your word-vector with this script (for example, we'll use model from
   `gensim-data <https://rare-technologies.com/new-download-api-for-pretrained-nlp-models-and-datasets-in-gensim/>`_) ::

    python -m gensim.downloader -d glove-wiki-gigaword-50  # download model in word2vec format
    python -m gensim.scripts.word2vec2tensor -i ~/gensim-data/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz \
                                             -o /tmp/my_model_prefix

#. Open http://projector.tensorflow.org/
#. Click "Load Data" button from the left menu.
#. Select "Choose file" in "Load a TSV file of vectors." and choose "/tmp/my_model_prefix_tensor.tsv" file.
#. Select "Choose file" in "Load a TSV file of metadata." and choose "/tmp/my_model_prefix_metadata.tsv" file.
#. ???
#. PROFIT!

For more information about TensorBoard TSV format please visit:
https://www.tensorflow.org/versions/master/how_tos/embedding_viz/


Command line arguments
----------------------

.. program-output:: python -m gensim.scripts.word2vec2tensor --help
   :ellipsis: 0, -7

"""

import os
import sys
import logging
import argparse

import gensim
from gensim import utils

logger = logging.getLogger(__name__)


def word2vec2tensor(word2vec_model_path, tensor_filename, binary=False):
    """Convert file in Word2Vec format and writes two files 2D tensor TSV file.

    File "tensor_filename"_tensor.tsv contains word-vectors, "tensor_filename"_metadata.tsv contains words.

    Parameters
    ----------
    word2vec_model_path : str
        Path to file in Word2Vec format.
    tensor_filename : str
        Prefix for output files.
    binary : bool, optional
        True if input file in binary format.

    """
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=binary)
    outfiletsv = tensor_filename + '_tensor.tsv'
    outfiletsvmeta = tensor_filename + '_metadata.tsv'

    with utils.open(outfiletsv, 'wb') as file_vector, utils.open(outfiletsvmeta, 'wb') as file_metadata:
        for word in model.index2word:
            file_metadata.write(gensim.utils.to_utf8(word) + gensim.utils.to_utf8('\n'))
            vector_row = '\t'.join(str(x) for x in model[word])
            file_vector.write(gensim.utils.to_utf8(vector_row) + gensim.utils.to_utf8('\n'))

    logger.info("2D tensor file saved to %s", outfiletsv)
    logger.info("Tensor metadata file saved to %s", outfiletsvmeta)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__[:-138])
    parser.add_argument("-i", "--input", required=True, help="Path to input file in word2vec format")
    parser.add_argument("-o", "--output", required=True, help="Prefix path for output files")
    parser.add_argument(
        "-b", "--binary", action='store_const', const=True, default=False,
        help="Set this flag if word2vec model in binary format (default: %(default)s)"
    )
    args = parser.parse_args()

    logger.info("running %s", ' '.join(sys.argv))
    word2vec2tensor(args.input, args.output, args.binary)
    logger.info("finished running %s", os.path.basename(sys.argv[0]))
