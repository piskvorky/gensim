#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
USAGE: %(program)s -train CORPUS -output VECTORS -size SIZE -window WINDOW
-cbow CBOW -sample SAMPLE -hs HS -negative NEGATIVE -threads THREADS -iter ITER
-min_count MIN-COUNT -alpha ALPHA -binary BINARY -accuracy FILE

Trains a neural embedding model on text file CORPUS.
Parameters essentially reproduce those used by the original C tool
(see https://code.google.com/archive/p/word2vec/).

Parameters for training:
        -train <file>
                Use text data from <file> to train the model
        -output <file>
                Use <file> to save the resulting word vectors / word clusters
        -size <int>
                Set size of word vectors; default is 100
        -window <int>
                Set max skip length between words; default is 5
        -sample <float>
                Set threshold for occurrence of words. Those that appear with higher frequency in the training data
                will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)
        -hs <int>
                Use Hierarchical Softmax; default is 0 (not used)
        -negative <int>
                Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
        -threads <int>
                Use <int> threads (default 3)
        -iter <int>
                Run more training iterations (default 5)
        -min_count <int>
                This will discard words that appear less than <int> times; default is 5
        -alpha <float>
                Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
        -binary <int>
                Save the resulting vectors in binary moded; default is 0 (off)
        -cbow <int>
                Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)
        -accuracy <file>
                Compute accuracy of the resulting model analogical inference power on questions file <file>
                See an example of questions file
                at https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt

Example: python -m gensim.scripts.word2vec_standalone -train data.txt \
         -output vec.txt -size 200 -sample 1e-4 -binary 0 -iter 3
"""


import logging
import os.path
import sys
import argparse
from numpy import seterr

from gensim.models.word2vec import Word2Vec, LineSentence  # avoid referencing __main__ in pickle

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s", " ".join(sys.argv))
    seterr(all='raise')  # don't ignore numpy errors

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", help="Use text data from file TRAIN to train the model", required=True)
    parser.add_argument("-output", help="Use file OUTPUT to save the resulting word vectors")
    parser.add_argument("-window", help="Set max skip length WINDOW between words; default is 5", type=int, default=5)
    parser.add_argument("-size", help="Set size of word vectors; default is 100", type=int, default=100)
    parser.add_argument(
        "-sample",
        help="Set threshold for occurrence of words. "
             "Those that appear with higher frequency in the training data will be randomly down-sampled; "
             "default is 1e-3, useful range is (0, 1e-5)",
        type=float, default=1e-3)
    parser.add_argument(
        "-hs", help="Use Hierarchical Softmax; default is 0 (not used)",
        type=int, default=0, choices=[0, 1]
    )
    parser.add_argument(
        "-negative", help="Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)",
        type=int, default=5
    )
    parser.add_argument("-threads", help="Use THREADS threads (default 3)", type=int, default=3)
    parser.add_argument("-iter", help="Run more training iterations (default 5)", type=int, default=5)
    parser.add_argument(
        "-min_count", help="This will discard words that appear less than MIN_COUNT times; default is 5",
        type=int, default=5
    )
    parser.add_argument(
        "-alpha", help="Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW",
        type=float
    )
    parser.add_argument(
        "-cbow", help="Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)",
        type=int, default=1, choices=[0, 1]
    )
    parser.add_argument(
        "-binary", help="Save the resulting vectors in binary mode; default is 0 (off)",
        type=int, default=0, choices=[0, 1]
    )
    parser.add_argument("-accuracy", help="Use questions from file ACCURACY to evaluate the model")

    args = parser.parse_args()

    if args.cbow == 0:
        skipgram = 1
        if not args.alpha:
            args.alpha = 0.025
    else:
        skipgram = 0
        if not args.alpha:
            args.alpha = 0.05

    corpus = LineSentence(args.train)

    model = Word2Vec(
        corpus, vector_size=args.size, min_count=args.min_count, workers=args.threads,
        window=args.window, sample=args.sample, alpha=args.alpha, sg=skipgram,
        hs=args.hs, negative=args.negative, cbow_mean=1, epochs=args.iter,
    )

    if args.output:
        outfile = args.output
        model.wv.save_word2vec_format(outfile, binary=args.binary)
    else:
        outfile = args.train.split('.')[0]
        model.save(outfile + '.model')
        if args.binary == 1:
            model.wv.save_word2vec_format(outfile + '.model.bin', binary=True)
        else:
            model.wv.save_word2vec_format(outfile + '.model.txt', binary=False)

    if args.accuracy:
        questions_file = args.accuracy
        model.accuracy(questions_file)

    logger.info("finished running %s", os.path.basename(sys.argv[0]))
