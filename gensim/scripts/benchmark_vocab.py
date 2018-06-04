from __future__ import unicode_literals
from __future__ import print_function

import logging
import argparse
import time
import os
import glob
import itertools

from gensim.models.word2vec import Word2Vec, LineSentence
from gensim.models.doc2vec import Doc2Vec, TaggedLineDocument
from gensim.models.fasttext import FastText


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GSOC Multistream-API: evaluate vocab performance '
                                                 'for word2vec')
    parser.add_argument('--input', type=str, help='Input file or regexp for multistream.')
    parser.add_argument('--size', type=int, default=300)
    parser.add_argument('--workers-grid', nargs='+', type=int, default=[1, 2, 3, 4, 5, 8, 10])
    parser.add_argument('--model', type=str, default='word2vec')
    parser.add_argument('--label', type=str, default='untitled')

    args = parser.parse_args()

    input_ = os.path.expanduser(args.input)
    input_files = glob.glob(input_)
    logger.info('Glob found {} input files. List: {}'.format(len(input_files), input_files))

    for workers in args.workers_grid:
        if args.model == 'word2vec':
            input_streams = [LineSentence(_) for _ in input_files]
            model = Word2Vec()
        elif args.model == 'doc2vec':
            input_streams = [TaggedLineDocument(_) for _ in input_files]
            model = Doc2Vec()
        elif args.model == 'fasttext':
            input_streams = [LineSentence(_) for _ in input_files]
            model = FastText()
        else:
            raise NotImplementedError("Model '{}' is not supported", args.model)

        if workers == 1:
            multistream = False
            input_streams = itertools.chain(*input_streams)
        else:
            multistream = True

        logger.info('Start building vocab with model={}, workers={}, multistream={}'.format(args.model, workers, multistream))
        start_time = time.time()
        model.build_vocab(input_streams, workers=workers, multistream=multistream)
        end_time = time.time()
        logger.info('Model = {}\tWorkers = {}\tVocab time = {:.2f} secs'.format(args.model, workers, end_time - start_time))