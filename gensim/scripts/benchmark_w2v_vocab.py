from __future__ import unicode_literals
from __future__ import print_function

import logging
import argparse
# import yappi
import os
import glob

from gensim.models import base_any2vec
from gensim.models.word2vec import Word2Vec, LineSentence


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GSOC Multistream-API: evaluate vocab performance '
                                                 'for word2vec')
    parser.add_argument('--input', type=str, help='Input file or regexp for multistream.')
    parser.add_argument('--size', type=int, default=300)
    parser.add_argument('--workers-grid', nargs='+', type=int, default=[1, 2, 3, 4, 5, 8, 10, 12, 14])
    parser.add_argument('--label', type=str, default='untitled')

    args = parser.parse_args()

    input_ = os.path.expanduser(args.input)
    input_streams = glob.glob(input_)
    logger.info('Glob found {} input streams. List: {}'.format(len(input_streams), input_streams))

    input_streams = [LineSentence(_) for _ in input_streams]
    for workers in args.workers_grid:
        model = Word2Vec()
        model.build_vocab(input_streams, workers=workers)
        logger.info('Workers = {}\tVocab time = {:.2f} secs'.format(workers,
                                                                    base_any2vec.PERFORMANCE_METRICS['vocab_time']))
