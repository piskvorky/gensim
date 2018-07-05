from __future__ import unicode_literals
from __future__ import print_function

import logging
import argparse
import time
import os

from gensim.models.word2vec import LineSentence


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)


def do_benchmark(input_):
    iter = LineSentence(input_)

    start_time = time.time()
    for _ in iter:
        pass
    end_time = time.time()

    logger.info('Finished benchmarking. Time elapsed: {:.2f} s.'.format(end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GSOC Multistream-API: evaluate performance '
                                                 'metrics for any2vec models')
    parser.add_argument('--input', type=str, help='Input file or regexp if `multistream` mode is on.')

    args = parser.parse_args()

    input_ = os.path.expanduser(args.input)

    do_benchmark(input_)
