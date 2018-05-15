from __future__ import unicode_literals
from __future__ import print_function

import logging
import argparse
import json
import copy

from gensim.models.base_any2vec import PERFORMANCE_METRICS
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import LineSentence


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)


SUPPORTED_MODELS = {
    'fasttext': FastText,
    'word2vec': Word2Vec,
    'doc2vec': Doc2Vec
}


def benchmark_model(input, model, window, workers, vector_size):
    if model == 'doc2vec':
        kwargs = {
            'documents': LineSentence(input)
        }
    else:
        kwargs = {
            'sentences': LineSentence(input)
        }

    kwargs['size'] = vector_size
    kwargs['window'] = window
    kwargs['workers'] = workers
    kwargs['iter'] = 1

    logger.info('Creating model with kwargs={}'.format(kwargs))

    # Training model for 1 epoch.
    SUPPORTED_MODELS[model](**kwargs)

    return copy.deepcopy(PERFORMANCE_METRICS)


def do_benchmarks(input, models_grid, vector_size, workers_grid, windows_grid, label):
    report = {}

    for model in models_grid:
        for window in windows_grid:
            for workers in workers_grid:
                model_str = '{}-{}-window{}-workers{}-size{}'.format(label, model, window, workers, vector_size)

                logger.info('Start benchmarking {}.'.format(model_str))
                results = benchmark_model(input, model, window, workers, vector_size)

                logger.info('--- MODEL {} RESULTS ---'.format(model_str).center(30))
                logger.info('* Total time: {} sec.'.format(results['total_time']))
                logger.info('* Avg queue size: {} elems.'.format(results['queue_size']))
                logger.info('* Processing speed: {} words/sec'.format(results['words_sec']))
                logger.info('* Avg CPU loads: {}'.format(results['cpu_load']))
                
                report[model_str] = results

    fout_name = '{}-results.json'.format(label)
    with open(fout_name, 'w') as fout:
        json.dump(report, fout)

    logger.info('Saved metrics report to {}.'.format(fout_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GSOC Multistream-API: evaluate performance '
                                                 'metrics for any2vec models')
    parser.add_argument('--input', type=str)
    parser.add_argument('--models-grid', nargs='+', type=str, default=SUPPORTED_MODELS.keys())
    parser.add_argument('--size', type=int, default=300)
    parser.add_argument('--workers-grid', nargs='+', type=int, default=[1, 4, 8, 10, 12, 14])
    parser.add_argument('--windows-grid', nargs='+', type=int, default=[10])
    parser.add_argument('--label', type=str, default='untitled')

    args = parser.parse_args()

    do_benchmarks(args.input, args.models_grid, args.size, args.workers_grid, args.windows_grid, args.label)
