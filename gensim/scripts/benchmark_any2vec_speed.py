from __future__ import unicode_literals
from __future__ import print_function

import logging
import argparse
import json
import copy
# import yappi
import os
import glob

from gensim.models import base_any2vec
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedLineDocument
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec_inner import CythonLineSentence


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)


SUPPORTED_MODELS = {
    'fasttext': FastText,
    'word2vec': Word2Vec,
    'doc2vec': Doc2Vec,
}


def print_results(model_str, results):
    logger.info('----- MODEL "{}" RESULTS -----'.format(model_str).center(50))
    logger.info('\t* Vocab time: {} sec.'.format(results['vocab_time']))
    logger.info('\t* Total epoch time: {} sec.'.format(results['total_time']))
    # logger.info('\t* Avg queue size: {} elems.'.format(results['queue_size']))
    logger.info('\t* Processing speed: {} words/sec'.format(results['words_sec']))
    logger.info('\t* Avg CPU loads: {}'.format(results['cpu_load']))
    logger.info('\t* Sum CPU load: {}'.format(results['cpu_load_sum']))


def benchmark_model(input_streams, model, window, workers, vector_size):
    if model == 'doc2vec':
        kwargs = {
            'input_streams': [TaggedLineDocument(inp) for inp in input_streams]
        }
    else:
        kwargs = {
            'input_streams': [LineSentence(inp) for inp in input_streams]
        }

    kwargs['size'] = vector_size

    if model != 'sent2vec':
        kwargs['window'] = window

    kwargs['workers'] = workers
    kwargs['iter'] = 1

    logger.info('Creating model with kwargs={}'.format(kwargs))

    # Training model for 1 epoch.
    # yappi.start()
    SUPPORTED_MODELS[model](**kwargs)
    # yappi.get_func_stats().print_all()
    # yappi.get_thread_stats().print_all()

    return copy.deepcopy(base_any2vec.PERFORMANCE_METRICS)


def do_benchmarks(input_streams, models_grid, vector_size, workers_grid, windows_grid, label):
    full_report = {}

    for model in models_grid:
        for window in windows_grid:
            for workers in workers_grid:
                model_str = '{}-{}-window-{:02d}-workers-{:02d}-size-{}'.format(label, model, window, workers, vector_size)

                logger.info('Start benchmarking {}.'.format(model_str))
                results = benchmark_model(input_streams, model, window, workers, vector_size)

                print_results(model_str, results)

                full_report[model_str] = results

    logger.info('Benchmarking completed. Here are the results:')
    for model_str in sorted(full_report.keys()):
        print_results(model_str, full_report[model_str])

    fout_name = '{}-report.json'.format(label)
    with open(fout_name, 'w') as fout:
        json.dump(full_report, fout)

    logger.info('Saved metrics report to {}.'.format(fout_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GSOC Multistream-API: evaluate performance '
                                                 'metrics for any2vec models')
    parser.add_argument('--input', type=str, help='Input file or regexp if `multistream` mode is on.')
    parser.add_argument('--models-grid', nargs='+', type=str, default=SUPPORTED_MODELS.keys())
    parser.add_argument('--size', type=int, default=300)
    parser.add_argument('--workers-grid', nargs='+', type=int, default=[1, 4, 8, 10, 12, 14])
    parser.add_argument('--windows-grid', nargs='+', type=int, default=[10])
    parser.add_argument('--label', type=str, default='untitled')

    args = parser.parse_args()

    input_ = os.path.expanduser(args.input)
    input_streams = glob.glob(input_)
    logger.info('Glob found {} input streams. List: {}'.format(len(input_streams), input_streams))

    do_benchmarks(input_streams, args.models_grid, args.size, args.workers_grid, args.windows_grid, args.label)
