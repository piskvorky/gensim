from __future__ import unicode_literals
from __future__ import print_function

import logging
import numpy as np
import sys

from gensim.models.word2vec import Text8Corpus, Word2Vec
from gensim.corpora import WikiCorpus
from gensim.models.callbacks import CallbackAny2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def print_accuracy(model, questions_file, num_bits=0):
    print('Evaluating...\n')
    orig_vectors = np.copy(model.wv.vectors)
    model.wv.quantize_vectors(num_bits=num_bits)
    model.init_sims(replace=True)

    acc = model.accuracy(questions_file)

    sem_correct = sum((len(acc[i]['correct']) for i in range(5)))
    sem_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5))
    sem_acc = 100 * float(sem_correct) / sem_total
    print('\nSemantic: {:d}/{:d}, Accuracy: {:.2f}%'.format(sem_correct, sem_total, sem_acc))

    syn_correct = sum((len(acc[i]['correct']) for i in range(5, len(acc) - 1)))
    syn_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5, len(acc) - 1))
    syn_acc = 100 * float(syn_correct) / syn_total
    print('Syntactic: {:d}/{:d}, Accuracy: {:.2f}%\n'.format(syn_correct, syn_total, syn_acc))

    model.wv.vectors = orig_vectors
    model.init_sims(replace=True)


class EpochSaver(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        str_name = 'enwiki_w2v_{}_{}epoch_{}bit.w2v'.format(model.vector_size, self.epoch, model.num_bits)
        model.save(str_name)
        self.epoch += 1


class EpochLogger(CallbackAny2Vec):
    def __init__(self, questions_path):
        self.epoch = 0
        self.questions_path = questions_path

    def on_epoch_end(self, model):
        print("Epoch #{} end.".format(self.epoch))
        print("\t---No quantization during training, accuracies---\t")
        print("No quantization:")
        print_accuracy(model, questions_path, num_bits=0)
        print("Quantized 1bit:")
        print_accuracy(model, questions_path, num_bits=1)
        print("Quantized 2bits:")
        print_accuracy(model, questions_path, num_bits=2)
        self.epoch += 1


class MyCorpus(object):
    def __init__(self, *args, **kwargs):
        self._corpus = WikiCorpus(*args, **kwargs)

    def __iter__(self):
        for text in self._corpus.get_texts():
            yield text


if 'wiki' in sys.argv[1]:
    corpus = MyCorpus(sys.argv[1], lemmatize=False, lower=False)
else:
    corpus = Text8Corpus(sys.argv[1])
questions_path = './docs/notebooks/datasets/questions-words.txt'


corpus = Text8Corpus(sys.argv[1])
questions_path = './docs/notebooks/datasets/questions-words.txt'

params = {
    'sentences': corpus,
    'iter': 5,
    'sg': 0,
    'size': 400,
    'workers': 8,
    'alpha': 0.05,
    'min_alpha': 0.0001,
    'window': 5,
    'min_count': 5,
    'negative': 12,
    'sample': 1e-4,
}

model = Word2Vec(
    num_bits=0,
    **params
)

model = Word2Vec(
    num_bits=1,
    **params
)

model = Word2Vec(
    num_bits=2,
    **params
)