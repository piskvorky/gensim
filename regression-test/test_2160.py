#
# Regression test for continue training on native data
#
# https://github.com/RaRe-Technologies/gensim/issues/2160
#
import os.path as P
from gensim.models import FastText
from gensim.test.utils import common_texts

curr_dir = P.dirname(P.abspath(__file__))


def train_gensim():
    path = P.join(curr_dir, 'toy-data.txt')
    with open(path) as fin:
        words = fin.read().strip().split(' ')

    model = FastText()
    model.build_vocab(words)
    model.train(words, total_examples=len(words), epochs=model.epochs)
    return model


def load_native():
    path = P.join(curr_dir, 'toy-model.bin')
    model = FastText.load_fasttext_format(path)
    # model.build_vocab(common_texts, update=True)  # this doesn't work, but should. See also https://github.com/RaRe-Technologies/gensim/issues/2139
    return model


trained = train_gensim()
native = load_native()
