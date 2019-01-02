#
# Triggers the bug.  Delete this file when done with issue 2160.
#
import sys


def train_gensim():
    #
    # The imports cause logs to be written, so we do them here.
    #
    from gensim.models.fasttext import FastText as FT_gensim
    from gensim.test.utils import datapath, common_texts as sentences

    path = datapath('toy-data.txt')
    with open(path) as fin:
        words = fin.read().strip().split(' ')

    sent = [words]

    model = FT_gensim(bucket=100)
    model.build_vocab(sent)
    model.train(sent, total_examples=len(sent), epochs=model.epochs)
    return model


def load_native():
    from gensim.models.fasttext import FastText as FT_gensim
    from gensim.test.utils import datapath, common_texts as sentences

    path = datapath('toy-model.bin')
    model = FT_gensim.load_fasttext_format(path)
    model.build_vocab(sentences, update=True)  # this doesn't work, but should. See also https://github.com/RaRe-Technologies/gensim/issues/2139
    model.train(sentences, total_examples=len(sentences), epochs=model.epochs)
    return model


def main():
    print('<log>')
    if len(sys.argv) == 2 and sys.argv[1] == 'native':
        model = load_native()
    else:
        model = train_gensim()
    print('</log>')
    return model


if __name__ == '__main__':
    model = main()
