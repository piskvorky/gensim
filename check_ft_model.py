import sys
from gensim.models import FastText
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    model_bin_path = sys.argv[1]
    word_to_predict = sys.argv[2]

    print(word_to_predict, type(word_to_predict))
    model = FastText.load_fasttext_format(model_bin_path)
    print(model[word_to_predict])